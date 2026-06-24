"""Quiz augmentation service using LiteLLM (OpenAI-compatible) API."""

import json
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.exceptions import OCRProcessingError
from app.core.logging import app_logger
from app.services.litellm_params import build_litellm_extra_params


class QuizAugmentService:
    """Service for paraphrasing and augmenting quiz questions."""

    def __init__(self):
        self.settings = get_settings()
        if not (self.settings.litellm_api_key or "").strip():
            raise OCRProcessingError(
                "LITELLM_API_KEY is required for quiz augmentation"
            )
        headers: Dict[str, str] = {}
        if self.settings.litellm_site_url:
            headers["HTTP-Referer"] = self.settings.litellm_site_url
        if self.settings.litellm_site_name:
            headers["X-Title"] = self.settings.litellm_site_name
        self.async_client = AsyncOpenAI(
            base_url=self.settings.litellm_base_url,
            api_key=self.settings.litellm_api_key,
            default_headers=headers or None,
        )

    async def augment_questions(
        self,
        questions: List[Dict[str, Any]],
        language: str = "th",
        num_questions: Optional[int] = None,
        num_sets: Optional[int] = None,
        mode: str = "transform",
        classify_topic_tag: bool = False,
        course_topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Paraphrase and augment questions with correct answer and explanation."""
        prompt = self._build_prompt(
            questions,
            language,
            num_questions,
            num_sets,
            mode,
            classify_topic_tag=classify_topic_tag,
            course_topics=course_topics or [],
        )
        app_logger.info("Calling LiteLLM for quiz augmentation")
        extra_params = build_litellm_extra_params(
            self.settings.litellm_model,
            reasoning_effort=self.settings.litellm_reasoning_effort,
        )
        response = await self.async_client.chat.completions.create(
            model=self.settings.litellm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            **extra_params,
        )
        content = response.choices[0].message.content if response.choices else ""
        if not content:
            raise OCRProcessingError("Empty response from LiteLLM API")

        parsed = self._extract_json(content)
        if not isinstance(parsed, dict):
            raise OCRProcessingError("Invalid augmentation response format")
        has_sets = isinstance(parsed.get("sets"), list)
        has_questions = isinstance(parsed.get("questions"), list)
        if not (has_sets or has_questions):
            raise OCRProcessingError("Invalid augmentation response format")
        if has_questions and not has_sets:
            parsed["sets"] = [{"questions": parsed["questions"]}]
        return parsed

    def _build_prompt(
        self,
        questions: List[Dict[str, Any]],
        language: str,
        num_questions: Optional[int],
        num_sets: Optional[int],
        mode: str,
        classify_topic_tag: bool = False,
        course_topics: Optional[List[str]] = None,
    ) -> str:
        target_count = (
            num_questions if num_questions and num_questions > 0 else len(questions)
        )
        target_sets = num_sets if num_sets and num_sets > 0 else 1
        if mode == "solve":
            normalized_mode = "solve"
        elif mode == "image_filter":
            normalized_mode = "image_filter"
        elif mode == "deconstruct":
            normalized_mode = "deconstruct"
        else:
            normalized_mode = "transform"
        allowed_topics = [
            str(item).strip() for item in (course_topics or []) if str(item).strip()
        ]
        use_topic_classification = (
            normalized_mode == "solve"
            and classify_topic_tag
            and len(allowed_topics) > 0
        )
        if normalized_mode == "transform":
            task_block = (
                "You are an expert exam author. Transform each question into a NEW question, "
                "not a paraphrase. Change the wording, numbers, and choices so it feels new, "
                "but keep the same skill/intent being tested. "
                "You may change the correct answer if needed. "
                "Add the correct answer index plus a short explanation written in Thai. "
                "Keep the same number of choices per question."
            )
        elif normalized_mode == "solve":
            task_block = (
                "You are an expert tutor. Keep each question and choices semantically the same as the input. "
                "Do not rewrite or invent new questions. Infer the best correct answer index and provide "
                "a short student-friendly explanation written in Thai."
            )
        elif normalized_mode == "image_filter":
            task_block = (
                "You are an exam question reviewer. For each input question, classify whether solving it "
                "requires visual information from an external image/diagram/graph/table that is NOT fully "
                "contained in the question text and choices."
            )
        else:
            task_block = (
                "You are an expert Educational Architect and Assessment Designer. "
                "Your task is to deconstruct each test question into its core academic skeleton."
            )
        topic_block = ""
        topic_field = ""
        topic_required = ""
        if use_topic_classification:
            topic_field = ',\n          "topic_tag": "one topic from allowed list"'
            topic_required = ', "topic_tag"'
            topic_block = (
                "\n\nTopic classification rules:\n"
                "1. For each question, assign exactly ONE topic_tag from the allowed list.\n"
                "2. Never invent a new topic_tag outside the allowed list.\n"
                "3. Choose the closest topic by mathematical skill tested in the question.\n"
                f"4. Allowed topic_tag list: {json.dumps(allowed_topics, ensure_ascii=False)}\n"
            )
        if normalized_mode == "image_filter":
            return (
                f"{task_block} Use the same language as the input "
                f"(language hint: {language}).\n\n"
                "Classification rules:\n"
                "1. requires_image = true when the question depends on an unseen figure, graph, table, chart, geometry drawing, or photo.\n"
                "2. requires_image = false when all required information is already present in text and choices.\n"
                "3. If uncertain, choose true (conservative filtering).\n"
                "4. Preserve question and choices exactly as input (only trim surrounding whitespace).\n\n"
                "Return ONLY valid JSON in this format:\n"
                "{\n"
                '  "sets": [\n'
                "    {\n"
                '      "questions": [\n'
                "        {\n"
                '          "question": "original question text",\n'
                '          "choices": ["choice 1", "choice 2", "choice 3", "choice 4"],\n'
                '          "requires_image": false,\n'
                '          "reason": "short reason"\n'
                "        }\n"
                "      ]\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                f"Return exactly {target_sets} sets.\n"
                f"Each set must contain exactly {target_count} questions.\n"
                "When returning each question object, include: question, choices, requires_image, reason.\n\n"
                f"Input questions JSON:\n{json.dumps(questions, ensure_ascii=False)}"
            )

        if normalized_mode == "deconstruct":
            deconstruct_count = len(questions)
            return (
                f"{task_block} Use the same language as the input "
                f"(language hint: {language}).\n\n"
                "CRITICAL RULES:\n"
                "1. Strip away all creative expressions, character names, specific scenarios, specific locations, and original narratives.\n"
                "2. Extract ONLY the underlying logic, scientific or mathematical principles, abstract variables, and cognitive skills being tested.\n"
                "3. Do not keep concrete numbers from the source as-is; convert them into abstract symbols/placeholders "
                "(for example a, b, n, k, min, max, rate, count).\n"
                "4. Do not quote, paraphrase closely, or preserve unique wording from the source question.\n"
                "5. If the source includes choices or solutions, use them only to infer distractor logic and academic structure.\n\n"
                "6. If an input question has a non-empty context field, extract a brief abstract context_guidance value "
                "that describes what kind of supporting context a new question should provide. Do NOT copy, quote, "
                "or closely paraphrase the raw context. Describe it generically, such as 'short reading passage about ...', "
                "'dialogue between speakers about ...', 'table of values showing ...', or 'shared instruction defining ...'.\n"
                "7. If an input question has no meaningful context, omit context_guidance from that extracted skeleton object.\n\n"
                "Return ONLY valid JSON in this format:\n"
                "{\n"
                '  "sets": [\n'
                "    {\n"
                '      "questions": [\n'
                "        {\n"
                '          "subject": "broad subject area",\n'
                '          "topic_tags": ["specific", "sub-topics"],\n'
                '          "learning_objective": "specific academic skill, rule, or knowledge being tested",\n'
                '          "core_logic_and_formulas": "step-by-step logical process or equations required to solve it",\n'
                '          "context_guidance": "optional abstract description of supporting context needed for this question; omit when none",\n'
                '          "variables": {\n'
                '            "given": ["abstract input variable 1", "abstract input variable 2"],\n'
                '            "target": "abstract output being requested"\n'
                "          },\n"
                '          "constraints_and_tricks": ["hidden trap, unit conversion, edge case, or exception"],\n'
                '          "distractor_logic": ["plausible wrong-answer logic students may follow"]\n'
                "        }\n"
                "      ]\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Ensure the JSON is valid and contains no markdown formatting outside the JSON block.\n"
                f"Return exactly 1 set with exactly {deconstruct_count} extracted skeleton objects.\n"
                "Each extracted object must include: subject, topic_tags, learning_objective, core_logic_and_formulas, "
                "variables, constraints_and_tricks, distractor_logic. Include context_guidance only when the input question has meaningful context.\n\n"
                f"Input questions JSON:\n{json.dumps(questions, ensure_ascii=False)}"
            )

        return (
            f"{task_block} Use the same language as the input "
            f"for question and choices (language hint: {language}). "
            "The explanation field must always be Thai.\n\n"
            "Return ONLY valid JSON in this format:\n"
            "{\n"
            '  "sets": [\n'
            "    {\n"
            '      "questions": [\n'
            "        {\n"
            '          "question": "new question text",\n'
            '          "choices": ["choice 1", "choice 2", "choice 3", "choice 4"],\n'
            '          "correct_answer": 0,\n'
            '          "explanation": "short Thai explanation"'
            f"{topic_field}\n"
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Return exactly {target_sets} sets.\n"
            f"Each set must contain exactly {target_count} questions.\n"
            "If input has more, select the most representative. "
            "If input has fewer, create additional variants.\n"
            "If you are unsure about the correct answer, choose the best option "
            "and explain briefly.\n"
            f"{topic_block}\n"
            f"When returning each question object, include: question, choices, correct_answer, explanation{topic_required}.\n\n"
            f"Input questions JSON:\n{json.dumps(questions, ensure_ascii=False)}"
        )

    def _extract_json(self, content: str) -> Dict[str, Any]:
        text = content.strip()
        # Strip code fences if present
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: extract first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise OCRProcessingError("Failed to parse JSON from LiteLLM response")
