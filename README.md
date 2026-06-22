# Tiwmai Student API

FastAPI service for the student web app. This repo is scoped to student-facing
routes for the Vercel migration round.

## Scope

Exposed route groups:

- `GET /api/v1/health`
- `/api/v1/student/auth/*`
- `/api/v1/student/courses*`
- `/api/v1/student/lessons/*`
- `/api/v1/student/quizzes*`
- `/api/v1/student/users/*`
- `/api/v1/student/payments/*`
- `/api/v1/student/chat*`
- `/api/v1/student/legal/*`

Not exposed in this deployment:

- tutor/admin APIs
- OCR upload/process APIs
- quiz generation jobs
- queue endpoints
- Prometheus `/metrics`

## Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs:

- `http://localhost:8000/api/docs`
- `http://localhost:8000/api/redoc`
- `http://localhost:8000/api/openapi.json`

## Environment

Required for production:

```env
SECRET_KEY=
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_JWT_SECRET=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=
ALLOWED_ORIGINS=https://your-student-web.vercel.app
STUDENT_WEB_APP_URL=https://your-student-web.vercel.app
```

OpenRouter uses the OpenAI-compatible API:

```env
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=https://your-student-web.vercel.app
OPENROUTER_SITE_NAME=Tiwmai
```

## Vercel

This repo includes:

- `api/index.py`
- `vercel.json`

Deploy it as a Vercel Python project, then point the student frontend env var to:

```env
REACT_APP_API_BASE_URL=https://your-tiwmai-api.vercel.app/api/v1
```

## Verification

```bash
pytest
```

The route-surface tests assert that admin/tutor/OCR/job/metrics routes are not
registered.
