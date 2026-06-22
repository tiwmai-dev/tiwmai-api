# AGENTS.md

## Repository Scope

This repository is the Tiwmai student backend API service.

Expected stack:
- FastAPI application under `app/`
- API routes under `app/api/`
- Business logic under `app/services/`
- Pydantic schemas under `app/models/`
- Tests under `tests/`

Keep this repository focused on the student web API: student auth, course
catalog/learning data, enrollment, payments, student chat, student legal pages,
and student profile storage. Do not add tutor/admin APIs, OCR/document
processing, quiz generation jobs, worker queues, monitoring stacks, or frontend
application code here.

## Architecture Rules

- This migration round is student-only. Add tutor/admin APIs in their own repos
  or a separately planned backend boundary, not as unused placeholders here.
- Keep public API routes under `/api/v1`.
- Keep student routes under `/api/v1/student`.
- Put request/response shapes in Pydantic schemas instead of returning ad hoc
  dictionaries from route handlers when behavior is shared or user-facing.
- Keep route handlers thin. Put durable business logic in `app/services/`.
- Do not move backend secrets, service-role keys, payment keys, or provider keys
  into frontend repos.

## Environment And Secrets

- Never commit `.env` files or real credentials.
- Keep `.env.example` updated when adding or renaming environment variables.
- Treat these as sensitive: Supabase service role keys, JWT secrets, Stripe
  private keys and webhook secrets, OpenRouter keys, Sentry DSNs when configured
  with PII, and Resend keys.
- Frontend apps should call the API through an environment-configured API base
  URL. Do not hardcode production domains into backend logic unless the setting
  is explicitly a backend callback or redirect URL.

## Local Development

Common setup:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Alternative:

```bash
python -m app.main
```

API docs are expected at:
- `http://localhost:8000/api/docs`
- `http://localhost:8000/api/redoc`
- `http://localhost:8000/api/openapi.json`

## Verification

For backend code changes, run the narrowest relevant tests first, then broaden
when shared behavior changes.

Useful commands:

```bash
pytest
pytest tests/test_api.py
pytest tests/test_supabase_service.py
pytest --cov=app tests/
```

For formatting and static checks when the tooling is installed:

```bash
black app/ tests/
isort app/ tests/
flake8 app/ tests/
mypy app/
```

If tests cannot be run because local services or credentials are missing,
report exactly what was not run and why.

## Change Discipline

- Keep changes scoped to the requested backend behavior.
- Add or update tests when changing student auth, payments, course access,
  enrollment, chat energy, or API response contracts.
- Preserve backward compatibility for existing frontend callers unless the task
  explicitly includes a coordinated frontend change.
- When changing API responses, search frontend callers before finalizing.
- Keep route registrations in `app/api/student_endpoints.py` and student handler
  behavior in `app/api/student_handlers.py`.
- Prefer small migration-safe data changes over destructive rewrites.

## Deployment Boundary

- This repo is the API service and should deploy separately from Vercel
  frontend projects.
- Vercel frontend repos should use `REACT_APP_API_BASE_URL` or
  `VITE_API_BASE_URL` to reach this API.
- Keep CORS settings explicit through `ALLOWED_ORIGINS`.
- Do not rely on Create React App or Vite dev proxy behavior in production.
