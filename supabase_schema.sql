-- Supabase schema for the Tanaijarn backend.
-- Run this in the Supabase SQL editor or through the Supabase CLI before
-- starting the migrated FastAPI backend.

create table if not exists public.profiles (
  user_id text primary key,
  email text,
  username text,
  name text,
  role text not null default 'student',
  status text not null default 'active',
  given_name text,
  family_name text,
  student_id text,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  data jsonb not null default '{}'::jsonb
);

create table if not exists public.courses (
  course_id text primary key,
  user_id text,
  instructor_id text,
  name text,
  title text,
  category text,
  status text not null default 'active',
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  data jsonb not null default '{}'::jsonb
);

create table if not exists public.lessons (
  lesson_id text primary key,
  course_id text not null,
  user_id text,
  status text not null default 'active',
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  data jsonb not null default '{}'::jsonb
);

create table if not exists public.quizzes (
  quiz_id text primary key,
  course_id text,
  user_id text,
  lesson_id text,
  document_id text,
  status text not null default 'active',
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  data jsonb not null default '{}'::jsonb
);

create table if not exists public.question_bank_items (
  item_id text primary key,
  course_id text not null,
  user_id text,
  source text,
  status text not null default 'active',
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  data jsonb not null default '{}'::jsonb
);

create table if not exists public.enrollments (
  enrollment_id text primary key,
  user_id text not null,
  course_id text not null,
  status text not null default 'active',
  enrolled_at timestamptz,
  expires_at timestamptz,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  data jsonb not null default '{}'::jsonb
);

create table if not exists public.quiz_results (
  result_id text primary key,
  user_id text not null,
  quiz_id text not null,
  course_id text,
  submitted_at timestamptz,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  data jsonb not null default '{}'::jsonb
);

create table if not exists public.files (
  file_id text primary key,
  user_id text,
  storage_key text,
  content_type text,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  data jsonb not null default '{}'::jsonb
);

create table if not exists public.chat_messages (
  message_id text primary key,
  conversation_id text,
  user_id text,
  role text,
  created_at timestamptz default now(),
  data jsonb not null default '{}'::jsonb
);

create table if not exists public.invitations (
  invitation_id text primary key,
  course_id text,
  instructor_id text,
  student_id text,
  status text not null default 'pending',
  created_at timestamptz default now(),
  expires_at timestamptz,
  data jsonb not null default '{}'::jsonb
);

create table if not exists public.platform_config (
  config_key text primary key,
  updated_at timestamptz default now(),
  data jsonb not null default '{}'::jsonb
);

create index if not exists idx_profiles_email on public.profiles (email);
create index if not exists idx_profiles_username on public.profiles (username);
create index if not exists idx_profiles_student_id on public.profiles (student_id);
create index if not exists idx_profiles_role on public.profiles (role);
create index if not exists idx_courses_user_id on public.courses (user_id);
create index if not exists idx_courses_instructor_id on public.courses (instructor_id);
create index if not exists idx_lessons_course_id on public.lessons (course_id);
create index if not exists idx_lessons_course_status_created on public.lessons (course_id, status, created_at);
create index if not exists idx_quizzes_course_id on public.quizzes (course_id);
create index if not exists idx_quizzes_course_status_created on public.quizzes (course_id, status, created_at desc);
create index if not exists idx_quizzes_user_course_status_created on public.quizzes (user_id, course_id, status, created_at desc);
create index if not exists idx_quizzes_document_id on public.quizzes (document_id);
create index if not exists idx_question_bank_course_updated on public.question_bank_items (course_id, updated_at desc);
create index if not exists idx_question_bank_course_status_updated on public.question_bank_items (course_id, status, updated_at desc);
create index if not exists idx_question_bank_user_id on public.question_bank_items (user_id);
create index if not exists idx_enrollments_user_id on public.enrollments (user_id);
create index if not exists idx_enrollments_user_status_enrolled on public.enrollments (user_id, status, enrolled_at desc);
create index if not exists idx_enrollments_course_id on public.enrollments (course_id);
create index if not exists idx_enrollments_course_status_enrolled on public.enrollments (course_id, status, enrolled_at desc);
create index if not exists idx_quiz_results_user_id on public.quiz_results (user_id);
create index if not exists idx_quiz_results_user_course_submitted on public.quiz_results (user_id, course_id, submitted_at desc);
create index if not exists idx_quiz_results_course_quiz_submitted on public.quiz_results (course_id, quiz_id, submitted_at desc);
create index if not exists idx_files_user_id on public.files (user_id);

-- Performance indexes for read-heavy student/tutor overview paths.
create index if not exists idx_courses_owner_active_created
  on public.courses (user_id, status, created_at desc)
  where status <> 'deleted';
create index if not exists idx_courses_instructor_active_created
  on public.courses (instructor_id, status, created_at desc)
  where status <> 'deleted';
create index if not exists idx_enrollments_user_course_active
  on public.enrollments (user_id, course_id, enrolled_at desc)
  where status in ('active', 'trial', 'paid');
create index if not exists idx_enrollments_course_user_active
  on public.enrollments (course_id, user_id, enrolled_at desc)
  where status <> 'deleted';
create index if not exists idx_quiz_results_user_course_quiz_submitted
  on public.quiz_results (user_id, course_id, quiz_id, submitted_at desc);
create index if not exists idx_quiz_results_course_user_submitted
  on public.quiz_results (course_id, user_id, submitted_at desc);
create index if not exists idx_question_bank_course_active_updated
  on public.question_bank_items (course_id, status, updated_at desc)
  where status <> 'deleted';

alter table public.profiles enable row level security;
alter table public.courses enable row level security;
alter table public.lessons enable row level security;
alter table public.quizzes enable row level security;
alter table public.question_bank_items enable row level security;
alter table public.enrollments enable row level security;
alter table public.quiz_results enable row level security;
alter table public.files enable row level security;
alter table public.chat_messages enable row level security;
alter table public.invitations enable row level security;
alter table public.platform_config enable row level security;

-- The FastAPI backend uses SUPABASE_SERVICE_ROLE_KEY and bypasses RLS.
-- Add user-facing anon policies later only for tables accessed directly by frontend.

-- Public buckets used for images rendered directly by the web apps.
insert into storage.buckets (
  id,
  name,
  public,
  file_size_limit,
  allowed_mime_types
)
values
  (
    'tiwmai-course-images',
    'tiwmai-course-images',
    true,
    10485760,
    array['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp', 'image/tiff']
  ),
  (
    'tiwmai-avatars',
    'tiwmai-avatars',
    true,
    1048576,
    array['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp', 'image/tiff']
  )
on conflict (id) do update
set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;
