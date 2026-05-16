PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    board_type TEXT NOT NULL CHECK (board_type IN ('CBSE', 'ICSE', 'STATE', 'OTHER')),
    state TEXT,
    district TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS teachers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    school_id INTEGER NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    google_account_email TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (school_id) REFERENCES schools (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    school_id INTEGER NOT NULL,
    teacher_id INTEGER NOT NULL,
    academic_year TEXT NOT NULL,
    grade TEXT NOT NULL,
    section TEXT NOT NULL,
    subject TEXT NOT NULL,
    medium TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (school_id) REFERENCES schools (id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES teachers (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS class_subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL,
    subject TEXT NOT NULL,
    medium TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
    UNIQUE (class_id, subject)
);

CREATE TABLE IF NOT EXISTS curriculum_subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    board_type TEXT NOT NULL CHECK (board_type IN ('CBSE', 'ICSE', 'STATE', 'OTHER')),
    grade TEXT NOT NULL,
    subject TEXT NOT NULL,
    default_medium TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (board_type, grade, subject)
);

CREATE TABLE IF NOT EXISTS curriculum_chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    curriculum_subject_id INTEGER NOT NULL,
    chapter_code TEXT NOT NULL,
    chapter_name TEXT NOT NULL,
    chapter_order INTEGER,
    term TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (curriculum_subject_id) REFERENCES curriculum_subjects (id) ON DELETE CASCADE,
    UNIQUE (curriculum_subject_id, chapter_code)
);

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    school_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    roll_number TEXT NOT NULL,
    full_name TEXT NOT NULL,
    email TEXT,
    gender TEXT,
    date_of_birth TEXT,
    guardian_name TEXT,
    guardian_phone TEXT,
    preferred_language TEXT,
    accessibility_notes TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'transferred')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (school_id) REFERENCES schools (id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
    UNIQUE (class_id, roll_number)
);

CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    board_type TEXT NOT NULL CHECK (board_type IN ('CBSE', 'ICSE', 'STATE', 'OTHER')),
    grade TEXT NOT NULL,
    subject TEXT NOT NULL,
    chapter_code TEXT NOT NULL,
    chapter_name TEXT NOT NULL,
    term TEXT,
    UNIQUE (board_type, grade, subject, chapter_code)
);

CREATE TABLE IF NOT EXISTS concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id INTEGER NOT NULL,
    concept_code TEXT NOT NULL,
    concept_name TEXT NOT NULL,
    description TEXT,
    difficulty_level TEXT,
    FOREIGN KEY (chapter_id) REFERENCES chapters (id) ON DELETE CASCADE,
    UNIQUE (chapter_id, concept_code)
);

CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL,
    chapter_id INTEGER NOT NULL,
    teacher_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    assessment_type TEXT NOT NULL CHECK (assessment_type IN ('practice', 'class_test', 'remedial', 'homework')),
    delivery_mode TEXT NOT NULL CHECK (delivery_mode IN ('google_form', 'local', 'manual')),
    google_form_id TEXT,
    google_form_url TEXT,
    language TEXT,
    total_marks REAL DEFAULT 0,
    assigned_at TEXT,
    due_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
    FOREIGN KEY (chapter_id) REFERENCES chapters (id) ON DELETE RESTRICT,
    FOREIGN KEY (teacher_id) REFERENCES teachers (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS assessment_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL,
    question_number INTEGER NOT NULL,
    question_text TEXT NOT NULL,
    question_type TEXT NOT NULL CHECK (question_type IN ('mcq', 'short_answer', 'checkbox', 'dropdown')),
    options_json TEXT,
    difficulty TEXT,
    bloom_level TEXT,
    marks REAL NOT NULL DEFAULT 1,
    correct_answer TEXT,
    explanation TEXT,
    google_question_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE CASCADE,
    UNIQUE (assessment_id, question_number)
);

CREATE TABLE IF NOT EXISTS question_concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_question_id INTEGER NOT NULL,
    concept_id INTEGER NOT NULL,
    weightage REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY (assessment_question_id) REFERENCES assessment_questions (id) ON DELETE CASCADE,
    FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
    UNIQUE (assessment_question_id, concept_id)
);

CREATE TABLE IF NOT EXISTS student_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'not_started' CHECK (status IN ('not_started', 'submitted', 'graded')),
    score_obtained REAL DEFAULT 0,
    percentage REAL,
    submitted_at TEXT,
    graded_at TEXT,
    FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
    UNIQUE (assessment_id, student_id)
);

CREATE TABLE IF NOT EXISTS student_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_assessment_id INTEGER NOT NULL,
    assessment_question_id INTEGER NOT NULL,
    raw_answer TEXT,
    normalized_answer TEXT,
    is_correct INTEGER CHECK (is_correct IN (0, 1)),
    score_awarded REAL DEFAULT 0,
    feedback TEXT,
    grading_reasoning TEXT,
    error_type TEXT CHECK (error_type IN ('concept_misunderstanding', 'careless_mistake', 'language_issue', 'incomplete_reasoning')),
    processed_at TEXT,
    FOREIGN KEY (student_assessment_id) REFERENCES student_assessments (id) ON DELETE CASCADE,
    FOREIGN KEY (assessment_question_id) REFERENCES assessment_questions (id) ON DELETE CASCADE,
    UNIQUE (student_assessment_id, assessment_question_id)
);

CREATE TABLE IF NOT EXISTS student_concept_mastery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    concept_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    mastery_score REAL NOT NULL DEFAULT 0,
    confidence_score REAL NOT NULL DEFAULT 0,
    questions_attempted INTEGER NOT NULL DEFAULT 0,
    questions_correct INTEGER NOT NULL DEFAULT 0,
    last_assessed_at TEXT,
    status TEXT NOT NULL DEFAULT 'developing' CHECK (status IN ('strong', 'developing', 'lagging')),
    FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
    FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
    UNIQUE (student_id, concept_id, class_id)
);

CREATE TABLE IF NOT EXISTS class_concept_mastery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL,
    concept_id INTEGER NOT NULL,
    average_mastery_score REAL NOT NULL DEFAULT 0,
    students_assessed INTEGER NOT NULL DEFAULT 0,
    students_lagging INTEGER NOT NULL DEFAULT 0,
    last_updated_at TEXT,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
    FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
    UNIQUE (class_id, concept_id)
);

CREATE TABLE IF NOT EXISTS remediation_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    assessment_id INTEGER NOT NULL,
    concept_id INTEGER NOT NULL,
    recommendation_type TEXT NOT NULL CHECK (recommendation_type IN ('reteach', 'practice_quiz', 'peer_learning', 'worksheet')),
    recommendation_text TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 1 CHECK (priority BETWEEN 1 AND 5),
    generated_by_model TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
    FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE CASCADE,
    FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS google_sync_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL,
    sync_type TEXT NOT NULL CHECK (sync_type IN ('form_create', 'response_fetch', 'grade_fetch')),
    external_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'success', 'failed')),
    message TEXT,
    synced_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS response_processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id INTEGER NOT NULL,
    response_id TEXT NOT NULL,
    respondent_email TEXT,
    submitted_at TEXT,
    raw_response_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TEXT,
    FOREIGN KEY (assessment_id) REFERENCES assessments (id) ON DELETE CASCADE,
    UNIQUE (assessment_id, response_id)
);

CREATE TABLE IF NOT EXISTS student_blueprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    strengths_json TEXT NOT NULL DEFAULT '[]',
    weaknesses_json TEXT NOT NULL DEFAULT '[]',
    opportunities_json TEXT NOT NULL DEFAULT '[]',
    threats_json TEXT NOT NULL DEFAULT '[]',
    recommendations_json TEXT NOT NULL DEFAULT '[]',
    narrative TEXT,
    generated_by_model TEXT,
    based_on_assessments INTEGER NOT NULL DEFAULT 0,
    last_submission_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
    UNIQUE (student_id, class_id)
);

CREATE TABLE IF NOT EXISTS attendance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    teacher_id INTEGER NOT NULL,
    attendance_date TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('present', 'absent')),
    source TEXT NOT NULL DEFAULT 'audio' CHECK (source IN ('audio', 'tool', 'manual')),
    raw_model_output TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES teachers (id) ON DELETE CASCADE,
    UNIQUE (class_id, student_id, attendance_date)
);

CREATE TABLE IF NOT EXISTS student_adaptation_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    profile_json TEXT NOT NULL DEFAULT '{}',
    summary TEXT,
    generated_by_model TEXT,
    based_on_assessments INTEGER NOT NULL DEFAULT 0,
    last_submission_at TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
    UNIQUE (student_id, class_id, subject)
);

CREATE TABLE IF NOT EXISTS source_materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    curriculum_subject_id INTEGER NOT NULL,
    uploaded_by_teacher_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('pdf', 'text', 'image')),
    original_filename TEXT,
    mime_type TEXT,
    storage_path TEXT,
    raw_text TEXT,
    extraction_summary TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (curriculum_subject_id) REFERENCES curriculum_subjects (id) ON DELETE CASCADE,
    FOREIGN KEY (uploaded_by_teacher_id) REFERENCES teachers (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS material_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_material_id INTEGER NOT NULL,
    curriculum_subject_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    section_heading TEXT,
    content_type TEXT NOT NULL DEFAULT 'text' CHECK (content_type IN ('text', 'image_ocr', 'multimodal_summary')),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_material_id) REFERENCES source_materials (id) ON DELETE CASCADE,
    FOREIGN KEY (curriculum_subject_id) REFERENCES curriculum_subjects (id) ON DELETE CASCADE,
    UNIQUE (source_material_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS material_ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_material_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processing', 'completed', 'failed')),
    extraction_summary TEXT,
    raw_structure_json TEXT,
    error_text TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_material_id) REFERENCES source_materials (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS academic_year_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    subject TEXT NOT NULL,
    academic_year TEXT NOT NULL,
    plan_title TEXT,
    raw_syllabus_text TEXT NOT NULL,
    planning_json TEXT NOT NULL DEFAULT '{}',
    generated_by_model TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('draft', 'active', 'archived')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (teacher_id) REFERENCES teachers (id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS academic_year_plan_units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id INTEGER NOT NULL,
    chapter_code TEXT,
    chapter_name TEXT NOT NULL,
    subtopics_json TEXT NOT NULL DEFAULT '[]',
    recommended_sessions INTEGER NOT NULL DEFAULT 1,
    target_month TEXT,
    term TEXT,
    sequence_order INTEGER NOT NULL DEFAULT 1,
    completed_subtopics_json TEXT NOT NULL DEFAULT '[]',
    completion_percent REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'not_started' CHECK (status IN ('not_started', 'in_progress', 'completed')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (plan_id) REFERENCES academic_year_plans (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS class_timetable_slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL,
    subject TEXT NOT NULL,
    weekday INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6),
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    auto_record_enabled INTEGER NOT NULL DEFAULT 1 CHECK (auto_record_enabled IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS class_coverage_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id INTEGER NOT NULL,
    teacher_id INTEGER NOT NULL,
    subject TEXT NOT NULL,
    plan_id INTEGER,
    timetable_slot_id INTEGER,
    session_date TEXT NOT NULL,
    scheduled_start TEXT,
    scheduled_end TEXT,
    actual_start TEXT,
    actual_end TEXT,
    source TEXT NOT NULL DEFAULT 'audio' CHECK (source IN ('audio', 'manual_note', 'audio_plus_note')),
    transcript_text TEXT,
    coverage_json TEXT NOT NULL DEFAULT '{}',
    confidence_score REAL NOT NULL DEFAULT 0,
    coverage_summary TEXT,
    processing_status TEXT NOT NULL DEFAULT 'completed' CHECK (processing_status IN ('processing', 'completed', 'failed')),
    processing_notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (class_id) REFERENCES classes (id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES teachers (id) ON DELETE CASCADE,
    FOREIGN KEY (plan_id) REFERENCES academic_year_plans (id) ON DELETE SET NULL,
    FOREIGN KEY (timetable_slot_id) REFERENCES class_timetable_slots (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_students_class_roll ON students (class_id, roll_number);
CREATE INDEX IF NOT EXISTS idx_class_subjects_class_subject ON class_subjects (class_id, subject);
CREATE INDEX IF NOT EXISTS idx_curriculum_subjects_grade_subject ON curriculum_subjects (grade, subject);
CREATE INDEX IF NOT EXISTS idx_curriculum_chapters_subject_order ON curriculum_chapters (curriculum_subject_id, chapter_order);
CREATE INDEX IF NOT EXISTS idx_assessments_class_chapter ON assessments (class_id, chapter_id);
CREATE INDEX IF NOT EXISTS idx_questions_assessment_number ON assessment_questions (assessment_id, question_number);
CREATE INDEX IF NOT EXISTS idx_student_assessments_student_assessment ON student_assessments (student_id, assessment_id);
CREATE INDEX IF NOT EXISTS idx_student_answers_attempt_question ON student_answers (student_assessment_id, assessment_question_id);
CREATE INDEX IF NOT EXISTS idx_student_concept_mastery_student_concept ON student_concept_mastery (student_id, concept_id);
CREATE INDEX IF NOT EXISTS idx_response_processing_queue_status ON response_processing_queue (status, created_at);
CREATE INDEX IF NOT EXISTS idx_student_blueprints_student_class ON student_blueprints (student_id, class_id, subject);
CREATE INDEX IF NOT EXISTS idx_attendance_class_date ON attendance_records (class_id, attendance_date);
CREATE INDEX IF NOT EXISTS idx_attendance_student_date ON attendance_records (student_id, attendance_date);
CREATE INDEX IF NOT EXISTS idx_student_adaptation_profiles_student_class ON student_adaptation_profiles (student_id, class_id, subject);
CREATE INDEX IF NOT EXISTS idx_source_materials_subject_created ON source_materials (curriculum_subject_id, created_at);
CREATE INDEX IF NOT EXISTS idx_material_chunks_subject_chunk ON material_chunks (curriculum_subject_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_material_ingestion_runs_material_created ON material_ingestion_runs (source_material_id, created_at);
CREATE INDEX IF NOT EXISTS idx_academic_year_plans_class_subject ON academic_year_plans (class_id, subject, academic_year);
CREATE INDEX IF NOT EXISTS idx_academic_year_plan_units_plan_order ON academic_year_plan_units (plan_id, sequence_order);
CREATE INDEX IF NOT EXISTS idx_class_timetable_slots_class_subject_day ON class_timetable_slots (class_id, subject, weekday);
CREATE INDEX IF NOT EXISTS idx_class_coverage_sessions_class_subject_date ON class_coverage_sessions (class_id, subject, session_date);
