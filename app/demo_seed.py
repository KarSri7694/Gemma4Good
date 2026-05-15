from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from app.db import get_connection


ROOT = Path(__file__).resolve().parent.parent
DEMO_SEED_MARKER_PATH = ROOT / "data" / "cache" / "demo_seed_initialized.json"
ACADEMIC_YEAR = "2026-27"
GRADE = "7"
SECTION = "A"
BASE_CLASS_SUBJECT = "Science"
DEMO_DATES = {
    "today": date(2026, 5, 15),
    "attendance_start": date(2026, 5, 1),
}

SUBJECTS = {
    "English": {
        "medium": "English",
        "chapters": [
            {
                "code": "ENG-7-01",
                "name": "Three Questions",
                "term": "Term 1",
                "month": "April",
                "sessions": 4,
                "subtopics": ["theme and moral", "character traits", "narrative sequence"],
            },
            {
                "code": "ENG-7-02",
                "name": "A Gift of Chappals",
                "term": "Term 1",
                "month": "May",
                "sessions": 5,
                "subtopics": ["dialogue reading", "humour and empathy", "chapter summary"],
            },
            {
                "code": "ENG-7-03",
                "name": "Gopal and the Hilsa-Fish",
                "term": "Term 2",
                "month": "July",
                "sessions": 4,
                "subtopics": ["comic setting", "plot inference", "retelling"],
            },
            {
                "code": "ENG-7-04",
                "name": "The Ashes That Made Trees Bloom",
                "term": "Term 2",
                "month": "August",
                "sessions": 5,
                "subtopics": ["setting and mood", "kindness and greed", "text evidence"],
            },
        ],
    },
    "Science": {
        "medium": "English + Hindi",
        "chapters": [
            {
                "code": "SCI-7-01",
                "name": "Nutrition in Plants",
                "term": "Term 1",
                "month": "April",
                "sessions": 6,
                "subtopics": ["photosynthesis", "stomata", "nutrient sources"],
            },
            {
                "code": "SCI-7-02",
                "name": "Heat",
                "term": "Term 1",
                "month": "May",
                "sessions": 5,
                "subtopics": ["clinical thermometer", "conduction convection radiation", "sea breeze and land breeze"],
            },
            {
                "code": "SCI-7-03",
                "name": "Acids Bases and Salts",
                "term": "Term 2",
                "month": "July",
                "sessions": 5,
                "subtopics": ["indicators", "neutralisation", "daily life examples"],
            },
            {
                "code": "SCI-7-04",
                "name": "Motion and Time",
                "term": "Term 2",
                "month": "August",
                "sessions": 5,
                "subtopics": ["speed and distance", "uniform motion", "graphs"],
            },
        ],
    },
    "Math": {
        "medium": "English + Hindi",
        "chapters": [
            {
                "code": "MAT-7-01",
                "name": "Integers",
                "term": "Term 1",
                "month": "April",
                "sessions": 5,
                "subtopics": ["positive and negative numbers", "number line operations", "properties of addition"],
            },
            {
                "code": "MAT-7-02",
                "name": "Fractions and Decimals",
                "term": "Term 1",
                "month": "May",
                "sessions": 6,
                "subtopics": ["fraction multiplication", "decimal operations", "word problems"],
            },
            {
                "code": "MAT-7-03",
                "name": "Simple Equations",
                "term": "Term 2",
                "month": "July",
                "sessions": 5,
                "subtopics": ["forming equations", "solving one-step equations", "checking solutions"],
            },
            {
                "code": "MAT-7-04",
                "name": "Lines and Angles",
                "term": "Term 2",
                "month": "August",
                "sessions": 5,
                "subtopics": ["complementary angles", "vertically opposite angles", "transversal examples"],
            },
        ],
    },
    "Computer Science": {
        "medium": "English",
        "chapters": [
            {
                "code": "CSC-7-01",
                "name": "Computer Fundamentals",
                "term": "Term 1",
                "month": "April",
                "sessions": 4,
                "subtopics": ["hardware and software", "input and output devices", "storage devices"],
            },
            {
                "code": "CSC-7-02",
                "name": "Number System",
                "term": "Term 1",
                "month": "May",
                "sessions": 4,
                "subtopics": ["binary numbers", "decimal to binary", "bits and bytes"],
            },
            {
                "code": "CSC-7-03",
                "name": "Internet Services",
                "term": "Term 2",
                "month": "July",
                "sessions": 4,
                "subtopics": ["web browser use", "email etiquette", "safe search"],
            },
            {
                "code": "CSC-7-04",
                "name": "Scratch Programming",
                "term": "Term 2",
                "month": "August",
                "sessions": 6,
                "subtopics": ["sprites and stage", "motion blocks", "simple animation"],
            },
        ],
    },
    "Social Science": {
        "medium": "English + Hindi",
        "chapters": [
            {
                "code": "SST-7-01",
                "name": "On Equality",
                "term": "Term 1",
                "month": "April",
                "sessions": 4,
                "subtopics": ["meaning of equality", "dignity", "examples from daily life"],
            },
            {
                "code": "SST-7-02",
                "name": "Role of the Government in Health",
                "term": "Term 1",
                "month": "May",
                "sessions": 5,
                "subtopics": ["public health services", "private health care", "equity in access"],
            },
            {
                "code": "SST-7-03",
                "name": "How the State Government Works",
                "term": "Term 2",
                "month": "July",
                "sessions": 5,
                "subtopics": ["MLA and assembly", "forming government", "question hour"],
            },
            {
                "code": "SST-7-04",
                "name": "Markets Around Us",
                "term": "Term 2",
                "month": "August",
                "sessions": 4,
                "subtopics": ["weekly markets", "shops in neighbourhood", "chain of markets"],
            },
        ],
    },
}

STUDENTS = [
    ("07A01", "Asha Verma", "Hindi", "Needs visual reinforcement"),
    ("07A02", "Rohan Gupta", "English", ""),
    ("07A03", "Meera Khan", "English + Hindi", "Slow processing speed"),
    ("07A04", "Ishaan Rao", "English", ""),
    ("07A05", "Priya Nair", "English", ""),
    ("07A06", "Arjun Mehta", "Hindi", "Benefits from worked examples"),
    ("07A07", "Sana Sheikh", "English + Hindi", ""),
    ("07A08", "Vivaan Joshi", "English", ""),
    ("07A09", "Ananya Das", "English", "Prefers short instructions"),
    ("07A10", "Kabir Ali", "Hindi", ""),
    ("07A11", "Nitya Paul", "English", ""),
    ("07A12", "Rahul Sethi", "English + Hindi", "Needs confidence-building feedback"),
]

SUBJECT_PROFICIENCY = {
    "07A01": {"English": 0.78, "Science": 0.64, "Math": 0.69, "Computer Science": 0.74, "Social Science": 0.71},
    "07A02": {"English": 0.90, "Science": 0.88, "Math": 0.86, "Computer Science": 0.84, "Social Science": 0.82},
    "07A03": {"English": 0.61, "Science": 0.45, "Math": 0.52, "Computer Science": 0.58, "Social Science": 0.57},
    "07A04": {"English": 0.72, "Science": 0.63, "Math": 0.66, "Computer Science": 0.71, "Social Science": 0.60},
    "07A05": {"English": 0.84, "Science": 0.76, "Math": 0.74, "Computer Science": 0.79, "Social Science": 0.80},
    "07A06": {"English": 0.58, "Science": 0.55, "Math": 0.68, "Computer Science": 0.63, "Social Science": 0.59},
    "07A07": {"English": 0.80, "Science": 0.73, "Math": 0.70, "Computer Science": 0.76, "Social Science": 0.83},
    "07A08": {"English": 0.69, "Science": 0.71, "Math": 0.77, "Computer Science": 0.81, "Social Science": 0.65},
    "07A09": {"English": 0.76, "Science": 0.68, "Math": 0.61, "Computer Science": 0.74, "Social Science": 0.78},
    "07A10": {"English": 0.62, "Science": 0.59, "Math": 0.57, "Computer Science": 0.60, "Social Science": 0.64},
    "07A11": {"English": 0.87, "Science": 0.79, "Math": 0.82, "Computer Science": 0.88, "Social Science": 0.81},
    "07A12": {"English": 0.66, "Science": 0.54, "Math": 0.60, "Computer Science": 0.67, "Social Science": 0.63},
}

ATTENDANCE_ABSENCES = {
    "07A03": {3, 8},
    "07A06": {5},
    "07A10": {2, 9},
    "07A12": {6},
}

TIMETABLE = [
    (0, "08:00", "08:40", "English"),
    (0, "08:45", "09:25", "Science"),
    (0, "09:35", "10:15", "Math"),
    (0, "10:35", "11:15", "Social Science"),
    (0, "11:20", "12:00", "Computer Science"),
    (0, "12:05", "12:45", "Science"),
    (1, "08:00", "08:40", "Math"),
    (1, "08:45", "09:25", "English"),
    (1, "09:35", "10:15", "Science"),
    (1, "10:35", "11:15", "Computer Science"),
    (1, "11:20", "12:00", "Social Science"),
    (1, "12:05", "12:45", "Math"),
    (2, "08:00", "08:40", "Science"),
    (2, "08:45", "09:25", "Social Science"),
    (2, "09:35", "10:15", "English"),
    (2, "10:35", "11:15", "Math"),
    (2, "11:20", "12:00", "Computer Science"),
    (2, "12:05", "12:45", "English"),
    (3, "08:00", "08:40", "Computer Science"),
    (3, "08:45", "09:25", "Math"),
    (3, "09:35", "10:15", "Science"),
    (3, "10:35", "11:15", "Social Science"),
    (3, "11:20", "12:00", "English"),
    (3, "12:05", "12:45", "Science"),
    (4, "08:00", "08:40", "English"),
    (4, "08:45", "09:25", "Science"),
    (4, "09:35", "10:15", "Math"),
    (4, "10:35", "11:15", "Social Science"),
    (4, "11:20", "12:00", "Computer Science"),
    (4, "12:05", "12:45", "Math"),
    (5, "08:00", "08:40", "Social Science"),
    (5, "08:45", "09:25", "English"),
    (5, "09:35", "10:15", "Science"),
    (5, "10:35", "11:15", "Computer Science"),
]

ASSESSMENT_BLUEPRINTS = [
    ("Science Quiz 1 - Nutrition in Plants", "Science", "SCI-7-01", "class_test", "google_form", "2026-04-12 08:00:00", "2026-04-12 18:00:00", 0.02),
    ("Science Quiz 2 - Heat", "Science", "SCI-7-02", "class_test", "google_form", "2026-05-08 08:00:00", "2026-05-08 18:00:00", -0.10),
    ("Math Quiz 1 - Integers", "Math", "MAT-7-01", "practice", "local", "2026-04-18 08:00:00", "2026-04-18 15:00:00", 0.03),
    ("Math Quiz 2 - Fractions and Decimals", "Math", "MAT-7-02", "class_test", "google_form", "2026-05-10 08:00:00", "2026-05-10 18:00:00", -0.09),
    ("English Quiz - Three Questions", "English", "ENG-7-01", "practice", "manual", "2026-04-20 08:00:00", "2026-04-20 15:00:00", 0.01),
    ("Computer Science Quiz - Number System", "Computer Science", "CSC-7-02", "practice", "local", "2026-05-05 08:00:00", "2026-05-05 15:00:00", 0.08),
    ("Social Science Quiz - On Equality", "Social Science", "SST-7-01", "class_test", "google_form", "2026-04-25 08:00:00", "2026-04-25 18:00:00", -0.04),
]


def ensure_demo_data() -> None:
    if DEMO_SEED_MARKER_PATH.exists():
        return
    with get_connection() as connection:
        existing_teacher = connection.execute("SELECT id FROM teachers ORDER BY id LIMIT 1").fetchone()
        if existing_teacher:
            _write_demo_seed_marker()
            return
        teacher_id, school_id = _ensure_teacher_and_school(connection)
        class_id = _ensure_canonical_class(connection, teacher_id, school_id)
        _ensure_class_subjects(connection, class_id)
        student_ids = _ensure_students(connection, class_id, school_id)
        chapter_map = _ensure_curriculum_and_chapters(connection, class_id)
        concept_map = _ensure_concepts(connection, chapter_map)
        _ensure_materials(connection, teacher_id, chapter_map)
        plan_ids = _ensure_academic_year_plans(connection, teacher_id, class_id)
        slot_ids = _ensure_timetable(connection, class_id)
        _ensure_coverage_sessions(connection, class_id, teacher_id, plan_ids, slot_ids)
        assessment_ids = _ensure_assessments(connection, class_id, teacher_id, student_ids, chapter_map, concept_map)
        _ensure_attendance(connection, class_id, teacher_id, student_ids)
        _ensure_mastery_analytics(connection, class_id, student_ids, concept_map)
        _ensure_remediation(connection, student_ids, assessment_ids, concept_map)
        _ensure_student_profiles(connection, class_id, student_ids)
        _ensure_queue_items(connection, assessment_ids)
        connection.commit()
    _write_demo_seed_marker()

def _write_demo_seed_marker() -> None:
    DEMO_SEED_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEMO_SEED_MARKER_PATH.write_text(
        json.dumps({"initialized": True, "academic_year": ACADEMIC_YEAR}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _ensure_teacher_and_school(connection) -> tuple[int, int]:
    teacher_row = connection.execute(
        "SELECT id, school_id FROM teachers ORDER BY id LIMIT 1"
    ).fetchone()
    if teacher_row:
        return int(teacher_row["id"]), int(teacher_row["school_id"])

    school_id = int(
        connection.execute(
            """
            INSERT INTO schools (name, board_type, state, district)
            VALUES (?, 'CBSE', ?, ?)
            """,
            ("Sarvodaya Public School", "Delhi", "South West Delhi"),
        ).lastrowid
    )
    teacher_id = int(
        connection.execute(
            """
            INSERT INTO teachers (school_id, full_name, email, google_account_email)
            VALUES (?, ?, ?, ?)
            """,
            (
                school_id,
                "Kartikeya Srivastava",
                "kartikeya.srivastava@example.edu",
                "kartikeya.srivastava@gmail.com",
            ),
        ).lastrowid
    )
    return teacher_id, school_id


def _ensure_canonical_class(connection, teacher_id: int, school_id: int) -> int:
    class_row = connection.execute(
        """
        SELECT id
        FROM classes
        WHERE teacher_id = ? AND academic_year = ? AND grade = ? AND section = ?
        ORDER BY id
        LIMIT 1
        """,
        (teacher_id, ACADEMIC_YEAR, GRADE, SECTION),
    ).fetchone()
    if class_row:
        class_id = int(class_row["id"])
        connection.execute(
            """
            UPDATE classes
            SET subject = ?, medium = ?
            WHERE id = ?
            """,
            (BASE_CLASS_SUBJECT, "English + Hindi", class_id),
        )
        return class_id

    return int(
        connection.execute(
            """
            INSERT INTO classes (school_id, teacher_id, academic_year, grade, section, subject, medium)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (school_id, teacher_id, ACADEMIC_YEAR, GRADE, SECTION, BASE_CLASS_SUBJECT, "English + Hindi"),
        ).lastrowid
    )


def _ensure_class_subjects(connection, class_id: int) -> None:
    allowed_subjects = tuple(SUBJECTS.keys())
    for subject, metadata in SUBJECTS.items():
        connection.execute(
            """
            INSERT INTO class_subjects (class_id, subject, medium)
            VALUES (?, ?, ?)
            ON CONFLICT(class_id, subject) DO UPDATE SET
                medium = excluded.medium
            """,
            (class_id, subject, metadata["medium"]),
        )
    placeholders = ", ".join("?" for _ in allowed_subjects)
    connection.execute(
        f"DELETE FROM class_subjects WHERE class_id = ? AND subject NOT IN ({placeholders})",
        (class_id, *allowed_subjects),
    )


def _ensure_students(connection, class_id: int, school_id: int) -> dict[str, int]:
    expected_rolls = {roll for roll, _, _, _ in STUDENTS}
    for roll_number, full_name, preferred_language, accessibility_notes in STUDENTS:
        email_local = full_name.lower().replace(" ", ".")
        existing = connection.execute(
            """
            SELECT id
            FROM students
            WHERE class_id = ? AND roll_number = ?
            LIMIT 1
            """,
            (class_id, roll_number),
        ).fetchone()
        if existing:
            connection.execute(
                """
                UPDATE students
                SET full_name = ?, email = ?, preferred_language = ?, accessibility_notes = ?, status = 'active'
                WHERE id = ?
                """,
                (
                    full_name,
                    f"{email_local}@student.demo",
                    preferred_language,
                    accessibility_notes,
                    int(existing["id"]),
                ),
            )
        else:
            connection.execute(
                """
                INSERT INTO students (
                    school_id, class_id, roll_number, full_name, email, preferred_language,
                    accessibility_notes, guardian_name, guardian_phone, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    school_id,
                    class_id,
                    roll_number,
                    full_name,
                    f"{email_local}@student.demo",
                    preferred_language,
                    accessibility_notes,
                    f"{full_name.split()[0]} Guardian",
                    f"99999{roll_number[-3:]}",
                ),
            )

    connection.execute(
        """
        UPDATE students
        SET status = 'inactive'
        WHERE class_id = ? AND roll_number NOT IN ({placeholders})
        """.format(placeholders=", ".join("?" for _ in expected_rolls)),
        (class_id, *sorted(expected_rolls)),
    )

    rows = connection.execute(
        "SELECT id, roll_number FROM students WHERE class_id = ?",
        (class_id,),
    ).fetchall()
    return {row["roll_number"]: int(row["id"]) for row in rows}


def _ensure_curriculum_and_chapters(connection, class_id: int) -> dict[str, dict[str, dict[str, int | str]]]:
    subject_map: dict[str, dict[str, dict[str, int | str]]] = {}
    for subject, metadata in SUBJECTS.items():
        connection.execute(
            """
            INSERT INTO curriculum_subjects (board_type, grade, subject, default_medium)
            VALUES ('CBSE', ?, ?, ?)
            ON CONFLICT(board_type, grade, subject) DO UPDATE SET
                default_medium = excluded.default_medium,
                updated_at = CURRENT_TIMESTAMP
            """,
            (GRADE, subject, metadata["medium"]),
        )
        curriculum_subject_id = int(
            connection.execute(
                """
                SELECT id
                FROM curriculum_subjects
                WHERE board_type = 'CBSE' AND grade = ? AND subject = ?
                LIMIT 1
                """,
                (GRADE, subject),
            ).fetchone()["id"]
        )
        subject_map[subject] = {}
        for order, chapter in enumerate(metadata["chapters"], start=1):
            connection.execute(
                """
                INSERT INTO curriculum_chapters (
                    curriculum_subject_id, chapter_code, chapter_name, chapter_order, term
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(curriculum_subject_id, chapter_code) DO UPDATE SET
                    chapter_name = excluded.chapter_name,
                    chapter_order = excluded.chapter_order,
                    term = excluded.term,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    curriculum_subject_id,
                    chapter["code"],
                    chapter["name"],
                    order,
                    chapter["term"],
                ),
            )
            connection.execute(
                """
                INSERT INTO chapters (board_type, grade, subject, chapter_code, chapter_name, term)
                VALUES ('CBSE', ?, ?, ?, ?, ?)
                ON CONFLICT(board_type, grade, subject, chapter_code) DO UPDATE SET
                    chapter_name = excluded.chapter_name,
                    term = excluded.term
                """,
                (GRADE, subject, chapter["code"], chapter["name"], chapter["term"]),
            )
            chapter_id = int(
                connection.execute(
                    """
                    SELECT id
                    FROM chapters
                    WHERE board_type = 'CBSE' AND grade = ? AND subject = ? AND chapter_code = ?
                    LIMIT 1
                    """,
                    (GRADE, subject, chapter["code"]),
                ).fetchone()["id"]
            )
            subject_map[subject][chapter["code"]] = {
                "id": chapter_id,
                "name": chapter["name"],
                "curriculum_subject_id": curriculum_subject_id,
                "chapter_order": order,
            }

        connection.execute(
            """
            INSERT INTO class_subjects (class_id, subject, medium)
            VALUES (?, ?, ?)
            ON CONFLICT(class_id, subject) DO UPDATE SET medium = excluded.medium
            """,
            (class_id, subject, metadata["medium"]),
        )
    return subject_map


def _ensure_concepts(connection, chapter_map) -> dict[tuple[str, str], list[dict[str, int | str]]]:
    concept_map: dict[tuple[str, str], list[dict[str, int | str]]] = {}
    for subject, metadata in SUBJECTS.items():
        for chapter in metadata["chapters"]:
            chapter_id = int(chapter_map[subject][chapter["code"]]["id"])
            chapter_concepts = []
            for index, subtopic in enumerate(chapter["subtopics"], start=1):
                concept_code = f"{chapter['code']}-C{index:02d}"
                concept_name = subtopic.title()
                connection.execute(
                    """
                    INSERT INTO concepts (chapter_id, concept_code, concept_name, description, difficulty_level)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chapter_id, concept_code) DO UPDATE SET
                        concept_name = excluded.concept_name,
                        description = excluded.description,
                        difficulty_level = excluded.difficulty_level
                    """,
                    (
                        chapter_id,
                        concept_code,
                        concept_name,
                        f"{concept_name} in {chapter['name']}",
                        "core" if index == 1 else "medium",
                    ),
                )
                concept_id = int(
                    connection.execute(
                        """
                        SELECT id
                        FROM concepts
                        WHERE chapter_id = ? AND concept_code = ?
                        LIMIT 1
                        """,
                        (chapter_id, concept_code),
                    ).fetchone()["id"]
                )
                chapter_concepts.append({"id": concept_id, "code": concept_code, "name": concept_name})
            concept_map[(subject, chapter["code"])] = chapter_concepts
    return concept_map


def _ensure_materials(connection, teacher_id: int, chapter_map) -> None:
    for subject, metadata in SUBJECTS.items():
        curriculum_subject_id = int(next(iter(chapter_map[subject].values()))["curriculum_subject_id"])
        title = f"Grade 7 {subject} Term Plan Pack"
        raw_text = "\n".join(
            f"{chapter['name']}: {', '.join(chapter['subtopics'])}"
            for chapter in metadata["chapters"]
        )
        existing = connection.execute(
            """
            SELECT id
            FROM source_materials
            WHERE curriculum_subject_id = ? AND title = ?
            LIMIT 1
            """,
            (curriculum_subject_id, title),
        ).fetchone()
        if existing:
            material_id = int(existing["id"])
            connection.execute(
                """
                UPDATE source_materials
                SET raw_text = ?, extraction_summary = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    raw_text,
                    f"Seeded pack for {subject} with {len(metadata['chapters'])} chapters.",
                    material_id,
                ),
            )
        else:
            material_id = int(
                connection.execute(
                    """
                    INSERT INTO source_materials (
                        curriculum_subject_id, uploaded_by_teacher_id, title, source_type,
                        original_filename, mime_type, raw_text, extraction_summary
                    )
                    VALUES (?, ?, ?, 'pdf', ?, 'application/pdf', ?, ?)
                    """,
                    (
                        curriculum_subject_id,
                        teacher_id,
                        title,
                        f"{subject.lower().replace(' ', '_')}_grade7_seed.pdf",
                        raw_text,
                        f"Seeded pack for {subject} with {len(metadata['chapters'])} chapters.",
                    ),
                ).lastrowid
            )
        for index, chapter in enumerate(metadata["chapters"], start=1):
            chunk_text = f"{chapter['name']}\n" + "\n".join(f"- {item}" for item in chapter["subtopics"])
            connection.execute(
                """
                INSERT INTO material_chunks (
                    source_material_id, curriculum_subject_id, chunk_index, chunk_text,
                    page_start, page_end, section_heading, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_material_id, chunk_index) DO UPDATE SET
                    chunk_text = excluded.chunk_text,
                    page_start = excluded.page_start,
                    page_end = excluded.page_end,
                    section_heading = excluded.section_heading,
                    metadata_json = excluded.metadata_json
                """,
                (
                    material_id,
                    curriculum_subject_id,
                    index,
                    chunk_text,
                    index,
                    index,
                    chapter["name"],
                    _json({"subject": subject, "chapter_code": chapter["code"]}),
                ),
            )
        run_row = connection.execute(
            """
            SELECT id
            FROM material_ingestion_runs
            WHERE source_material_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (material_id,),
        ).fetchone()
        if run_row:
            connection.execute(
                """
                UPDATE material_ingestion_runs
                SET status = 'completed', extraction_summary = ?, raw_structure_json = ?, error_text = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    f"Structured {subject} chapters into chunks.",
                    _json({"chapters": [chapter["name"] for chapter in metadata["chapters"]]}),
                    int(run_row["id"]),
                ),
            )
        else:
            connection.execute(
                """
                INSERT INTO material_ingestion_runs (
                    source_material_id, status, extraction_summary, raw_structure_json
                )
                VALUES (?, 'completed', ?, ?)
                """,
                (
                    material_id,
                    f"Structured {subject} chapters into chunks.",
                    _json({"chapters": [chapter["name"] for chapter in metadata["chapters"]]}),
                ),
            )


def _ensure_academic_year_plans(connection, teacher_id: int, class_id: int) -> dict[str, int]:
    completion_targets = {
        "English": [100.0, 45.0, 0.0, 0.0],
        "Science": [100.0, 58.0, 0.0, 0.0],
        "Math": [96.0, 62.0, 0.0, 0.0],
        "Computer Science": [100.0, 38.0, 0.0, 0.0],
        "Social Science": [82.0, 34.0, 0.0, 0.0],
    }
    plan_ids: dict[str, int] = {}
    for subject, metadata in SUBJECTS.items():
        planning = {
            "subject": subject,
            "board": "CBSE",
            "grade": GRADE,
            "recommended_pacing_note": f"{subject} pacing is seeded for weekly classroom use.",
            "next_focus": metadata["chapters"][1]["subtopics"][0],
        }
        raw_syllabus_text = "\n".join(
            f"{item['code']} {item['name']}: {', '.join(item['subtopics'])}"
            for item in metadata["chapters"]
        )
        existing = connection.execute(
            """
            SELECT id
            FROM academic_year_plans
            WHERE class_id = ? AND LOWER(subject) = LOWER(?) AND academic_year = ? AND status = 'active'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (class_id, subject, ACADEMIC_YEAR),
        ).fetchone()
        if existing:
            plan_id = int(existing["id"])
            connection.execute(
                """
                UPDATE academic_year_plans
                SET teacher_id = ?, plan_title = ?, raw_syllabus_text = ?, planning_json = ?,
                    generated_by_model = ?, status = 'active', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    teacher_id,
                    f"{subject} Year Plan",
                    raw_syllabus_text,
                    _json(planning),
                    "mock-gemma-planner",
                    plan_id,
                ),
            )
        else:
            plan_id = int(
                connection.execute(
                    """
                    INSERT INTO academic_year_plans (
                        teacher_id, class_id, subject, academic_year, plan_title,
                        raw_syllabus_text, planning_json, generated_by_model, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
                    """,
                    (
                        teacher_id,
                        class_id,
                        subject,
                        ACADEMIC_YEAR,
                        f"{subject} Year Plan",
                        raw_syllabus_text,
                        _json(planning),
                        "mock-gemma-planner",
                    ),
                ).lastrowid
            )
        plan_ids[subject] = plan_id
        connection.execute("DELETE FROM academic_year_plan_units WHERE plan_id = ?", (plan_id,))
        for index, chapter in enumerate(metadata["chapters"], start=1):
            completion_percent = completion_targets[subject][index - 1]
            completed_count = round((completion_percent / 100.0) * len(chapter["subtopics"]))
            completed = chapter["subtopics"][:completed_count]
            status = "completed" if completion_percent >= 99 else "in_progress" if completion_percent > 0 else "not_started"
            connection.execute(
                """
                INSERT INTO academic_year_plan_units (
                    plan_id, chapter_code, chapter_name, subtopics_json, recommended_sessions,
                    target_month, term, sequence_order, completed_subtopics_json,
                    completion_percent, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    chapter["code"],
                    chapter["name"],
                    _json(chapter["subtopics"]),
                    chapter["sessions"],
                    chapter["month"],
                    chapter["term"],
                    index,
                    _json(completed),
                    completion_percent,
                    status,
                ),
            )
    return plan_ids


def _ensure_timetable(connection, class_id: int) -> dict[tuple[str, int, str, str], int]:
    slot_ids: dict[tuple[str, int, str, str], int] = {}
    for weekday, start_time, end_time, subject in TIMETABLE:
        existing = connection.execute(
            """
            SELECT id
            FROM class_timetable_slots
            WHERE class_id = ? AND LOWER(subject) = LOWER(?) AND weekday = ? AND start_time = ? AND end_time = ?
            LIMIT 1
            """,
            (class_id, subject, weekday, start_time, end_time),
        ).fetchone()
        if existing:
            slot_id = int(existing["id"])
            connection.execute(
                """
                UPDATE class_timetable_slots
                SET auto_record_enabled = 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (slot_id,),
            )
        else:
            slot_id = int(
                connection.execute(
                    """
                    INSERT INTO class_timetable_slots (
                        class_id, subject, weekday, start_time, end_time, auto_record_enabled
                    )
                    VALUES (?, ?, ?, ?, ?, 1)
                    """,
                    (class_id, subject, weekday, start_time, end_time),
                ).lastrowid
            )
        slot_ids[(subject, weekday, start_time, end_time)] = slot_id
    return slot_ids


def _ensure_coverage_sessions(connection, class_id: int, teacher_id: int, plan_ids: dict[str, int], slot_ids) -> None:
    session_specs = [
        ("Science", "2026-05-14", 3, "09:35", "10:15", "Heat", ["clinical thermometer", "temperature reading"], ["sea breeze and land breeze"], ["conduction convection radiation"], 0.71),
        ("Math", "2026-05-13", 2, "10:35", "11:15", "Fractions and Decimals", ["fraction multiplication"], ["word problems"], ["decimal operations"], 0.76),
        ("English", "2026-05-12", 0, "08:00", "08:40", "A Gift of Chappals", ["dialogue reading", "chapter summary"], [], ["humour and empathy"], 0.84),
        ("Computer Science", "2026-05-12", 1, "10:35", "11:15", "Number System", ["binary numbers"], [], ["decimal to binary"], 0.88),
        ("Social Science", "2026-05-11", 5, "08:00", "08:40", "Role of the Government in Health", ["public health services"], ["equity in access"], ["private health care"], 0.74),
    ]
    for subject, session_date, weekday, start_time, end_time, chapter_name, covered, reteach, next_topics, confidence in session_specs:
        existing = connection.execute(
            """
            SELECT id
            FROM class_coverage_sessions
            WHERE class_id = ? AND LOWER(subject) = LOWER(?) AND session_date = ? AND scheduled_start = ?
            LIMIT 1
            """,
            (class_id, subject, session_date, start_time),
        ).fetchone()
        slot_id = slot_ids.get((subject, weekday, start_time, end_time))
        coverage = {
            "chapter_name": chapter_name,
            "subtopics_covered": covered,
            "reteach_topics": reteach,
            "next_topics": next_topics,
            "coverage_percent": round((len(covered) / max(1, len(covered) + len(next_topics))) * 100, 1),
        }
        transcript = (
            f"Teacher revised {chapter_name}, explained {', '.join(covered)}"
            + (f", and flagged {', '.join(reteach)} for reteach." if reteach else ".")
        )
        summary = f"{subject}: covered {', '.join(covered)}. Next up: {', '.join(next_topics)}."
        if existing:
            connection.execute(
                """
                UPDATE class_coverage_sessions
                SET teacher_id = ?, plan_id = ?, timetable_slot_id = ?, scheduled_end = ?, actual_start = ?,
                    actual_end = ?, source = 'audio_plus_note', transcript_text = ?, coverage_json = ?,
                    confidence_score = ?, coverage_summary = ?, processing_status = 'completed',
                    processing_notes = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    teacher_id,
                    plan_ids[subject],
                    slot_id,
                    end_time,
                    start_time,
                    end_time,
                    transcript,
                    _json(coverage),
                    confidence,
                    summary,
                    "Seeded from combined audio and teacher note flow.",
                    int(existing["id"]),
                ),
            )
        else:
            connection.execute(
                """
                INSERT INTO class_coverage_sessions (
                    class_id, teacher_id, subject, plan_id, timetable_slot_id, session_date,
                    scheduled_start, scheduled_end, actual_start, actual_end, source, transcript_text,
                    coverage_json, confidence_score, coverage_summary, processing_status, processing_notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'audio_plus_note', ?, ?, ?, ?, 'completed', ?)
                """,
                (
                    class_id,
                    teacher_id,
                    subject,
                    plan_ids[subject],
                    slot_id,
                    session_date,
                    start_time,
                    end_time,
                    start_time,
                    end_time,
                    transcript,
                    _json(coverage),
                    confidence,
                    summary,
                    "Seeded from combined audio and teacher note flow.",
                ),
            )


def _ensure_assessments(connection, class_id: int, teacher_id: int, student_ids, chapter_map, concept_map) -> dict[str, int]:
    assessment_ids: dict[str, int] = {}
    for index, (title, subject, chapter_code, assessment_type, delivery_mode, assigned_at, due_at, difficulty_shift) in enumerate(ASSESSMENT_BLUEPRINTS, start=1):
        chapter_id = int(chapter_map[subject][chapter_code]["id"])
        total_marks = 18.0
        google_form_id = f"demo-form-{index:03d}" if delivery_mode == "google_form" else None
        google_form_url = f"https://docs.google.com/forms/d/demo-form-{index:03d}/edit" if google_form_id else None
        existing = connection.execute(
            """
            SELECT id
            FROM assessments
            WHERE class_id = ? AND title = ?
            LIMIT 1
            """,
            (class_id, title),
        ).fetchone()
        if existing:
            assessment_id = int(existing["id"])
            connection.execute(
                """
                UPDATE assessments
                SET chapter_id = ?, teacher_id = ?, assessment_type = ?, delivery_mode = ?,
                    google_form_id = ?, google_form_url = ?, language = ?, total_marks = ?,
                    assigned_at = ?, due_at = ?, created_at = ?
                WHERE id = ?
                """,
                (
                    chapter_id,
                    teacher_id,
                    assessment_type,
                    delivery_mode,
                    google_form_id,
                    google_form_url,
                    SUBJECTS[subject]["medium"],
                    total_marks,
                    assigned_at,
                    due_at,
                    assigned_at,
                    assessment_id,
                ),
            )
        else:
            assessment_id = int(
                connection.execute(
                    """
                    INSERT INTO assessments (
                        class_id, chapter_id, teacher_id, title, assessment_type, delivery_mode,
                        google_form_id, google_form_url, language, total_marks, assigned_at, due_at, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        class_id,
                        chapter_id,
                        teacher_id,
                        title,
                        assessment_type,
                        delivery_mode,
                        google_form_id,
                        google_form_url,
                        SUBJECTS[subject]["medium"],
                        total_marks,
                        assigned_at,
                        due_at,
                        assigned_at,
                    ),
                ).lastrowid
            )
        assessment_ids[title] = assessment_id

        concepts = concept_map[(subject, chapter_code)]
        question_ids = []
        for question_number, concept in enumerate(concepts, start=1):
            question_text = f"Explain or apply: {concept['name']} in {chapter_map[subject][chapter_code]['name']}."
            correct_answer = f"Correct explanation of {concept['name']}."
            connection.execute(
                """
                INSERT INTO assessment_questions (
                    assessment_id, question_number, question_text, question_type, difficulty,
                    bloom_level, marks, correct_answer, explanation, google_question_id
                )
                VALUES (?, ?, ?, 'short_answer', ?, ?, 6, ?, ?, ?)
                ON CONFLICT(assessment_id, question_number) DO UPDATE SET
                    question_text = excluded.question_text,
                    difficulty = excluded.difficulty,
                    bloom_level = excluded.bloom_level,
                    marks = excluded.marks,
                    correct_answer = excluded.correct_answer,
                    explanation = excluded.explanation,
                    google_question_id = excluded.google_question_id
                """,
                (
                    assessment_id,
                    question_number,
                    question_text,
                    "easy" if question_number == 1 else "medium",
                    "remember" if question_number == 1 else "understand",
                    correct_answer,
                    f"Seeded answer key for {concept['name']}.",
                    f"gq-{assessment_id:03d}-{question_number:02d}",
                ),
            )
            question_id = int(
                connection.execute(
                    """
                    SELECT id
                    FROM assessment_questions
                    WHERE assessment_id = ? AND question_number = ?
                    LIMIT 1
                    """,
                    (assessment_id, question_number),
                ).fetchone()["id"]
            )
            question_ids.append(question_id)
            connection.execute(
                """
                INSERT INTO question_concepts (assessment_question_id, concept_id, weightage)
                VALUES (?, ?, 1.0)
                ON CONFLICT(assessment_question_id, concept_id) DO UPDATE SET weightage = excluded.weightage
                """,
                (question_id, int(concept["id"])),
            )

        for roll_number, student_id in student_ids.items():
            base_ratio = SUBJECT_PROFICIENCY.get(roll_number, {}).get(subject, 0.65) + difficulty_shift
            base_ratio = max(0.22, min(0.97, base_ratio))
            per_question_scores = []
            for offset, question_id in enumerate(question_ids):
                ratio = max(0.0, min(1.0, base_ratio + 0.08 - (offset * 0.05)))
                awarded = round(6 * ratio, 1)
                per_question_scores.append((question_id, awarded, ratio))
            score_obtained = round(sum(item[1] for item in per_question_scores), 1)
            percentage = round((score_obtained / total_marks) * 100, 1)
            submitted_at = due_at.replace("18:00:00", "13:10:00").replace("15:00:00", "13:10:00")
            connection.execute(
                """
                INSERT INTO student_assessments (
                    assessment_id, student_id, status, score_obtained, percentage, submitted_at, graded_at
                )
                VALUES (?, ?, 'graded', ?, ?, ?, ?)
                ON CONFLICT(assessment_id, student_id) DO UPDATE SET
                    status = excluded.status,
                    score_obtained = excluded.score_obtained,
                    percentage = excluded.percentage,
                    submitted_at = excluded.submitted_at,
                    graded_at = excluded.graded_at
                """,
                (
                    assessment_id,
                    student_id,
                    score_obtained,
                    percentage,
                    submitted_at,
                    due_at,
                ),
            )
            student_assessment_id = int(
                connection.execute(
                    """
                    SELECT id
                    FROM student_assessments
                    WHERE assessment_id = ? AND student_id = ?
                    LIMIT 1
                    """,
                    (assessment_id, student_id),
                ).fetchone()["id"]
            )
            for question_position, (question_id, awarded, ratio) in enumerate(per_question_scores, start=1):
                is_correct = 1 if ratio >= 0.7 else 0
                raw_answer = (
                    f"Accurate answer about {concepts[question_position - 1]['name']}"
                    if is_correct
                    else f"Partial answer on {concepts[question_position - 1]['name']}"
                )
                error_type = None
                if ratio < 0.45:
                    error_type = "concept_misunderstanding"
                elif ratio < 0.7:
                    error_type = "incomplete_reasoning"
                connection.execute(
                    """
                    INSERT INTO student_answers (
                        student_assessment_id, assessment_question_id, raw_answer, normalized_answer,
                        is_correct, score_awarded, feedback, grading_reasoning, error_type, processed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(student_assessment_id, assessment_question_id) DO UPDATE SET
                        raw_answer = excluded.raw_answer,
                        normalized_answer = excluded.normalized_answer,
                        is_correct = excluded.is_correct,
                        score_awarded = excluded.score_awarded,
                        feedback = excluded.feedback,
                        grading_reasoning = excluded.grading_reasoning,
                        error_type = excluded.error_type,
                        processed_at = excluded.processed_at
                    """,
                    (
                        student_assessment_id,
                        question_id,
                        raw_answer,
                        raw_answer.lower(),
                        is_correct,
                        awarded,
                        "Shows clear understanding." if is_correct else "Needs clearer explanation and examples.",
                        f"Seeded grading based on {subject} proficiency ratio {ratio:.2f}.",
                        error_type,
                        due_at,
                    ),
                )

        if delivery_mode == "google_form":
            for sync_type, external_suffix in [("form_create", "form"), ("response_fetch", "responses")]:
                external_id = f"{external_suffix}-{assessment_id}"
                existing_log = connection.execute(
                    """
                    SELECT id
                    FROM google_sync_logs
                    WHERE assessment_id = ? AND sync_type = ? AND external_id = ?
                    LIMIT 1
                    """,
                    (assessment_id, sync_type, external_id),
                ).fetchone()
                if existing_log:
                    connection.execute(
                        """
                        UPDATE google_sync_logs
                        SET status = 'success', message = ?
                        WHERE id = ?
                        """,
                        (f"Seeded {sync_type} state for {title}.", int(existing_log["id"])),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO google_sync_logs (assessment_id, sync_type, external_id, status, message)
                        VALUES (?, ?, ?, 'success', ?)
                        """,
                        (assessment_id, sync_type, external_id, f"Seeded {sync_type} state for {title}."),
                    )
    return assessment_ids


def _ensure_attendance(connection, class_id: int, teacher_id: int, student_ids: dict[str, int]) -> None:
    for day_offset in range(10):
        attendance_date = (DEMO_DATES["attendance_start"] + timedelta(days=day_offset)).isoformat()
        for roll_number, student_id in student_ids.items():
            status = "absent" if day_offset in ATTENDANCE_ABSENCES.get(roll_number, set()) else "present"
            connection.execute(
                """
                INSERT INTO attendance_records (
                    class_id, student_id, teacher_id, attendance_date, status, source, raw_model_output
                )
                VALUES (?, ?, ?, ?, ?, 'tool', ?)
                ON CONFLICT(class_id, student_id, attendance_date) DO UPDATE SET
                    status = excluded.status,
                    source = excluded.source,
                    raw_model_output = excluded.raw_model_output,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    class_id,
                    student_id,
                    teacher_id,
                    attendance_date,
                    status,
                    f"Seeded attendance for {roll_number} on {attendance_date}.",
                ),
            )


def _ensure_mastery_analytics(connection, class_id: int, student_ids: dict[str, int], concept_map) -> None:
    class_scores: dict[int, list[float]] = defaultdict(list)
    class_lagging: dict[int, int] = defaultdict(int)
    for (subject, _chapter_code), concepts in concept_map.items():
        for concept_index, concept in enumerate(concepts):
            concept_id = int(concept["id"])
            for roll_number, student_id in student_ids.items():
                base = SUBJECT_PROFICIENCY.get(roll_number, {}).get(subject, 0.65)
                mastery_score = max(0.18, min(0.98, base - (concept_index * 0.04)))
                confidence_score = max(0.35, min(0.97, mastery_score + 0.09))
                attempted = 2 + concept_index
                correct = max(0, min(attempted, round(attempted * mastery_score)))
                if mastery_score >= 0.8:
                    status = "strong"
                elif mastery_score >= 0.55:
                    status = "developing"
                else:
                    status = "lagging"
                    class_lagging[concept_id] += 1
                class_scores[concept_id].append(mastery_score)
                connection.execute(
                    """
                    INSERT INTO student_concept_mastery (
                        student_id, concept_id, class_id, mastery_score, confidence_score,
                        questions_attempted, questions_correct, last_assessed_at, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(student_id, concept_id, class_id) DO UPDATE SET
                        mastery_score = excluded.mastery_score,
                        confidence_score = excluded.confidence_score,
                        questions_attempted = excluded.questions_attempted,
                        questions_correct = excluded.questions_correct,
                        last_assessed_at = excluded.last_assessed_at,
                        status = excluded.status
                    """,
                    (
                        student_id,
                        concept_id,
                        class_id,
                        round(mastery_score, 3),
                        round(confidence_score, 3),
                        attempted,
                        correct,
                        "2026-05-12 12:00:00",
                        status,
                    ),
                )

    for concept_id, scores in class_scores.items():
        average_score = round(sum(scores) / len(scores), 3)
        connection.execute(
            """
            INSERT INTO class_concept_mastery (
                class_id, concept_id, average_mastery_score, students_assessed,
                students_lagging, last_updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(class_id, concept_id) DO UPDATE SET
                average_mastery_score = excluded.average_mastery_score,
                students_assessed = excluded.students_assessed,
                students_lagging = excluded.students_lagging,
                last_updated_at = excluded.last_updated_at
            """,
            (
                class_id,
                concept_id,
                average_score,
                len(scores),
                class_lagging.get(concept_id, 0),
                "2026-05-12 12:00:00",
            ),
        )


def _ensure_remediation(connection, student_ids: dict[str, int], assessment_ids: dict[str, int], concept_map) -> None:
    remediation_specs = [
        ("07A03", "Science Quiz 2 - Heat", ("Science", "SCI-7-02"), 0, "reteach", "Re-teach heat transfer with daily-life examples and a diagram."),
        ("07A10", "Math Quiz 2 - Fractions and Decimals", ("Math", "MAT-7-02"), 2, "practice_quiz", "Assign a 5-question mini-quiz on decimal operations and word problems."),
        ("07A12", "Social Science Quiz - On Equality", ("Social Science", "SST-7-01"), 1, "worksheet", "Use a comparison worksheet on equality and dignity in school settings."),
        ("07A06", "Science Quiz 2 - Heat", ("Science", "SCI-7-02"), 1, "peer_learning", "Pair with Rohan to discuss conduction, convection, and radiation."),
    ]
    for roll_number, assessment_title, concept_key, concept_index, recommendation_type, recommendation_text in remediation_specs:
        concept_id = int(concept_map[concept_key][concept_index]["id"])
        student_id = student_ids[roll_number]
        assessment_id = assessment_ids[assessment_title]
        existing = connection.execute(
            """
            SELECT id
            FROM remediation_recommendations
            WHERE student_id = ? AND assessment_id = ? AND concept_id = ? AND recommendation_type = ?
            LIMIT 1
            """,
            (student_id, assessment_id, concept_id, recommendation_type),
        ).fetchone()
        if existing:
            connection.execute(
                """
                UPDATE remediation_recommendations
                SET recommendation_text = ?, priority = 4, generated_by_model = ?
                WHERE id = ?
                """,
                (recommendation_text, "mock-gemma-remediation", int(existing["id"])),
            )
        else:
            connection.execute(
                """
                INSERT INTO remediation_recommendations (
                    student_id, assessment_id, concept_id, recommendation_type,
                    recommendation_text, priority, generated_by_model
                )
                VALUES (?, ?, ?, ?, ?, 4, ?)
                """,
                (
                    student_id,
                    assessment_id,
                    concept_id,
                    recommendation_type,
                    recommendation_text,
                    "mock-gemma-remediation",
                ),
            )


def _ensure_student_profiles(connection, class_id: int, student_ids: dict[str, int]) -> None:
    subject_rankings = {}
    for roll_number, subject_scores in SUBJECT_PROFICIENCY.items():
        ordered = sorted(subject_scores.items(), key=lambda item: item[1], reverse=True)
        subject_rankings[roll_number] = ordered

    for roll_number, student_id in student_ids.items():
        rankings = subject_rankings[roll_number]
        strongest_subject, strongest_score = rankings[0]
        weakest_subject, weakest_score = rankings[-1]
        connection.execute(
            """
            INSERT INTO student_blueprints (
                student_id, class_id, subject, strengths_json, weaknesses_json, opportunities_json,
                threats_json, recommendations_json, narrative, generated_by_model,
                based_on_assessments, last_submission_at
            )
            VALUES (?, ?, 'Science', ?, ?, ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(student_id, class_id) DO UPDATE SET
                strengths_json = excluded.strengths_json,
                weaknesses_json = excluded.weaknesses_json,
                opportunities_json = excluded.opportunities_json,
                threats_json = excluded.threats_json,
                recommendations_json = excluded.recommendations_json,
                narrative = excluded.narrative,
                generated_by_model = excluded.generated_by_model,
                based_on_assessments = excluded.based_on_assessments,
                last_submission_at = excluded.last_submission_at,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                student_id,
                class_id,
                _json([f"Strongest current subject: {strongest_subject}", "Participates in guided revision tasks"]),
                _json([f"Weakest current subject: {weakest_subject}", "Needs support when questions require explanation"]),
                _json([f"Can deepen mastery in {strongest_subject}", f"Can improve confidence through small wins in {weakest_subject}"]),
                _json([f"Low clarity in {weakest_subject} may affect future assessments"]),
                _json([f"Plan one reteach follow-up in {weakest_subject}", f"Give extension work in {strongest_subject}"]),
                (
                    f"{roll_number} is currently strongest in {strongest_subject} ({strongest_score:.0%}) "
                    f"and needs the most support in {weakest_subject} ({weakest_score:.0%})."
                ),
                "mock-gemma-blueprint",
                "2026-05-12 13:00:00",
            ),
        )

        for subject, score in SUBJECT_PROFICIENCY[roll_number].items():
            learning_style = "visual" if subject in {"Science", "Computer Science"} else "verbal"
            pacing = "slow" if score < 0.6 else "medium" if score < 0.8 else "fast"
            intervention_needed = score < 0.6
            connection.execute(
                """
                INSERT INTO student_adaptation_profiles (
                    student_id, class_id, subject, profile_json, summary, generated_by_model,
                    based_on_assessments, last_submission_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(student_id, class_id, subject) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    summary = excluded.summary,
                    generated_by_model = excluded.generated_by_model,
                    based_on_assessments = excluded.based_on_assessments,
                    last_submission_at = excluded.last_submission_at,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    student_id,
                    class_id,
                    subject,
                    _json(
                        {
                            "learning_style": learning_style,
                            "recommended_pacing": pacing,
                            "intervention_needed": intervention_needed,
                            "seat_preference": "front row" if intervention_needed else "flexible",
                        }
                    ),
                    f"{subject}: {roll_number} works best with {learning_style} support and {pacing} pacing.",
                    "mock-gemma-adaptation",
                    "2026-05-12 13:00:00",
                ),
            )


def _ensure_queue_items(connection, assessment_ids: dict[str, int]) -> None:
    queue_specs = [
        ("Science Quiz 2 - Heat", "resp-seed-heat-01", "queued"),
        ("Math Quiz 2 - Fractions and Decimals", "resp-seed-math-01", "completed"),
    ]
    for title, response_id, status in queue_specs:
        assessment_id = assessment_ids[title]
        connection.execute(
            """
            INSERT INTO response_processing_queue (
                assessment_id, response_id, respondent_email, submitted_at, raw_response_json,
                status, error_message, processed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT(assessment_id, response_id) DO UPDATE SET
                respondent_email = excluded.respondent_email,
                submitted_at = excluded.submitted_at,
                raw_response_json = excluded.raw_response_json,
                status = excluded.status,
                error_message = excluded.error_message,
                processed_at = excluded.processed_at
            """,
            (
                assessment_id,
                response_id,
                f"{response_id}@student.demo",
                "2026-05-14 13:00:00",
                _json({"response_id": response_id, "seeded": True}),
                status,
                None if status != "completed" else "2026-05-14 13:10:00",
            ),
        )


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)
