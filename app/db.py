from __future__ import annotations

import sqlite3
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "pathshala_play.db"
SCHEMA_PATH = ROOT / "db" / "schema.sql"


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def ensure_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.executescript(schema_sql)
        _run_migrations(connection)
        connection.commit()


def _run_migrations(connection: sqlite3.Connection) -> None:
    """Apply lightweight demo-safe schema migrations."""

    _ensure_column(connection, "assessment_questions", "options_json", "TEXT")
    _ensure_column(connection, "students", "email", "TEXT")
    _ensure_column(connection, "student_answers", "grading_reasoning", "TEXT")
    _ensure_column(connection, "student_blueprints", "subject", "TEXT NOT NULL DEFAULT ''")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_students_email ON students (email)")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_response_processing_queue_status "
        "ON response_processing_queue (status, created_at)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_student_blueprints_student_class "
        "ON student_blueprints (student_id, class_id, subject)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_attendance_class_date "
        "ON attendance_records (class_id, attendance_date)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_attendance_student_date "
        "ON attendance_records (student_id, attendance_date)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_student_adaptation_profiles_student_class "
        "ON student_adaptation_profiles (student_id, class_id, subject)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_class_subjects_class_subject "
        "ON class_subjects (class_id, subject)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_curriculum_subjects_grade_subject "
        "ON curriculum_subjects (grade, subject)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_curriculum_chapters_subject_order "
        "ON curriculum_chapters (curriculum_subject_id, chapter_order)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_source_materials_subject_created "
        "ON source_materials (curriculum_subject_id, created_at)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_material_chunks_subject_chunk "
        "ON material_chunks (curriculum_subject_id, chunk_index)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_material_ingestion_runs_material_created "
        "ON material_ingestion_runs (source_material_id, created_at)"
    )
    connection.execute(
        """
        UPDATE student_blueprints
        SET subject = COALESCE(
            NULLIF(subject, ''),
            (SELECT subject FROM classes WHERE classes.id = student_blueprints.class_id),
            ''
        )
        WHERE subject = '' OR subject IS NULL
        """
    )
    connection.execute(
        """
        INSERT INTO class_subjects (class_id, subject, medium)
        SELECT c.id, c.subject, c.medium
        FROM classes c
        WHERE c.subject IS NOT NULL AND c.subject != ''
          AND NOT EXISTS (
              SELECT 1
              FROM class_subjects cs
              WHERE cs.class_id = c.id AND cs.subject = c.subject
          )
        """
    )
    connection.execute(
        """
        INSERT INTO curriculum_subjects (board_type, grade, subject, default_medium)
        SELECT DISTINCT s.board_type, c.grade, cs.subject, cs.medium
        FROM class_subjects cs
        JOIN classes c ON c.id = cs.class_id
        JOIN schools s ON s.id = c.school_id
        WHERE NOT EXISTS (
            SELECT 1
            FROM curriculum_subjects cur
            WHERE cur.board_type = s.board_type
              AND cur.grade = c.grade
              AND cur.subject = cs.subject
        )
        """
    )
    connection.execute(
        """
        INSERT INTO curriculum_chapters (curriculum_subject_id, chapter_code, chapter_name, chapter_order, term)
        SELECT cur.id, ch.chapter_code, ch.chapter_name,
               ROW_NUMBER() OVER (PARTITION BY cur.id ORDER BY ch.id), ch.term
        FROM chapters ch
        JOIN curriculum_subjects cur
          ON cur.board_type = ch.board_type
         AND cur.grade = ch.grade
         AND cur.subject = ch.subject
        WHERE NOT EXISTS (
            SELECT 1
            FROM curriculum_chapters cch
            WHERE cch.curriculum_subject_id = cur.id
              AND cch.chapter_code = ch.chapter_code
          )
        """
    )
    _cleanup_decimal_subtopic_chapters(connection)


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def _cleanup_decimal_subtopic_chapters(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT cc.id, cc.chapter_code, cc.chapter_name, cur.board_type, cur.grade, cur.subject
        FROM curriculum_chapters cc
        JOIN curriculum_subjects cur ON cur.id = cc.curriculum_subject_id
        """
    ).fetchall()
    ids_to_delete: list[int] = []
    legacy_rows_to_delete: list[tuple[str, str, str, str]] = []
    for row in rows:
        chapter_id, chapter_code, chapter_name, board_type, grade, subject = row
        code = str(chapter_code or "").strip()
        name = str(chapter_name or "").strip()
        is_decimal_code = bool(re.fullmatch(r"\d+\.\d+(?:\.\d+)*", code))
        is_decimal_name = bool(re.match(r"^\d+\.\d+(?:\.\d+)*\s*[|:-]?\s*", name))
        if is_decimal_code or is_decimal_name:
            ids_to_delete.append(int(chapter_id))
            legacy_rows_to_delete.append(
                (str(board_type), str(grade), str(subject), code)
            )
    if ids_to_delete:
        connection.executemany(
            "DELETE FROM curriculum_chapters WHERE id = ?",
            [(item_id,) for item_id in ids_to_delete],
        )
    if legacy_rows_to_delete:
        connection.executemany(
            """
            DELETE FROM chapters
            WHERE board_type = ? AND grade = ? AND subject = ? AND chapter_code = ?
            """,
            legacy_rows_to_delete,
        )
