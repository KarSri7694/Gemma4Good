from __future__ import annotations

import sqlite3
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


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")
