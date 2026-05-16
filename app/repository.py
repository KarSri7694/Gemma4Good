from __future__ import annotations

"""Repository layer for persistent classroom data access.

Keeping SQL here prevents the Streamlit UI and AI service layers from each
implementing their own storage rules.
"""

import json
from typing import Any

from app.db import ensure_database, get_connection


def get_teacher() -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT t.id, t.full_name, t.google_account_email, s.name AS school_name
            FROM teachers t
            JOIN schools s ON s.id = t.school_id
            ORDER BY t.id
            LIMIT 1
            """
        ).fetchone()
    return dict(row) if row else None


def create_class_for_teacher(
    *,
    teacher_id: int,
    academic_year: str,
    grade: str,
    section: str,
    subject: str,
    medium: str,
) -> int:
    """Create a class row and seed its primary subject mapping."""
    with get_connection() as connection:
        teacher_row = connection.execute(
            "SELECT school_id FROM teachers WHERE id = ?",
            (teacher_id,),
        ).fetchone()
        if not teacher_row:
            raise ValueError("Teacher not found.")
        class_id = connection.execute(
            """
            INSERT INTO classes (school_id, teacher_id, academic_year, grade, section, subject, medium)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                teacher_row["school_id"],
                teacher_id,
                academic_year.strip(),
                grade.strip(),
                section.strip(),
                subject.strip(),
                medium.strip(),
            ),
        ).lastrowid
        connection.execute(
            """
            INSERT OR IGNORE INTO class_subjects (class_id, subject, medium)
            VALUES (?, ?, ?)
            """,
            (class_id, subject.strip(), medium.strip()),
        )
        connection.commit()
    return int(class_id)


def update_class_details(
    *,
    class_id: int,
    academic_year: str,
    grade: str,
    section: str,
    subject: str,
    medium: str,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE classes
            SET academic_year = ?, grade = ?, section = ?, subject = ?, medium = ?
            WHERE id = ?
            """,
            (
                academic_year.strip(),
                grade.strip(),
                section.strip(),
                subject.strip(),
                medium.strip(),
                class_id,
            ),
        )
        connection.commit()


def list_teacher_classes(teacher_id: int) -> list[dict[str, Any]]:
    """Return classes with small aggregates for selector and dashboard use."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.id,
                c.grade,
                c.section,
                c.subject,
                c.medium,
                c.academic_year,
                COALESCE(
                    (SELECT COUNT(*) FROM class_subjects cs WHERE cs.class_id = c.id),
                    0
                ) AS subject_count,
                COALESCE(
                    (SELECT GROUP_CONCAT(cs.subject, ', ') FROM class_subjects cs WHERE cs.class_id = c.id),
                    c.subject
                ) AS subjects_csv,
                COUNT(DISTINCT s.id) AS student_count,
                COUNT(DISTINCT a.id) AS assessment_count
            FROM classes c
            LEFT JOIN students s ON s.class_id = c.id AND s.status = 'active'
            LEFT JOIN assessments a ON a.class_id = c.id
            WHERE c.teacher_id = ?
            GROUP BY c.id, c.grade, c.section, c.subject, c.medium, c.academic_year
            ORDER BY assessment_count DESC, c.subject, c.grade, c.section
            """,
            (teacher_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_class_overview(class_id: int) -> dict[str, Any]:
    """Return a single summary payload reused across multiple screens."""
    with get_connection() as connection:
        overview = connection.execute(
            """
            SELECT
                c.id,
                c.grade,
                c.section,
                c.subject,
                c.medium,
                c.academic_year,
                COALESCE(
                    (SELECT COUNT(*) FROM class_subjects cs WHERE cs.class_id = c.id),
                    0
                ) AS subject_count,
                COALESCE(
                    (SELECT GROUP_CONCAT(cs.subject, ', ') FROM class_subjects cs WHERE cs.class_id = c.id),
                    c.subject
                ) AS subjects_csv,
                COUNT(DISTINCT s.id) AS student_count,
                COUNT(DISTINCT a.id) AS assessment_count,
                ROUND(AVG(sa.percentage), 1) AS avg_percentage
            FROM classes c
            LEFT JOIN students s ON s.class_id = c.id AND s.status = 'active'
            LEFT JOIN assessments a ON a.class_id = c.id
            LEFT JOIN student_assessments sa ON sa.assessment_id = a.id
            WHERE c.id = ?
            GROUP BY c.id, c.grade, c.section, c.subject, c.medium, c.academic_year
            """,
            (class_id,),
        ).fetchone()
    return dict(overview) if overview else {}


def list_class_students(class_id: int) -> list[dict[str, Any]]:
    """Return active students with assessment and attendance rollups."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                s.id,
                s.roll_number,
                s.full_name,
                s.preferred_language,
                s.accessibility_notes,
                ROUND(AVG(sa.percentage), 1) AS avg_percentage,
                SUM(CASE WHEN scm.status = 'lagging' THEN 1 ELSE 0 END) AS lagging_concepts,
                (
                    SELECT COUNT(*)
                    FROM attendance_records ar
                    WHERE ar.student_id = s.id
                ) AS attendance_days_recorded,
                (
                    SELECT COUNT(*)
                    FROM attendance_records ar
                    WHERE ar.student_id = s.id AND ar.status = 'present'
                ) AS attendance_days_present,
                (
                    SELECT MIN(ar.attendance_date)
                    FROM attendance_records ar
                    WHERE ar.student_id = s.id
                ) AS attendance_started_on
            FROM students s
            LEFT JOIN student_assessments sa ON sa.student_id = s.id
            LEFT JOIN student_concept_mastery scm ON scm.student_id = s.id AND scm.class_id = s.class_id
            WHERE s.class_id = ? AND s.status = 'active'
            GROUP BY s.id, s.roll_number, s.full_name, s.preferred_language, s.accessibility_notes
            ORDER BY s.roll_number
            """,
            (class_id,),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        recorded = item.get("attendance_days_recorded") or 0
        present = item.get("attendance_days_present") or 0
        # Compute the percentage once in the repository layer so every UI view
        # shows the same attendance value.
        item["attendance_percentage"] = round((present / recorded) * 100, 1) if recorded else None
        items.append(item)
    return items


def list_class_roster(class_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, roll_number, full_name, preferred_language
            FROM students
            WHERE class_id = ? AND status = 'active'
            ORDER BY roll_number
            """,
            (class_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_inactive_class_students(class_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, roll_number, full_name, preferred_language
            FROM students
            WHERE class_id = ? AND status = 'inactive'
            ORDER BY roll_number
            """,
            (class_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def find_student_id(student_name: str = "", roll_number: str = "") -> int | None:
    normalized_name = student_name.strip().lower()
    normalized_roll = roll_number.strip()
    with get_connection() as connection:
        if normalized_roll:
            row = connection.execute(
                """
                SELECT id
                FROM students
                WHERE roll_number = ? AND status = 'active'
                ORDER BY id
                LIMIT 1
                """,
                (normalized_roll,),
            ).fetchone()
            if row:
                return int(row["id"])

        if normalized_name:
            row = connection.execute(
                """
                SELECT id
                FROM students
                WHERE LOWER(full_name) = ? AND status = 'active'
                ORDER BY id
                LIMIT 1
                """,
                (normalized_name,),
            ).fetchone()
            if row:
                return int(row["id"])

            row = connection.execute(
                """
                SELECT id
                FROM students
                WHERE LOWER(full_name) LIKE ? AND status = 'active'
                ORDER BY id
                LIMIT 1
                """,
                (f"%{normalized_name}%",),
            ).fetchone()
            if row:
                return int(row["id"])
    return None


def list_class_subjects(class_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, class_id, subject, medium
            FROM class_subjects
            WHERE class_id = ?
            ORDER BY subject
            """,
            (class_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def find_teacher_and_class_for_subject(subject: str) -> tuple[int | None, dict[str, Any] | None]:
    normalized_subject = subject.strip().lower()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                t.id AS teacher_id,
                c.id AS class_id,
                c.grade,
                c.section,
                cs.subject
            FROM classes c
            JOIN teachers t ON t.id = c.teacher_id
            JOIN class_subjects cs ON cs.class_id = c.id
            WHERE LOWER(cs.subject) = ?
            ORDER BY c.id
            LIMIT 1
            """,
            (normalized_subject,),
        ).fetchone()
    if not row:
        return None, None
    return int(row["teacher_id"]), dict(row)


def find_class_for_attendance(grade: str, section: str, subject: str) -> tuple[int | None, dict[str, Any] | None]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                t.id AS teacher_id,
                c.id AS class_id,
                c.grade,
                c.section,
                cs.subject
            FROM classes c
            JOIN teachers t ON t.id = c.teacher_id
            JOIN class_subjects cs ON cs.class_id = c.id
            WHERE c.grade = ? AND LOWER(c.section) = ? AND LOWER(cs.subject) = ?
            ORDER BY c.id
            LIMIT 1
            """,
            (grade.strip(), section.strip().lower(), subject.strip().lower()),
        ).fetchone()
    if not row:
        return None, None
    return int(row["teacher_id"]), dict(row)


def find_class_for_management(
    grade: str,
    section: str,
    subject: str = "",
) -> tuple[int | None, dict[str, Any] | None]:
    normalized_subject = subject.strip().lower()
    with get_connection() as connection:
        if normalized_subject:
            row = connection.execute(
                """
                SELECT
                    t.id AS teacher_id,
                    c.id AS class_id,
                    c.grade,
                    c.section,
                    cs.subject,
                    c.academic_year,
                    COALESCE(cs.medium, c.medium) AS medium
                FROM classes c
                JOIN teachers t ON t.id = c.teacher_id
                JOIN class_subjects cs ON cs.class_id = c.id
                WHERE c.grade = ? AND LOWER(c.section) = ? AND LOWER(cs.subject) = ?
                ORDER BY c.id
                LIMIT 1
                """,
                (grade.strip(), section.strip().lower(), normalized_subject),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT
                    t.id AS teacher_id,
                    c.id AS class_id,
                    c.grade,
                    c.section,
                    c.subject,
                    c.academic_year,
                    c.medium
                FROM classes c
                JOIN teachers t ON t.id = c.teacher_id
                WHERE c.grade = ? AND LOWER(c.section) = ?
                ORDER BY c.id
                LIMIT 1
                """,
                (grade.strip(), section.strip().lower()),
            ).fetchone()
    if not row:
        return None, None
    return int(row["teacher_id"]), dict(row)


def list_grade_curriculum_subjects(grade: str) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, board_type, grade, subject, default_medium, created_at, updated_at
            FROM curriculum_subjects
            WHERE grade = ?
            ORDER BY subject
            """,
            (grade.strip(),),
        ).fetchall()
    return [dict(row) for row in rows]


def get_curriculum_subject(*, grade: str, subject: str) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, board_type, grade, subject, default_medium, created_at, updated_at
            FROM curriculum_subjects
            WHERE grade = ? AND LOWER(subject) = LOWER(?)
            LIMIT 1
            """,
            (grade.strip(), subject.strip()),
        ).fetchone()
    return dict(row) if row else None


def ensure_curriculum_subject(
    *,
    board_type: str,
    grade: str,
    subject: str,
    default_medium: str = "",
) -> int:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO curriculum_subjects (board_type, grade, subject, default_medium)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(board_type, grade, subject) DO UPDATE SET
                default_medium = COALESCE(NULLIF(excluded.default_medium, ''), curriculum_subjects.default_medium),
                updated_at = CURRENT_TIMESTAMP
            """,
            (board_type.strip(), grade.strip(), subject.strip(), default_medium.strip()),
        )
        row = connection.execute(
            """
            SELECT id
            FROM curriculum_subjects
            WHERE board_type = ? AND grade = ? AND subject = ?
            LIMIT 1
            """,
            (board_type.strip(), grade.strip(), subject.strip()),
        ).fetchone()
        if row:
            class_rows = connection.execute(
                """
                SELECT id
                FROM classes
                WHERE grade = ?
                """,
                (grade.strip(),),
            ).fetchall()
            for class_row in class_rows:
                connection.execute(
                    """
                    INSERT INTO class_subjects (class_id, subject, medium)
                    VALUES (?, ?, ?)
                    ON CONFLICT(class_id, subject) DO UPDATE SET
                        medium = COALESCE(NULLIF(excluded.medium, ''), class_subjects.medium)
                    """,
                    (class_row["id"], subject.strip(), default_medium.strip()),
                )
            connection.commit()
            return int(row["id"])
        connection.commit()
    raise ValueError("Unable to create or resolve curriculum subject.")


def upsert_curriculum_chapters(
    *,
    curriculum_subject_id: int,
    board_type: str,
    grade: str,
    subject: str,
    chapters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    created_rows: list[dict[str, Any]] = []
    with get_connection() as connection:
        for index, chapter in enumerate(chapters, start=1):
            chapter_code = (chapter.get("chapter_code") or f"{subject[:3].upper()}-{grade}-{index:02d}").strip()
            chapter_name = (chapter.get("chapter_name") or f"Chapter {index}").strip()
            term = str(chapter.get("term") or "").strip()
            chapter_order = int(chapter.get("chapter_order") or index)
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
                (curriculum_subject_id, chapter_code, chapter_name, chapter_order, term),
            )
            connection.execute(
                """
                INSERT INTO chapters (board_type, grade, subject, chapter_code, chapter_name, term)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(board_type, grade, subject, chapter_code) DO UPDATE SET
                    chapter_name = excluded.chapter_name,
                    term = excluded.term
                """,
                (board_type.strip(), grade.strip(), subject.strip(), chapter_code, chapter_name, term),
            )
            row = connection.execute(
                """
                SELECT id, chapter_code, chapter_name, chapter_order, term
                FROM curriculum_chapters
                WHERE curriculum_subject_id = ? AND chapter_code = ?
                LIMIT 1
                """,
                (curriculum_subject_id, chapter_code),
            ).fetchone()
            if row:
                created_rows.append(dict(row))
        connection.commit()
    return created_rows


def list_curriculum_chapters(curriculum_subject_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, chapter_code, chapter_name, chapter_order, term, created_at, updated_at
            FROM curriculum_chapters
            WHERE curriculum_subject_id = ?
            ORDER BY COALESCE(chapter_order, 9999), chapter_name
            """,
            (curriculum_subject_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_subject_to_class(
    *,
    class_id: int,
    subject: str,
    medium: str = "",
) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO class_subjects (class_id, subject, medium)
            VALUES (?, ?, ?)
            """,
            (class_id, subject.strip(), medium.strip()),
        )
        connection.commit()
    return int(cursor.lastrowid)


def update_class_subject_details(
    *,
    class_subject_id: int,
    subject: str,
    medium: str = "",
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE class_subjects
            SET subject = ?, medium = ?
            WHERE id = ?
            """,
            (subject.strip(), medium.strip(), class_subject_id),
        )
        connection.commit()


def add_student_to_class(
    *,
    class_id: int,
    roll_number: str,
    full_name: str,
    email: str = "",
    preferred_language: str = "",
    accessibility_notes: str = "",
) -> int:
    with get_connection() as connection:
        class_row = connection.execute(
            "SELECT school_id FROM classes WHERE id = ?",
            (class_id,),
        ).fetchone()
        if not class_row:
            raise ValueError("Class not found.")
        student_id = connection.execute(
            """
            INSERT INTO students (
                school_id, class_id, roll_number, full_name, email, preferred_language, accessibility_notes, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                class_row["school_id"],
                class_id,
                roll_number.strip(),
                full_name.strip(),
                email.strip(),
                preferred_language.strip(),
                accessibility_notes.strip(),
            ),
        ).lastrowid
        connection.commit()
    return int(student_id)


def update_student_details(
    *,
    student_id: int,
    roll_number: str,
    full_name: str,
    email: str = "",
    preferred_language: str = "",
    accessibility_notes: str = "",
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE students
            SET roll_number = ?, full_name = ?, email = ?, preferred_language = ?, accessibility_notes = ?
            WHERE id = ?
            """,
            (
                roll_number.strip(),
                full_name.strip(),
                email.strip(),
                preferred_language.strip(),
                accessibility_notes.strip(),
                student_id,
            ),
        )
        connection.commit()


def deactivate_student(student_id: int) -> None:
    with get_connection() as connection:
        connection.execute(
            "UPDATE students SET status = 'inactive' WHERE id = ?",
            (student_id,),
        )
        connection.commit()


def reactivate_student(student_id: int) -> None:
    with get_connection() as connection:
        connection.execute(
            "UPDATE students SET status = 'active' WHERE id = ?",
            (student_id,),
        )
        connection.commit()


def list_class_assessments(class_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                a.id,
                a.title,
                a.assessment_type,
                a.delivery_mode,
                a.language,
                a.total_marks,
                a.google_form_url,
                a.created_at,
                ch.chapter_name,
                ROUND(AVG(sa.percentage), 1) AS avg_percentage,
                COUNT(sa.id) AS submissions
            FROM assessments a
            JOIN chapters ch ON ch.id = a.chapter_id
            LEFT JOIN student_assessments sa ON sa.assessment_id = a.id
            WHERE a.class_id = ?
            GROUP BY a.id, a.title, a.assessment_type, a.delivery_mode, a.language,
                     a.total_marks, a.google_form_url, a.created_at, ch.chapter_name
            ORDER BY a.created_at DESC
            """,
            (class_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def list_class_assessment_history(class_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                a.id,
                a.title,
                a.assessment_type,
                a.created_at,
                a.due_at,
                ch.subject,
                ch.chapter_name,
                ROUND(AVG(sa.percentage), 1) AS avg_percentage,
                COUNT(sa.id) AS submissions
            FROM assessments a
            JOIN chapters ch ON ch.id = a.chapter_id
            LEFT JOIN student_assessments sa ON sa.assessment_id = a.id
            WHERE a.class_id = ?
            GROUP BY a.id, a.title, a.assessment_type, a.created_at, a.due_at, ch.subject, ch.chapter_name
            ORDER BY a.created_at DESC
            """,
            (class_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_chapter_for_class(
    *,
    class_id: int,
    subject: str,
    chapter_code: str,
    chapter_name: str,
    term: str = "",
) -> int:
    with get_connection() as connection:
        class_row = connection.execute(
            """
            SELECT c.grade, s.board_type
            FROM classes c
            JOIN schools s ON s.id = c.school_id
            WHERE c.id = ?
            """,
            (class_id,),
        ).fetchone()
        if not class_row:
            raise ValueError("Class not found.")
        chapter_id = connection.execute(
            """
            INSERT INTO chapters (board_type, grade, subject, chapter_code, chapter_name, term)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                class_row["board_type"],
                class_row["grade"],
                subject.strip(),
                chapter_code.strip(),
                chapter_name.strip(),
                term.strip(),
            ),
        ).lastrowid
        connection.commit()
    return int(chapter_id)


def update_chapter_details(
    *,
    chapter_id: int,
    chapter_code: str,
    chapter_name: str,
    term: str = "",
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE chapters
            SET chapter_code = ?, chapter_name = ?, term = ?
            WHERE id = ?
            """,
            (
                chapter_code.strip(),
                chapter_name.strip(),
                term.strip(),
                chapter_id,
            ),
        )
        connection.commit()


def delete_chapter_if_unused(chapter_id: int) -> tuple[bool, str]:
    with get_connection() as connection:
        assessment_row = connection.execute(
            "SELECT COUNT(*) AS count FROM assessments WHERE chapter_id = ?",
            (chapter_id,),
        ).fetchone()
        assessment_count = int(assessment_row["count"]) if assessment_row else 0
        if assessment_count > 0:
            return False, "Chapter is linked to existing assessments and cannot be deleted safely."

        connection.execute("DELETE FROM concepts WHERE chapter_id = ?", (chapter_id,))
        cursor = connection.execute("DELETE FROM chapters WHERE id = ?", (chapter_id,))
        connection.commit()
    if cursor.rowcount:
        return True, "Chapter deleted."
    return False, "Chapter not found."


def upsert_class_attendance(
    *,
    class_id: int,
    teacher_id: int,
    attendance_date: str,
    absent_student_ids: list[int],
    source: str,
    raw_model_output: str = "",
) -> dict[str, Any]:
    roster = list_class_roster(class_id)
    absent_id_set = set(absent_student_ids)
    if not roster:
        return {
            "attendance_date": attendance_date,
            "present_count": 0,
            "absent_count": 0,
            "records": [],
        }

    records: list[dict[str, Any]] = []
    with get_connection() as connection:
        for student in roster:
            status = "absent" if student["id"] in absent_id_set else "present"
            connection.execute(
                """
                INSERT INTO attendance_records (
                    class_id, student_id, teacher_id, attendance_date, status, source, raw_model_output
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(class_id, student_id, attendance_date) DO UPDATE SET
                    teacher_id = excluded.teacher_id,
                    status = excluded.status,
                    source = excluded.source,
                    raw_model_output = excluded.raw_model_output,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    class_id,
                    student["id"],
                    teacher_id,
                    attendance_date,
                    status,
                    source,
                    raw_model_output,
                ),
            )
            records.append(
                {
                    "student_id": student["id"],
                    "roll_number": student["roll_number"],
                    "full_name": student["full_name"],
                    "status": status,
                }
            )
        connection.commit()

    absent_records = [item for item in records if item["status"] == "absent"]
    return {
        "attendance_date": attendance_date,
        "present_count": len(records) - len(absent_records),
        "absent_count": len(absent_records),
        "records": records,
        "absent_students": absent_records,
    }


def list_attendance_for_date(class_id: int, attendance_date: str) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                ar.student_id,
                s.roll_number,
                s.full_name,
                ar.status,
                ar.source,
                ar.updated_at
            FROM attendance_records ar
            JOIN students s ON s.id = ar.student_id
            WHERE ar.class_id = ? AND ar.attendance_date = ?
            ORDER BY s.roll_number
            """,
            (class_id, attendance_date),
        ).fetchall()
    return [dict(row) for row in rows]


def get_attendance_overview(class_id: int, attendance_date: str) -> dict[str, Any]:
    rows = list_attendance_for_date(class_id, attendance_date)
    absent_count = sum(1 for row in rows if row["status"] == "absent")
    present_count = sum(1 for row in rows if row["status"] == "present")
    return {
        "attendance_date": attendance_date,
        "present_count": present_count,
        "absent_count": absent_count,
        "total_students": len(rows),
    }


def get_class_attendance_stats(class_id: int) -> dict[int, dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                s.id AS student_id,
                COUNT(ar.id) AS attendance_days_recorded,
                SUM(CASE WHEN ar.status = 'present' THEN 1 ELSE 0 END) AS attendance_days_present,
                MIN(ar.attendance_date) AS attendance_started_on,
                MAX(ar.attendance_date) AS attendance_last_marked_on
            FROM students s
            LEFT JOIN attendance_records ar ON ar.student_id = s.id AND ar.class_id = s.class_id
            WHERE s.class_id = ? AND s.status = 'active'
            GROUP BY s.id
            """,
            (class_id,),
        ).fetchall()
    stats: dict[int, dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        recorded = item.get("attendance_days_recorded") or 0
        present = item.get("attendance_days_present") or 0
        item["attendance_percentage"] = round((present / recorded) * 100, 1) if recorded else None
        stats[int(item["student_id"])] = item
    return stats


def update_student_attendance_status(
    *,
    class_id: int,
    student_id: int,
    teacher_id: int,
    attendance_date: str,
    status: str,
    source: str = "manual",
    raw_model_output: str = "",
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO attendance_records (
                class_id, student_id, teacher_id, attendance_date, status, source, raw_model_output
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(class_id, student_id, attendance_date) DO UPDATE SET
                teacher_id = excluded.teacher_id,
                status = excluded.status,
                source = excluded.source,
                raw_model_output = excluded.raw_model_output,
                updated_at = CURRENT_TIMESTAMP
            """,
            (class_id, student_id, teacher_id, attendance_date, status, source, raw_model_output),
        )
        connection.commit()


def clear_student_attendance(student_id: int, class_id: int | None = None) -> int:
    with get_connection() as connection:
        if class_id is None:
            cursor = connection.execute(
                "DELETE FROM attendance_records WHERE student_id = ?",
                (student_id,),
            )
        else:
            cursor = connection.execute(
                "DELETE FROM attendance_records WHERE student_id = ? AND class_id = ?",
                (student_id, class_id),
            )
        connection.commit()
    return cursor.rowcount


def clear_class_attendance(class_id: int) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM attendance_records WHERE class_id = ?",
            (class_id,),
        )
        connection.commit()
    return cursor.rowcount


def list_class_concept_gaps(class_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                ccm.concept_id,
                c.concept_name,
                c.description,
                ROUND(ccm.average_mastery_score * 100, 1) AS mastery_percent,
                ccm.students_assessed,
                ccm.students_lagging,
                ccm.last_updated_at
            FROM class_concept_mastery ccm
            JOIN concepts c ON c.id = ccm.concept_id
            WHERE ccm.class_id = ?
            ORDER BY ccm.average_mastery_score ASC, ccm.students_lagging DESC
            """,
            (class_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_student_detail(student_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        student = connection.execute(
            """
            SELECT
                s.id,
                s.full_name,
                s.roll_number,
                s.email,
                s.preferred_language,
                s.accessibility_notes,
                c.grade,
                c.section,
                c.subject
            FROM students s
            JOIN classes c ON c.id = s.class_id
            WHERE s.id = ?
            """,
            (student_id,),
        ).fetchone()

        mastery_rows = connection.execute(
            """
            SELECT
                c.concept_name,
                ROUND(scm.mastery_score * 100, 1) AS mastery_percent,
                ROUND(scm.confidence_score * 100, 1) AS confidence_percent,
                scm.questions_attempted,
                scm.questions_correct,
                scm.status
            FROM student_concept_mastery scm
            JOIN concepts c ON c.id = scm.concept_id
            WHERE scm.student_id = ?
            ORDER BY scm.mastery_score ASC
            """,
            (student_id,),
        ).fetchall()

        assessments = connection.execute(
            """
            SELECT
                a.title,
                ch.chapter_name,
                sa.score_obtained,
                sa.percentage,
                sa.submitted_at
            FROM student_assessments sa
            JOIN assessments a ON a.id = sa.assessment_id
            JOIN chapters ch ON ch.id = a.chapter_id
            WHERE sa.student_id = ?
            ORDER BY sa.submitted_at DESC
            """,
            (student_id,),
        ).fetchall()

        recommendations = connection.execute(
            """
            SELECT
                c.concept_name,
                rr.recommendation_type,
                rr.recommendation_text,
                rr.priority
            FROM remediation_recommendations rr
            JOIN concepts c ON c.id = rr.concept_id
            WHERE rr.student_id = ?
            ORDER BY rr.priority DESC, rr.created_at DESC
            """,
            (student_id,),
        ).fetchall()

        blueprint = connection.execute(
            """
            SELECT subject, strengths_json, weaknesses_json, opportunities_json, threats_json,
                   recommendations_json, narrative, generated_by_model,
                   based_on_assessments, last_submission_at, updated_at
            FROM student_blueprints
            WHERE student_id = ?
            ORDER BY subject
            """,
            (student_id,),
        ).fetchall()

        attendance_summary = connection.execute(
            """
            SELECT
                COUNT(*) AS attendance_days_recorded,
                SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) AS attendance_days_present,
                MIN(attendance_date) AS attendance_started_on,
                MAX(attendance_date) AS attendance_last_marked_on
            FROM attendance_records
            WHERE student_id = ?
            """,
            (student_id,),
        ).fetchone()

        attendance_history = connection.execute(
            """
            SELECT attendance_date, status, source, updated_at
            FROM attendance_records
            WHERE student_id = ?
            ORDER BY attendance_date DESC
            LIMIT 60
            """,
            (student_id,),
        ).fetchall()

        adaptation_profiles = connection.execute(
            """
            SELECT subject, profile_json, summary, generated_by_model,
                   based_on_assessments, last_submission_at, updated_at
            FROM student_adaptation_profiles
            WHERE student_id = ?
            ORDER BY subject
            """,
            (student_id,),
        ).fetchall()

    attendance_summary_dict = dict(attendance_summary) if attendance_summary else {}
    attendance_days_recorded = attendance_summary_dict.get("attendance_days_recorded") or 0
    attendance_days_present = attendance_summary_dict.get("attendance_days_present") or 0
    return {
        "student": dict(student) if student else None,
        "mastery": [dict(row) for row in mastery_rows],
        "assessments": [dict(row) for row in assessments],
        "recommendations": [dict(row) for row in recommendations],
        "attendance_summary": {
            **attendance_summary_dict,
            "attendance_percentage": (
                round((attendance_days_present / attendance_days_recorded) * 100, 1)
                if attendance_days_recorded
                else None
            ),
        },
        "attendance_history": [dict(row) for row in attendance_history],
        "adaptation_profiles": [
            {
                **dict(row),
                "profile": json.loads(row["profile_json"] or "{}"),
            }
            for row in adaptation_profiles
        ],
        "blueprints": [
            {
                **dict(row),
                "strengths": json.loads(row["strengths_json"] or "[]"),
                "weaknesses": json.loads(row["weaknesses_json"] or "[]"),
                "opportunities": json.loads(row["opportunities_json"] or "[]"),
                "threats": json.loads(row["threats_json"] or "[]"),
                "recommendations": json.loads(row["recommendations_json"] or "[]"),
            }
            for row in blueprint
        ],
    }


def list_chapters_for_class(class_id: int, subject: str | None = None) -> list[dict[str, Any]]:
    with get_connection() as connection:
        class_row = connection.execute(
            "SELECT grade, subject FROM classes WHERE id = ?",
            (class_id,),
        ).fetchone()
        if not class_row:
            return []

        resolved_subject = (subject or class_row["subject"]).strip()

        rows = connection.execute(
            """
            SELECT id, chapter_code, chapter_name, term
            FROM chapters
            WHERE grade = ? AND subject = ?
            ORDER BY chapter_name
            """,
            (class_row["grade"], resolved_subject),
        ).fetchall()
    return [dict(row) for row in rows]


def find_chapter_for_class_topic(class_id: int, topic: str) -> dict[str, Any] | None:
    normalized_topic = topic.strip().lower()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, chapter_name
            FROM chapters
            WHERE id IN (
                SELECT ch.id
                FROM chapters ch
                JOIN classes c ON c.grade = ch.grade
                JOIN class_subjects cs ON cs.class_id = c.id AND cs.subject = ch.subject
                WHERE c.id = ?
            )
            AND LOWER(chapter_name) = ?
            ORDER BY chapter_name
            LIMIT 1
            """,
            (class_id, normalized_topic),
        ).fetchone()
        if row:
            return dict(row)

        row = connection.execute(
            """
            SELECT id, chapter_name
            FROM chapters
            WHERE id IN (
                SELECT ch.id
                FROM chapters ch
                JOIN classes c ON c.grade = ch.grade
                JOIN class_subjects cs ON cs.class_id = c.id AND cs.subject = ch.subject
                WHERE c.id = ?
            )
            AND LOWER(chapter_name) LIKE ?
            ORDER BY chapter_name
            LIMIT 1
            """,
            (class_id, f"%{normalized_topic}%"),
        ).fetchone()
    return dict(row) if row else None


def find_fallback_chapter_for_class(class_id: int) -> dict[str, Any] | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT ch.id, ch.chapter_name
            FROM chapters ch
            JOIN classes c ON c.grade = ch.grade AND c.subject = ch.subject
            WHERE c.id = ?
            ORDER BY ch.chapter_name
            LIMIT 1
            """,
            (class_id,),
        ).fetchone()
    return dict(row) if row else None


def find_chapter_for_management(class_id: int, chapter_name: str) -> dict[str, Any] | None:
    normalized_name = chapter_name.strip().lower()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, chapter_code, chapter_name, term
            FROM chapters
            WHERE id IN (
                SELECT ch.id
                FROM chapters ch
                JOIN classes c ON c.grade = ch.grade AND c.subject = ch.subject
                WHERE c.id = ?
            )
            AND LOWER(chapter_name) = ?
            ORDER BY chapter_name
            LIMIT 1
            """,
            (class_id, normalized_name),
        ).fetchone()
        if row:
            return dict(row)
        row = connection.execute(
            """
            SELECT id, chapter_code, chapter_name, term
            FROM chapters
            WHERE id IN (
                SELECT ch.id
                FROM chapters ch
                JOIN classes c ON c.grade = ch.grade AND c.subject = ch.subject
                WHERE c.id = ?
            )
            AND LOWER(chapter_name) LIKE ?
            ORDER BY chapter_name
            LIMIT 1
            """,
            (class_id, f"%{normalized_name}%"),
        ).fetchone()
    return dict(row) if row else None


def create_source_material(
    *,
    curriculum_subject_id: int,
    uploaded_by_teacher_id: int,
    title: str,
    source_type: str,
    original_filename: str = "",
    mime_type: str = "",
    storage_path: str = "",
    raw_text: str = "",
    extraction_summary: str = "",
) -> int:
    with get_connection() as connection:
        material_id = connection.execute(
            """
            INSERT INTO source_materials (
                curriculum_subject_id, uploaded_by_teacher_id, title, source_type,
                original_filename, mime_type, storage_path, raw_text, extraction_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                curriculum_subject_id,
                uploaded_by_teacher_id,
                title.strip(),
                source_type.strip(),
                original_filename.strip(),
                mime_type.strip(),
                storage_path.strip(),
                raw_text,
                extraction_summary.strip(),
            ),
        ).lastrowid
        connection.commit()
    return int(material_id)


def update_source_material(
    *,
    material_id: int,
    extraction_summary: str = "",
    raw_text: str = "",
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE source_materials
            SET extraction_summary = ?, raw_text = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (extraction_summary.strip(), raw_text, material_id),
        )
        connection.commit()


def create_ingestion_run(
    *,
    source_material_id: int,
    status: str,
    extraction_summary: str = "",
    raw_structure_json: str = "",
    error_text: str = "",
) -> int:
    with get_connection() as connection:
        run_id = connection.execute(
            """
            INSERT INTO material_ingestion_runs (
                source_material_id, status, extraction_summary, raw_structure_json, error_text
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                source_material_id,
                status.strip(),
                extraction_summary.strip(),
                raw_structure_json,
                error_text,
            ),
        ).lastrowid
        connection.commit()
    return int(run_id)


def update_ingestion_run(
    *,
    run_id: int,
    status: str,
    extraction_summary: str = "",
    raw_structure_json: str = "",
    error_text: str = "",
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE material_ingestion_runs
            SET status = ?, extraction_summary = ?, raw_structure_json = ?, error_text = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status.strip(), extraction_summary.strip(), raw_structure_json, error_text, run_id),
        )
        connection.commit()


def replace_material_chunks(
    *,
    source_material_id: int,
    curriculum_subject_id: int,
    chunks: list[dict[str, Any]],
) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM material_chunks WHERE source_material_id = ?", (source_material_id,))
        for index, chunk in enumerate(chunks):
            connection.execute(
                """
                INSERT INTO material_chunks (
                    source_material_id, curriculum_subject_id, chunk_index, chunk_text,
                    page_start, page_end, section_heading, content_type, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_material_id,
                    curriculum_subject_id,
                    int(chunk.get("chunk_index", index)),
                    chunk.get("chunk_text", ""),
                    chunk.get("page_start"),
                    chunk.get("page_end"),
                    chunk.get("section_heading"),
                    chunk.get("content_type", "text"),
                    json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                ),
            )
        connection.commit()


def list_subject_materials(*, grade: str, subject: str) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT sm.id, sm.title, sm.source_type, sm.original_filename, sm.mime_type,
                   sm.storage_path, sm.extraction_summary, sm.created_at, sm.updated_at,
                   cur.id AS curriculum_subject_id, cur.grade, cur.subject
            FROM source_materials sm
            JOIN curriculum_subjects cur ON cur.id = sm.curriculum_subject_id
            WHERE cur.grade = ? AND LOWER(cur.subject) = LOWER(?)
            ORDER BY sm.created_at DESC
            """,
            (grade.strip(), subject.strip()),
        ).fetchall()
    return [dict(row) for row in rows]


def list_material_chunks_for_subject(*, grade: str, subject: str) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT mc.id, mc.source_material_id, mc.curriculum_subject_id, mc.chunk_index, mc.chunk_text,
                   mc.page_start, mc.page_end, mc.section_heading, mc.content_type, mc.metadata_json,
                   sm.title AS source_title
            FROM material_chunks mc
            JOIN curriculum_subjects cur ON cur.id = mc.curriculum_subject_id
            JOIN source_materials sm ON sm.id = mc.source_material_id
            WHERE cur.grade = ? AND LOWER(cur.subject) = LOWER(?)
            ORDER BY sm.created_at DESC, mc.chunk_index ASC
            """,
            (grade.strip(), subject.strip()),
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item["metadata_json"] or "{}")
        items.append(item)
    return items


def list_recent_ingestion_runs(limit: int = 20) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT mir.id, mir.source_material_id, mir.status, mir.extraction_summary,
                   mir.error_text, mir.created_at, mir.updated_at, sm.title
            FROM material_ingestion_runs mir
            JOIN source_materials sm ON sm.id = mir.source_material_id
            ORDER BY mir.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_assessment(
    *,
    class_id: int,
    chapter_id: int,
    teacher_id: int,
    title: str,
    language: str,
    assessment_type: str,
    questions: list[dict[str, Any]],
    due_at: str | None = None,
) -> int:
    total_marks = sum(question["marks"] for question in questions)

    with get_connection() as connection:
        assessment_id = connection.execute(
            """
            INSERT INTO assessments (
                class_id, chapter_id, teacher_id, title, assessment_type,
                delivery_mode, language, total_marks, due_at
            )
            VALUES (?, ?, ?, ?, ?, 'google_form', ?, ?, ?)
            """,
            (class_id, chapter_id, teacher_id, title, assessment_type, language, total_marks, due_at),
        ).lastrowid

        for index, question in enumerate(questions, start=1):
            connection.execute(
                """
                INSERT INTO assessment_questions (
                    assessment_id, question_number, question_text, question_type, options_json,
                    difficulty, bloom_level, marks, correct_answer, explanation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment_id,
                    index,
                    question["question_text"],
                    question["question_type"],
                    json.dumps(question.get("options", {}), ensure_ascii=False),
                    question["difficulty"],
                    question["bloom_level"],
                    question["marks"],
                    question["correct_answer"],
                    question["explanation"],
                ),
            )

        connection.execute(
            """
            INSERT INTO google_sync_logs (assessment_id, sync_type, status, message)
            VALUES (?, 'form_create', 'pending', ?)
            """,
            (assessment_id, "Quiz drafted locally. Ready for Google Forms sync."),
        )
        connection.commit()

    return assessment_id


def update_assessment_google_form_info(
    *,
    assessment_id: int,
    google_form_id: str,
    google_form_url: str,
    question_id_map: list[dict[str, Any]],
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE assessments
            SET google_form_id = ?, google_form_url = ?
            WHERE id = ?
            """,
            (google_form_id, google_form_url, assessment_id),
        )

        for row in question_id_map:
            connection.execute(
                """
                UPDATE assessment_questions
                SET google_question_id = ?
                WHERE assessment_id = ? AND question_number = ?
                """,
                (row["google_question_id"], assessment_id, row["question_number"]),
            )

        connection.execute(
            """
            INSERT INTO google_sync_logs (assessment_id, sync_type, external_id, status, message)
            VALUES (?, 'form_create', ?, 'success', ?)
            """,
            (assessment_id, google_form_id, "Google Form draft created and question IDs stored."),
        )
        connection.commit()


def list_assessment_questions(assessment_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, question_number, question_text, question_type, options_json,
                   difficulty, bloom_level, marks, correct_answer, explanation, google_question_id
            FROM assessment_questions
            WHERE assessment_id = ?
            ORDER BY question_number
            """,
            (assessment_id,),
        ).fetchall()

    items = []
    for row in rows:
        item = dict(row)
        item["options"] = json.loads(item["options_json"] or "{}")
        items.append(item)
    return items


def list_assessments_for_sync(class_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title, google_form_id, google_form_url, language, created_at
            FROM assessments
            WHERE class_id = ? AND google_form_id IS NOT NULL
            ORDER BY created_at DESC
            """,
            (class_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_assessment_sync_bundle(assessment_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        assessment = connection.execute(
            """
            SELECT a.id, a.class_id, a.chapter_id, a.title, a.google_form_id, a.google_form_url, c.subject
            FROM assessments a
            JOIN classes c ON c.id = a.class_id
            WHERE a.id = ?
            """,
            (assessment_id,),
        ).fetchone()
        if not assessment:
            return {}

        students = connection.execute(
            """
            SELECT id, full_name, email
            FROM students
            WHERE class_id = ? AND status = 'active'
            """,
            (assessment["class_id"],),
        ).fetchall()

        questions = connection.execute(
            """
            SELECT aq.id, aq.question_number, aq.question_text, aq.question_type,
                   aq.marks, aq.correct_answer, aq.explanation,
                   aq.google_question_id, aq.options_json
            FROM assessment_questions aq
            WHERE aq.assessment_id = ?
            ORDER BY aq.question_number
            """,
            (assessment_id,),
        ).fetchall()

        question_concepts = connection.execute(
            """
            SELECT aq.id AS assessment_question_id, qc.concept_id, qc.weightage
            FROM assessment_questions aq
            JOIN question_concepts qc ON qc.assessment_question_id = aq.id
            WHERE aq.assessment_id = ?
            """,
            (assessment_id,),
        ).fetchall()

    return {
        "assessment": dict(assessment),
        "students_by_email": {
            row["email"].strip().lower(): dict(row)
            for row in students
            if row["email"]
        },
        "questions_by_google_id": {
            row["google_question_id"]: {
                **dict(row),
                "options": json.loads(row["options_json"] or "{}"),
            }
            for row in questions
            if row["google_question_id"]
        },
        "question_concepts": [dict(row) for row in question_concepts],
    }


def upsert_google_form_response_sync(
    *,
    assessment_id: int,
    student_id: int,
    response_id: str,
    submitted_at: str | None,
    score_obtained: float,
    percentage: float,
    answers: list[dict[str, Any]],
) -> None:
    with get_connection() as connection:
        student_assessment = connection.execute(
            """
            INSERT INTO student_assessments (
                assessment_id, student_id, status, score_obtained, percentage, submitted_at, graded_at
            )
            VALUES (?, ?, 'graded', ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(assessment_id, student_id) DO UPDATE SET
                status = 'graded',
                score_obtained = excluded.score_obtained,
                percentage = excluded.percentage,
                submitted_at = excluded.submitted_at,
                graded_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (assessment_id, student_id, score_obtained, percentage, submitted_at),
        ).fetchone()
        student_assessment_id = student_assessment["id"]

        for answer in answers:
            connection.execute(
                """
                INSERT INTO student_answers (
                    student_assessment_id, assessment_question_id, raw_answer, normalized_answer,
                    is_correct, score_awarded, feedback, grading_reasoning, error_type, processed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(student_assessment_id, assessment_question_id) DO UPDATE SET
                    raw_answer = excluded.raw_answer,
                    normalized_answer = excluded.normalized_answer,
                    is_correct = excluded.is_correct,
                    score_awarded = excluded.score_awarded,
                    feedback = excluded.feedback,
                    grading_reasoning = excluded.grading_reasoning,
                    error_type = excluded.error_type,
                    processed_at = CURRENT_TIMESTAMP
                """,
                (
                    student_assessment_id,
                    answer["assessment_question_id"],
                    answer["raw_answer"],
                    answer["normalized_answer"],
                    answer["is_correct"],
                    answer["score_awarded"],
                    answer["feedback"],
                    answer.get("grading_reasoning", ""),
                    answer["error_type"],
                ),
            )

        connection.execute(
            """
            INSERT INTO google_sync_logs (assessment_id, sync_type, external_id, status, message)
            VALUES (?, 'response_fetch', ?, 'success', ?)
            """,
            (assessment_id, response_id, f"Synced response {response_id} into analytics."),
        )
        connection.commit()


def replace_mastery_snapshots(
    *,
    class_id: int,
    student_mastery_rows: list[dict[str, Any]],
    class_mastery_rows: list[dict[str, Any]],
) -> None:
    with get_connection() as connection:
        connection.execute("DELETE FROM student_concept_mastery WHERE class_id = ?", (class_id,))
        connection.execute("DELETE FROM class_concept_mastery WHERE class_id = ?", (class_id,))

        for row in student_mastery_rows:
            connection.execute(
                """
                INSERT INTO student_concept_mastery (
                    student_id, concept_id, class_id, mastery_score, confidence_score,
                    questions_attempted, questions_correct, last_assessed_at, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (
                    row["student_id"],
                    row["concept_id"],
                    class_id,
                    row["mastery_score"],
                    row["confidence_score"],
                    row["questions_attempted"],
                    row["questions_correct"],
                    row["status"],
                ),
            )

        for row in class_mastery_rows:
            connection.execute(
                """
                INSERT INTO class_concept_mastery (
                    class_id, concept_id, average_mastery_score, students_assessed,
                    students_lagging, last_updated_at
                )
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    class_id,
                    row["concept_id"],
                    row["average_mastery_score"],
                    row["students_assessed"],
                    row["students_lagging"],
                ),
            )
        connection.commit()


def list_attempted_students_for_assessment(assessment_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT s.id AS student_id, s.full_name, s.roll_number, sa.percentage, sa.score_obtained
            FROM student_assessments sa
            JOIN students s ON s.id = sa.student_id
            WHERE sa.assessment_id = ? AND sa.status = 'graded'
            ORDER BY s.roll_number
            """,
            (assessment_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_student_assessment_review(assessment_id: int, student_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        summary = connection.execute(
            """
            SELECT s.full_name, s.roll_number, sa.score_obtained, sa.percentage, a.title
            FROM student_assessments sa
            JOIN students s ON s.id = sa.student_id
            JOIN assessments a ON a.id = sa.assessment_id
            WHERE sa.assessment_id = ? AND sa.student_id = ?
            """,
            (assessment_id, student_id),
        ).fetchone()

        rows = connection.execute(
            """
            SELECT aq.question_number, aq.question_text, aq.question_type, aq.options_json,
                   aq.marks, aq.correct_answer, sa.raw_answer, sa.score_awarded,
                   sa.feedback, sa.grading_reasoning, sa.is_correct
            FROM student_answers sa
            JOIN student_assessments sat ON sat.id = sa.student_assessment_id
            JOIN assessment_questions aq ON aq.id = sa.assessment_question_id
            WHERE sat.assessment_id = ? AND sat.student_id = ?
            ORDER BY aq.question_number
            """,
            (assessment_id, student_id),
        ).fetchall()

    details = []
    for row in rows:
        item = dict(row)
        item["options"] = json.loads(item["options_json"] or "{}")
        details.append(item)

    return {
        "summary": dict(summary) if summary else None,
        "answers": details,
    }


def get_student_blueprint_context(student_id: int, subject: str) -> dict[str, Any]:
    with get_connection() as connection:
        student = connection.execute(
            """
            SELECT s.id, s.full_name, s.roll_number, s.preferred_language,
                   c.class_id, c.grade, c.section, c.subject
            FROM (
                SELECT s.id, s.full_name, s.roll_number, s.preferred_language, s.class_id
                FROM students s
                WHERE s.id = ?
            ) s
            JOIN (
                SELECT id AS class_id, grade, section, subject
                FROM classes
            ) c ON c.class_id = s.class_id
            """,
            (student_id,),
        ).fetchone()

        assessments = connection.execute(
            """
            SELECT a.title, ch.chapter_name, sa.score_obtained, sa.percentage, sa.submitted_at
            FROM student_assessments sa
            JOIN assessments a ON a.id = sa.assessment_id
            JOIN classes c ON c.id = a.class_id
            JOIN chapters ch ON ch.id = a.chapter_id
            WHERE sa.student_id = ? AND c.subject = ?
            ORDER BY sa.submitted_at DESC
            """,
            (student_id, subject),
        ).fetchall()

        mastery = connection.execute(
            """
            SELECT c.concept_name, scm.mastery_score, scm.status
            FROM student_concept_mastery scm
            JOIN concepts c ON c.id = scm.concept_id
            JOIN classes cl ON cl.id = scm.class_id
            WHERE scm.student_id = ? AND cl.subject = ?
            ORDER BY scm.mastery_score ASC
            """,
            (student_id, subject),
        ).fetchall()

        answer_samples = connection.execute(
            """
            SELECT aq.question_text, sa.raw_answer, sa.score_awarded, sa.grading_reasoning
            FROM student_answers sa
            JOIN student_assessments sat ON sat.id = sa.student_assessment_id
            JOIN assessments a ON a.id = sat.assessment_id
            JOIN classes c ON c.id = a.class_id
            JOIN assessment_questions aq ON aq.id = sa.assessment_question_id
            WHERE sat.student_id = ? AND c.subject = ?
            ORDER BY sat.submitted_at DESC, aq.question_number ASC
            LIMIT 12
            """,
            (student_id, subject),
        ).fetchall()

    mastery_rows = [dict(row) for row in mastery]
    return {
        "student": dict(student) if student else None,
        "assessment_history": [dict(row) for row in assessments],
        "strong_concepts": [row["concept_name"] for row in mastery_rows if row["status"] == "strong"],
        "lagging_concepts": [row["concept_name"] for row in mastery_rows if row["status"] == "lagging"],
        "developing_concepts": [row["concept_name"] for row in mastery_rows if row["status"] == "developing"],
        "recent_answer_samples": [dict(row) for row in answer_samples],
    }


def get_student_adaptation_profile_context(student_id: int, subject: str) -> dict[str, Any]:
    with get_connection() as connection:
        student = connection.execute(
            """
            SELECT s.id, s.full_name, s.roll_number, s.email, s.preferred_language,
                   s.accessibility_notes, c.id AS class_id, c.grade, c.section, c.subject
            FROM students s
            JOIN classes c ON c.id = s.class_id
            WHERE s.id = ?
            """,
            (student_id,),
        ).fetchone()

        mastery = connection.execute(
            """
            SELECT c.concept_name, ROUND(scm.mastery_score * 100, 1) AS mastery_percent,
                   ROUND(scm.confidence_score * 100, 1) AS confidence_percent,
                   scm.questions_attempted, scm.questions_correct, scm.status
            FROM student_concept_mastery scm
            JOIN concepts c ON c.id = scm.concept_id
            JOIN classes cl ON cl.id = scm.class_id
            WHERE scm.student_id = ? AND cl.subject = ?
            ORDER BY scm.mastery_score ASC, c.concept_name
            """,
            (student_id, subject),
        ).fetchall()

        assessments = connection.execute(
            """
            SELECT a.title, ch.chapter_name, sa.score_obtained, sa.percentage, sa.submitted_at
            FROM student_assessments sa
            JOIN assessments a ON a.id = sa.assessment_id
            JOIN classes c ON c.id = a.class_id
            JOIN chapters ch ON ch.id = a.chapter_id
            WHERE sa.student_id = ? AND c.subject = ?
            ORDER BY sa.submitted_at DESC
            """,
            (student_id, subject),
        ).fetchall()

        answer_samples = connection.execute(
            """
            SELECT aq.question_text, aq.question_type, aq.correct_answer,
                   sa.raw_answer, sa.score_awarded, sa.feedback, sa.grading_reasoning, sa.error_type
            FROM student_answers sa
            JOIN student_assessments sat ON sat.id = sa.student_assessment_id
            JOIN assessments a ON a.id = sat.assessment_id
            JOIN classes c ON c.id = a.class_id
            JOIN assessment_questions aq ON aq.id = sa.assessment_question_id
            WHERE sat.student_id = ? AND c.subject = ?
            ORDER BY sat.submitted_at DESC, aq.question_number ASC
            LIMIT 20
            """,
            (student_id, subject),
        ).fetchall()

        recommendations = connection.execute(
            """
            SELECT rr.recommendation_type, rr.recommendation_text, rr.priority, co.concept_name
            FROM remediation_recommendations rr
            JOIN concepts co ON co.id = rr.concept_id
            WHERE rr.student_id = ?
            ORDER BY rr.priority DESC, rr.created_at DESC
            LIMIT 12
            """,
            (student_id,),
        ).fetchall()

        attendance_summary = connection.execute(
            """
            SELECT
                COUNT(*) AS attendance_days_recorded,
                SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) AS attendance_days_present,
                MIN(attendance_date) AS attendance_started_on,
                MAX(attendance_date) AS attendance_last_marked_on
            FROM attendance_records
            WHERE student_id = ?
            """,
            (student_id,),
        ).fetchone()

    mastery_rows = [dict(row) for row in mastery]
    answer_rows = [dict(row) for row in answer_samples]
    recommendation_rows = [dict(row) for row in recommendations]
    attendance_dict = dict(attendance_summary) if attendance_summary else {}
    attendance_recorded = attendance_dict.get("attendance_days_recorded") or 0
    attendance_present = attendance_dict.get("attendance_days_present") or 0

    mcq_scores = [row["score_awarded"] for row in answer_rows if row.get("question_type") == "mcq"]
    short_scores = [row["score_awarded"] for row in answer_rows if row.get("question_type") == "short_answer"]
    best_formats = []
    needs_more_support_in = []
    if mcq_scores and (not short_scores or sum(mcq_scores) >= sum(short_scores)):
        best_formats.append("mcq")
    if short_scores and sum(short_scores) > sum(mcq_scores or [0]):
        best_formats.append("short_answer")
    if short_scores and any(score == 0 for score in short_scores):
        needs_more_support_in.append("written explanation")
    if mcq_scores and any(score == 0 for score in mcq_scores):
        needs_more_support_in.append("option discrimination")

    support_preferences = {
        "preferred_language": (dict(student) if student else {}).get("preferred_language", ""),
        "accessibility_support": (
            [str((dict(student) if student else {}).get("accessibility_notes", "")).strip()]
            if (dict(student) if student else {}).get("accessibility_notes")
            else []
        ),
        "pace_support": (
            "Needs slower pacing and shorter steps."
            if (dict(student) if student else {}).get("accessibility_notes")
            else "Use normal classroom pacing with checks for understanding."
        ),
        "explanation_style": ["simple explanation", "worked examples"],
        "response_support": ["step-by-step prompting"],
    }

    return {
        "student": dict(student) if student else None,
        "assessment_history": [dict(row) for row in assessments],
        "mastery_map": mastery_rows,
        "strong_concepts": [row["concept_name"] for row in mastery_rows if row["status"] == "strong"],
        "lagging_concepts": [row["concept_name"] for row in mastery_rows if row["status"] == "lagging"],
        "developing_concepts": [row["concept_name"] for row in mastery_rows if row["status"] == "developing"],
        "recent_answer_samples": answer_rows,
        "support_preferences": support_preferences,
        "answer_format_summary": {
            "best_formats": best_formats,
            "needs_more_support_in": needs_more_support_in,
        },
        "intervention_history": recommendation_rows,
        "attendance_signal": {
            **attendance_dict,
            "attendance_percentage": (
                round((attendance_present / attendance_recorded) * 100, 1)
                if attendance_recorded
                else None
            ),
        },
    }


def upsert_student_blueprint(
    *,
    student_id: int,
    class_id: int,
    subject: str,
    blueprint: dict[str, Any],
    generated_by_model: str,
    based_on_assessments: int,
    last_submission_at: str | None,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO student_blueprints (
                student_id, class_id, subject, strengths_json, weaknesses_json, opportunities_json,
                threats_json, recommendations_json, narrative, generated_by_model,
                based_on_assessments, last_submission_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
                subject,
                json.dumps(blueprint.get("strengths", []), ensure_ascii=False),
                json.dumps(blueprint.get("weaknesses", []), ensure_ascii=False),
                json.dumps(blueprint.get("opportunities", []), ensure_ascii=False),
                json.dumps(blueprint.get("threats", []), ensure_ascii=False),
                json.dumps(blueprint.get("recommendations", []), ensure_ascii=False),
                blueprint.get("narrative", ""),
                generated_by_model,
                based_on_assessments,
                last_submission_at,
            ),
        )
        connection.commit()


def upsert_student_adaptation_profile(
    *,
    student_id: int,
    class_id: int,
    subject: str,
    profile: dict[str, Any],
    summary: str,
    generated_by_model: str,
    based_on_assessments: int,
    last_submission_at: str | None,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO student_adaptation_profiles (
                student_id, class_id, subject, profile_json, summary, generated_by_model,
                based_on_assessments, last_submission_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
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
                json.dumps(profile, ensure_ascii=False),
                summary,
                generated_by_model,
                based_on_assessments,
                last_submission_at,
            ),
        )
        connection.commit()


def get_student_adaptation_profile(student_id: int, subject: str | None = None) -> dict[str, Any] | None:
    with get_connection() as connection:
        if subject:
            row = connection.execute(
                """
                SELECT student_id, class_id, subject, profile_json, summary, generated_by_model,
                       based_on_assessments, last_submission_at, updated_at
                FROM student_adaptation_profiles
                WHERE student_id = ? AND subject = ?
                LIMIT 1
                """,
                (student_id, subject),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT student_id, class_id, subject, profile_json, summary, generated_by_model,
                       based_on_assessments, last_submission_at, updated_at
                FROM student_adaptation_profiles
                WHERE student_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (student_id,),
            ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["profile"] = json.loads(item["profile_json"] or "{}")
    return item


def list_google_linked_assessments() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, class_id, title, google_form_id, google_form_url
            FROM assessments
            WHERE google_form_id IS NOT NULL
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def enqueue_response_for_processing(
    *,
    assessment_id: int,
    response_id: str,
    respondent_email: str,
    submitted_at: str | None,
    raw_response_json: str,
) -> bool:
    ensure_database()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO response_processing_queue (
                assessment_id, response_id, respondent_email, submitted_at, raw_response_json, status
            )
            VALUES (?, ?, ?, ?, ?, 'queued')
            """,
            (assessment_id, response_id, respondent_email, submitted_at, raw_response_json),
        )
        connection.commit()
    return cursor.rowcount > 0


def list_queue_items(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    ensure_database()
    with get_connection() as connection:
        if status:
            rows = connection.execute(
                """
                SELECT q.id, q.assessment_id, a.title, q.response_id, q.respondent_email,
                       q.submitted_at, q.status, q.error_message, q.created_at, q.processed_at
                FROM response_processing_queue q
                JOIN assessments a ON a.id = q.assessment_id
                WHERE q.status = ?
                ORDER BY q.created_at DESC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT q.id, q.assessment_id, a.title, q.response_id, q.respondent_email,
                       q.submitted_at, q.status, q.error_message, q.created_at, q.processed_at
                FROM response_processing_queue q
                JOIN assessments a ON a.id = q.assessment_id
                ORDER BY q.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return [dict(row) for row in rows]


def claim_next_queued_response() -> dict[str, Any] | None:
    ensure_database()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, assessment_id, response_id, respondent_email, submitted_at, raw_response_json
            FROM response_processing_queue
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None

        connection.execute(
            """
            UPDATE response_processing_queue
            SET status = 'processing', error_message = NULL
            WHERE id = ? AND status = 'queued'
            """,
            (row["id"],),
        )
        connection.commit()
    return dict(row)


def mark_queue_item_completed(queue_id: int) -> None:
    ensure_database()
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE response_processing_queue
            SET status = 'completed', processed_at = CURRENT_TIMESTAMP, error_message = NULL
            WHERE id = ?
            """,
            (queue_id,),
        )
        connection.commit()


def mark_queue_item_failed(queue_id: int, error_message: str) -> None:
    ensure_database()
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE response_processing_queue
            SET status = 'failed', processed_at = CURRENT_TIMESTAMP, error_message = ?
            WHERE id = ?
            """,
            (error_message[:1000], queue_id),
        )
        connection.commit()


def upsert_academic_year_plan(
    *,
    teacher_id: int,
    class_id: int,
    subject: str,
    academic_year: str,
    raw_syllabus_text: str,
    plan_title: str,
    planning: dict[str, Any],
    generated_by_model: str,
    status: str = "active",
) -> int:
    ensure_database()
    with get_connection() as connection:
        existing = connection.execute(
            """
            SELECT id
            FROM academic_year_plans
            WHERE class_id = ? AND LOWER(subject) = LOWER(?) AND academic_year = ? AND status != 'archived'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (class_id, subject.strip(), academic_year.strip()),
        ).fetchone()
        if existing:
            connection.execute(
                """
                UPDATE academic_year_plans
                SET teacher_id = ?, plan_title = ?, raw_syllabus_text = ?, planning_json = ?,
                    generated_by_model = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    teacher_id,
                    plan_title.strip(),
                    raw_syllabus_text,
                    json.dumps(planning, ensure_ascii=False),
                    generated_by_model.strip(),
                    status,
                    int(existing["id"]),
                ),
            )
            plan_id = int(existing["id"])
        else:
            plan_id = int(
                connection.execute(
                    """
                    INSERT INTO academic_year_plans (
                        teacher_id, class_id, subject, academic_year, plan_title,
                        raw_syllabus_text, planning_json, generated_by_model, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        teacher_id,
                        class_id,
                        subject.strip(),
                        academic_year.strip(),
                        plan_title.strip(),
                        raw_syllabus_text,
                        json.dumps(planning, ensure_ascii=False),
                        generated_by_model.strip(),
                        status,
                    ),
                ).lastrowid
            )
        connection.commit()
    return plan_id


def replace_academic_year_plan_units(plan_id: int, units: list[dict[str, Any]]) -> None:
    ensure_database()
    with get_connection() as connection:
        connection.execute("DELETE FROM academic_year_plan_units WHERE plan_id = ?", (plan_id,))
        for index, unit in enumerate(units, start=1):
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
                    str(unit.get("chapter_code") or "").strip(),
                    str(unit.get("chapter_name") or f"Unit {index}").strip(),
                    json.dumps(unit.get("subtopics", []), ensure_ascii=False),
                    int(unit.get("recommended_sessions") or 1),
                    str(unit.get("target_month") or "").strip(),
                    str(unit.get("term") or "").strip(),
                    int(unit.get("sequence_order") or index),
                    json.dumps(unit.get("completed_subtopics", []), ensure_ascii=False),
                    float(unit.get("completion_percent") or 0.0),
                    str(unit.get("status") or "not_started").strip(),
                ),
            )
        connection.commit()


def get_active_academic_year_plan(
    *,
    class_id: int,
    subject: str,
    academic_year: str = "",
) -> dict[str, Any] | None:
    ensure_database()
    with get_connection() as connection:
        if academic_year.strip():
            row = connection.execute(
                """
                SELECT *
                FROM academic_year_plans
                WHERE class_id = ? AND LOWER(subject) = LOWER(?) AND academic_year = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (class_id, subject.strip(), academic_year.strip()),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT *
                FROM academic_year_plans
                WHERE class_id = ? AND LOWER(subject) = LOWER(?) AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (class_id, subject.strip()),
            ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["planning"] = json.loads(item.get("planning_json") or "{}")
    return item


def list_academic_year_plan_units(plan_id: int) -> list[dict[str, Any]]:
    ensure_database()
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM academic_year_plan_units
            WHERE plan_id = ?
            ORDER BY sequence_order, id
            """,
            (plan_id,),
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["subtopics"] = json.loads(item.get("subtopics_json") or "[]")
        item["completed_subtopics"] = json.loads(item.get("completed_subtopics_json") or "[]")
        items.append(item)
    return items


def update_academic_year_plan_unit_progress(
    *,
    plan_unit_id: int,
    completed_subtopics: list[str],
    completion_percent: float,
    status: str,
) -> None:
    ensure_database()
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE academic_year_plan_units
            SET completed_subtopics_json = ?, completion_percent = ?, status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                json.dumps(completed_subtopics, ensure_ascii=False),
                float(max(0.0, min(100.0, completion_percent))),
                status.strip(),
                plan_unit_id,
            ),
        )
        connection.commit()


def upsert_class_timetable_slot(
    *,
    class_id: int,
    subject: str,
    weekday: int,
    start_time: str,
    end_time: str,
    auto_record_enabled: bool = True,
) -> int:
    ensure_database()
    with get_connection() as connection:
        existing = connection.execute(
            """
            SELECT id
            FROM class_timetable_slots
            WHERE class_id = ? AND LOWER(subject) = LOWER(?) AND weekday = ? AND start_time = ? AND end_time = ?
            LIMIT 1
            """,
            (class_id, subject.strip(), weekday, start_time.strip(), end_time.strip()),
        ).fetchone()
        if existing:
            connection.execute(
                """
                UPDATE class_timetable_slots
                SET auto_record_enabled = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (1 if auto_record_enabled else 0, int(existing["id"])),
            )
            slot_id = int(existing["id"])
        else:
            slot_id = int(
                connection.execute(
                    """
                    INSERT INTO class_timetable_slots (
                        class_id, subject, weekday, start_time, end_time, auto_record_enabled
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        class_id,
                        subject.strip(),
                        weekday,
                        start_time.strip(),
                        end_time.strip(),
                        1 if auto_record_enabled else 0,
                    ),
                ).lastrowid
            )
        connection.commit()
    return slot_id


def list_class_timetable_slots(class_id: int, subject: str = "") -> list[dict[str, Any]]:
    ensure_database()
    with get_connection() as connection:
        if subject.strip():
            rows = connection.execute(
                """
                SELECT *
                FROM class_timetable_slots
                WHERE class_id = ? AND LOWER(subject) = LOWER(?)
                ORDER BY weekday, start_time, end_time
                """,
                (class_id, subject.strip()),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM class_timetable_slots
                WHERE class_id = ?
                ORDER BY subject, weekday, start_time, end_time
                """,
                (class_id,),
            ).fetchall()
    return [dict(row) for row in rows]


def delete_class_timetable_slot(slot_id: int) -> None:
    ensure_database()
    with get_connection() as connection:
        connection.execute("DELETE FROM class_timetable_slots WHERE id = ?", (slot_id,))
        connection.commit()


def create_class_coverage_session(
    *,
    class_id: int,
    teacher_id: int,
    subject: str,
    plan_id: int | None,
    timetable_slot_id: int | None,
    session_date: str,
    scheduled_start: str = "",
    scheduled_end: str = "",
    actual_start: str = "",
    actual_end: str = "",
    source: str = "audio",
    transcript_text: str = "",
    coverage: dict[str, Any] | None = None,
    confidence_score: float = 0.0,
    coverage_summary: str = "",
    processing_status: str = "completed",
    processing_notes: str = "",
) -> int:
    ensure_database()
    with get_connection() as connection:
        session_id = int(
            connection.execute(
                """
                INSERT INTO class_coverage_sessions (
                    class_id, teacher_id, subject, plan_id, timetable_slot_id, session_date,
                    scheduled_start, scheduled_end, actual_start, actual_end, source,
                    transcript_text, coverage_json, confidence_score, coverage_summary,
                    processing_status, processing_notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    class_id,
                    teacher_id,
                    subject.strip(),
                    plan_id,
                    timetable_slot_id,
                    session_date.strip(),
                    scheduled_start.strip(),
                    scheduled_end.strip(),
                    actual_start.strip(),
                    actual_end.strip(),
                    source.strip(),
                    transcript_text,
                    json.dumps(coverage or {}, ensure_ascii=False),
                    float(max(0.0, min(1.0, confidence_score))),
                    coverage_summary.strip(),
                    processing_status.strip(),
                    processing_notes.strip(),
                ),
            ).lastrowid
        )
        connection.commit()
    return session_id


def list_class_coverage_sessions(
    *,
    class_id: int,
    subject: str = "",
    limit: int = 20,
) -> list[dict[str, Any]]:
    ensure_database()
    with get_connection() as connection:
        if subject.strip():
            rows = connection.execute(
                """
                SELECT *
                FROM class_coverage_sessions
                WHERE class_id = ? AND LOWER(subject) = LOWER(?)
                ORDER BY session_date DESC, created_at DESC
                LIMIT ?
                """,
                (class_id, subject.strip(), limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM class_coverage_sessions
                WHERE class_id = ?
                ORDER BY session_date DESC, created_at DESC
                LIMIT ?
                """,
                (class_id, limit),
            ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["coverage"] = json.loads(item.get("coverage_json") or "{}")
        items.append(item)
    return items


class TeacherClassRepository:
    get_teacher = staticmethod(get_teacher)
    create_class_for_teacher = staticmethod(create_class_for_teacher)
    update_class_details = staticmethod(update_class_details)
    list_teacher_classes = staticmethod(list_teacher_classes)
    get_class_overview = staticmethod(get_class_overview)
    list_class_subjects = staticmethod(list_class_subjects)
    find_teacher_and_class_for_subject = staticmethod(find_teacher_and_class_for_subject)
    find_class_for_attendance = staticmethod(find_class_for_attendance)
    find_class_for_management = staticmethod(find_class_for_management)
    add_subject_to_class = staticmethod(add_subject_to_class)
    update_class_subject_details = staticmethod(update_class_subject_details)


class CurriculumRepository:
    list_grade_curriculum_subjects = staticmethod(list_grade_curriculum_subjects)
    get_curriculum_subject = staticmethod(get_curriculum_subject)
    ensure_curriculum_subject = staticmethod(ensure_curriculum_subject)
    upsert_curriculum_chapters = staticmethod(upsert_curriculum_chapters)
    list_curriculum_chapters = staticmethod(list_curriculum_chapters)
    list_chapters_for_class = staticmethod(list_chapters_for_class)
    find_chapter_for_class_topic = staticmethod(find_chapter_for_class_topic)
    find_fallback_chapter_for_class = staticmethod(find_fallback_chapter_for_class)
    find_chapter_for_management = staticmethod(find_chapter_for_management)
    create_chapter_for_class = staticmethod(create_chapter_for_class)
    update_chapter_details = staticmethod(update_chapter_details)
    delete_chapter_if_unused = staticmethod(delete_chapter_if_unused)


class StudentRepository:
    list_class_students = staticmethod(list_class_students)
    list_class_roster = staticmethod(list_class_roster)
    list_inactive_class_students = staticmethod(list_inactive_class_students)
    find_student_id = staticmethod(find_student_id)
    add_student_to_class = staticmethod(add_student_to_class)
    update_student_details = staticmethod(update_student_details)
    deactivate_student = staticmethod(deactivate_student)
    reactivate_student = staticmethod(reactivate_student)
    get_student_detail = staticmethod(get_student_detail)
    get_student_blueprint_context = staticmethod(get_student_blueprint_context)
    get_student_adaptation_profile_context = staticmethod(get_student_adaptation_profile_context)
    upsert_student_blueprint = staticmethod(upsert_student_blueprint)
    upsert_student_adaptation_profile = staticmethod(upsert_student_adaptation_profile)
    get_student_adaptation_profile = staticmethod(get_student_adaptation_profile)


class AttendanceRepository:
    upsert_class_attendance = staticmethod(upsert_class_attendance)
    list_attendance_for_date = staticmethod(list_attendance_for_date)
    get_attendance_overview = staticmethod(get_attendance_overview)
    get_class_attendance_stats = staticmethod(get_class_attendance_stats)
    update_student_attendance_status = staticmethod(update_student_attendance_status)
    clear_student_attendance = staticmethod(clear_student_attendance)
    clear_class_attendance = staticmethod(clear_class_attendance)


class MaterialRepository:
    create_source_material = staticmethod(create_source_material)
    update_source_material = staticmethod(update_source_material)
    create_ingestion_run = staticmethod(create_ingestion_run)
    update_ingestion_run = staticmethod(update_ingestion_run)
    replace_material_chunks = staticmethod(replace_material_chunks)
    list_subject_materials = staticmethod(list_subject_materials)
    list_material_chunks_for_subject = staticmethod(list_material_chunks_for_subject)
    list_recent_ingestion_runs = staticmethod(list_recent_ingestion_runs)


class AssessmentRepository:
    list_class_assessments = staticmethod(list_class_assessments)
    list_class_assessment_history = staticmethod(list_class_assessment_history)
    create_assessment = staticmethod(create_assessment)
    update_assessment_google_form_info = staticmethod(update_assessment_google_form_info)
    list_assessment_questions = staticmethod(list_assessment_questions)
    list_assessments_for_sync = staticmethod(list_assessments_for_sync)
    get_assessment_sync_bundle = staticmethod(get_assessment_sync_bundle)
    upsert_google_form_response_sync = staticmethod(upsert_google_form_response_sync)
    list_attempted_students_for_assessment = staticmethod(list_attempted_students_for_assessment)
    get_student_assessment_review = staticmethod(get_student_assessment_review)
    list_google_linked_assessments = staticmethod(list_google_linked_assessments)


class AnalyticsRepository:
    list_class_concept_gaps = staticmethod(list_class_concept_gaps)
    replace_mastery_snapshots = staticmethod(replace_mastery_snapshots)


class QueueRepository:
    enqueue_response_for_processing = staticmethod(enqueue_response_for_processing)
    list_queue_items = staticmethod(list_queue_items)
    claim_next_queued_response = staticmethod(claim_next_queued_response)
    mark_queue_item_completed = staticmethod(mark_queue_item_completed)
    mark_queue_item_failed = staticmethod(mark_queue_item_failed)


class PlanningRepository:
    upsert_academic_year_plan = staticmethod(upsert_academic_year_plan)
    replace_academic_year_plan_units = staticmethod(replace_academic_year_plan_units)
    get_active_academic_year_plan = staticmethod(get_active_academic_year_plan)
    list_academic_year_plan_units = staticmethod(list_academic_year_plan_units)
    update_academic_year_plan_unit_progress = staticmethod(update_academic_year_plan_unit_progress)


class TimetableRepository:
    upsert_class_timetable_slot = staticmethod(upsert_class_timetable_slot)
    list_class_timetable_slots = staticmethod(list_class_timetable_slots)
    delete_class_timetable_slot = staticmethod(delete_class_timetable_slot)


class CoverageRepository:
    create_class_coverage_session = staticmethod(create_class_coverage_session)
    list_class_coverage_sessions = staticmethod(list_class_coverage_sessions)


class RepositoryContainer:
    def __init__(self) -> None:
        self.teacher_classes = TeacherClassRepository()
        self.curriculum = CurriculumRepository()
        self.students = StudentRepository()
        self.attendance = AttendanceRepository()
        self.materials = MaterialRepository()
        self.assessments = AssessmentRepository()
        self.analytics = AnalyticsRepository()
        self.queue = QueueRepository()
        self.planning = PlanningRepository()
        self.timetable = TimetableRepository()
        self.coverage = CoverageRepository()


teacher_class_repository = TeacherClassRepository()
curriculum_repository = CurriculumRepository()
student_repository = StudentRepository()
attendance_repository = AttendanceRepository()
material_repository = MaterialRepository()
assessment_repository = AssessmentRepository()
analytics_repository = AnalyticsRepository()
queue_repository = QueueRepository()
planning_repository = PlanningRepository()
timetable_repository = TimetableRepository()
coverage_repository = CoverageRepository()
repositories = RepositoryContainer()
