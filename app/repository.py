from __future__ import annotations

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


def list_teacher_classes(teacher_id: int) -> list[dict[str, Any]]:
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
                SUM(CASE WHEN scm.status = 'lagging' THEN 1 ELSE 0 END) AS lagging_concepts
            FROM students s
            LEFT JOIN student_assessments sa ON sa.student_id = s.id
            LEFT JOIN student_concept_mastery scm ON scm.student_id = s.id AND scm.class_id = s.class_id
            WHERE s.class_id = ? AND s.status = 'active'
            GROUP BY s.id, s.roll_number, s.full_name, s.preferred_language, s.accessibility_notes
            ORDER BY s.roll_number
            """,
            (class_id,),
        ).fetchall()
    return [dict(row) for row in rows]


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

    return {
        "student": dict(student) if student else None,
        "mastery": [dict(row) for row in mastery_rows],
        "assessments": [dict(row) for row in assessments],
        "recommendations": [dict(row) for row in recommendations],
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


def list_chapters_for_class(class_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        class_row = connection.execute(
            "SELECT grade, subject FROM classes WHERE id = ?",
            (class_id,),
        ).fetchone()
        if not class_row:
            return []

        rows = connection.execute(
            """
            SELECT id, chapter_code, chapter_name, term
            FROM chapters
            WHERE grade = ? AND subject = ?
            ORDER BY chapter_name
            """,
            (class_row["grade"], class_row["subject"]),
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
