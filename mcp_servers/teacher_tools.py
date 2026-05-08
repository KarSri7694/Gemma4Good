from __future__ import annotations
from typing import Annotated

import os
import sys
import json
from pathlib import Path
from math import prod
from datetime import datetime

from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llama_client import LlamaServerClient, LlamaServerConfig
from app.attendance import mark_attendance_from_identifiers
from app.gemma_adaptation_profile import generate_student_adaptation_profile_with_gemma
from app.model_control import choose_quiz_generation_mode, load_model_sampling_config
from app.material_ingestion import ingest_reading_material
from app.rag import build_retrieval_context, search_subject_materials as search_rag_subject_materials
from app.repository import (
    add_student_to_class,
    clear_class_attendance,
    clear_student_attendance,
    create_assessment,
    create_chapter_for_class,
    create_class_for_teacher,
    deactivate_student,
    delete_chapter_if_unused,
    get_curriculum_subject,
    get_student_adaptation_profile,
    get_student_adaptation_profile_context,
    get_student_detail,
    get_teacher,
    list_curriculum_chapters,
    list_subject_materials,
    reactivate_student,
    update_assessment_google_form_info,
    update_chapter_details,
    update_class_details,
    update_student_details,
    update_student_attendance_status,
    upsert_student_adaptation_profile,
)
from app.db import ensure_database, get_connection
from app.generator import build_quiz_questions
from app.google_forms import create_google_form_quiz
from app.quiz_engine import generate_quiz_with_llama


mcp = FastMCP("Science Teacher Tools")
ensure_database()


def _tool_settings() -> dict[str, str]:
    return {
        "llama_base_url": os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080"),
        "llama_model": os.getenv("LLAMA_MODEL", "Gemma-4-E4B-Q4_K_M"),
        "subject": os.getenv("TEACHER_TOOL_SUBJECT", "Science"),
        "grade_band": os.getenv("TEACHER_TOOL_GRADE_BAND", "6-8"),
    }


def _extract_response_text(response: dict) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content") or choices[0].get("text") or ""
        return str(content).strip()
    return str(response.get("content") or response.get("text") or "").strip()


def _ask_llama(system_prompt: str, user_prompt: str) -> str:
    settings = _tool_settings()
    sampling = load_model_sampling_config()
    client = LlamaServerClient(LlamaServerConfig(base_url=settings["llama_base_url"]))
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        extra_payload={"model": settings["llama_model"]},
    )
    return _extract_response_text(response) or "The model returned an empty response."


def _find_student_id(student_name: str = "", roll_number: str = "") -> int | None:
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


def _find_teacher_and_class_for_subject(subject: str) -> tuple[int | None, dict | None]:
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


def _find_class_for_attendance(grade: str, section: str, subject: str) -> tuple[int | None, dict | None]:
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


def _find_class_for_management(grade: str, section: str, subject: str = "") -> tuple[int | None, dict | None]:
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


def _find_chapter_for_class_topic(class_id: int, topic: str) -> dict | None:
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


def _find_fallback_chapter_for_class(class_id: int) -> dict | None:
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


def _generate_and_store_adaptation_profile(
    *,
    student_id: int,
    subject: str,
    llama_client: LlamaServerClient,
    llama_model_name: str,
) -> dict:
    context = get_student_adaptation_profile_context(student_id, subject)
    student_row = context.get("student") or {}
    generated = generate_student_adaptation_profile_with_gemma(
        client=llama_client,
        model_name=llama_model_name,
        temperature=load_model_sampling_config().temperature,
        top_p=load_model_sampling_config().top_p,
        top_k=load_model_sampling_config().top_k,
        student_context=context,
    )
    final_profile = {
        "mastery_map": context.get("mastery_map", []),
        "attendance_signal": context.get("attendance_signal", {}),
        "intervention_history": context.get("intervention_history", []),
        **generated,
    }
    upsert_student_adaptation_profile(
        student_id=student_id,
        class_id=int(student_row.get("class_id", 0)),
        subject=subject,
        profile=final_profile,
        summary=generated.get("summary", ""),
        generated_by_model=llama_model_name,
        based_on_assessments=len(context.get("assessment_history", [])),
        last_submission_at=context.get("attendance_signal", {}).get("attendance_last_marked_on"),
    )
    profile_row = get_student_adaptation_profile(student_id, subject)
    return profile_row or {
        "student_id": student_id,
        "subject": subject,
        "profile": final_profile,
        "summary": generated.get("summary", ""),
    }


def _find_chapter_for_management(class_id: int, chapter_name: str) -> dict | None:
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


@mcp.tool(description="Check the configured model endpoint for this teacher tool server.")
def server_health() -> str:
    settings = _tool_settings()
    return (
        f"Server is configured for subject '{settings['subject']}', grade band '{settings['grade_band']}', "
        f"model '{settings['llama_model']}' at '{settings['llama_base_url']}'."
    )


@mcp.tool(description="Answer teacher questions about the configured subject and grade band.")
def answer_subject_question(question: str, class_context: str = "", preferred_language: str = "English") -> str:
    settings = _tool_settings()
    system_prompt = (
        "You are a subject expert helping a teacher in India. "
        "Answer clearly, correctly, and at classroom level. "
        "Keep the answer practical and avoid unnecessary jargon."
    )
    user_prompt = (
        f"Subject: {settings['subject']}\n"
        f"Grade band: {settings['grade_band']}\n"
        f"Preferred language: {preferred_language}\n"
        f"Class context: {class_context or 'Not provided'}\n"
        f"Teacher question: {question}\n\n"
        "Respond for a teacher, not a student. Include examples only if they help teaching."
    )
    return _ask_llama(system_prompt, user_prompt)


@mcp.tool(description="Create a short reteach plan for a weak concept.")
def create_reteach_plan(concept_name: str, misconceptions: str = "", preferred_language: str = "English") -> str:
    settings = _tool_settings()
    system_prompt = (
        "You are a classroom remediation planner. "
        "Create short, actionable reteach plans for a teacher in India."
    )
    user_prompt = (
        f"Subject: {settings['subject']}\n"
        f"Grade band: {settings['grade_band']}\n"
        f"Preferred language: {preferred_language}\n"
        f"Concept to reteach: {concept_name}\n"
        f"Observed misconceptions: {misconceptions or 'Not provided'}\n\n"
        "Create a concise reteach plan with:\n"
        "- goal\n"
        "- board explanation\n"
        "- quick activity\n"
        "- exit check"
    )
    return _ask_llama(system_prompt, user_prompt)


@mcp.tool(description="Generate a teacher-facing concept explanation.")
def explain_concept_for_class(
    concept_name: str,
    learner_level: str = "mixed",
    preferred_language: str = "English",
) -> str:
    settings = _tool_settings()
    system_prompt = (
        "You are a teacher explanation copilot. "
        "Generate board-ready explanations for classroom teaching."
    )
    user_prompt = (
        f"Subject: {settings['subject']}\n"
        f"Grade band: {settings['grade_band']}\n"
        f"Preferred language: {preferred_language}\n"
        f"Concept: {concept_name}\n"
        f"Learner level: {learner_level}\n\n"
        "Explain this for a teacher to present in class. Include:\n"
        "- simple explanation\n"
        "- one example\n"
        "- one likely misconception\n"
        "- one check-for-understanding question"
    )
    return _ask_llama(system_prompt, user_prompt)


@mcp.tool(description="Perform a basic arithmetic calculation for testing tool use.")
def calculator(operation: str, numbers: list[float]) -> str:
    normalized_operation = operation.strip().lower()
    if not numbers:
        return "No numbers provided."
    if normalized_operation == "add":
        result = sum(numbers)
    elif normalized_operation == "subtract":
        result = numbers[0]
        for value in numbers[1:]:
            result -= value
    elif normalized_operation == "multiply":
        result = prod(numbers)
    elif normalized_operation == "divide":
        result = numbers[0]
        for value in numbers[1:]:
            if value == 0:
                return "Division by zero is not allowed."
            result /= value
    else:
        return "Unsupported operation. Use add, subtract, multiply, or divide."
    return str(result)


@mcp.tool(description="Fetch the full student profile using student name or roll number.")
def get_student_information(
    student_name: Annotated[str, "The name of the student"] = "",
    roll_number: Annotated[str, "The roll number of the student"] = "",
) -> str:
    student_id = _find_student_id(student_name=student_name, roll_number=roll_number)
    if student_id is None:
        return "No active student found for the provided student name or roll number."

    student_detail = get_student_detail(student_id)
    if not student_detail.get("student"):
        return "Student record was found, but detailed student data is unavailable."

    return json.dumps(student_detail, indent=2, ensure_ascii=False)


@mcp.tool(description="Fetch the structured subject-specific adaptation profile for a student.")
def get_student_adaptation_profile_tool(
    student_name: Annotated[str, "The name of the student"] = "",
    roll_number: Annotated[str, "The roll number of the student"] = "",
    subject: Annotated[str, "The subject for the profile"] = "",
) -> str:
    student_id = _find_student_id(student_name=student_name, roll_number=roll_number)
    if student_id is None:
        return "No active student found for the provided student name or roll number."
    profile = get_student_adaptation_profile(student_id, subject.strip() or None)
    if not profile:
        return "No adaptation profile exists yet for the requested student and subject."
    return json.dumps(profile, indent=2, ensure_ascii=False)


@mcp.tool(description="Regenerate and store the structured adaptation profile for a student and subject.")
def regenerate_student_adaptation_profile(
    student_name: Annotated[str, "The name of the student"] = "",
    roll_number: Annotated[str, "The roll number of the student"] = "",
    subject: Annotated[str, "The subject for the profile"] = "",
) -> str:
    student_id = _find_student_id(student_name=student_name, roll_number=roll_number)
    if student_id is None:
        return "No active student found for the provided student name or roll number."
    student_detail = get_student_detail(student_id)
    student_row = student_detail.get("student") or {}
    resolved_subject = subject.strip() or student_row.get("subject", "")
    if not resolved_subject:
        return "No subject was provided and the student's class subject could not be resolved."
    settings = _tool_settings()
    client = LlamaServerClient(LlamaServerConfig(base_url=settings["llama_base_url"]))
    profile = _generate_and_store_adaptation_profile(
        student_id=student_id,
        subject=resolved_subject,
        llama_client=client,
        llama_model_name=settings["llama_model"],
    )
    return json.dumps(profile, indent=2, ensure_ascii=False)


@mcp.tool(description="Create a remedial plan for a student using the adaptation profile.")
def create_student_remedial_plan(
    student_name: Annotated[str, "The name of the student"] = "",
    roll_number: Annotated[str, "The roll number of the student"] = "",
    subject: Annotated[str, "The subject for the remedial plan"] = "",
    preferred_language: Annotated[str, "Preferred output language"] = "English",
) -> str:
    student_id = _find_student_id(student_name=student_name, roll_number=roll_number)
    if student_id is None:
        return "No active student found for the provided student name or roll number."
    student_detail = get_student_detail(student_id)
    student_row = student_detail.get("student") or {}
    resolved_subject = subject.strip() or student_row.get("subject", "")
    if not resolved_subject:
        return "No subject was provided and the student's class subject could not be resolved."
    profile = get_student_adaptation_profile(student_id, resolved_subject)
    if not profile:
        settings = _tool_settings()
        client = LlamaServerClient(LlamaServerConfig(base_url=settings["llama_base_url"]))
        profile = _generate_and_store_adaptation_profile(
            student_id=student_id,
            subject=resolved_subject,
            llama_client=client,
            llama_model_name=settings["llama_model"],
        )
    system_prompt = (
        "You are a remedial teaching copilot for Indian classrooms. "
        "Create a short student-specific remedial plan from the adaptation profile."
    )
    user_prompt = (
        f"Preferred language: {preferred_language}\n"
        f"Student: {student_row.get('full_name', '')}\n"
        f"Subject: {resolved_subject}\n"
        f"Adaptation profile: {json.dumps(profile.get('profile', {}), ensure_ascii=False)}\n\n"
        "Create a concise plan with: priority targets, teaching approach, practice type, and success check."
    )
    return _ask_llama(system_prompt, user_prompt)


@mcp.tool(description="Create a personalized remedial quiz for one student using the adaptation profile.")
def create_personalized_student_quiz(
    student_name: Annotated[str, "The name of the student"] = "",
    roll_number: Annotated[str, "The roll number of the student"] = "",
    subject: Annotated[str, "The subject for the quiz"] = "",
    due_date: Annotated[str, "Quiz due date in DD/MM/YYYY format"] = "",
    due_time: Annotated[str, "Quiz due time in 12-hour format like 05:30 PM"] = "",
    question_count: Annotated[int, "Number of quiz questions to generate"] = 5,
    preferred_language: Annotated[str, "Preferred quiz language"] = "English",
    teacher_instructions: Annotated[str, "Additional teacher instructions for Gemma while generating the quiz"] = "",
) -> str:
    if not due_date.strip() or not due_time.strip():
        return (
            "Missing quiz due date or due time. "
            "Ask the teacher to provide both a due date in DD/MM/YYYY format and a due time in 12-hour format."
        )
    student_id = _find_student_id(student_name=student_name, roll_number=roll_number)
    if student_id is None:
        return "No active student found for the provided student name or roll number."
    student_detail = get_student_detail(student_id)
    student_row = student_detail.get("student") or {}
    resolved_subject = subject.strip() or student_row.get("subject", "")
    if not resolved_subject:
        return "No subject was provided and the student's class subject could not be resolved."
    settings = _tool_settings()
    client = LlamaServerClient(LlamaServerConfig(base_url=settings["llama_base_url"]))
    profile = get_student_adaptation_profile(student_id, resolved_subject)
    if not profile:
        profile = _generate_and_store_adaptation_profile(
            student_id=student_id,
            subject=resolved_subject,
            llama_client=client,
            llama_model_name=settings["llama_model"],
        )
    teacher_id, class_row = _find_teacher_and_class_for_subject(resolved_subject)
    if teacher_id is None or class_row is None:
        return f"No class/teacher mapping was found for subject '{resolved_subject}'."
    topic = "; ".join((profile.get("profile", {}) or {}).get("priority_targets", [])[:3]) or "concept reinforcement"
    chapter_row = _find_fallback_chapter_for_class(int(class_row["class_id"]))
    if chapter_row is None:
        return f"No chapter record was found for subject '{resolved_subject}'."
    normalized_language = preferred_language.strip() or "English"
    sampling = load_model_sampling_config()
    generation_strategy, _ = choose_quiz_generation_mode(
        settings["llama_model"],
        sampling.quiz_question_generation_mode,
    )
    questions, generation_note, raw_outputs = generate_quiz_with_llama(
        client=client,
        base_url=settings["llama_base_url"],
        model_name=settings["llama_model"],
        generation_strategy=generation_strategy,
        subject=resolved_subject,
        grade=str(class_row["grade"]),
        chapter_name=chapter_row["chapter_name"],
        learner_profile=profile.get("summary", "") or "Target weak concepts and reinforce understanding.",
        source_material=(
            "Generate a personalized remedial quiz using this adaptation profile: "
            + json.dumps(profile.get("profile", {}), ensure_ascii=False)
        ),
        teacher_instructions=teacher_instructions,
        language=normalized_language,
        question_count=max(1, question_count),
    )
    if not questions:
        questions = build_quiz_questions(topic, [topic, resolved_subject], normalized_language)[: max(1, question_count)]
    due_at = datetime.strptime(
        f"{due_date.strip()} {due_time.strip()}",
        "%d/%m/%Y %I:%M %p",
    ).strftime("%Y-%m-%d %H:%M:%S")
    assessment_title = f"{student_row.get('full_name', 'Student')} Personalized Quiz"
    assessment_id = create_assessment(
        class_id=int(class_row["class_id"]),
        chapter_id=int(chapter_row["id"]),
        teacher_id=teacher_id,
        title=assessment_title,
        language=normalized_language,
        assessment_type="remedial",
        questions=questions,
        due_at=due_at,
    )
    form_result = create_google_form_quiz(
        title=assessment_title,
        description=(
            f"Student: {student_row.get('full_name', '')}\n"
            f"Subject: {resolved_subject}\n"
            f"Submit before: {due_date.strip()} at {due_time.strip()}"
        ),
        questions=questions,
    )
    update_assessment_google_form_info(
        assessment_id=assessment_id,
        google_form_id=form_result["form_id"],
        google_form_url=form_result["edit_uri"],
        question_id_map=form_result["question_id_map"],
    )
    return json.dumps(
        {
            "assessment_id": assessment_id,
            "student_id": student_id,
            "subject": resolved_subject,
            "topic": topic,
            "generation_note": generation_note if 'generation_note' in locals() else "Generated using local fallback logic.",
            "raw_model_outputs": raw_outputs if 'raw_outputs' in locals() else [],
            "edit_form_link": form_result["edit_uri"],
            "student_form_link": form_result.get("responder_uri") or "",
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Create a quiz for a given subject and topic. If due date or due time is missing, ask the teacher for it first.")
def create_quiz_for_topic(
    subject: Annotated[str, "The subject for which the quiz should be created"],
    topic: Annotated[str, "The topic or chapter name for the quiz"],
    due_date: Annotated[str, "Quiz due date in DD/MM/YYYY format"] = "",
    due_time: Annotated[str, "Quiz due time in 12-hour format like 05:30 PM"] = "",
    question_count: Annotated[int, "Number of quiz questions to generate"] = 5,
    preferred_language: Annotated[str, "Preferred quiz language"] = "English",
    teacher_instructions: Annotated[str, "Additional teacher instructions for Gemma while generating the quiz"] = "",
) -> str:
    if not due_date.strip() or not due_time.strip():
        return (
            "Missing quiz due date or due time. "
            "Ask the teacher to provide both a due date in DD/MM/YYYY format and a due time in 12-hour format."
        )

    teacher_id, class_row = _find_teacher_and_class_for_subject(subject)
    if teacher_id is None or class_row is None:
        return f"No class/teacher mapping was found for subject '{subject.strip()}'."

    chapter_row = _find_chapter_for_class_topic(int(class_row["class_id"]), topic)
    chapter_mapping_note = ""
    if chapter_row is None:
        chapter_row = _find_fallback_chapter_for_class(int(class_row["class_id"]))
        if chapter_row is None:
            return (
                f"No chapter mapping was found for topic '{topic.strip()}' in subject '{subject.strip()}'. "
                "Create at least one chapter record for this class subject first."
            )
        chapter_mapping_note = (
            f"No exact chapter title matched '{topic.strip()}'. "
            f"The quiz was generated for the requested topic and stored under the closest available class chapter "
            f"'{chapter_row['chapter_name']}'."
        )

    normalized_language = preferred_language.strip() or "English"
    settings = _tool_settings()
    sampling = load_model_sampling_config()
    generation_strategy, _ = choose_quiz_generation_mode(
        settings["llama_model"],
        sampling.quiz_question_generation_mode,
    )
    client = LlamaServerClient(LlamaServerConfig(base_url=settings["llama_base_url"]))
    rag_context = build_retrieval_context(
        grade=str(class_row["grade"]),
        subject=subject.strip(),
        query=f"{topic.strip()} classroom quiz",
    )
    questions, generation_note, raw_outputs = generate_quiz_with_llama(
        client=client,
        base_url=settings["llama_base_url"],
        model_name=settings["llama_model"],
        generation_strategy=generation_strategy,
        subject=subject.strip(),
        grade=str(class_row["grade"]),
        chapter_name=chapter_row["chapter_name"],
        learner_profile="Mixed-ability classroom.",
        source_material=(
            f"Create a quiz for {topic.strip()} in {subject.strip()}."
            + (f"\n\nRetrieved subject material:\n{rag_context}" if rag_context else "")
        ),
        teacher_instructions=teacher_instructions,
        language=normalized_language,
        question_count=max(1, question_count),
    )
    if not questions:
        questions = build_quiz_questions(topic.strip(), [topic.strip(), subject.strip()], normalized_language)
        if question_count > len(questions):
            while len(questions) < question_count:
                questions.extend(
                    build_quiz_questions(
                        topic.strip(),
                        [topic.strip(), subject.strip()],
                        normalized_language,
                    )
                )
        questions = questions[: max(1, question_count)]

    try:
        due_at = datetime.strptime(
            f"{due_date.strip()} {due_time.strip()}",
            "%d/%m/%Y %I:%M %p",
        ).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return (
            "Invalid due date or due time format. "
            "Ask the teacher to provide the due date in DD/MM/YYYY format and the due time in 12-hour format like 05:30 PM."
        )
    assessment_title = f"{topic.strip()} Quiz"
    assessment_id = create_assessment(
        class_id=int(class_row["class_id"]),
        chapter_id=int(chapter_row["id"]),
        teacher_id=teacher_id,
        title=assessment_title,
        language=normalized_language,
        assessment_type="class_test",
        questions=questions,
        due_at=due_at,
    )
    form_result = create_google_form_quiz(
        title=assessment_title,
        description=(
            f"Subject: {subject.strip()}\n"
            f"Topic: {topic.strip()}\n"
            f"Submit before: {due_date.strip()} at {due_time.strip()}"
        ),
        questions=questions,
    )
    update_assessment_google_form_info(
        assessment_id=assessment_id,
        google_form_id=form_result["form_id"],
        google_form_url=form_result["edit_uri"],
        question_id_map=form_result["question_id_map"],
    )

    quiz_payload = {
        "assessment_id": assessment_id,
        "subject": subject.strip(),
        "topic": topic.strip(),
        "stored_chapter_name": chapter_row["chapter_name"],
        "chapter_mapping_note": chapter_mapping_note,
        "preferred_language": normalized_language,
        "due_date": due_date.strip(),
        "due_time": due_time.strip(),
        "question_count": len(questions),
        "generation_note": generation_note if 'generation_note' in locals() else "Generated using local fallback logic.",
        "raw_model_outputs": raw_outputs if 'raw_outputs' in locals() else [],
        "retrieval_context": rag_context,
        "edit_form_link": form_result["edit_uri"],
        "student_form_link": form_result.get("responder_uri") or "",
    }
    return json.dumps(quiz_payload, indent=2, ensure_ascii=False)


@mcp.tool(description="Ingest reading material into the grade-subject library using pasted text or a local file path.")
def ingest_subject_material(
    grade: Annotated[str, "The class grade, for example 7"],
    title: Annotated[str, "The reading material title"],
    text_content: Annotated[str, "Pasted reading material text"] = "",
    local_file_path: Annotated[str, "Optional local file path to a PDF or image"] = "",
) -> str:
    file_bytes = None
    source_type = "text"
    mime_type = "text/plain"
    original_filename = ""
    if local_file_path.strip():
        file_path = Path(local_file_path.strip())
        if not file_path.exists():
            return f"Material file not found at '{local_file_path.strip()}'."
        file_bytes = file_path.read_bytes()
        original_filename = file_path.name
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            source_type = "pdf"
            mime_type = "application/pdf"
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            source_type = "image"
            mime_type = f"image/{suffix.lstrip('.') if suffix != '.jpg' else 'jpeg'}"
        else:
            return "Unsupported file type. Use a PDF or image file, or provide pasted text."
    elif text_content.strip():
        source_type = "text"
    else:
        return "Provide either pasted text content or a local file path."

    teacher_row = get_teacher()
    if not teacher_row:
        return "No teacher record is configured."

    try:
        result = ingest_reading_material(
            teacher_id=int(teacher_row["id"]),
            board_type="CBSE",
            grade=grade.strip(),
            title=title.strip(),
            source_type=source_type,
            content_bytes=file_bytes,
            text_content=text_content,
            original_filename=original_filename,
            mime_type=mime_type,
            llama_base_url=_tool_settings()["llama_base_url"],
            llama_model_name=_tool_settings()["llama_model"],
        )
        curriculum_subject = get_curriculum_subject(grade=grade.strip(), subject=result["subject"])
        chapters = list_curriculum_chapters(curriculum_subject["id"]) if curriculum_subject else []
        materials = list_subject_materials(grade=grade.strip(), subject=result["subject"])
        return json.dumps(
            {
                "grade": grade.strip(),
                "subject": result["subject"],
                "material_id": result["material_id"],
                "chunk_count": result["chunk_count"],
                "indexed_count": result["indexed_count"],
                "summary": result["summary"],
                "chapters": chapters,
                "materials_count": len(materials),
            },
            indent=2,
            ensure_ascii=False,
        )
    except Exception as exc:
        return f"Failed to ingest reading material: {exc}"


@mcp.tool(description="Search uploaded subject reading materials for a given grade and subject.")
def search_subject_material_library(
    grade: Annotated[str, "The class grade, for example 7"],
    subject: Annotated[str, "The subject to search, for example Science"],
    query: Annotated[str, "The search query to run against uploaded materials"],
) -> str:
    try:
        hits = search_rag_subject_materials(
            grade=grade.strip(),
            subject=subject.strip(),
            query=query.strip(),
        )
    except Exception as exc:
        return f"Failed to search reading materials: {exc}"
    return json.dumps({"grade": grade.strip(), "subject": subject.strip(), "hits": hits}, indent=2, ensure_ascii=False)


@mcp.tool(description="Explain a concept using uploaded subject reading materials.")
def explain_from_subject_materials(
    grade: Annotated[str, "The class grade, for example 7"],
    subject: Annotated[str, "The subject to use, for example Science"],
    question: Annotated[str, "The concept or teacher question to answer from uploaded materials"],
    preferred_language: Annotated[str, "Preferred answer language"] = "English",
) -> str:
    context = build_retrieval_context(
        grade=grade.strip(),
        subject=subject.strip(),
        query=question.strip(),
    )
    if not context:
        return (
            f"No uploaded materials were found for Grade {grade.strip()} {subject.strip()} "
            "or no relevant chunks matched the query."
        )
    answer = _ask_llama(
        "You are a teacher copilot. Answer only from the retrieved textbook material when possible.",
        (
            f"Grade: {grade.strip()}\n"
            f"Subject: {subject.strip()}\n"
            f"Preferred language: {preferred_language.strip() or 'English'}\n"
            f"Teacher question: {question.strip()}\n\n"
            f"Retrieved material:\n{context}"
        ),
    )
    return json.dumps(
        {
            "grade": grade.strip(),
            "subject": subject.strip(),
            "question": question.strip(),
            "answer": answer,
            "retrieval_context": context,
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Mark class attendance. Provide absent roll numbers or absent student names, and all other students will be marked present.")
def mark_class_attendance(
    grade: Annotated[str, "The class grade, for example 7"],
    section: Annotated[str, "The class section, for example A"],
    subject: Annotated[str, "The class subject, for example Science"],
    attendance_date: Annotated[str, "Attendance date in YYYY-MM-DD format. Leave blank to use today's date."] = "",
    absent_roll_numbers: Annotated[str, "Comma-separated absent roll numbers"] = "",
    absent_student_names: Annotated[str, "Comma-separated absent student names"] = "",
) -> str:
    if not absent_roll_numbers.strip() and not absent_student_names.strip():
        return (
            "Missing absent student details. Ask the teacher to provide the absent roll numbers, "
            "absent student names, or both."
        )

    teacher_id, class_row = _find_class_for_attendance(grade, section, subject)
    if teacher_id is None or class_row is None:
        return (
            f"No class mapping was found for Grade {grade.strip()}-{section.strip()} "
            f"and subject '{subject.strip()}'."
        )

    result = mark_attendance_from_identifiers(
        class_id=int(class_row["class_id"]),
        teacher_id=teacher_id,
        attendance_date=attendance_date.strip(),
        absent_roll_numbers=absent_roll_numbers,
        absent_student_names=absent_student_names,
        source="tool",
        raw_model_output="Marked via MCP tool call.",
    )
    return json.dumps(
        {
            "grade": class_row["grade"],
            "section": class_row["section"],
            "subject": class_row["subject"],
            "attendance_date": result["attendance_date"],
            "present_count": result["present_count"],
            "absent_count": result["absent_count"],
            "matched_roll_numbers": result["matched_roll_numbers"],
            "matched_names": result["matched_names"],
            "unresolved_mentions": result["unresolved_mentions"],
            "absent_students": result["absent_students"],
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Edit attendance for one student on one date.")
def edit_student_attendance(
    grade: Annotated[str, "The class grade, for example 7"],
    section: Annotated[str, "The class section, for example A"],
    subject: Annotated[str, "The class subject, for example Science"],
    attendance_date: Annotated[str, "Attendance date in YYYY-MM-DD format"],
    status: Annotated[str, "Attendance status: present or absent"],
    student_name: Annotated[str, "The student's full or partial name"] = "",
    roll_number: Annotated[str, "The student's roll number"] = "",
) -> str:
    if not attendance_date.strip():
        return "Missing attendance date. Ask the teacher to provide the date in YYYY-MM-DD format."
    normalized_status = status.strip().lower()
    if normalized_status not in {"present", "absent"}:
        return "Invalid attendance status. Use either present or absent."
    student_id = _find_student_id(student_name=student_name, roll_number=roll_number)
    if student_id is None:
        return "No active student found for the provided student name or roll number."
    teacher_id, class_row = _find_class_for_attendance(grade, section, subject)
    if teacher_id is None or class_row is None:
        return (
            f"No class mapping was found for Grade {grade.strip()}-{section.strip()} "
            f"and subject '{subject.strip()}'."
        )
    update_student_attendance_status(
        class_id=int(class_row["class_id"]),
        student_id=student_id,
        teacher_id=teacher_id,
        attendance_date=attendance_date.strip(),
        status=normalized_status,
        source="tool",
        raw_model_output="Edited via MCP tool call.",
    )
    return json.dumps(
        {
            "grade": class_row["grade"],
            "section": class_row["section"],
            "subject": class_row["subject"],
            "attendance_date": attendance_date.strip(),
            "status": normalized_status,
            "student_id": student_id,
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Create a new class for the teacher.")
def create_class(
    academic_year: Annotated[str, "Academic year like 2026-27"],
    grade: Annotated[str, "The class grade, for example 7"],
    section: Annotated[str, "The class section, for example A"],
    subject: Annotated[str, "The subject, for example Science"],
    medium: Annotated[str, "The teaching medium, for example English"] = "",
) -> str:
    with get_connection() as connection:
        teacher_row = connection.execute("SELECT id FROM teachers ORDER BY id LIMIT 1").fetchone()
    if not teacher_row:
        return "No teacher record was found."
    class_id = create_class_for_teacher(
        teacher_id=int(teacher_row["id"]),
        academic_year=academic_year,
        grade=grade,
        section=section,
        subject=subject,
        medium=medium,
    )
    return json.dumps(
        {
            "class_id": class_id,
            "academic_year": academic_year.strip(),
            "grade": grade.strip(),
            "section": section.strip(),
            "subject": subject.strip(),
            "medium": medium.strip(),
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Add a student to an existing class.")
def add_student(
    grade: Annotated[str, "The class grade, for example 7"],
    section: Annotated[str, "The class section, for example A"],
    subject: Annotated[str, "The class subject, for example Science"],
    roll_number: Annotated[str, "The student's roll number"],
    full_name: Annotated[str, "The student's full name"],
    email: Annotated[str, "The student's email"] = "",
    preferred_language: Annotated[str, "The student's preferred language"] = "",
    accessibility_notes: Annotated[str, "Accessibility notes for the student"] = "",
) -> str:
    teacher_id, class_row = _find_class_for_management(grade, section, subject)
    if teacher_id is None or class_row is None:
        return (
            f"No class mapping was found for Grade {grade.strip()}-{section.strip()} "
            f"and subject '{subject.strip()}'."
        )
    student_id = add_student_to_class(
        class_id=int(class_row["class_id"]),
        roll_number=roll_number,
        full_name=full_name,
        email=email,
        preferred_language=preferred_language,
        accessibility_notes=accessibility_notes,
    )
    return json.dumps(
        {
            "student_id": student_id,
            "class_id": int(class_row["class_id"]),
            "roll_number": roll_number.strip(),
            "full_name": full_name.strip(),
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Remove a student from the active class roster by deactivating the student record.")
def remove_student(
    student_name: Annotated[str, "The student's full or partial name"] = "",
    roll_number: Annotated[str, "The student's roll number"] = "",
) -> str:
    student_id = _find_student_id(student_name=student_name, roll_number=roll_number)
    if student_id is None:
        return "No active student found for the provided student name or roll number."
    deactivate_student(student_id)
    return json.dumps(
        {
            "student_id": student_id,
            "status": "inactive",
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Add a chapter to an existing class subject.")
def add_subject_chapter(
    grade: Annotated[str, "The class grade, for example 7"],
    section: Annotated[str, "The class section, for example A"],
    subject: Annotated[str, "The class subject, for example Science"],
    chapter_code: Annotated[str, "The chapter code"],
    chapter_name: Annotated[str, "The chapter name"],
    term: Annotated[str, "The term, for example Term 1"] = "",
) -> str:
    teacher_id, class_row = _find_class_for_management(grade, section, subject)
    if teacher_id is None or class_row is None:
        return (
            f"No class mapping was found for Grade {grade.strip()}-{section.strip()} "
            f"and subject '{subject.strip()}'."
        )
    chapter_id = create_chapter_for_class(
        class_id=int(class_row["class_id"]),
        subject=subject,
        chapter_code=chapter_code,
        chapter_name=chapter_name,
        term=term,
    )
    return json.dumps(
        {
            "chapter_id": chapter_id,
            "class_id": int(class_row["class_id"]),
            "chapter_code": chapter_code.strip(),
            "chapter_name": chapter_name.strip(),
            "term": term.strip(),
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Clear attendance history for one student or for all students in a class.")
def clear_attendance_records(
    scope: Annotated[str, "Use student or class"],
    grade: Annotated[str, "The class grade, for example 7"] = "",
    section: Annotated[str, "The class section, for example A"] = "",
    subject: Annotated[str, "The class subject, for example Science"] = "",
    student_name: Annotated[str, "The student's full or partial name"] = "",
    roll_number: Annotated[str, "The student's roll number"] = "",
) -> str:
    normalized_scope = scope.strip().lower()
    if normalized_scope == "student":
        student_id = _find_student_id(student_name=student_name, roll_number=roll_number)
        if student_id is None:
            return "No active student found for the provided student name or roll number."
        teacher_id, class_row = _find_class_for_management(grade, section, subject) if grade.strip() and section.strip() else (None, None)
        deleted = clear_student_attendance(student_id, int(class_row["class_id"]) if class_row else None)
        return json.dumps(
            {
                "scope": "student",
                "student_id": student_id,
                "deleted_records": deleted,
            },
            indent=2,
            ensure_ascii=False,
        )
    if normalized_scope == "class":
        teacher_id, class_row = _find_class_for_management(grade, section, subject)
        if teacher_id is None or class_row is None:
            return (
                f"No class mapping was found for Grade {grade.strip()}-{section.strip()} "
                f"and subject '{subject.strip()}'."
            )
        deleted = clear_class_attendance(int(class_row["class_id"]))
        return json.dumps(
            {
                "scope": "class",
                "class_id": int(class_row["class_id"]),
                "deleted_records": deleted,
            },
            indent=2,
            ensure_ascii=False,
        )
    return "Invalid scope. Use either student or class."


@mcp.tool(description="Update metadata for an existing class.")
def update_class(
    grade: Annotated[str, "The current class grade, for example 7"],
    section: Annotated[str, "The current class section, for example A"],
    subject: Annotated[str, "The current class subject, for example Science"],
    new_academic_year: Annotated[str, "New academic year like 2026-27"] = "",
    new_grade: Annotated[str, "New class grade"] = "",
    new_section: Annotated[str, "New class section"] = "",
    new_subject: Annotated[str, "New class subject"] = "",
    new_medium: Annotated[str, "New teaching medium"] = "",
) -> str:
    teacher_id, class_row = _find_class_for_management(grade, section, subject)
    if teacher_id is None or class_row is None:
        return (
            f"No class mapping was found for Grade {grade.strip()}-{section.strip()} "
            f"and subject '{subject.strip()}'."
        )
    update_class_details(
        class_id=int(class_row["class_id"]),
        academic_year=new_academic_year or class_row["academic_year"],
        grade=new_grade or class_row["grade"],
        section=new_section or class_row["section"],
        subject=new_subject or class_row["subject"],
        medium=new_medium or class_row["medium"] or "",
    )
    return json.dumps(
        {
            "class_id": int(class_row["class_id"]),
            "academic_year": (new_academic_year or class_row["academic_year"]).strip(),
            "grade": (new_grade or class_row["grade"]).strip(),
            "section": (new_section or class_row["section"]).strip(),
            "subject": (new_subject or class_row["subject"]).strip(),
            "medium": (new_medium or class_row["medium"] or "").strip(),
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Update details for an existing student.")
def update_student(
    student_name: Annotated[str, "The student's current full or partial name"] = "",
    roll_number: Annotated[str, "The student's current roll number"] = "",
    new_roll_number: Annotated[str, "New roll number"] = "",
    new_full_name: Annotated[str, "New full name"] = "",
    new_email: Annotated[str, "New email address"] = "",
    new_preferred_language: Annotated[str, "New preferred language"] = "",
    new_accessibility_notes: Annotated[str, "New accessibility notes"] = "",
) -> str:
    student_id = _find_student_id(student_name=student_name, roll_number=roll_number)
    if student_id is None:
        return "No active student found for the provided student name or roll number."
    student_detail = get_student_detail(student_id)
    student_row = student_detail.get("student")
    if not student_row:
        return "Student record was found, but detailed student data is unavailable."
    update_student_details(
        student_id=student_id,
        roll_number=new_roll_number or student_row["roll_number"],
        full_name=new_full_name or student_row["full_name"],
        email=new_email or student_row.get("email", "") or "",
        preferred_language=new_preferred_language or student_row["preferred_language"] or "",
        accessibility_notes=new_accessibility_notes or student_row["accessibility_notes"] or "",
    )
    return json.dumps(
        {
            "student_id": student_id,
            "roll_number": (new_roll_number or student_row["roll_number"]).strip(),
            "full_name": (new_full_name or student_row["full_name"]).strip(),
            "email": (new_email or student_row.get("email", "") or "").strip(),
            "preferred_language": (new_preferred_language or student_row["preferred_language"] or "").strip(),
            "accessibility_notes": (new_accessibility_notes or student_row["accessibility_notes"] or "").strip(),
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Reactivate a previously removed student.")
def restore_student(
    student_id: Annotated[int, "The student id to reactivate"],
) -> str:
    reactivate_student(student_id)
    return json.dumps(
        {
            "student_id": student_id,
            "status": "active",
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Update an existing chapter for a class subject.")
def update_subject_chapter(
    grade: Annotated[str, "The class grade, for example 7"],
    section: Annotated[str, "The class section, for example A"],
    subject: Annotated[str, "The class subject, for example Science"],
    current_chapter_name: Annotated[str, "The current chapter name"],
    new_chapter_code: Annotated[str, "The new chapter code"] = "",
    new_chapter_name: Annotated[str, "The new chapter name"] = "",
    new_term: Annotated[str, "The new term"] = "",
) -> str:
    teacher_id, class_row = _find_class_for_management(grade, section, subject)
    if teacher_id is None or class_row is None:
        return (
            f"No class mapping was found for Grade {grade.strip()}-{section.strip()} "
            f"and subject '{subject.strip()}'."
        )
    chapter_row = _find_chapter_for_management(int(class_row["class_id"]), current_chapter_name)
    if not chapter_row:
        return f"No chapter mapping was found for '{current_chapter_name.strip()}'."
    update_chapter_details(
        chapter_id=int(chapter_row["id"]),
        chapter_code=new_chapter_code or chapter_row["chapter_code"],
        chapter_name=new_chapter_name or chapter_row["chapter_name"],
        term=new_term or chapter_row["term"] or "",
    )
    return json.dumps(
        {
            "chapter_id": int(chapter_row["id"]),
            "chapter_code": (new_chapter_code or chapter_row["chapter_code"]).strip(),
            "chapter_name": (new_chapter_name or chapter_row["chapter_name"]).strip(),
            "term": (new_term or chapter_row["term"] or "").strip(),
        },
        indent=2,
        ensure_ascii=False,
    )


@mcp.tool(description="Delete a chapter only if it is not linked to any assessments.")
def delete_subject_chapter(
    grade: Annotated[str, "The class grade, for example 7"],
    section: Annotated[str, "The class section, for example A"],
    subject: Annotated[str, "The class subject, for example Science"],
    chapter_name: Annotated[str, "The chapter name to delete"],
) -> str:
    teacher_id, class_row = _find_class_for_management(grade, section, subject)
    if teacher_id is None or class_row is None:
        return (
            f"No class mapping was found for Grade {grade.strip()}-{section.strip()} "
            f"and subject '{subject.strip()}'."
        )
    chapter_row = _find_chapter_for_management(int(class_row["class_id"]), chapter_name)
    if not chapter_row:
        return f"No chapter mapping was found for '{chapter_name.strip()}'."
    deleted, message = delete_chapter_if_unused(int(chapter_row["id"]))
    return json.dumps(
        {
            "chapter_id": int(chapter_row["id"]),
            "deleted": deleted,
            "message": message,
        },
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
