from __future__ import annotations

import atexit
import asyncio
import json
from pathlib import Path
import signal
import sys
import threading
from datetime import datetime, timedelta

import streamlit as st
from requests import RequestException

ROOT = Path(__file__).resolve().parent.parent
CONVERSATION_LOG_PATH = ROOT / "conversation.txt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import ensure_database
from app.daily_brief import DEFAULT_DAILY_LOOP_SERVICE
from app.demo_seed import ensure_demo_data
from app.assessment_sync import sync_google_form_assessment
from app.attendance import mark_attendance_from_identifiers, parse_absent_students_from_audio, resolve_absent_students
from app.gemma_adaptation_profile import generate_student_adaptation_profile_with_gemma
from app.generator import build_lesson_pack, build_quiz_questions
from app.llama_client import LlamaServerClient, LlamaServerConfig
from app.material_ingestion import ingest_reading_material
from app.model_control import choose_quiz_generation_mode, load_model_sampling_config
from app.personalization import build_student_generation_context, build_student_retrieval_query
from app.pdf_export import build_quiz_pdf_bytes
from app.rag import build_retrieval_context, search_subject_materials
from app.teaching_progress import DEFAULT_TEACHING_PLANNER_SERVICE, WEEKDAY_LABELS, WEEKDAY_OPTIONS
from app.ui_shell import GEMMA_ANALYSIS_LABEL, apply_global_ui_theme, render_section_intro, render_workspace_banner
from mcp_servers.llama_MCP_bridge import cleanup as cleanup_mcp_bridge
from mcp_servers.llama_MCP_bridge import execute_tool as execute_mcp_tool
from mcp_servers.llama_MCP_bridge import get_all_mcp_tools
from mcp_servers.llama_MCP_bridge import start_servers as start_mcp_servers
from app.repository import (
    assessment_repository,
    attendance_repository,
    analytics_repository,
    coverage_repository,
    curriculum_repository,
    material_repository,
    planning_repository,
    queue_repository,
    student_repository,
    teacher_class_repository,
    timetable_repository,
)


st.set_page_config(
    page_title="Pathshala Play",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Pathshala Play — an AI-powered classroom copilot for Indian teachers."},
)
apply_global_ui_theme()

ensure_database()
ensure_demo_data()
MODEL_SAMPLING = load_model_sampling_config()
_SHUTDOWN_HOOKS_REGISTERED = False


def _cleanup_mcp_bridge_sync() -> None:
    try:
        if DEFAULT_TEACHING_PLANNER_SERVICE.get_local_recorder_status().get("active"):
            DEFAULT_TEACHING_PLANNER_SERVICE.stop_local_microphone_recording()
    except Exception:
        pass
    try:
        asyncio.run(cleanup_mcp_bridge())
    except RuntimeError:
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(cleanup_mcp_bridge())
            finally:
                loop.close()
        except Exception:
            pass
    except Exception:
        pass


def _handle_app_shutdown(_signum=None, _frame=None) -> None:
    _cleanup_mcp_bridge_sync()
    raise SystemExit(0)


def _register_shutdown_hooks() -> None:
    global _SHUTDOWN_HOOKS_REGISTERED
    if _SHUTDOWN_HOOKS_REGISTERED:
        return

    atexit.register(_cleanup_mcp_bridge_sync)
    if threading.current_thread() is threading.main_thread():
        for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            signal_value = getattr(signal, signal_name, None)
            if signal_value is None:
                continue
            try:
                signal.signal(signal_value, _handle_app_shutdown)
            except (ValueError, OSError):
                continue
    _SHUTDOWN_HOOKS_REGISTERED = True


_register_shutdown_hooks()


def queue_teacher_chat_prompt(class_id: int, prompt_override: str = "") -> None:
    input_key = f"teacher_chat_input_{class_id}"
    pending_user_prompt_key = f"teacher_chat_pending_user_prompt_{class_id}"
    prompt = prompt_override.strip() or str(st.session_state.get(input_key, "")).strip()
    if prompt:
        st.session_state[pending_user_prompt_key] = prompt


def build_teacher_chat_prompt_from_inputs(
    *,
    class_id: int,
    subject_name: str,
    uploaded_chat_files,
    recorded_chat_audio,
    llama_base_url: str,
    llama_model_name: str,
    use_llama_server: bool,
) -> str:
    input_key = f"teacher_chat_input_{class_id}"
    prompt_parts: list[str] = []
    typed_prompt = str(st.session_state.get(input_key, "")).strip()
    if typed_prompt:
        prompt_parts.append(typed_prompt)

    if recorded_chat_audio:
        transcript = DEFAULT_TEACHING_PLANNER_SERVICE.transcribe_audio_to_text(
            audio_bytes=recorded_chat_audio.getvalue(),
            audio_mime_type=recorded_chat_audio.type or "audio/wav",
            llama_base_url=llama_base_url,
            llama_model_name=llama_model_name,
            prompt_hint=f"Teacher chat voice note for subject {subject_name}",
        )
        prompt_parts.append(f"Teacher voice note transcript:\n{transcript}")

    extracted_file_parts: list[str] = []
    for uploaded_file in uploaded_chat_files or []:
        extracted_text = DEFAULT_TEACHING_PLANNER_SERVICE.extract_uploaded_text(
            content_bytes=uploaded_file.getvalue(),
            mime_type=uploaded_file.type or "",
            original_filename=uploaded_file.name,
            llama_base_url=llama_base_url,
            llama_model_name=llama_model_name,
            use_llama_server=use_llama_server,
        )
        extracted_file_parts.append(
            f"Attached file: {uploaded_file.name}\nExtracted text:\n{extracted_text[:16000]}"
        )
    if extracted_file_parts:
        prompt_parts.append("\n\n".join(extracted_file_parts))

    return "\n\n".join(part for part in prompt_parts if part.strip()).strip()


def choose_auto_quiz_target(
    *,
    class_id: int,
    class_row: dict,
    class_subjects: list[dict],
    llama_base_url: str,
    llama_model_name: str,
    use_llama_server: bool,
) -> dict[str, str]:
    subject_chapter_map: dict[str, list[dict[str, str]]] = {}
    for item in class_subjects:
        subject_name = str(item.get("subject") or "").strip()
        if not subject_name:
            continue
        subject_chapter_map[subject_name] = curriculum_repository.list_chapters_for_class(class_id, subject_name)

    assessment_history = assessment_repository.list_class_assessment_history(class_id)
    compact_history = [
        {
            "subject": row.get("subject", ""),
            "topic": row.get("chapter_name", ""),
            "avg_percentage": row.get("avg_percentage"),
            "submissions": row.get("submissions"),
            "created_at": row.get("created_at", ""),
        }
        for row in assessment_history
    ]

    if use_llama_server:
        try:
            client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
            response = client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a teacher assessment planner. "
                            "Choose the next best quiz target for the class. "
                            "Use only the available subject chapters and the summarized academic-year quiz history. "
                            "Prefer either an untested topic or a previously weak topic with low scores. "
                            "Return strict JSON only with this shape: "
                            "{\"subject\":\"string\",\"topic\":\"string\",\"reason\":\"string\",\"selection_mode\":\"untested|reteach\"}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Class: Grade {class_row.get('grade', '')}-{class_row.get('section', '')}\n"
                            f"Academic year: {class_row.get('academic_year', '')}\n\n"
                            "Available subject chapters:\n"
                            f"{json.dumps(subject_chapter_map, indent=2, ensure_ascii=False)}\n\n"
                            "Quiz history summaries for this academic year:\n"
                            f"{json.dumps(compact_history, indent=2, ensure_ascii=False)}"
                        ),
                    },
                ],
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                response_format={"type": "json_object"},
                extra_payload={"model": llama_model_name},
            )
            parsed = parse_json_content(extract_llama_content(response) or "")
            if parsed and parsed.get("subject") and parsed.get("topic"):
                return {
                    "subject": str(parsed.get("subject") or "").strip(),
                    "topic": str(parsed.get("topic") or "").strip(),
                    "reason": str(parsed.get("reason") or "").strip(),
                    "selection_mode": str(parsed.get("selection_mode") or "untested").strip(),
                }
        except (RequestException, ValueError, json.JSONDecodeError):
            pass

    tested_pairs = {
        (str(item.get("subject") or "").strip().lower(), str(item.get("topic") or "").strip().lower())
        for item in compact_history
    }
    weak_history = sorted(
        [item for item in compact_history if item.get("avg_percentage") is not None],
        key=lambda item: (item.get("avg_percentage", 100), -(item.get("submissions") or 0)),
    )
    if weak_history and float(weak_history[0].get("avg_percentage") or 100) < 60:
        return {
            "subject": str(weak_history[0].get("subject") or "").strip(),
            "topic": str(weak_history[0].get("topic") or "").strip(),
            "reason": "Selected a previously weak topic because the last average score was low.",
            "selection_mode": "reteach",
        }

    for subject_name, chapters in subject_chapter_map.items():
        for chapter in chapters:
            pair = (subject_name.strip().lower(), str(chapter.get("chapter_name") or "").strip().lower())
            if pair not in tested_pairs:
                return {
                    "subject": subject_name,
                    "topic": str(chapter.get("chapter_name") or "").strip(),
                    "reason": "Selected an untested chapter for coverage expansion.",
                    "selection_mode": "untested",
                }

    fallback_subject = str(class_subjects[0].get("subject") or class_row.get("subject") or "").strip() if class_subjects else str(class_row.get("subject") or "").strip()
    fallback_chapters = subject_chapter_map.get(fallback_subject, [])
    fallback_topic = str(fallback_chapters[0].get("chapter_name") or "Concept revision").strip() if fallback_chapters else "Concept revision"
    return {
        "subject": fallback_subject,
        "topic": fallback_topic,
        "reason": "Used the first available chapter as fallback.",
        "selection_mode": "untested",
    }


def stop_teacher_chat_run(class_id: int) -> None:
    pending_agent_key = f"teacher_chat_pending_agent_{class_id}"
    pending_user_prompt_key = f"teacher_chat_pending_user_prompt_{class_id}"
    running_key = f"teacher_chat_running_{class_id}"
    stop_requested_key = f"teacher_chat_stop_requested_{class_id}"
    st.session_state.pop(pending_agent_key, None)
    st.session_state.pop(pending_user_prompt_key, None)
    st.session_state[running_key] = False
    st.session_state[stop_requested_key] = True


def build_quiz_generation_messages(
    *,
    subject: str,
    grade: str,
    chapter_name: str,
    learner_profile: str,
    source_material: str,
    teacher_instructions: str,
    language: str,
    question_count: int,
) -> list[dict[str, str]]:
    schema_hint = {
        "questions": [
            {
                "question_text": "string",
                "question_type": "mcq or short_answer",
                "options": {"A": "string", "B": "string", "C": "string", "D": "string"},
                "difficulty": "easy/medium/hard",
                "bloom_level": "remember/understand/apply/analyze",
                "marks": 2,
                "correct_answer": "string",
                "explanation": "string",
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an Indian school teacher assessment copilot. "
                "Generate a short chapter quiz for a mixed-ability classroom. "
                "Return strict JSON only. Do not include markdown fences, commentary, or trailing text. "
                "Return valid JSON only with this shape: "
                f"{json.dumps(schema_hint)}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Subject: {subject}\n"
                f"Grade: {grade}\n"
                f"Chapter: {chapter_name}\n"
                f"Language: {language}\n"
                f"Learner profile: {learner_profile}\n"
                f"Teacher instructions: {teacher_instructions or 'None provided'}\n"
                f"Source material: {source_material}\n\n"
                "Requirements:\n"
                f"- Create exactly {question_count} questions.\n"
                "- Use only these question_type values: mcq, short_answer.\n"
                "- For every mcq question, create exactly four options with keys A, B, C, D.\n"
                "- For short_answer questions, set options to an empty object {}.\n"
                "- Use only these difficulty values: easy, medium, hard.\n"
                "- Use only these bloom_level values: remember, understand, apply, analyze.\n"
                "- Keep wording age-appropriate for Indian classrooms.\n"
                "- Questions should reveal misconceptions, not only memorization.\n"
                "- Each marks value must be numeric.\n"
                "- Output JSON only."
            ),
        },
    ]


def build_single_question_messages(
    *,
    subject: str,
    grade: str,
    chapter_name: str,
    learner_profile: str,
    source_material: str,
    teacher_instructions: str,
    language: str,
    question_number: int,
    total_questions: int,
    prior_questions: list[dict],
) -> list[dict[str, str]]:
    schema_hint = {
        "question": {
            "question_text": "string",
            "question_type": "mcq or short_answer",
            "options": {"A": "string", "B": "string", "C": "string", "D": "string"},
            "difficulty": "easy/medium/hard",
            "bloom_level": "remember/understand/apply/analyze",
            "marks": 2,
            "correct_answer": "string",
            "explanation": "string",
        }
    }
    prior_text = "\n".join(
        f"- {index}. {question['question_text']}" for index, question in enumerate(prior_questions, start=1)
    ) or "- None yet."
    return [
        {
            "role": "system",
            "content": (
                "You are an Indian school teacher assessment copilot. "
                "Generate exactly one quiz question. "
                "Return strict JSON only. Do not include markdown fences, commentary, or trailing text. "
                f"Return valid JSON only with this shape: {json.dumps(schema_hint)}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Subject: {subject}\n"
                f"Grade: {grade}\n"
                f"Chapter: {chapter_name}\n"
                f"Language: {language}\n"
                f"Learner profile: {learner_profile}\n"
                f"Teacher instructions: {teacher_instructions or 'None provided'}\n"
                f"Source material: {source_material}\n"
                f"Generate question {question_number} of {total_questions}.\n\n"
                "Already generated questions. Do not repeat them or make near-duplicates:\n"
                f"{prior_text}\n\n"
                "Requirements:\n"
                "- Use only these question_type values: mcq, short_answer.\n"
                "- If the question_type is mcq, create exactly four options with keys A, B, C, D.\n"
                "- If the question_type is short_answer, set options to an empty object {}.\n"
                "- Use only these difficulty values: easy, medium, hard.\n"
                "- Use only these bloom_level values: remember, understand, apply, analyze.\n"
                "- Make this question test a different angle or misconception from earlier questions.\n"
                "- Each marks value must be numeric.\n"
                "- Output JSON only."
            ),
        },
    ]


def extract_llama_content(response: dict) -> str:
    content = ""

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "") or choices[0].get("text", "")
    elif "content" in response:
        content = response.get("content", "")
    elif "text" in response:
        content = response.get("text", "")

    if not content:
        return ""

    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


def extract_llama_message(response: dict) -> dict:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    return {}


def extract_tool_calls(response: dict) -> list[dict]:
    message = extract_llama_message(response)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        return [item for item in tool_calls if isinstance(item, dict)]

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        return [
            {
                "id": "function_call_1",
                "type": "function",
                "function": function_call,
            }
        ]
    return []


def append_conversation_log(label: str, content: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with CONVERSATION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {label}\n")
        log_file.write(f"{content}\n\n")


def parse_json_content(content: str) -> dict | None:
    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        repaired = content[start : end + 1]
        repaired = repaired.replace(",}", "}").replace(",]", "]")
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


def normalize_question(question: dict) -> dict | None:
    question_type_map = {
        "multiple choice": "mcq",
        "mcq": "mcq",
        "fill in the blank": "short_answer",
        "short answer": "short_answer",
        "short_answer": "short_answer",
        "checkbox": "checkbox",
        "dropdown": "dropdown",
    }
    bloom_map = {
        "remembering": "remember",
        "remember": "remember",
        "understanding": "understand",
        "understand": "understand",
        "applying": "apply",
        "apply": "apply",
        "analyzing": "analyze",
        "analyse": "analyze",
        "analyze": "analyze",
    }
    raw_question_type = str(question.get("question_type", "short_answer")).strip().lower()
    raw_bloom = str(question.get("bloom_level", "understand")).strip().lower()
    raw_difficulty = str(question.get("difficulty", "medium")).strip().lower()
    raw_options = question.get("options", {}) or {}
    normalized_question_type = question_type_map.get(raw_question_type, "short_answer")
    normalized_options = {}
    if normalized_question_type == "mcq":
        normalized_options = {
            "A": str(raw_options.get("A", "")).strip(),
            "B": str(raw_options.get("B", "")).strip(),
            "C": str(raw_options.get("C", "")).strip(),
            "D": str(raw_options.get("D", "")).strip(),
        }

    normalized = {
        "question_text": str(question.get("question_text", "")).strip(),
        "question_type": normalized_question_type,
        "options": normalized_options,
        "difficulty": raw_difficulty if raw_difficulty in {"easy", "medium", "hard"} else "medium",
        "bloom_level": bloom_map.get(raw_bloom, "understand"),
        "marks": float(question.get("marks", 1)),
        "correct_answer": str(question.get("correct_answer", "")).strip(),
        "explanation": str(question.get("explanation", "")).strip(),
    }
    if normalized["question_type"] == "mcq" and not all(normalized_options.values()):
        return None
    return normalized if normalized["question_text"] else None


def parse_llama_quiz_response(response: dict) -> list[dict] | None:
    content = extract_llama_content(response)
    parsed = parse_json_content(content)
    if not parsed:
        return None

    questions = parsed.get("questions")
    if not isinstance(questions, list) or not questions:
        return None

    normalized_questions = []
    for question in questions:
        if not isinstance(question, dict):
            return None
        normalized = normalize_question(question)
        if not normalized:
            return None
        normalized_questions.append(normalized)

    return normalized_questions


def parse_llama_single_question_response(response: dict) -> dict | None:
    content = extract_llama_content(response)
    parsed = parse_json_content(content)
    if not parsed:
        return None

    question = parsed.get("question")
    if not isinstance(question, dict):
        questions = parsed.get("questions")
        if isinstance(questions, list) and questions:
            question = questions[0]
        else:
            question = parsed
    if not isinstance(question, dict):
        return None
    return normalize_question(question)


def normalize_raw_llama_output(response: dict) -> str:
    content = extract_llama_content(response)
    if content:
        return content
    return json.dumps(response, indent=2, ensure_ascii=False)


def generate_quiz_with_llama(
    *,
    client: LlamaServerClient,
    base_url: str,
    model_name: str,
    generation_strategy: str,
    subject: str,
    grade: str,
    chapter_name: str,
    learner_profile: str,
    source_material: str,
    teacher_instructions: str,
    language: str,
    question_count: int,
) -> tuple[list[dict] | None, str, list[str]]:
    raw_outputs: list[str] = []
    if generation_strategy == "one_by_one":
        questions: list[dict] = []
        for question_number in range(1, question_count + 1):
            response = client.chat_completion(
                messages=build_single_question_messages(
                    subject=subject,
                    grade=grade,
                    chapter_name=chapter_name,
                    learner_profile=learner_profile,
                    source_material=source_material,
                    teacher_instructions=teacher_instructions,
                    language=language,
                    question_number=question_number,
                    total_questions=question_count,
                    prior_questions=questions,
                ),
                temperature=MODEL_SAMPLING.temperature,
                top_p=MODEL_SAMPLING.top_p,
                top_k=MODEL_SAMPLING.top_k,
                response_format={"type": "json_object"},
                extra_payload={"model": model_name},
            )
            raw_outputs.append(normalize_raw_llama_output(response))
            parsed_question = parse_llama_single_question_response(response)
            if not parsed_question:
                return None, (
                    f"llama-server generated question {question_number}, but the JSON could not be parsed."
                ), raw_outputs
            questions.append(parsed_question)
        return questions, (
            f"Generated via llama-server at {base_url} using model {model_name} "
            f"in one-by-one mode."
        ), raw_outputs

    response = client.chat_completion(
        messages=build_quiz_generation_messages(
            subject=subject,
            grade=grade,
            chapter_name=chapter_name,
            learner_profile=learner_profile,
            source_material=source_material,
            teacher_instructions=teacher_instructions,
            language=language,
            question_count=question_count,
        ),
        temperature=MODEL_SAMPLING.temperature,
        top_p=MODEL_SAMPLING.top_p,
        top_k=MODEL_SAMPLING.top_k,
        response_format={"type": "json_object"},
        extra_payload={"model": model_name},
    )
    raw_outputs.append(normalize_raw_llama_output(response))
    parsed_questions = parse_llama_quiz_response(response)
    if not parsed_questions:
        return None, "llama-server responded, but the one-shot JSON could not be parsed.", raw_outputs
    return parsed_questions, (
        f"Generated via llama-server at {base_url} using model {model_name} "
        f"in one-shot mode."
    ), raw_outputs


def format_class_label(class_row: dict) -> str:
    subject_label = class_row.get("subjects_csv") or class_row.get("subject", "")
    subject_count = class_row.get("subject_count") or 0
    if subject_count > 1:
        subject_label = f"{subject_count} subjects"
    return (
        f"Grade {class_row['grade']}-{class_row['section']} | "
        f"{subject_label} | {class_row['student_count']} students"
    )


def build_student_quiz_topic(student_detail: dict, subject: str | None = None) -> str:
    adaptation_profile = get_subject_adaptation_profile(
        student_detail,
        subject or (student_detail.get("student") or {}).get("subject"),
    )
    priority_targets = ((adaptation_profile or {}).get("profile") or {}).get("priority_targets", [])
    if priority_targets:
        return "; ".join(priority_targets[:3])

    blueprint = get_subject_blueprint(student_detail, subject or (student_detail.get("student") or {}).get("subject"))
    weaknesses = blueprint.get("weaknesses", [])
    if weaknesses:
        return "; ".join(weaknesses[:3])

    mastery = student_detail.get("mastery", [])
    lagging = [row["concept_name"] for row in mastery if row.get("status") == "lagging"]
    if lagging:
        return "; ".join(lagging[:3])

    return "chapter revision and concept reinforcement"


def get_subject_blueprint(student_detail: dict, subject: str | None) -> dict:
    blueprints = student_detail.get("blueprints") or []
    if subject:
        match = next((item for item in blueprints if item.get("subject") == subject), None)
        if match:
            return match
    return blueprints[0] if blueprints else {}


def get_subject_adaptation_profile(student_detail: dict, subject: str | None) -> dict:
    profiles = student_detail.get("adaptation_profiles") or []
    if subject:
        match = next((item for item in profiles if item.get("subject") == subject), None)
        if match:
            return match
    return profiles[0] if profiles else {}


def get_available_student_subjects(student_detail: dict) -> list[str]:
    subjects = [
        item.get("subject")
        for item in ((student_detail.get("adaptation_profiles") or []) + (student_detail.get("blueprints") or []))
        if item.get("subject")
    ]
    student_subject = (student_detail.get("student") or {}).get("subject")
    if student_subject and student_subject not in subjects:
        subjects.insert(0, student_subject)
    return subjects or ([student_subject] if student_subject else [])


def combine_due_datetime(date_value, time_value) -> str:
    combined = datetime.combine(date_value, time_value)
    return combined.strftime("%Y-%m-%d %H:%M:%S")


def format_due_datetime_for_message(due_at: str | None) -> str:
    if not due_at:
        return "Not set"
    try:
        parsed = datetime.strptime(due_at, "%Y-%m-%d %H:%M:%S")
        return parsed.strftime("%d/%m/%Y at %I:%M %p")
    except ValueError:
        return due_at


def format_date_time_preview(date_value, time_value) -> str:
    return datetime.combine(date_value, time_value).strftime("%d/%m/%Y at %I:%M %p")


def build_quiz_share_message(*, topic: str, subject: str, due_at: str | None, quiz_link: str) -> str:
    return (
        "A new quiz has been assigned to you\n"
        f"Quiz Topic: {topic}\n"
        f"Subject: {subject}\n"
        f"Submit before: {format_due_datetime_for_message(due_at)}\n"
        f"Quiz Link: {quiz_link}"
    )


def format_attendance_date(value) -> str:
    return value.strftime("%Y-%m-%d")


def get_planner_local_audio_keys(class_id: int, subject_name: str) -> tuple[str, str]:
    return (
        f"planner_local_audio_bytes_{class_id}_{subject_name}",
        f"planner_local_audio_mime_{class_id}_{subject_name}",
    )


def maybe_auto_stop_local_recording(
    *,
    class_id: int,
    subject_name: str,
    info_message: str = "Local microphone recording auto-stopped 5 minutes after the scheduled end time.",
) -> None:
    planner_local_audio_bytes_key, planner_local_audio_mime_key = get_planner_local_audio_keys(class_id, subject_name)
    recorder_status = DEFAULT_TEACHING_PLANNER_SERVICE.get_local_recorder_status()
    recorder_session = recorder_status.get("session") or {}
    if (
        recorder_status.get("active")
        and recorder_session.get("class_id") == class_id
        and recorder_session.get("subject") == subject_name
        and recorder_session.get("scheduled_end")
    ):
        try:
            stop_deadline = datetime.strptime(
                f"{datetime.now().date().isoformat()} {recorder_session['scheduled_end']}",
                "%Y-%m-%d %H:%M",
            )
            if datetime.now() >= stop_deadline + timedelta(minutes=5):
                finished_capture = DEFAULT_TEACHING_PLANNER_SERVICE.stop_local_microphone_recording()
                st.session_state[planner_local_audio_bytes_key] = finished_capture["audio_bytes"]
                st.session_state[planner_local_audio_mime_key] = finished_capture["audio_mime_type"]
                st.info(info_message)
                st.rerun()
        except Exception:
            pass


def render_global_mic_controls(
    *,
    class_id: int,
    subject_name: str,
    active_timetable_slot: dict | None,
    capture_support_status: dict,
) -> None:
    planner_local_audio_bytes_key, planner_local_audio_mime_key = get_planner_local_audio_keys(class_id, subject_name)
    recorder_status = DEFAULT_TEACHING_PLANNER_SERVICE.get_local_recorder_status()
    recorder_session = recorder_status.get("session") or {}

    with st.container(border=True):
        st.markdown("**Quick Actions**")
        control_col1, control_col2, control_col3 = st.columns([1, 1, 2])
        with control_col1:
            if st.button(
                "Start Mic Recording",
                key=f"global_start_local_mic_{class_id}_{subject_name}",
                disabled=not capture_support_status.get("automatic_capture_available") or bool(recorder_status.get("active")),
                use_container_width=True,
                type="primary",
            ):
                try:
                    DEFAULT_TEACHING_PLANNER_SERVICE.start_local_microphone_recording(
                        class_id=class_id,
                        subject=subject_name,
                        timetable_slot_id=int(active_timetable_slot["id"]) if active_timetable_slot else None,
                        scheduled_end=str((active_timetable_slot or {}).get("end_time") or ""),
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to start local microphone recording: {exc}")
        with control_col2:
            if st.button(
                "Stop Mic Recording",
                key=f"global_stop_local_mic_{class_id}_{subject_name}",
                disabled=not bool(recorder_status.get("active")),
                use_container_width=True,
            ):
                try:
                    finished_capture = DEFAULT_TEACHING_PLANNER_SERVICE.stop_local_microphone_recording()
                    target_class_id = int((recorder_session or {}).get("class_id") or class_id)
                    target_subject = str((recorder_session or {}).get("subject") or subject_name)
                    target_audio_bytes_key, target_audio_mime_key = get_planner_local_audio_keys(target_class_id, target_subject)
                    st.session_state[target_audio_bytes_key] = finished_capture["audio_bytes"]
                    st.session_state[target_audio_mime_key] = finished_capture["audio_mime_type"]
                    st.success(f"Local microphone recording saved for {target_subject}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to stop local microphone recording: {exc}")
        with control_col3:
            if recorder_status.get("active"):
                st.success(
                    f"Recording is active for {recorder_session.get('subject', subject_name)}"
                    f"{' until ' + recorder_session.get('scheduled_end', '') if recorder_session.get('scheduled_end') else ''}.",
                    icon=":material/mic:",
                )
            elif st.session_state.get(planner_local_audio_bytes_key):
                st.info(
                    f"Saved clip ready for {subject_name}: {len(st.session_state[planner_local_audio_bytes_key])} bytes.",
                    icon=":material/check:",
                )
            else:
                st.caption(capture_support_status.get("reason", "Microphone capture status unavailable."))


def build_teacher_chat_messages(
    *,
    class_row: dict,
    overview: dict,
    concept_gaps: list[dict],
    roster: list[dict],
    chapters: list[dict],
    history: list[dict],
    subject_name: str,
    retrieval_context: str = "",
    daily_brief: dict | None = None,
) -> list[dict[str, str]]:
    current_timestamp = datetime.now().astimezone()
    top_concepts = concept_gaps[:5]
    concept_summary = "\n".join(
        f"- {item['concept_name']}: {item['mastery_percent']}% mastery, {item['students_lagging']} students lagging"
        for item in top_concepts
    ) or "- No concept gap data available yet."
    roster_summary = "\n".join(
        f"- {item.get('roll_number', 'No roll number')} | {item.get('full_name', 'Unknown student')}"
        for item in roster
    ) or "- No active students found."
    chapter_summary = "\n".join(
        f"- {item.get('chapter_code', 'No code')} | {item.get('chapter_name', 'Untitled chapter')}"
        for item in chapters
    ) or "- No chapters found for the selected subject."
    system_prompt = (
        "You are Gemma, acting as a classroom copilot for an Indian teacher. "
        "Give concise, practical, teacher-facing answers. "
        "When the teacher asks you to perform an action, prefer using the available tools instead of only describing what should be done. "
        "You can help with class management, quizzes, attendance, uploaded materials, grade-wide syllabus import, timetable import, and teaching-progress updates. "
        "Use the class context below. If you make recommendations, tie them to the subject, chapter misconceptions, "
        "assessment planning, or remediation strategy.\n\n"
        f"Current date and time: {current_timestamp.strftime('%Y-%m-%d %I:%M %p %Z')}\n"
        f"Current class: Grade {overview.get('grade', class_row.get('grade', ''))}-{overview.get('section', class_row.get('section', ''))}\n"
        f"Class: Grade {overview.get('grade', class_row.get('grade', ''))}-{overview.get('section', class_row.get('section', ''))}\n"
        f"Current subject selected: {subject_name or class_row.get('subject', 'Not set')}\n"
        f"Subject: {subject_name or class_row.get('subject', 'Not set')}\n"
        f"Medium: {overview.get('medium', class_row.get('medium', 'Not set'))}\n"
        f"Academic year: {overview.get('academic_year', class_row.get('academic_year', 'Not set'))}\n"
        f"Student count: {overview.get('student_count', class_row.get('student_count', 0))}\n"
        f"Assessment count: {overview.get('assessment_count', 0)}\n"
        "Students in this class:\n"
        f"{roster_summary}\n"
        "Subject chapters for the current class and selected subject:\n"
        f"{chapter_summary}\n"
        "Top concept gaps:\n"
        f"{concept_summary}"
    )
    if retrieval_context:
        system_prompt += f"\n\nRetrieved subject material:\n{retrieval_context}"
    if daily_brief:
        system_prompt += (
            "\n\nLatest Gemma daily loop brief:\n"
            f"{json.dumps(daily_brief, indent=2, ensure_ascii=False)}"
        )
    return [{"role": "system", "content": system_prompt}, *history]


def stringify_mcp_tool_result(result) -> str:
    structured = getattr(result, "structuredContent", None)
    if structured:
        try:
            return json.dumps(structured, indent=2, ensure_ascii=False)
        except TypeError:
            return str(structured)

    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
            else:
                try:
                    parts.append(json.dumps(item.model_dump(), ensure_ascii=False))
                except Exception:
                    parts.append(str(item))
        if parts:
            return "\n".join(parts)

    try:
        return json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    except Exception:
        return str(result)


async def run_teacher_chat_tool_loop_async(
    *,
    llama_base_url: str,
    llama_model_name: str,
    selected_class: dict,
    overview: dict,
    concept_gaps: list[dict],
    roster: list[dict],
    chapters: list[dict],
    history: list[dict],
    max_agent_iterations: int,
    subject_name: str,
    retrieval_context: str = "",
    daily_brief: dict | None = None,
    base_messages: list[dict] | None = None,
    prior_tool_trace: list[str] | None = None,
) -> dict:
    client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
    messages = list(base_messages) if base_messages else build_teacher_chat_messages(
        class_row=selected_class,
        overview=overview,
        concept_gaps=concept_gaps,
        roster=roster,
        chapters=chapters,
        history=history,
        subject_name=subject_name,
        retrieval_context=retrieval_context,
        daily_brief=daily_brief,
    )
    mcp_config_path = str(ROOT / "mcp.json")
    await start_mcp_servers(mcp_config_path)
    tool_schemas = await get_all_mcp_tools()
    tool_trace: list[str] = list(prior_tool_trace or [])
    tool_calls_used = 0

    try:
        while tool_calls_used < max_agent_iterations:
            response = client.chat_completion(
                messages=messages,
                temperature=MODEL_SAMPLING.temperature,
                top_p=MODEL_SAMPLING.top_p,
                top_k=MODEL_SAMPLING.top_k,
                extra_payload={
                    "model": llama_model_name,
                    "tools": tool_schemas,
                    "tool_choice": "auto",
                },
            )
            append_conversation_log("MODEL_TOOL_LOOP_RAW", json.dumps(response, indent=2, ensure_ascii=False))
            assistant_message = extract_llama_message(response)
            assistant_content = str(assistant_message.get("content", "") or "").strip()
            tool_calls = extract_tool_calls(response)
            if not tool_calls:
                return {
                    "status": "ready_for_final_answer",
                    "assistant_text": assistant_content or "",
                    "tool_trace": tool_trace,
                    "messages": messages,
                    "tool_calls_used": tool_calls_used,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                if tool_calls_used >= max_agent_iterations:
                    return {
                        "status": "limit_reached",
                        "assistant_text": assistant_content or "",
                        "tool_trace": tool_trace,
                        "messages": messages,
                        "tool_calls_used": tool_calls_used,
                    }
                function_payload = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
                tool_name = function_payload.get("name")
                raw_arguments = function_payload.get("arguments", "{}")
                if not tool_name:
                    continue
                if isinstance(raw_arguments, str):
                    try:
                        tool_arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
                    except json.JSONDecodeError:
                        tool_arguments = {}
                elif isinstance(raw_arguments, dict):
                    tool_arguments = raw_arguments
                else:
                    tool_arguments = {}

                tool_result = await execute_mcp_tool(tool_name, tool_arguments)
                tool_text = stringify_mcp_tool_result(tool_result)
                append_conversation_log("TOOL_RESULT", f"{tool_name}\n{tool_text}")
                tool_trace.append(f"{tool_name}: {tool_text}")
                tool_calls_used += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", tool_name),
                        "name": tool_name,
                        "content": tool_text,
                    }
                )

        return {
            "status": "limit_reached",
            "assistant_text": "",
            "tool_trace": tool_trace,
            "messages": messages,
            "tool_calls_used": tool_calls_used,
        }
    finally:
        await cleanup_mcp_bridge()


def run_teacher_chat_tool_loop(
    *,
    llama_base_url: str,
    llama_model_name: str,
    selected_class: dict,
    overview: dict,
    concept_gaps: list[dict],
    roster: list[dict],
    chapters: list[dict],
    history: list[dict],
    max_agent_iterations: int,
    subject_name: str,
    retrieval_context: str = "",
    daily_brief: dict | None = None,
    base_messages: list[dict] | None = None,
    prior_tool_trace: list[str] | None = None,
) -> dict:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(
        run_teacher_chat_tool_loop_async(
            llama_base_url=llama_base_url,
            llama_model_name=llama_model_name,
            selected_class=selected_class,
            overview=overview,
            concept_gaps=concept_gaps,
            roster=roster,
            chapters=chapters,
            history=history,
            max_agent_iterations=max_agent_iterations,
            subject_name=subject_name,
            retrieval_context=retrieval_context,
            daily_brief=daily_brief,
            base_messages=base_messages,
            prior_tool_trace=prior_tool_trace,
        )
    )


def build_final_teacher_answer_messages(messages: list[dict]) -> list[dict]:
    return [
        *messages,
        {
            "role": "system",
            "content": (
                "Provide the final answer to the teacher now. "
                "Use any tool results already available in the conversation. "
                "Do not call tools. Be concise, practical, and classroom-focused."
            ),
        },
    ]


def stream_teacher_final_answer(*, llama_base_url: str, llama_model_name: str, messages: list[dict]):
    client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
    return client.stream_chat_completion(
        messages=build_final_teacher_answer_messages(messages),
        temperature=MODEL_SAMPLING.temperature,
        top_p=MODEL_SAMPLING.top_p,
        top_k=MODEL_SAMPLING.top_k,
        extra_payload={"model": llama_model_name},
    )


def render_streamed_teacher_final_answer(*, llama_base_url: str, llama_model_name: str, messages: list[dict]) -> tuple[str, str]:
    answer_text = ""
    reasoning_text = ""
    reasoning_placeholder = None
    if MODEL_SAMPLING.show_reasoning:
        with st.expander("Reasoning", expanded=False):
            reasoning_placeholder = st.empty()
    answer_placeholder = st.empty()

    for event in stream_teacher_final_answer(
        llama_base_url=llama_base_url,
        llama_model_name=llama_model_name,
        messages=messages,
    ):
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        event_text = str(event.get("text", ""))
        if not event_text:
            continue
        if event_type == "reasoning":
            reasoning_text += event_text
            if reasoning_placeholder is not None:
                reasoning_placeholder.code(reasoning_text)
        elif event_type == "content":
            answer_text += event_text
            answer_placeholder.markdown(answer_text)

    if not answer_text:
        answer_placeholder.markdown("Gemma returned an empty response.")
        answer_text = "Gemma returned an empty response."
    if reasoning_text:
        append_conversation_log("MODEL_REASONING", reasoning_text)
    append_conversation_log("MODEL_FINAL_OUTPUT", answer_text)
    return answer_text, reasoning_text


teacher = teacher_class_repository.get_teacher()

st.markdown(
    """
    <div style="margin-bottom:0.25rem;">
        <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:2rem; font-weight:700;
                    color:#4F46E5; letter-spacing:-0.02em; line-height:1.2;">
            🎓 Pathshala Play
        </div>
        <div style="color:#6B7280; font-size:0.88rem; margin-top:0.2rem;">
            A calmer teaching workspace for live class decisions, student support, and lesson follow-through.
        </div>
    </div>
    <hr style="border:none; border-top:1.5px solid #EEF2FF; margin:0.6rem 0 1rem 0;">
    """,
    unsafe_allow_html=True,
)

if not teacher:
    st.error("No teacher record found in the database.")
    st.stop()

with st.sidebar:
    st.markdown(
        f"""
        <div style="margin-bottom:1rem;">
            <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.35rem;">
                <div style="width:38px;height:38px;border-radius:50%;
                            background:linear-gradient(135deg,#818CF8,#C4B5FD);
                            display:flex;align-items:center;justify-content:center;
                            font-size:1rem;font-weight:700;color:white;flex-shrink:0;">
                    {teacher['full_name'][0].upper()}
                </div>
                <div>
                    <div style="color:white;font-weight:600;font-size:0.95rem;line-height:1.2;">
                        {teacher['full_name']}
                    </div>
                    <div style="color:rgba(199,210,254,0.75);font-size:0.72rem;">
                        {teacher['school_name']}
                    </div>
                </div>
            </div>
            <div style="color:rgba(199,210,254,0.6);font-size:0.72rem;padding-left:2px;">
                {teacher['google_account_email']}
            </div>
        </div>
        <hr style="border:none;border-top:1px solid rgba(199,210,254,0.2);margin:0 0 0.75rem 0;">
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Model Settings", expanded=False):
        llama_base_url = st.text_input("llama-server URL", value=MODEL_SAMPLING.llama_base_url)
        llama_model_name = st.text_input("Model Name", value=MODEL_SAMPLING.llama_model_name)
        use_llama_server = st.toggle("Use llama-server", value=True)
        auto_generation_strategy, auto_generation_reason = choose_quiz_generation_mode(
            llama_model_name,
            MODEL_SAMPLING.quiz_question_generation_mode,
        )
        st.caption(
            f"Sampling: temp={MODEL_SAMPLING.temperature}, "
            f"top_p={MODEL_SAMPLING.top_p}, top_k={MODEL_SAMPLING.top_k}"
        )
        st.caption(f"Auto grading poll interval: {MODEL_SAMPLING.auto_grade_poll_interval_seconds}s")
        st.caption(
            "Quiz generation mode: "
            f"{'One by one' if auto_generation_strategy == 'one_by_one' else 'One shot'}"
        )
        st.caption(auto_generation_reason)

    classes = teacher_class_repository.list_teacher_classes(teacher["id"])
    if not classes:
        st.warning("No classes found for this teacher.")
        st.stop()

    selected_class = st.selectbox(
        "Select Class",
        options=classes,
        format_func=format_class_label,
    )

class_id = selected_class["id"]
overview = teacher_class_repository.get_class_overview(class_id)
students = student_repository.list_class_students(class_id)
attendance_stats = attendance_repository.get_class_attendance_stats(class_id)
assessments = assessment_repository.list_class_assessments(class_id)
concept_gaps = analytics_repository.list_class_concept_gaps(class_id)
latest_sync_result = st.session_state.get("latest_sync_result")
latest_attendance_result = st.session_state.get(f"attendance_result_{class_id}")
class_subject_options = teacher_class_repository.list_class_subjects(class_id)
current_subject_row = next(
    (item for item in class_subject_options if item["subject"] == selected_class.get("subject")),
    class_subject_options[0] if class_subject_options else {"subject": selected_class.get("subject", ""), "medium": selected_class.get("medium", "")},
)
selected_subject_name = current_subject_row.get("subject", selected_class.get("subject", ""))
if class_subject_options:
    selected_subject_name = st.selectbox(
        "Subject Workspace",
        options=[item["subject"] for item in class_subject_options],
        index=max(0, [item["subject"] for item in class_subject_options].index(current_subject_row["subject"]))
        if current_subject_row.get("subject") in [item["subject"] for item in class_subject_options]
        else 0,
        key=f"class_subject_workspace_{class_id}",
    )
    current_subject_row = next(
        (item for item in class_subject_options if item["subject"] == selected_subject_name),
        current_subject_row,
    )
current_subject_medium = current_subject_row.get("medium") or selected_class.get("medium", "")
plan_snapshot = DEFAULT_TEACHING_PLANNER_SERVICE.get_plan_snapshot(
    class_id=class_id,
    subject=selected_subject_name,
    academic_year=selected_class.get("academic_year", ""),
)
timetable_slots = timetable_repository.list_class_timetable_slots(class_id, selected_subject_name)
all_timetable_slots = timetable_repository.list_class_timetable_slots(class_id)
active_timetable_slot = DEFAULT_TEACHING_PLANNER_SERVICE.find_active_timetable_slot(
    class_id=class_id,
    subject=selected_subject_name,
)
coverage_sessions = coverage_repository.list_class_coverage_sessions(
    class_id=class_id,
    subject=selected_subject_name,
    limit=10,
)
capture_support_status = DEFAULT_TEACHING_PLANNER_SERVICE.get_capture_support_status()
maybe_auto_stop_local_recording(class_id=class_id, subject_name=selected_subject_name)
dashboard_date = datetime.now().date().isoformat()
dashboard_attendance_rows = attendance_repository.list_attendance_for_date(class_id, dashboard_date)
if (
    latest_attendance_result
    and latest_attendance_result.get("attendance_date") == dashboard_date
    and latest_attendance_result.get("records")
):
    dashboard_attendance_rows = latest_attendance_result["records"]
render_workspace_banner(
    class_label=f"Grade {overview.get('grade', '')}-{overview.get('section', '')}",
    subject_name=selected_subject_name,
    academic_year=overview.get("academic_year", "") or selected_class.get("academic_year", ""),
    medium=current_subject_medium,
    present_count=sum(1 for item in dashboard_attendance_rows if item.get("status") == "present"),
    plan_completion=plan_snapshot.get("completion_percent", 0),
    has_active_slot=bool(active_timetable_slot),
)
render_global_mic_controls(
    class_id=class_id,
    subject_name=selected_subject_name,
    active_timetable_slot=active_timetable_slot,
    capture_support_status=capture_support_status,
)
daily_dashboard_state_key = f"daily_dashboard_brief_{class_id}_{selected_subject_name}_{dashboard_date}"
if daily_dashboard_state_key not in st.session_state:
    cached_daily_brief = DEFAULT_DAILY_LOOP_SERVICE.load_cached_daily_dashboard_brief(
        class_id=class_id,
        subject_name=selected_subject_name,
    )
    if cached_daily_brief:
        st.session_state[daily_dashboard_state_key] = cached_daily_brief
    else:
        initial_daily_brief = DEFAULT_DAILY_LOOP_SERVICE.build_local_daily_dashboard_brief(
            selected_class=selected_class,
            overview=overview,
            subject_name=selected_subject_name,
            students=students,
            attendance_rows=dashboard_attendance_rows,
            concept_gaps=concept_gaps,
            assessments=assessments,
            coverage_sessions=coverage_sessions,
            plan_snapshot=plan_snapshot,
            for_date=dashboard_date,
        )
        DEFAULT_DAILY_LOOP_SERVICE.save_cached_daily_dashboard_brief(
            class_id=class_id,
            subject_name=selected_subject_name,
            brief_payload=initial_daily_brief,
        )
        st.session_state[daily_dashboard_state_key] = initial_daily_brief
daily_dashboard_brief = st.session_state.get(daily_dashboard_state_key, {})
brief_payload = daily_dashboard_brief.get("brief", {})

dashboard_tab, teaching_tab, students_tab, assessments_tab, settings_tab = st.tabs(
    [
        "📊 Dashboard",
        "📖 Teaching",
        "👥 Students",
        "📝 Assessments",
        "⚙️ Settings & Data",
    ]
)

with dashboard_tab:
    tab_dashboard_summary, tab_chat = st.tabs(["Class Summary", "🤖 Chat With Gemma"])
with teaching_tab:
    tab_planner, tab_materials = st.tabs(["Teaching Progress", "📚 Reading Materials"])
with students_tab:
    tab_overview, tab_students = st.tabs(["Class Overview", "🎓 Student Learning"])
with assessments_tab:
    tab_quiz, tab_class_gaps, tab_assessments = st.tabs(["Create Quiz", "💡 Misconceptions", "Assessment Review"])
with settings_tab:
    tab_attendance, tab_management = st.tabs(["Attendance", "Class Management"])

with tab_dashboard_summary:
    render_section_intro(
        "Class Health",
        f"Focused daily view for Grade {overview.get('grade', '')}-{overview.get('section', '')} and the active subject workspace.",
    )
    with st.container(border=True):
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Students", overview.get("student_count", 0))
        metric_2.metric("Assessments", overview.get("assessment_count", 0))
        metric_3.metric("Average Score", f"{overview.get('avg_percentage') or 0}%")
        metric_4.metric("Concepts To Reteach", sum(1 for row in concept_gaps if row["mastery_percent"] < 60))

    with st.container(border=True):
        brief_col1, brief_col2 = st.columns([2.2, 1.3])
        with brief_col1:
            st.markdown(f"**{GEMMA_ANALYSIS_LABEL}**")
            brief_context_date = (
                (daily_dashboard_brief.get("context") or {}).get("date")
                or dashboard_date
            )
            st.caption(
                f"Summary for {brief_context_date}. Generated by {daily_dashboard_brief.get('generated_by_model', 'fallback')}."
            )
            cache_metadata = daily_dashboard_brief.get("cache_metadata") or {}
            if cache_metadata.get("saved_at"):
                st.caption(f"Loaded from saved cache. Last saved on {cache_metadata['saved_at']}.")
            st.write(brief_payload.get("daily_summary") or f"{GEMMA_ANALYSIS_LABEL} has not generated a summary yet.")
            next_concept = brief_payload.get("next_concept_to_teach") or "No next concept recommendation yet."
            st.write(f"Next concept to teach: {next_concept}")
        with brief_col2:
            if st.button(f"Refresh {GEMMA_ANALYSIS_LABEL}", key=f"refresh_daily_loop_{class_id}_{selected_subject_name}", use_container_width=True):
                refreshed_daily_brief = DEFAULT_DAILY_LOOP_SERVICE.generate_daily_dashboard_brief(
                    selected_class=selected_class,
                    overview=overview,
                    subject_name=selected_subject_name,
                    students=students,
                    attendance_rows=dashboard_attendance_rows,
                    concept_gaps=concept_gaps,
                    assessments=assessments,
                    coverage_sessions=coverage_sessions,
                    plan_snapshot=plan_snapshot,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    use_llama_server=use_llama_server,
                    for_date=dashboard_date,
                )
                DEFAULT_DAILY_LOOP_SERVICE.save_cached_daily_dashboard_brief(
                    class_id=class_id,
                    subject_name=selected_subject_name,
                    brief_payload=refreshed_daily_brief,
                )
                st.session_state[daily_dashboard_state_key] = refreshed_daily_brief
                st.rerun()
            st.metric("Plan Completion", f"{plan_snapshot.get('completion_percent', 0)}%")
            st.metric("Present Today", sum(1 for item in dashboard_attendance_rows if item.get("status") == "present"))

    with st.container(border=True):
        daily_loop_detail_col1, daily_loop_detail_col2, daily_loop_detail_col3, daily_loop_detail_col4 = st.columns(4)
        with daily_loop_detail_col1:
            st.markdown("**Reteach Priorities**")
            reteach_items = brief_payload.get("reteach_concepts") or []
            if reteach_items:
                for item in reteach_items[:4]:
                    st.write(f"- {item}")
            else:
                st.write("- No reteach concepts flagged.")
        with daily_loop_detail_col2:
            st.markdown("**Speed Up**")
            speed_up_items = brief_payload.get("speed_up_areas") or []
            if speed_up_items:
                for item in speed_up_items[:4]:
                    st.write(f"- {item}")
            else:
                st.write("- No speed-up recommendation yet.")
        with daily_loop_detail_col3:
            st.markdown("**Watch List**")
            watch_items = brief_payload.get("students_to_watch") or []
            if watch_items:
                for item in watch_items[:4]:
                    st.write(f"- {item}")
            else:
                st.write("- No student watch list yet.")
        with daily_loop_detail_col4:
            st.markdown("**Teacher Actions**")
            teacher_actions = brief_payload.get("teacher_actions") or []
            if teacher_actions:
                for item in teacher_actions[:4]:
                    st.write(f"- {item}")
            else:
                st.write("- No teacher actions yet.")

    if latest_sync_result and latest_sync_result.get("class_id") == class_id:
        st.success(
            "Google Form sync completed. "
            f"Seen: {latest_sync_result['responses_seen']}, "
            f"matched students: {latest_sync_result['students_synced']}, "
            f"skipped: {latest_sync_result['responses_skipped']}, "
            f"Gemma-graded answers: {latest_sync_result.get('gemma_graded_answers', 0)}."
        )
        insight_col1, insight_col2 = st.columns(2)
        with insight_col1:
            st.markdown("**Post-Sync Class Insights**")
            if concept_gaps:
                for concept in concept_gaps[:3]:
                    st.write(
                        f"- {concept['concept_name']}: {concept['mastery_percent']}% mastery, "
                        f"{concept['students_lagging']} students lagging"
                    )
            else:
                st.write("- No concept insights available yet.")
        with insight_col2:
            st.markdown("**Students Needing Support**")
            lagging_students = sorted(
                students,
                key=lambda item: (
                    -(item["lagging_concepts"] or 0),
                    item["avg_percentage"] if item["avg_percentage"] is not None else 100,
                ),
            )
            for student in lagging_students[:3]:
                st.write(
                    f"- {student['full_name']}: "
                    f"{student['lagging_concepts'] or 0} lagging concepts, "
                    f"avg {student['avg_percentage'] or 0}%"
                )

with tab_overview:
    overview_attendance_date_key = f"overview_attendance_date_{class_id}"
    overview_attendance_date_pending_key = f"overview_attendance_date_pending_{class_id}"
    if overview_attendance_date_pending_key in st.session_state:
        st.session_state[overview_attendance_date_key] = st.session_state.pop(overview_attendance_date_pending_key)
    overview_attendance_date_value = st.date_input(
        "Overview Attendance Date",
        key=overview_attendance_date_key,
    )
    overview_attendance_date = format_attendance_date(overview_attendance_date_value)
    overview_attendance_rows = attendance_repository.list_attendance_for_date(class_id, overview_attendance_date)
    if (
        latest_attendance_result
        and latest_attendance_result.get("attendance_date") == overview_attendance_date
        and latest_attendance_result.get("records")
    ):
        overview_attendance_rows = latest_attendance_result["records"]
    attendance_by_student_id = {
        row["student_id"]: row
        for row in overview_attendance_rows
    }
    selected_overview_student_id = st.session_state.get("overview_selected_student_id")
    if selected_overview_student_id:
        student_detail = student_repository.get_student_detail(selected_overview_student_id)
        student = student_detail.get("student")
        if student:
            top_col1, top_col2 = st.columns([1, 4])
            with top_col1:
                if st.button("Back To Class Overview", key="back_to_class_overview"):
                    st.session_state.pop("overview_selected_student_id", None)
                    st.session_state.pop("overview_personalized_quiz", None)
                    st.rerun()
            with top_col2:
                st.markdown(f"**{student['full_name']}**")
                st.caption(
                    f"Roll {student['roll_number']} | Grade {student['grade']}-{student['section']} | "
                    f"{student['subject']}"
                )
                attendance_row = attendance_by_student_id.get(student["id"])
                if attendance_row:
                    st.caption(
                        f"Attendance on {overview_attendance_date}: {attendance_row['status'].title()} "
                        f"via {attendance_row['source']}"
                    )
                else:
                    st.caption(f"Attendance on {overview_attendance_date}: Not marked")
                student_attendance_summary = attendance_stats.get(student["id"], {})
                attendance_percentage = student_attendance_summary.get("attendance_percentage")
                if attendance_percentage is not None:
                    st.caption(
                        f"Overall attendance: {attendance_percentage}% "
                        f"since {student_attendance_summary.get('attendance_started_on')}"
                    )
                else:
                    st.caption("Overall attendance: Not available yet")

            profile_col1, profile_col2 = st.columns([1.1, 1])
            with profile_col1:
                st.write(f"Preferred language: {student['preferred_language'] or 'Not set'}")
                st.write(f"Accessibility notes: {student['accessibility_notes'] or 'None'}")
                student_attendance_summary = student_detail.get("attendance_summary", {})
                attendance_percentage = student_attendance_summary.get("attendance_percentage")
                st.write(
                    "Attendance summary: "
                    + (
                        f"{attendance_percentage}% since {student_attendance_summary.get('attendance_started_on')}"
                        if attendance_percentage is not None
                        else "Not available yet"
                    )
                )
                st.markdown("**Concept Mastery**")
                st.dataframe(student_detail["mastery"], use_container_width=True, hide_index=True)
                st.markdown("**Assessment History**")
                st.dataframe(student_detail["assessments"], use_container_width=True, hide_index=True)
                if student_detail.get("attendance_history"):
                    st.markdown("**Recent Attendance**")
                    st.dataframe(student_detail["attendance_history"], use_container_width=True, hide_index=True)
                adaptation_profile = get_subject_adaptation_profile(student_detail, student["subject"])
                if adaptation_profile:
                    profile_payload = adaptation_profile.get("profile", {})
                    st.markdown("**Adaptation Profile**")
                    st.write("Priority Targets")
                    for item in profile_payload.get("priority_targets", []):
                        st.write(f"- {item}")
                    st.write("Recommended Interventions")
                    for item in profile_payload.get("recommended_interventions", []):
                        st.write(f"- {item}")
                    st.write("Support Preferences")
                    support_preferences = profile_payload.get("support_preferences", {})
                    st.write(f"Language: {support_preferences.get('preferred_language') or 'Not set'}")
                    st.write(f"Pace: {support_preferences.get('pace_support') or 'Not set'}")
                    if adaptation_profile.get("summary"):
                        st.text_area(
                            "Adaptation Summary",
                            value=adaptation_profile["summary"],
                            height=140,
                            key=f"overview_adaptation_summary_{student['id']}",
                        )

            with profile_col2:
                available_subjects = get_available_student_subjects(student_detail)
                selected_profile_subject = st.selectbox(
                    "Subject",
                    options=available_subjects,
                    index=available_subjects.index(student["subject"]) if student["subject"] in available_subjects else 0,
                    key=f"overview_subject_{student['id']}",
                ) if available_subjects else student["subject"]
                blueprint = get_subject_blueprint(student_detail, selected_profile_subject)
                adaptation_profile = get_subject_adaptation_profile(student_detail, selected_profile_subject)
                if adaptation_profile:
                    profile_payload = adaptation_profile.get("profile", {})
                    st.markdown("**Student Adaptation Profile**")
                    st.caption(
                        f"Based on {adaptation_profile.get('based_on_assessments', 0)} assessments | "
                        f"Updated {adaptation_profile.get('updated_at', '')}"
                    )
                    st.write("Priority Targets")
                    for item in profile_payload.get("priority_targets", []):
                        st.write(f"- {item}")
                    st.write("Misconceptions")
                    for item in profile_payload.get("misconception_map", []):
                        st.write(f"- {item.get('concept', 'Concept')}: {item.get('issue', '')}")
                    if st.button("Regenerate Adaptation Profile", key=f"regenerate_profile_{student['id']}"):
                        try:
                            client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
                            context = student_repository.get_student_adaptation_profile_context(student["id"], selected_profile_subject or student["subject"])
                            generated_profile = generate_student_adaptation_profile_with_gemma(
                                client=client,
                                model_name=llama_model_name,
                                temperature=MODEL_SAMPLING.temperature,
                                top_p=MODEL_SAMPLING.top_p,
                                top_k=MODEL_SAMPLING.top_k,
                                student_context=context,
                            )
                            final_profile = {
                                "mastery_map": context.get("mastery_map", []),
                                "attendance_signal": context.get("attendance_signal", {}),
                                "intervention_history": context.get("intervention_history", []),
                                **generated_profile,
                            }
                            student_repository.upsert_student_adaptation_profile(
                                student_id=student["id"],
                                class_id=class_id,
                                subject=selected_profile_subject or student["subject"],
                                profile=final_profile,
                                summary=generated_profile.get("summary", ""),
                                generated_by_model=llama_model_name,
                                based_on_assessments=len(context.get("assessment_history", [])),
                                last_submission_at=context.get("attendance_signal", {}).get("attendance_last_marked_on"),
                            )
                            st.success("Adaptation profile regenerated.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed to regenerate adaptation profile: {exc}")
                st.markdown("**Strengths And Weaknesses**")
                if blueprint:
                    st.caption(f"Subject: {blueprint.get('subject', selected_profile_subject or student['subject'])}")
                    strengths = blueprint.get("strengths", [])
                    weaknesses = blueprint.get("weaknesses", [])
                    opportunities = blueprint.get("opportunities", [])
                    threats = blueprint.get("threats", [])
                    recommendations = blueprint.get("recommendations", [])
                    st.write("Strengths")
                    for item in strengths:
                        st.write(f"- {item}")
                    st.write("Weaknesses")
                    for item in weaknesses:
                        st.write(f"- {item}")
                    st.write("Opportunities")
                    for item in opportunities:
                        st.write(f"- {item}")
                    st.write("Threats")
                    for item in threats:
                        st.write(f"- {item}")
                    st.write("Recommendations")
                    for item in recommendations:
                        st.write(f"- {item}")
                    if blueprint.get("narrative"):
                        st.text_area(
                            "Student Narrative",
                            value=blueprint["narrative"],
                            height=180,
                            key=f"overview_blueprint_{student['id']}",
                        )
                else:
                    st.info("No blueprint available yet for this student.")

                student_quiz_language = st.text_input(
                    "Preferred Quiz Language",
                    value="English",
                    key=f"student_quiz_language_{student['id']}",
                )
                student_teacher_instructions = st.text_area(
                    "Teacher Instructions For This Quiz",
                    value="",
                    key=f"student_teacher_instructions_{student['id']}",
                    height=100,
                    help="Add any extra instructions for Gemma for this student's quiz.",
                )
                student_due_date = st.date_input(
                    "Student Quiz Due Date",
                    key=f"student_quiz_due_date_{student['id']}",
                )
                student_due_time = st.time_input(
                    "Student Quiz Due Time",
                    key=f"student_quiz_due_time_{student['id']}",
                )
                student_due_at = combine_due_datetime(student_due_date, student_due_time)
                st.caption(f"Selected due time: {format_date_time_preview(student_due_date, student_due_time)}")

                if st.button(
                    "Generate Quiz For This Student",
                    key=f"student_quiz_{student['id']}",
                    type="primary",
                ):
                    topic = build_student_quiz_topic(student_detail, selected_profile_subject)
                    student_adaptation_context = student_repository.get_student_adaptation_profile_context(
                        student["id"],
                        selected_profile_subject or student["subject"],
                    )
                    personalized_questions = None
                    personalized_note = "Generated using local mock logic."
                    personalized_retrieval_query = build_student_retrieval_query(
                        subject=selected_profile_subject or student["subject"],
                        topic_hint=topic,
                        adaptation_profile=adaptation_profile,
                        student_context=student_adaptation_context,
                    )
                    personalized_rag_context = build_retrieval_context(
                        grade=student["grade"],
                        subject=selected_profile_subject or student["subject"],
                        query=personalized_retrieval_query,
                        top_k=MODEL_SAMPLING.rag_top_k,
                    )
                    learner_profile, personalized_source_material = build_student_generation_context(
                        subject=selected_profile_subject or student["subject"],
                        topic_hint=topic,
                        adaptation_profile=adaptation_profile,
                        student_context=student_adaptation_context,
                        retrieval_context=personalized_rag_context,
                    )
                    if use_llama_server:
                        client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
                        try:
                            personalized_questions, personalized_note, _ = generate_quiz_with_llama(
                                client=client,
                                base_url=llama_base_url,
                                model_name=llama_model_name,
                                generation_strategy=auto_generation_strategy,
                                subject=selected_profile_subject or student["subject"],
                                grade=student["grade"],
                                chapter_name=topic,
                                learner_profile=learner_profile,
                                source_material=personalized_source_material,
                                teacher_instructions=student_teacher_instructions,
                                language=student_quiz_language.strip() or "English",
                                question_count=5,
                            )
                        except (RequestException, ValueError, json.JSONDecodeError) as exc:
                            personalized_note = f"llama-server unavailable or invalid response: {exc}"

                    if not personalized_questions:
                        personalized_questions = build_quiz_questions(
                            topic,
                            [topic],
                            student_quiz_language.strip() or "English",
                        )[:5]

                    st.session_state["overview_personalized_quiz"] = {
                        "student_id": student["id"],
                        "student_name": student["full_name"],
                        "note": personalized_note,
                        "questions": personalized_questions,
                        "due_at": student_due_at,
                        "subject": selected_profile_subject or student["subject"],
                        "topic": topic,
                        "title": f"{student['full_name']} Personalized Quiz",
                        "language": student_quiz_language.strip() or "English",
                        "teacher_instructions": student_teacher_instructions,
                    }

            personalized_quiz = st.session_state.get("overview_personalized_quiz")
            if personalized_quiz and personalized_quiz.get("student_id") == student["id"]:
                st.markdown("**Personalized Quiz Preview**")
                st.caption(personalized_quiz["note"])
                for index, question in enumerate(personalized_quiz["questions"], start=1):
                    with st.expander(f"Quiz Question {index}", expanded=True):
                        st.write(question["question_text"])
                        if question["question_type"] == "mcq" and question.get("options"):
                            st.write(f"A. {question['options'].get('A', '')}")
                            st.write(f"B. {question['options'].get('B', '')}")
                            st.write(f"C. {question['options'].get('C', '')}")
                            st.write(f"D. {question['options'].get('D', '')}")
                        st.write(
                            f"Type: {question['question_type']} | "
                            f"Difficulty: {question['difficulty']} | "
                            f"Marks: {question['marks']}"
                        )
                        st.write(f"Correct answer / rubric: {question['correct_answer']}")
                if st.button(
                    "Create Google Form Draft For This Student",
                    key=f"student_google_form_{student['id']}",
                ):
                    try:
                        from app.google_forms import create_google_form_quiz

                        student_chapters = curriculum_repository.list_chapters_for_class(class_id)
                        chapter_id = student_chapters[0]["id"] if student_chapters else preview["chapter_id"] if "preview" in locals() else 1
                        assessment_id = assessment_repository.create_assessment(
                            class_id=class_id,
                            chapter_id=chapter_id,
                            teacher_id=teacher["id"],
                            title=personalized_quiz["title"],
                            language=personalized_quiz["language"],
                            assessment_type="remedial",
                            questions=personalized_quiz["questions"],
                            due_at=personalized_quiz.get("due_at"),
                        )
                        form_result = create_google_form_quiz(
                            title=personalized_quiz["title"],
                            description=f"Personalized quiz for {student['full_name']}",
                            questions=personalized_quiz["questions"],
                        )
                        assessment_repository.update_assessment_google_form_info(
                            assessment_id=assessment_id,
                            google_form_id=form_result["form_id"],
                            google_form_url=form_result["edit_uri"],
                            question_id_map=form_result["question_id_map"],
                        )
                        st.session_state["overview_personalized_form_result"] = form_result
                        st.session_state["overview_personalized_share_message"] = build_quiz_share_message(
                            topic=personalized_quiz["topic"],
                            subject=personalized_quiz["subject"],
                            due_at=personalized_quiz.get("due_at"),
                            quiz_link=form_result.get("responder_uri") or form_result["edit_uri"],
                        )
                        st.success("Personalized Google Form draft created.")
                    except Exception as exc:
                        st.error(f"Failed to create personalized Google Form draft: {exc}")

                personalized_form_result = st.session_state.get("overview_personalized_form_result")
                if personalized_form_result:
                    st.write(f"Form edit URL: {personalized_form_result['edit_uri']}")
                    if personalized_form_result.get("responder_uri"):
                        st.write(f"Responder URL: {personalized_form_result['responder_uri']}")
                    personalized_share_message = st.session_state.get("overview_personalized_share_message")
                    if personalized_share_message:
                        st.text_area(
                            "Student Quiz Share Message",
                            value=personalized_share_message,
                            height=140,
                            key=f"personalized_share_message_{student['id']}",
                        )
    else:
        left, right = st.columns([1.2, 1])

        with left:
            st.markdown("**Roster Snapshot**")
            st.caption("Click a student name to open the full profile.")
            st.caption(f"Attendance shown for {overview_attendance_date}.")
            for student in students:
                attendance_row = attendance_by_student_id.get(student["id"])
                attendance_label = attendance_row["status"].title() if attendance_row else "Not marked"
                attendance_percentage = student.get("attendance_percentage")
                attendance_percentage_label = f"{attendance_percentage}%" if attendance_percentage is not None else "N/A"
                row_col1, row_col2, row_col3, row_col4, row_col5, row_col6 = st.columns([2.2, 1, 1, 1, 1.2, 1])
                with row_col1:
                    if st.button(
                        f"{student['roll_number']} | {student['full_name']}",
                        key=f"overview_student_{student['id']}",
                        use_container_width=True,
                    ):
                        st.session_state["overview_selected_student_id"] = student["id"]
                        st.rerun()
                with row_col2:
                    st.write(student["preferred_language"] or "Not set")
                with row_col3:
                    st.write(f"{student['avg_percentage'] or 0}%")
                with row_col4:
                    st.write(f"{student['lagging_concepts'] or 0} gaps")
                with row_col5:
                    st.write(attendance_label)
                with row_col6:
                    st.write(attendance_percentage_label)

        with right:
            st.empty()

with tab_quiz:
    st.markdown("**Draft a quiz for the selected class**")

    quiz_mode = st.radio(
        "Quiz Target Selection",
        options=["Manual", "Gemma Auto Select"],
        horizontal=True,
        key=f"quiz_target_mode_{class_id}",
    )
    chapter_options = curriculum_repository.list_chapters_for_class(class_id, selected_subject_name)
    if quiz_mode == "Manual" and not chapter_options:
        st.warning("No chapters found for the selected subject and grade.")
    else:
        quiz_col1, quiz_col2 = st.columns(2)
        auto_quiz_target = None
        auto_subject_chapters: list[dict] = []
        auto_generation_reason = ""
        resolved_quiz_subject_name = selected_subject_name
        with quiz_col1:
            if quiz_mode == "Manual":
                selected_chapter = st.selectbox(
                    "Chapter",
                    options=chapter_options,
                    format_func=lambda chapter: chapter["chapter_name"],
                    key="quiz_chapter",
                )
            else:
                auto_quiz_target = choose_auto_quiz_target(
                    class_id=class_id,
                    class_row=selected_class,
                    class_subjects=class_subject_options,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    use_llama_server=use_llama_server,
                )
                resolved_quiz_subject_name = auto_quiz_target.get("subject") or selected_subject_name
                auto_subject_chapters = curriculum_repository.list_chapters_for_class(class_id, resolved_quiz_subject_name)
                selected_chapter = next(
                    (
                        chapter
                        for chapter in auto_subject_chapters
                        if str(chapter.get("chapter_name") or "").strip().lower() == auto_quiz_target.get("topic", "").strip().lower()
                    ),
                    auto_subject_chapters[0] if auto_subject_chapters else {"id": 0, "chapter_name": auto_quiz_target.get("topic", "Concept revision")},
                )
                auto_generation_reason = auto_quiz_target.get("reason", "")
                st.caption(
                    f"Gemma selected subject: {resolved_quiz_subject_name} | topic: {selected_chapter['chapter_name']}"
                )
                if auto_generation_reason:
                    st.caption(auto_generation_reason)
                if not auto_subject_chapters:
                    st.warning(
                        f"No chapter records exist yet for auto-selected subject '{resolved_quiz_subject_name}'. "
                        "Import the syllabus first or add chapters for that subject."
                    )
            quiz_title = st.text_input(
                "Quiz Title",
                value=f"{selected_chapter['chapter_name']} Quick Check",
            )
            quiz_language = st.text_input("Preferred Quiz Language", value="English")
        with quiz_col2:
            assessment_type = st.selectbox(
                "Assessment Type",
                ["practice", "class_test", "remedial", "homework"],
            )
            question_count = st.number_input(
                "Number of Questions",
                min_value=1,
                max_value=15,
                value=10 if auto_generation_strategy == "one_by_one" else 5,
                step=1,
            )
            learner_profile = st.text_area(
                "Learner Notes",
                value="Mixed pace learners, some students need bilingual cues and visual reinforcement.",
                height=100,
            )
            source_material = st.text_area(
                "Lesson Notes",
                value=f"Create a short quiz for {selected_chapter['chapter_name']}.",
                height=100,
            )
            teacher_instructions = st.text_area(
                "Teacher Instructions",
                value="",
                height=100,
                help="Add any additional instructions for Gemma, such as question style, coverage, or constraints.",
            )
            due_date = st.date_input("Quiz Due Date", key="general_quiz_due_date")
            due_time = st.time_input("Quiz Due Time", key="general_quiz_due_time")
            general_due_at = combine_due_datetime(due_date, due_time)
            st.caption(f"Selected due time: {format_date_time_preview(due_date, due_time)}")

        if st.button("Generate Quiz Draft", type="primary"):
            generation_subject_name = resolved_quiz_subject_name if quiz_mode != "Manual" else selected_subject_name
            if int(selected_chapter.get("id", 0) or 0) == 0:
                st.error("No valid chapter mapping exists for the selected quiz target yet.")
                st.stop()
            concept_names = [row["concept_name"] for row in concept_gaps] or [selected_chapter["chapter_name"]]
            lesson_pack = build_lesson_pack(
                request=type(
                    "LessonRequestProxy",
                    (),
                    {
                        "subject": generation_subject_name,
                        "grade_band": f"{selected_class['grade']}",
                        "topic": selected_chapter["chapter_name"],
                        "class_profile": learner_profile,
                        "source_material": source_material,
                        "accessibility_need": "bilingual and visual support",
                        "language": quiz_language,
                    },
                )()
            )
            questions = None
            generation_mode = "mock"
            generation_note = "Generated using local mock logic."
            raw_llama_outputs: list[str] = []

            rag_context = build_retrieval_context(
                grade=selected_class["grade"],
                subject=generation_subject_name,
                query=f"{selected_chapter['chapter_name']} {quiz_title} {learner_profile} {source_material}",
                top_k=MODEL_SAMPLING.rag_top_k,
            )
            generation_source_material = source_material
            if rag_context:
                generation_source_material = (
                    f"{source_material}\n\nRetrieved subject material:\n{rag_context}"
                    if source_material.strip()
                    else f"Retrieved subject material:\n{rag_context}"
                )
            if use_llama_server:
                client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
                try:
                    questions, generation_note, raw_llama_outputs = generate_quiz_with_llama(
                        client=client,
                        base_url=llama_base_url,
                        model_name=llama_model_name,
                        generation_strategy=auto_generation_strategy,
                        subject=generation_subject_name,
                        grade=selected_class["grade"],
                        chapter_name=selected_chapter["chapter_name"],
                        learner_profile=learner_profile,
                        source_material=generation_source_material,
                        teacher_instructions=teacher_instructions,
                        language=quiz_language,
                        question_count=question_count,
                    )
                    if questions:
                        generation_mode = "llama-server"
                    else:
                        generation_mode = "llama-server-raw"
                except (RequestException, ValueError, json.JSONDecodeError) as exc:
                    generation_note = f"llama-server unavailable or invalid response: {exc}. Used mock fallback."

            if not questions:
                if generation_mode != "llama-server-raw":
                    questions = build_quiz_questions(
                        selected_chapter["chapter_name"], concept_names[:3], quiz_language
                    )
                    if question_count > len(questions):
                        questions.extend(
                            build_quiz_questions(
                                selected_chapter["chapter_name"], concept_names[:3], quiz_language
                            )[: question_count - len(questions)]
                        )
                    questions = questions[:question_count]

            st.session_state["quiz_preview"] = {
                "chapter_id": selected_chapter["id"],
                "title": quiz_title,
                "language": quiz_language,
                "assessment_type": assessment_type,
                "questions": questions,
                "teacher_summary": lesson_pack["teacher_summary"],
                "teacher_instructions": teacher_instructions,
                "generation_mode": generation_mode,
                "generation_note": generation_note,
                "generation_strategy": auto_generation_strategy,
                "raw_llama_outputs": raw_llama_outputs,
                "due_at": general_due_at,
                "topic": selected_chapter["chapter_name"],
                "subject": generation_subject_name,
                "retrieval_context": rag_context,
                "target_selection_mode": "auto" if quiz_mode != "Manual" else "manual",
                "target_selection_reason": auto_generation_reason,
            }

    preview = st.session_state.get("quiz_preview")
    if preview:
        st.markdown("**Quiz Draft Preview**")
        st.caption(preview["teacher_summary"])
        st.caption(preview["generation_note"])
        if preview["generation_mode"] == "llama-server-raw":
            st.warning("Gemma responded, but the app could not parse valid JSON. Raw model output is shown below.")
            raw_outputs = preview.get("raw_llama_outputs", [])
            for index, raw_output in enumerate(raw_outputs, start=1):
                label = (
                    f"Raw Output {index}"
                    if preview.get("generation_strategy") == "one_by_one"
                    else "Raw Model Output"
                )
                st.text_area(label, value=raw_output, height=220, key=f"raw_llama_output_{index}")
        else:
            edited_questions = []
            for index, question in enumerate(preview["questions"], start=1):
                with st.expander(f"Question {index}", expanded=True):
                    edited_question_text = st.text_area(
                        f"Question Text {index}",
                        value=question["question_text"],
                        key=f"question_text_{index}",
                        height=100,
                    )
                    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                    with meta_col1:
                        edited_question_type = st.selectbox(
                            f"Question Type {index}",
                            options=["mcq", "short_answer", "checkbox", "dropdown"],
                            index=["mcq", "short_answer", "checkbox", "dropdown"].index(question["question_type"])
                            if question["question_type"] in ["mcq", "short_answer", "checkbox", "dropdown"]
                            else 1,
                            key=f"question_type_{index}",
                        )
                    with meta_col2:
                        edited_difficulty = st.selectbox(
                            f"Difficulty {index}",
                            options=["easy", "medium", "hard"],
                            index=["easy", "medium", "hard"].index(question["difficulty"])
                            if question["difficulty"] in ["easy", "medium", "hard"]
                            else 1,
                            key=f"difficulty_{index}",
                        )
                    with meta_col3:
                        edited_bloom = st.selectbox(
                            f"Bloom Level {index}",
                            options=["remember", "understand", "apply", "analyze"],
                            index=["remember", "understand", "apply", "analyze"].index(question["bloom_level"])
                            if question["bloom_level"] in ["remember", "understand", "apply", "analyze"]
                            else 1,
                            key=f"bloom_{index}",
                        )
                    with meta_col4:
                        edited_marks = st.number_input(
                            f"Marks {index}",
                            min_value=1.0,
                            max_value=20.0,
                            value=float(question["marks"]),
                            step=1.0,
                            key=f"marks_{index}",
                        )
                    edited_options = {}
                    if edited_question_type == "mcq":
                        st.markdown("**Options**")
                        option_col1, option_col2 = st.columns(2)
                        current_options = question.get("options", {})
                        with option_col1:
                            edited_options["A"] = st.text_input(
                                f"Option A {index}",
                                value=current_options.get("A", ""),
                                key=f"option_a_{index}",
                            )
                            edited_options["B"] = st.text_input(
                                f"Option B {index}",
                                value=current_options.get("B", ""),
                                key=f"option_b_{index}",
                            )
                        with option_col2:
                            edited_options["C"] = st.text_input(
                                f"Option C {index}",
                                value=current_options.get("C", ""),
                                key=f"option_c_{index}",
                            )
                            edited_options["D"] = st.text_input(
                                f"Option D {index}",
                                value=current_options.get("D", ""),
                                key=f"option_d_{index}",
                            )
                    edited_correct_answer = st.text_area(
                        f"Correct Answer {index}",
                        value=question["correct_answer"],
                        key=f"correct_answer_{index}",
                        height=80,
                    )
                    edited_explanation = st.text_area(
                        f"Explanation {index}",
                        value=question["explanation"],
                        key=f"explanation_{index}",
                        height=80,
                    )
                    edited_questions.append(
                        {
                            "question_text": edited_question_text.strip(),
                            "question_type": edited_question_type,
                            "options": {key: value.strip() for key, value in edited_options.items()},
                            "difficulty": edited_difficulty,
                            "bloom_level": edited_bloom,
                            "marks": float(edited_marks),
                            "correct_answer": edited_correct_answer.strip(),
                            "explanation": edited_explanation.strip(),
                        }
                    )

            preview["questions"] = edited_questions

            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                if st.button("Save Quiz Draft To Database", use_container_width=True):
                    assessment_id = assessment_repository.create_assessment(
                        class_id=class_id,
                        chapter_id=preview["chapter_id"],
                        teacher_id=teacher["id"],
                        title=preview["title"],
                        language=preview["language"],
                        assessment_type=preview["assessment_type"],
                        questions=preview["questions"],
                        due_at=preview.get("due_at"),
                    )
                    preview["assessment_id"] = assessment_id
                    st.success(f"Saved quiz draft as assessment #{assessment_id}.")

            with action_col2:
                try:
                    pdf_bytes = build_quiz_pdf_bytes(
                        title=preview["title"],
                        subject=preview.get("subject", selected_subject_name),
                        chapter_name=preview.get("topic", preview["title"]),
                        language=preview["language"],
                        questions=preview["questions"],
                        due_at=preview.get("due_at"),
                    )
                    pdf_file_name = f"{preview['title'].strip().replace(' ', '_') or 'quiz'}_questions.pdf"
                    st.download_button(
                        "Generate PDF",
                        data=pdf_bytes,
                        file_name=pdf_file_name,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Failed to prepare quiz PDF: {exc}")

            with action_col3:
                if st.button("Create Google Form Draft", use_container_width=True):
                    try:
                        from app.google_forms import create_google_form_quiz

                        assessment_id = preview.get("assessment_id")
                        if not assessment_id:
                            assessment_id = assessment_repository.create_assessment(
                                class_id=class_id,
                                chapter_id=preview["chapter_id"],
                                teacher_id=teacher["id"],
                                title=preview["title"],
                                language=preview["language"],
                                assessment_type=preview["assessment_type"],
                                questions=preview["questions"],
                                due_at=preview.get("due_at"),
                            )
                            preview["assessment_id"] = assessment_id

                        form_result = create_google_form_quiz(
                            title=preview["title"],
                            description=preview["teacher_summary"],
                            questions=preview["questions"],
                        )
                        assessment_repository.update_assessment_google_form_info(
                            assessment_id=assessment_id,
                            google_form_id=form_result["form_id"],
                            google_form_url=form_result["edit_uri"],
                            question_id_map=form_result["question_id_map"],
                        )
                        st.session_state["google_form_result"] = form_result
                        st.session_state["google_form_share_message"] = build_quiz_share_message(
                            topic=preview.get("topic", preview["title"]),
                            subject=preview.get("subject", selected_subject_name),
                            due_at=preview.get("due_at"),
                            quiz_link=form_result.get("responder_uri") or form_result["edit_uri"],
                        )
                        st.success(f"Google Form draft created for assessment #{assessment_id}.")
                    except Exception as exc:
                        st.error(f"Failed to create Google Form draft: {exc}")

            google_form_result = st.session_state.get("google_form_result")
            if google_form_result:
                st.write(f"Form edit URL: {google_form_result['edit_uri']}")
                if google_form_result.get("responder_uri"):
                    st.write(f"Responder URL: {google_form_result['responder_uri']}")
                share_message = st.session_state.get("google_form_share_message")
                if share_message:
                    st.text_area(
                        "Share Message",
                        value=share_message,
                        height=200,
                        key="general_quiz_share_message",
                    )

with tab_students:
    if not students:
        st.info("No students found in this class.")
    else:
        selected_student = st.selectbox(
            "Select Student",
            options=students,
            format_func=lambda student: f"{student['roll_number']} | {student['full_name']}",
        )
        student_detail = student_repository.get_student_detail(selected_student["id"])
        student = student_detail["student"]

        if student:
            header_left, header_right = st.columns([1.2, 1])
            with header_left:
                st.markdown(f"**{student['full_name']}**")
                st.caption(
                    f"Roll {student['roll_number']} | Grade {student['grade']}-{student['section']} | "
                    f"{student['subject']}"
                )
            with header_right:
                st.write(f"Preferred language: {student['preferred_language'] or 'Not set'}")
                st.write(f"Accessibility notes: {student['accessibility_notes'] or 'None'}")

            mastery_col, history_col = st.columns(2)
            with mastery_col:
                st.markdown("**Concept Mastery**")
                st.dataframe(student_detail["mastery"], use_container_width=True, hide_index=True)

            with history_col:
                st.markdown("**Assessment History**")
                st.dataframe(student_detail["assessments"], use_container_width=True, hide_index=True)

            st.markdown("**Recommended Interventions**")
            if student_detail["recommendations"]:
                for item in student_detail["recommendations"]:
                    st.write(
                        f"- [{item['priority']}/5] {item['concept_name']} | "
                        f"{item['recommendation_type']}: {item['recommendation_text']}"
                    )
            else:
                st.info("No remediation recommendations for this student yet.")

            st.markdown("**Student Blueprint**")
            blueprints = student_detail.get("blueprints") or []
            adaptation_profiles = student_detail.get("adaptation_profiles") or []
            if adaptation_profiles:
                selected_adaptation_subject = st.selectbox(
                    "Adaptation Subject",
                    options=get_available_student_subjects(student_detail),
                    index=0,
                    key=f"student_adaptation_subject_{student['id']}",
                )
                adaptation_profile = get_subject_adaptation_profile(student_detail, selected_adaptation_subject)
                profile_payload = adaptation_profile.get("profile", {})
                st.markdown(f"**{selected_adaptation_subject} Adaptation Profile**")
                st.caption(
                    f"Based on {adaptation_profile.get('based_on_assessments', 0)} assessments | "
                    f"Updated {adaptation_profile.get('updated_at', '')}"
                )
                st.write("Priority Targets")
                for item in profile_payload.get("priority_targets", []):
                    st.write(f"- {item}")
                st.write("Response Style")
                response_style = profile_payload.get("response_style", {})
                for item in response_style.get("best_formats", []):
                    st.write(f"- Stronger in: {item}")
                for item in response_style.get("needs_more_support_in", []):
                    st.write(f"- Needs support in: {item}")
                if adaptation_profile.get("summary"):
                    st.text_area(
                        "Adaptation Summary",
                        value=adaptation_profile["summary"],
                        height=140,
                        key=f"student_adaptation_summary_{student['id']}",
                    )
            if blueprints:
                available_subjects = get_available_student_subjects(student_detail)
                selected_student_subject = st.selectbox(
                    "Subject",
                    options=available_subjects,
                    index=available_subjects.index(student["subject"]) if student["subject"] in available_subjects else 0,
                    key=f"student_learning_subject_{student['id']}",
                ) if available_subjects else student["subject"]
                blueprint = get_subject_blueprint(student_detail, selected_student_subject)
                st.markdown(f"**{(selected_student_subject or student.get('subject') or 'Subject')} SWOT**")
                st.caption(
                    f"Based on {blueprint['based_on_assessments']} assessments | "
                    f"Updated {blueprint['updated_at']}"
                )
                bp_col1, bp_col2 = st.columns(2)
                with bp_col1:
                    st.write("Strengths")
                    for item in blueprint.get("strengths", []):
                        st.write(f"- {item}")
                    st.write("Weaknesses")
                    for item in blueprint.get("weaknesses", []):
                        st.write(f"- {item}")
                    st.write("Opportunities")
                    for item in blueprint.get("opportunities", []):
                        st.write(f"- {item}")
                with bp_col2:
                    st.write("Threats")
                    for item in blueprint.get("threats", []):
                        st.write(f"- {item}")
                    st.write("Recommendations")
                    for item in blueprint.get("recommendations", []):
                        st.write(f"- {item}")
                if blueprint.get("narrative"):
                    st.text_area(
                        f"{blueprint.get('subject', 'Subject')} Blueprint Narrative",
                        value=blueprint["narrative"],
                        height=160,
                        key=f"blueprint_narrative_{student['id']}_{blueprint.get('subject', 'subject')}",
                    )
            else:
                st.info("No student blueprint has been generated yet. It will appear after graded responses are processed.")

with tab_class_gaps:
    st.markdown("**Concepts the class is struggling with**")
    if not concept_gaps:
        st.info("No concept mastery data found for this class.")
    else:
        st.dataframe(
            [
                {
                    "Concept": row["concept_name"],
                    "Mastery %": row["mastery_percent"],
                    "Students Assessed": row["students_assessed"],
                    "Students Lagging": row["students_lagging"],
                    "Why reteach": row["description"],
                }
                for row in concept_gaps
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Immediate reteach recommendation**")
        lowest = concept_gaps[0]
        st.write(
            f"Reteach `{lowest['concept_name']}` first. It has only "
            f"{lowest['mastery_percent']}% class mastery and {lowest['students_lagging']} students are behind."
        )

with tab_assessments:
    st.markdown("**Recent assessments for this class**")
    if not assessments:
        st.info("No assessments yet.")
    else:
        st.dataframe(
            [
                {
                    "Title": item["title"],
                    "Chapter": item["chapter_name"],
                    "Type": item["assessment_type"],
                    "Mode": item["delivery_mode"],
                    "Submissions": item["submissions"],
                    "Avg %": item["avg_percentage"] or 0,
                    "Language": item["language"],
                }
                for item in assessments
            ],
            use_container_width=True,
            hide_index=True,
        )

        latest = assessments[0]
        st.markdown("**Latest assessment**")
        st.write(f"Title: {latest['title']}")
        st.write(f"Chapter: {latest['chapter_name']}")
        st.write(f"Average class score: {latest['avg_percentage'] or 0}%")
        if latest["google_form_url"]:
            st.write(f"Google Form: {latest['google_form_url']}")

    st.markdown("**Sync Google Form Responses**")
    syncable_assessments = assessment_repository.list_assessments_for_sync(class_id)
    if not syncable_assessments:
        st.info("No Google Form linked assessments available for sync.")
    else:
        selected_sync_assessment = st.selectbox(
            "Assessment To Sync",
            options=syncable_assessments,
            format_func=lambda item: item["title"],
            key="sync_assessment_select",
        )
        if st.button("Fetch Responses And Refresh Insights"):
            try:
                sync_result = sync_google_form_assessment(
                    selected_sync_assessment["id"],
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                )
                st.session_state["latest_sync_result"] = {
                    **sync_result,
                    "class_id": class_id,
                    "assessment_id": selected_sync_assessment["id"],
                }
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to sync Google Form responses: {exc}")

        attempted_students = assessment_repository.list_attempted_students_for_assessment(selected_sync_assessment["id"])
        if attempted_students:
            st.markdown("**Review Attempted Students**")
            selected_attempted_student = st.selectbox(
                "Student Attempts",
                options=attempted_students,
                format_func=lambda item: (
                    f"{item['roll_number']} | {item['full_name']} | "
                    f"{item['score_obtained'] or 0} marks | {item['percentage'] or 0}%"
                ),
                key="attempted_student_select",
            )
            review = assessment_repository.get_student_assessment_review(
                selected_sync_assessment["id"],
                selected_attempted_student["student_id"],
            )
            if review["summary"]:
                st.caption(
                    f"{review['summary']['full_name']} | Roll {review['summary']['roll_number']} | "
                    f"Score {review['summary']['score_obtained'] or 0} | "
                    f"{review['summary']['percentage'] or 0}%"
                )
            for answer in review["answers"]:
                with st.expander(f"Question {answer['question_number']}", expanded=True):
                    st.write(answer["question_text"])
                    if answer["question_type"] == "mcq" and answer["options"]:
                        st.write("Options:")
                        st.write(f"A. {answer['options'].get('A', '')}")
                        st.write(f"B. {answer['options'].get('B', '')}")
                        st.write(f"C. {answer['options'].get('C', '')}")
                        st.write(f"D. {answer['options'].get('D', '')}")
                    st.write(f"Student answer: {answer['raw_answer'] or '(blank)'}")
                    st.write(f"Marks awarded: {answer['score_awarded']} / {answer['marks']}")
                    st.write(f"Correct answer / rubric: {answer['correct_answer']}")
                    st.write(f"Gemma reasoning: {answer['grading_reasoning'] or answer['feedback'] or 'No reasoning available.'}")
        else:
            st.info("No graded student attempts found for this assessment yet.")

    st.markdown("**Auto Grading Queue**")
    queue_items = queue_repository.list_queue_items(limit=10)
    if queue_items:
        st.dataframe(
            [
                {
                    "Assessment": item["title"],
                    "Response ID": item["response_id"],
                    "Email": item["respondent_email"],
                    "Status": item["status"],
                    "Submitted": item["submitted_at"],
                    "Error": item["error_message"] or "",
                }
                for item in queue_items
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("The response queue is empty.")
    st.caption("Run `python scripts/auto_grade_worker.py` to automatically watch Google Forms and grade new submissions.")

with tab_attendance:
    st.markdown("**Attendance**")
    attendance_result_key = f"attendance_result_{class_id}"
    attendance_proposal_key = f"attendance_proposal_{class_id}"
    overview_attendance_date_pending_key = f"overview_attendance_date_pending_{class_id}"
    attendance_date_value = st.date_input(
        "Attendance Date",
        value=datetime.now().date(),
        key=f"attendance_date_{class_id}",
    )
    attendance_date = format_attendance_date(attendance_date_value)
    roster = student_repository.list_class_roster(class_id)
    attendance_overview = attendance_repository.get_attendance_overview(class_id, attendance_date)
    attendance_rows = attendance_repository.list_attendance_for_date(class_id, attendance_date)
    if (
        latest_attendance_result
        and latest_attendance_result.get("attendance_date") == attendance_date
        and latest_attendance_result.get("records")
    ):
        attendance_rows = latest_attendance_result["records"]
        attendance_overview = {
            "attendance_date": attendance_date,
            "present_count": int(latest_attendance_result.get("present_count", 0)),
            "absent_count": int(latest_attendance_result.get("absent_count", 0)),
            "total_students": len(attendance_rows),
        }

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Total Students", len(roster))
    metric_col2.metric("Present", attendance_overview["present_count"])
    metric_col3.metric("Absent", attendance_overview["absent_count"])

    st.markdown("**Attendance Controls**")
    edit_col1, edit_col2 = st.columns(2)
    with edit_col1:
        selected_attendance_student = st.selectbox(
            "Edit Student Attendance",
            options=roster,
            format_func=lambda item: f"{item['roll_number']} | {item['full_name']}",
            key=f"attendance_edit_student_{class_id}",
        ) if roster else None
        selected_student_attendance_row = (
            next((row for row in attendance_rows if row["student_id"] == selected_attendance_student["id"]), None)
            if selected_attendance_student
            else None
        )
        edited_status = st.selectbox(
            "Attendance Status For Selected Date",
            options=["present", "absent"],
            index=0 if not selected_student_attendance_row or selected_student_attendance_row["status"] == "present" else 1,
            key=f"attendance_edit_status_{class_id}",
        )
        if st.button("Save Attendance Edit", key=f"attendance_edit_save_{class_id}"):
            if selected_attendance_student:
                attendance_repository.update_student_attendance_status(
                    class_id=class_id,
                    student_id=selected_attendance_student["id"],
                    teacher_id=teacher["id"],
                    attendance_date=attendance_date,
                    status=edited_status,
                    source="manual",
                    raw_model_output="Edited manually by teacher.",
                )
                st.session_state[attendance_result_key] = {
                    "attendance_date": attendance_date,
                    "present_count": attendance_repository.get_attendance_overview(class_id, attendance_date)["present_count"],
                    "absent_count": attendance_repository.get_attendance_overview(class_id, attendance_date)["absent_count"],
                }
                st.session_state[overview_attendance_date_pending_key] = attendance_date_value
                st.rerun()
    with edit_col2:
        clear_student_confirm = st.checkbox(
            "Confirm clearing this student's full attendance history",
            key=f"attendance_clear_student_confirm_{class_id}",
        )
        if st.button("Clear Selected Student Attendance", key=f"attendance_clear_student_{class_id}"):
            if selected_attendance_student and clear_student_confirm:
                attendance_repository.clear_student_attendance(selected_attendance_student["id"], class_id)
                st.session_state.pop(attendance_result_key, None)
                st.rerun()
        clear_class_confirm = st.checkbox(
            "Confirm clearing attendance of all students in this class",
            key=f"attendance_clear_class_confirm_{class_id}",
        )
        if st.button("Clear All Class Attendance", key=f"attendance_clear_class_{class_id}"):
            if clear_class_confirm:
                attendance_repository.clear_class_attendance(class_id)
                st.session_state.pop(attendance_result_key, None)
                st.session_state.pop(attendance_proposal_key, None)
                st.rerun()

    st.caption(
        "Press the mic button, say the roll numbers or names of students who are absent, "
        "and Gemma will mark the rest present."
    )
    attendance_audio_bytes_key = f"attendance_audio_bytes_{class_id}"
    attendance_audio_mime_key = f"attendance_audio_mime_{class_id}"
    recorded_audio = st.audio_input(
        "Record absent students",
        key=f"attendance_audio_{class_id}_{attendance_date}",
    )
    if recorded_audio:
        st.session_state[attendance_audio_bytes_key] = recorded_audio.getvalue()
        st.session_state[attendance_audio_mime_key] = recorded_audio.type or "audio/wav"
        st.audio(recorded_audio)
        st.caption(
            f"Recorded audio: {len(st.session_state[attendance_audio_bytes_key])} bytes | "
            f"{st.session_state[attendance_audio_mime_key]}"
        )

    if st.button("Detect Absentees From Audio", type="primary", key=f"attendance_process_{class_id}"):
        audio_bytes = st.session_state.get(attendance_audio_bytes_key, b"")
        audio_mime_type = st.session_state.get(attendance_audio_mime_key, "audio/wav")
        if not audio_bytes:
            st.error("Record attendance audio first.")
        elif not use_llama_server:
            st.error("Enable llama-server to use Gemma audio attendance.")
        else:
            client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
            try:
                parsed_audio = parse_absent_students_from_audio(
                    client=client,
                    model_name=llama_model_name,
                    class_label=(
                        f"Grade {selected_class['grade']}-{selected_class['section']} | "
                        f"{selected_subject_name}"
                    ),
                    students=roster,
                    audio_bytes=audio_bytes,
                    audio_mime_type=audio_mime_type,
                )
                resolved_absentees = resolve_absent_students(
                    students=roster,
                    absent_roll_numbers=parsed_audio["absent_roll_numbers"],
                    absent_student_names=parsed_audio["absent_student_names"],
                )
                st.session_state[attendance_proposal_key] = {
                    **parsed_audio,
                    **resolved_absentees,
                    "attendance_date": attendance_date,
                    "class_id": class_id,
                    "teacher_id": teacher["id"],
                    "present_count": len(roster) - len(resolved_absentees["absent_student_ids"]),
                    "absent_count": len(resolved_absentees["absent_student_ids"]),
                }
                st.rerun()
            except (RequestException, ValueError, json.JSONDecodeError) as exc:
                st.error(f"Failed to process attendance audio: {exc}")

    latest_attendance_proposal = st.session_state.get(attendance_proposal_key)
    if (
        latest_attendance_proposal
        and latest_attendance_proposal.get("attendance_date") == attendance_date
        and latest_attendance_proposal.get("class_id") == class_id
    ):
        st.markdown("**Review Detected Absentees**")
        st.caption(
            f"Gemma detected {latest_attendance_proposal['absent_count']} absent students. "
            "Review this before saving attendance."
        )
        if latest_attendance_proposal.get("spoken_summary"):
            st.caption(f"Gemma heard: {latest_attendance_proposal['spoken_summary']}")
        review_col1, review_col2 = st.columns(2)
        with review_col1:
            st.write("Matched roll numbers")
            for item in latest_attendance_proposal.get("matched_roll_numbers", []):
                st.write(f"- {item}")
            st.write("Matched student names")
            for item in latest_attendance_proposal.get("matched_names", []):
                st.write(f"- {item}")
        with review_col2:
            st.write("Students marked absent")
            absent_students = [
                student
                for student in roster
                if student["id"] in set(latest_attendance_proposal.get("absent_student_ids", []))
            ]
            if absent_students:
                for item in absent_students:
                    st.write(f"- {item['roll_number']} | {item['full_name']}")
            else:
                st.write("- None")
            if latest_attendance_proposal.get("unresolved_mentions"):
                st.warning(
                    "Unresolved mentions: "
                    + ", ".join(latest_attendance_proposal["unresolved_mentions"])
                )
        with st.expander("Gemma Attendance Output", expanded=False):
            st.code(latest_attendance_proposal.get("raw_model_output", ""))

        confirm_col, discard_col = st.columns(2)
        with confirm_col:
            if st.button("Confirm And Save Attendance", type="primary", key=f"attendance_confirm_{class_id}"):
                marked_attendance = mark_attendance_from_identifiers(
                    class_id=class_id,
                    teacher_id=teacher["id"],
                    attendance_date=attendance_date,
                    absent_roll_numbers=", ".join(latest_attendance_proposal.get("matched_roll_numbers", [])),
                    absent_student_names=", ".join(latest_attendance_proposal.get("matched_names", [])),
                    source="audio",
                    raw_model_output=latest_attendance_proposal.get("raw_model_output", ""),
                )
                st.session_state[attendance_result_key] = {
                    **latest_attendance_proposal,
                    **marked_attendance,
                }
                st.session_state[overview_attendance_date_pending_key] = attendance_date_value
                st.session_state.pop(attendance_proposal_key, None)
                st.rerun()
        with discard_col:
            if st.button("Discard Detection", key=f"attendance_discard_{class_id}"):
                st.session_state.pop(attendance_proposal_key, None)
                st.rerun()

    if latest_attendance_result and latest_attendance_result.get("attendance_date") == attendance_date:
        st.success(
            f"Attendance marked for {attendance_date}. "
            f"Present: {latest_attendance_result['present_count']} | "
            f"Absent: {latest_attendance_result['absent_count']}"
        )
        if latest_attendance_result.get("spoken_summary"):
            st.caption(f"Gemma heard: {latest_attendance_result['spoken_summary']}")
        if latest_attendance_result.get("unresolved_mentions"):
            st.warning(
                "Unresolved mentions: "
                + ", ".join(latest_attendance_result["unresolved_mentions"])
            )
        with st.expander("Gemma Attendance Output", expanded=False):
            st.code(latest_attendance_result.get("raw_model_output", ""))

    st.markdown("**Attendance Register**")
    current_rows = attendance_repository.list_attendance_for_date(class_id, attendance_date)
    if current_rows:
        st.dataframe(
            [
                {
                    "Roll Number": row["roll_number"],
                    "Student": row["full_name"],
                    "Status": row["status"].title(),
                    "Source": row["source"],
                    "Updated At": row["updated_at"],
                    "Overall Attendance %": (
                        f"{attendance_stats.get(row['student_id'], {}).get('attendance_percentage')}%"
                        if attendance_stats.get(row["student_id"], {}).get("attendance_percentage") is not None
                        else "N/A"
                    ),
                }
                for row in current_rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No attendance has been marked for this class on the selected date yet.")

with tab_management:
    st.markdown("**Class Management**")
    management_mode_key = f"class_management_mode_{teacher['id']}"
    managed_class_id = selected_class["id"]
    class_subjects = teacher_class_repository.list_class_subjects(managed_class_id)
    selected_management_subject = st.selectbox(
        "Subject Workspace",
        options=class_subjects,
        format_func=lambda item: item["subject"],
        key=f"management_workspace_{teacher['id']}",
    ) if class_subjects else None
    management_roster = student_repository.list_class_roster(managed_class_id)
    inactive_students = student_repository.list_inactive_class_students(managed_class_id)
    current_management_chapters = curriculum_repository.list_chapters_for_class(
        managed_class_id,
        selected_management_subject["subject"] if selected_management_subject else None,
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Active Students", len(management_roster))
    summary_col2.metric("Chapters", len(current_management_chapters))
    summary_col3.metric("Subjects", len(class_subjects))
    st.caption(
        f"Managing Grade {selected_class['grade']}-{selected_class['section']} | "
        f"Subjects: {selected_class.get('subjects_csv') or selected_class.get('subject', '')}"
    )
    if selected_management_subject:
        st.caption(
            f"Active subject workspace: {selected_management_subject['subject']} | "
            f"Medium: {selected_management_subject.get('medium') or 'Not set'}"
        )

    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    with action_col1:
        if st.button("Add New Subject", key=f"manage_add_subject_{teacher['id']}", use_container_width=True):
            st.session_state[management_mode_key] = "add_subject"
    with action_col2:
        if st.button("Edit Subject", key=f"manage_edit_subject_{managed_class_id}", use_container_width=True):
            st.session_state[management_mode_key] = "edit_subject"
    with action_col3:
        if st.button("Manage Students", key=f"manage_students_{managed_class_id}", use_container_width=True):
            st.session_state[management_mode_key] = "students"
    with action_col4:
        if st.button("Manage Chapters", key=f"manage_chapters_{managed_class_id}", use_container_width=True):
            st.session_state[management_mode_key] = "chapters"

    management_mode = st.session_state.get(management_mode_key, "students")
    if st.button("Hide Panel", key=f"manage_hide_{teacher['id']}"):
        st.session_state[management_mode_key] = ""
        st.rerun()
    management_mode = st.session_state.get(management_mode_key, "")

    if management_mode == "add_subject":
        st.markdown("**Add New Subject**")
        with st.form(f"create_class_form_{teacher['id']}"):
            new_subject = st.text_input("Subject", value="")
            new_medium = st.text_input(
                "Medium",
                value=(selected_management_subject or {}).get("medium", "") or selected_class.get("medium", "") or "",
            )
            create_class_submitted = st.form_submit_button("Create Subject")
        if create_class_submitted:
            try:
                teacher_class_repository.add_subject_to_class(class_id=managed_class_id, subject=new_subject, medium=new_medium)
                st.success("Subject created.")
                st.session_state[management_mode_key] = ""
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to create subject: {exc}")

    elif management_mode == "edit_subject":
        if selected_management_subject:
            st.markdown("**Edit Subject Details**")
            with st.form(f"edit_class_form_{managed_class_id}_{selected_management_subject['id']}"):
                edit_subject = st.text_input("Subject", value=selected_management_subject["subject"])
                edit_medium = st.text_input("Medium", value=selected_management_subject.get("medium", "") or "")
                edit_class_submitted = st.form_submit_button("Save Subject Changes")
            if edit_class_submitted:
                try:
                    teacher_class_repository.update_class_subject_details(
                        class_subject_id=selected_management_subject["id"],
                        subject=edit_subject,
                        medium=edit_medium,
                    )
                    st.success("Subject updated.")
                    st.session_state[management_mode_key] = ""
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to update subject: {exc}")
        else:
            st.info("Add a subject first to edit its details.")

    elif management_mode == "students":
        student_manage_tab1, student_manage_tab2, student_manage_tab3, student_manage_tab4 = st.tabs(
            ["Add Student", "Edit Student", "Remove Student", "Restore Student"]
        )
        with student_manage_tab1:
            with st.form(f"add_student_form_{managed_class_id}"):
                new_roll_number = st.text_input("Roll Number")
                new_student_name = st.text_input("Student Name")
                new_student_email = st.text_input("Student Email")
                new_student_language = st.text_input("Preferred Language", value="English")
                new_student_accessibility = st.text_area("Accessibility Notes", height=80)
                add_student_submitted = st.form_submit_button("Add Student")
            if add_student_submitted:
                try:
                    student_repository.add_student_to_class(
                        class_id=managed_class_id,
                        roll_number=new_roll_number,
                        full_name=new_student_name,
                        email=new_student_email,
                        preferred_language=new_student_language,
                        accessibility_notes=new_student_accessibility,
                    )
                    st.success("Student added to subject roster.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to add student: {exc}")

        with student_manage_tab2:
            if management_roster:
                selected_edit_student = st.selectbox(
                    "Select Student To Edit",
                    options=management_roster,
                    format_func=lambda item: f"{item['roll_number']} | {item['full_name']}",
                    key=f"edit_student_select_{managed_class_id}",
                )
                selected_edit_student_detail = student_repository.get_student_detail(selected_edit_student["id"]).get("student") if selected_edit_student else None
                with st.form(f"edit_student_form_{managed_class_id}_{selected_edit_student['id']}"):
                    edit_student_roll = st.text_input("Roll Number", value=selected_edit_student_detail["roll_number"] if selected_edit_student_detail else "")
                    edit_student_name = st.text_input("Student Name", value=selected_edit_student_detail["full_name"] if selected_edit_student_detail else "")
                    edit_student_email = st.text_input("Student Email", value=selected_edit_student_detail.get("email", "") if selected_edit_student_detail else "")
                    edit_student_language = st.text_input("Preferred Language", value=selected_edit_student_detail.get("preferred_language", "") if selected_edit_student_detail else "")
                    edit_student_accessibility = st.text_area("Accessibility Notes", value=selected_edit_student_detail.get("accessibility_notes", "") if selected_edit_student_detail else "", height=80)
                    edit_student_submitted = st.form_submit_button("Save Student Changes")
                if edit_student_submitted:
                    try:
                        student_repository.update_student_details(
                            student_id=selected_edit_student["id"],
                            roll_number=edit_student_roll,
                            full_name=edit_student_name,
                            email=edit_student_email,
                            preferred_language=edit_student_language,
                            accessibility_notes=edit_student_accessibility,
                        )
                        st.success("Student updated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to update student: {exc}")
            else:
                st.info("No active students in this class yet.")

        with student_manage_tab3:
            if management_roster:
                selected_manage_student = st.selectbox(
                    "Select Student To Remove",
                    options=management_roster,
                    format_func=lambda item: f"{item['roll_number']} | {item['full_name']}",
                    key=f"manage_student_select_{managed_class_id}",
                )
                remove_student_confirm = st.checkbox(
                    "Confirm removing this student from the active roster",
                    key=f"remove_student_confirm_{managed_class_id}",
                )
                if st.button("Remove Student", key=f"remove_student_button_{managed_class_id}"):
                    if remove_student_confirm:
                        try:
                            student_repository.deactivate_student(selected_manage_student["id"])
                            st.success("Student removed from active roster.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed to remove student: {exc}")
                    else:
                        st.warning("Confirm student removal first.")
            else:
                st.info("No active students in this class yet.")

        with student_manage_tab4:
            if inactive_students:
                selected_inactive_student = st.selectbox(
                    "Select Student To Restore",
                    options=inactive_students,
                    format_func=lambda item: f"{item['roll_number']} | {item['full_name']}",
                    key=f"restore_student_select_{managed_class_id}",
                )
                if st.button("Restore Student", key=f"restore_student_button_{managed_class_id}"):
                    try:
                        student_repository.reactivate_student(selected_inactive_student["id"])
                        st.success("Student restored to active roster.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to restore student: {exc}")
            else:
                st.info("No inactive students in this class.")

    elif management_mode == "chapters":
        if not selected_management_subject:
            st.info("Add a subject first to manage its chapters.")
        else:
            st.markdown(f"**Chapter Management For {selected_management_subject['subject']}**")
        chapter_manage_tab1, chapter_manage_tab2, chapter_manage_tab3 = st.tabs(
            ["Add Chapter", "Edit Chapter", "Delete Chapter"]
        )
        with chapter_manage_tab1:
            with st.form(f"add_chapter_form_{managed_class_id}"):
                chapter_code_value = st.text_input(
                    "Chapter Code",
                    value=f"{selected_management_subject['subject'][:3].upper()}-{selected_class['grade']}-" if selected_management_subject else "",
                )
                chapter_name_value = st.text_input("Chapter Name")
                chapter_term_value = st.text_input("Term", value="Term 1")
                add_chapter_submitted = st.form_submit_button("Add Chapter")
            if add_chapter_submitted:
                try:
                    if not selected_management_subject:
                        raise ValueError("Select a subject workspace first.")
                    curriculum_repository.create_chapter_for_class(
                        class_id=managed_class_id,
                        subject=selected_management_subject["subject"],
                        chapter_code=chapter_code_value,
                        chapter_name=chapter_name_value,
                        term=chapter_term_value,
                    )
                    st.success("Chapter added.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to add chapter: {exc}")

        with chapter_manage_tab2:
            if current_management_chapters:
                selected_chapter = st.selectbox(
                    "Select Chapter To Edit",
                    options=current_management_chapters,
                    format_func=lambda item: f"{item['chapter_code']} | {item['chapter_name']}",
                    key=f"manage_chapter_select_{managed_class_id}",
                )
                with st.form(f"edit_chapter_form_{managed_class_id}_{selected_chapter['id']}"):
                    edit_chapter_code = st.text_input("Chapter Code", value=selected_chapter["chapter_code"])
                    edit_chapter_name = st.text_input("Chapter Name", value=selected_chapter["chapter_name"])
                    edit_chapter_term = st.text_input("Term", value=selected_chapter.get("term", "") or "")
                    edit_chapter_submitted = st.form_submit_button("Save Chapter Changes")
                if edit_chapter_submitted:
                    try:
                        curriculum_repository.update_chapter_details(
                            chapter_id=selected_chapter["id"],
                            chapter_code=edit_chapter_code,
                            chapter_name=edit_chapter_name,
                            term=edit_chapter_term,
                        )
                        st.success("Chapter updated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to update chapter: {exc}")
            else:
                st.info("No chapters have been added for this subject yet.")

        with chapter_manage_tab3:
            if current_management_chapters:
                selected_delete_chapter = st.selectbox(
                    "Select Chapter To Delete",
                    options=current_management_chapters,
                    format_func=lambda item: f"{item['chapter_code']} | {item['chapter_name']}",
                    key=f"delete_chapter_select_{managed_class_id}",
                )
                delete_chapter_confirm = st.checkbox(
                    "Confirm deleting selected chapter if it is unused by assessments",
                    key=f"delete_chapter_confirm_{managed_class_id}",
                )
                if st.button("Delete Chapter", key=f"delete_chapter_button_{managed_class_id}"):
                    if delete_chapter_confirm:
                        deleted, message = curriculum_repository.delete_chapter_if_unused(selected_delete_chapter["id"])
                        if deleted:
                            st.success(message)
                            st.rerun()
                        else:
                            st.warning(message)
                    else:
                        st.warning("Confirm chapter deletion first.")
            else:
                st.info("No chapters have been added for this subject yet.")

    st.markdown("**Current Teacher Classes**")
    st.dataframe(classes, use_container_width=True, hide_index=True)

with tab_materials:
    st.markdown("**Reading Materials**")
    st.caption(
        "Upload books, notes, or images for the selected grade. Subjects and chapters are extracted automatically and shared across sections of the same grade."
    )
    grade_subjects = curriculum_repository.list_grade_curriculum_subjects(selected_class["grade"])
    current_curriculum_subject = curriculum_repository.get_curriculum_subject(grade=selected_class["grade"], subject=selected_subject_name)
    current_curriculum_chapters = (
        curriculum_repository.list_curriculum_chapters(current_curriculum_subject["id"])
        if current_curriculum_subject
        else []
    )
    materials = material_repository.list_subject_materials(grade=selected_class["grade"], subject=selected_subject_name) if selected_subject_name else []
    ingestion_runs = material_repository.list_recent_ingestion_runs(limit=10)

    material_col1, material_col2 = st.columns([2, 1])
    with material_col1:
        upload_mode = st.radio(
            "Material Type",
            options=["PDF", "Text", "Image"],
            horizontal=True,
            key=f"material_upload_mode_{class_id}",
        )
        with st.form(f"material_ingestion_form_{class_id}"):
            material_title = st.text_input("Material Title", value=f"Grade {selected_class['grade']} {selected_subject_name} Material")
            if upload_mode == "PDF":
                uploaded_file = st.file_uploader(
                    "Upload PDF",
                    type=["pdf"],
                    key=f"material_pdf_uploader_{class_id}",
                )
                text_material = ""
            elif upload_mode == "Image":
                uploaded_file = st.file_uploader(
                    "Upload Image",
                    type=["png", "jpg", "jpeg", "webp"],
                    key=f"material_image_uploader_{class_id}",
                )
                text_material = ""
            else:
                uploaded_file = None
                text_material = st.text_area("Paste Reading Material", height=220)
            submit_material = st.form_submit_button("Ingest Material", type="primary")
        if submit_material:
            try:
                source_type = upload_mode.lower()
                content_bytes = uploaded_file.getvalue() if uploaded_file else None
                mime_type = uploaded_file.type if uploaded_file else "text/plain"
                original_filename = uploaded_file.name if uploaded_file else ""
                progress_bar = st.progress(0, text="Preparing ingestion...")
                progress_status = st.empty()

                def update_ingestion_progress(ratio: float, stage: str, detail: str) -> None:
                    progress_bar.progress(int(ratio * 100), text=stage)
                    if detail.strip():
                        progress_status.caption(detail)
                    else:
                        progress_status.caption(stage)

                result = ingest_reading_material(
                    teacher_id=teacher["id"],
                    board_type="CBSE",
                    grade=selected_class["grade"],
                    title=material_title,
                    source_type=source_type,
                    content_bytes=content_bytes,
                    text_content=text_material,
                    original_filename=original_filename,
                    mime_type=mime_type,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    progress_callback=update_ingestion_progress,
                )
                progress_bar.progress(100, text="Completed")
                progress_status.caption(
                    f"Finished processing {material_title or original_filename or 'material'}."
                )
                st.success(
                    f"Ingested material for {result['subject']}. "
                    f"Created or updated {len(result['chapters'])} chapters and indexed {result['indexed_count']} chunks."
                )
                if result.get("embedding_warning"):
                    st.warning(result["embedding_warning"])
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to ingest reading material: {exc}")

    with material_col2:
        st.markdown("**Grade Subject Library**")
        if grade_subjects:
            for item in grade_subjects:
                st.write(f"- {item['subject']}")
        else:
            st.info("No grade-level subjects have been extracted yet.")

        st.markdown("**Current Subject Chapters**")
        if current_curriculum_chapters:
            for chapter in current_curriculum_chapters:
                st.write(
                    f"- {chapter['chapter_code']} | {chapter['chapter_name']}"
                )
        else:
            st.info("No extracted chapters yet for this subject.")

    st.markdown("**Current Subject Materials**")
    if materials:
        st.dataframe(
            [
                {
                    "Title": item["title"],
                    "Type": item["source_type"],
                    "File": item["original_filename"] or "",
                    "Summary": item["extraction_summary"] or "",
                    "Created At": item["created_at"],
                }
                for item in materials
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No reading materials have been uploaded for this subject yet.")

    retrieval_query = st.text_input(
        "Search Current Subject Materials",
        value="",
        key=f"material_search_query_{class_id}",
        placeholder=f"Search {selected_subject_name} materials",
    )
    if retrieval_query.strip():
        try:
            hits = search_subject_materials(
                grade=selected_class["grade"],
                subject=selected_subject_name,
                query=retrieval_query,
                top_k=MODEL_SAMPLING.rag_top_k,
            )
            if hits:
                for index, hit in enumerate(hits, start=1):
                    metadata = hit.get("metadata", {})
                    with st.expander(
                        f"Hit {index} | score {hit['score']} | {metadata.get('source_title', 'Material')}",
                        expanded=index == 1,
                    ):
                        st.caption(
                            f"Section: {metadata.get('section_heading') or 'General'} | "
                            f"Content type: {metadata.get('content_type') or 'text'}"
                        )
                        st.write(hit["text"])
            else:
                st.info("No matching chunks found for this subject.")
        except Exception as exc:
            st.error(f"Material search failed: {exc}")

    st.markdown("**Recent Ingestion Runs**")
    if ingestion_runs:
        st.dataframe(ingestion_runs, use_container_width=True, hide_index=True)
    else:
        st.info("No ingestion runs yet.")

with tab_planner:
    st.markdown("**Teaching Progress Workspace**")
    st.caption(
        "Create a yearly subject plan, map weekly periods, and process classroom recordings so Gemma can update what was actually taught and suggest the next sessions automatically."
    )
    if plan_snapshot.get("plan"):
        planner_metric_1, planner_metric_2, planner_metric_3 = st.columns(3)
        planner_metric_1.metric("Plan Completion", f"{plan_snapshot.get('completion_percent', 0)}%")
        planner_metric_2.metric("Upcoming Units", len(plan_snapshot.get("upcoming_units", [])))
        planner_metric_3.metric("Recorded Sessions", len(coverage_sessions))
    else:
        st.info("No academic year plan exists yet for this class subject.")

    if active_timetable_slot:
        st.caption(
            f"Active scheduled period detected: {WEEKDAY_LABELS.get(int(active_timetable_slot['weekday']), 'Day')} "
            f"{active_timetable_slot['start_time']}-{active_timetable_slot['end_time']} for {selected_subject_name}."
        )
    else:
        st.caption("No active scheduled period is detected right now for this subject.")

    if capture_support_status.get("automatic_capture_available"):
        st.caption(capture_support_status["reason"])
    else:
        st.warning(capture_support_status["reason"])

    planner_setup_tab, planner_schedule_tab, planner_coverage_tab, planner_next_tab = st.tabs(
        ["Year Plan", "Timetable", "Process Class", "Next Classes"]
    )

    with planner_setup_tab:
        existing_plan = plan_snapshot.get("plan")
        syllabus_state_key = f"planner_syllabus_text_{class_id}_{selected_subject_name}"
        if syllabus_state_key not in st.session_state:
            st.session_state[syllabus_state_key] = (existing_plan or {}).get("raw_syllabus_text", "")
        uploaded_syllabus_file = st.file_uploader(
            "Upload full grade syllabus as PDF or image",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            key=f"planner_syllabus_file_{class_id}_{selected_subject_name}",
        )
        extract_syllabus = st.button(
            "Extract Grade Syllabus Text",
            key=f"planner_extract_syllabus_{class_id}_{selected_subject_name}",
            disabled=not uploaded_syllabus_file,
        )
        if extract_syllabus and uploaded_syllabus_file:
            try:
                st.session_state[syllabus_state_key] = DEFAULT_TEACHING_PLANNER_SERVICE.extract_uploaded_text(
                    content_bytes=uploaded_syllabus_file.getvalue(),
                    mime_type=uploaded_syllabus_file.type or "",
                    original_filename=uploaded_syllabus_file.name,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    use_llama_server=use_llama_server,
                )
                st.success("Syllabus text extracted from file.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to extract syllabus text: {exc}")
        import_full_grade_syllabus = st.button(
            "Import Whole Grade Syllabus Into Subjects",
            key=f"planner_import_full_grade_syllabus_{class_id}_{selected_subject_name}",
            disabled=not st.session_state.get(syllabus_state_key, "").strip(),
        )
        if import_full_grade_syllabus:
            try:
                import_result = DEFAULT_TEACHING_PLANNER_SERVICE.import_grade_syllabus_document(
                    teacher_id=teacher["id"],
                    class_id=class_id,
                    academic_year=selected_class.get("academic_year", ""),
                    grade=selected_class["grade"],
                    board_type="CBSE",
                    syllabus_text=st.session_state.get(syllabus_state_key, ""),
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    use_llama_server=use_llama_server,
                )
                st.session_state[f"planner_grade_syllabus_import_{class_id}"] = import_result
                st.success(f"Imported {import_result['subject_count']} subjects from the grade syllabus.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to import the grade syllabus: {exc}")
        latest_grade_syllabus_import = st.session_state.get(f"planner_grade_syllabus_import_{class_id}")
        if latest_grade_syllabus_import:
            with st.expander("Latest Grade Syllabus Import", expanded=False):
                for item in latest_grade_syllabus_import.get("subjects_imported", []):
                    st.write(
                        f"- {item['subject']}: {item['units_count']} chapters | "
                        f"{', '.join(item['chapter_names'][:4])}"
                    )
        with st.form(f"academic_year_plan_form_{class_id}_{selected_subject_name}"):
            syllabus_text = st.text_area(
                "Paste or edit syllabus text",
                value=st.session_state.get(syllabus_state_key, ""),
                height=220,
                placeholder="Paste the full grade syllabus or a single-subject syllabus here.",
            )
            generate_plan = st.form_submit_button("Generate Or Refresh Year Plan", type="primary")
        if generate_plan:
            try:
                st.session_state[syllabus_state_key] = syllabus_text
                DEFAULT_TEACHING_PLANNER_SERVICE.generate_year_plan(
                    teacher_id=teacher["id"],
                    class_id=class_id,
                    academic_year=selected_class.get("academic_year", ""),
                    grade=selected_class["grade"],
                    subject=selected_subject_name,
                    board_type="CBSE",
                    syllabus_text=syllabus_text,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    use_llama_server=use_llama_server,
                )
                st.success("Academic year plan generated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to generate year plan: {exc}")

        refreshed_plan = planning_repository.get_active_academic_year_plan(
            class_id=class_id,
            subject=selected_subject_name,
            academic_year=selected_class.get("academic_year", ""),
        )
        if refreshed_plan:
            refreshed_units = planning_repository.list_academic_year_plan_units(int(refreshed_plan["id"]))
            st.markdown(f"**{refreshed_plan.get('plan_title') or 'Academic Year Plan'}**")
            st.dataframe(
                [
                    {
                        "Order": item["sequence_order"],
                        "Chapter": item["chapter_name"],
                        "Subtopics": ", ".join(item.get("subtopics", [])[:5]),
                        "Recommended Sessions": item["recommended_sessions"],
                        "Completion %": item["completion_percent"],
                        "Status": item["status"],
                    }
                    for item in refreshed_units
                ],
                use_container_width=True,
                hide_index=True,
            )

    with planner_schedule_tab:
        uploaded_timetable_file = st.file_uploader(
            "Upload whole weekly timetable as PDF or image",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            key=f"planner_timetable_file_{class_id}_{selected_subject_name}",
        )
        import_timetable = st.button(
            "Import Whole Timetable Grid",
            key=f"planner_import_timetable_{class_id}_{selected_subject_name}",
            disabled=not uploaded_timetable_file,
        )
        if import_timetable and uploaded_timetable_file:
            try:
                timetable_text = DEFAULT_TEACHING_PLANNER_SERVICE.extract_uploaded_text(
                    content_bytes=uploaded_timetable_file.getvalue(),
                    mime_type=uploaded_timetable_file.type or "",
                    original_filename=uploaded_timetable_file.name,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    use_llama_server=use_llama_server,
                )
                imported_slots = DEFAULT_TEACHING_PLANNER_SERVICE.import_timetable_grid_document(
                    class_id=class_id,
                    grade=selected_class["grade"],
                    timetable_text=timetable_text,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    use_llama_server=use_llama_server,
                )
                st.session_state[f"planner_timetable_extracted_{class_id}_{selected_subject_name}"] = timetable_text
                st.success(f"Imported {len(imported_slots)} timetable slot(s) across subjects.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to import timetable: {exc}")
        extracted_timetable_text = st.session_state.get(f"planner_timetable_extracted_{class_id}_{selected_subject_name}", "")
        if extracted_timetable_text:
            with st.expander("Extracted Timetable Text", expanded=False):
                st.text(extracted_timetable_text)
        with st.form(f"planner_timetable_form_{class_id}_{selected_subject_name}"):
            schedule_col1, schedule_col2, schedule_col3, schedule_col4 = st.columns(4)
            with schedule_col1:
                weekday_label = st.selectbox(
                    "Weekday",
                    options=[label for _, label in WEEKDAY_OPTIONS],
                    key=f"planner_weekday_{class_id}_{selected_subject_name}",
                )
            with schedule_col2:
                slot_start = st.text_input(
                    "Start Time",
                    value="09:00",
                    key=f"planner_slot_start_{class_id}_{selected_subject_name}",
                )
            with schedule_col3:
                slot_end = st.text_input(
                    "End Time",
                    value="09:45",
                    key=f"planner_slot_end_{class_id}_{selected_subject_name}",
                )
            with schedule_col4:
                auto_record_enabled = st.checkbox(
                    "Auto-record target",
                    value=True,
                    key=f"planner_auto_record_{class_id}_{selected_subject_name}",
                )
            save_slot = st.form_submit_button("Add Timetable Slot", type="primary")
        if save_slot:
            try:
                weekday_value = next(key for key, label in WEEKDAY_OPTIONS if label == weekday_label)
                DEFAULT_TEACHING_PLANNER_SERVICE.add_timetable_slot(
                    class_id=class_id,
                    subject=selected_subject_name,
                    weekday=weekday_value,
                    start_time=slot_start,
                    end_time=slot_end,
                    auto_record_enabled=auto_record_enabled,
                )
                st.success("Timetable slot saved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to save timetable slot: {exc}")

        if timetable_slots:
            st.markdown("**Saved Weekly Slots For Current Subject**")
            for slot in timetable_slots:
                slot_col1, slot_col2 = st.columns([6, 1])
                with slot_col1:
                    st.write(
                        f"{WEEKDAY_LABELS.get(int(slot['weekday']), 'Day')} | {slot['start_time']}-{slot['end_time']} | "
                        f"{'Auto-record target' if slot['auto_record_enabled'] else 'Manual only'}"
                    )
                with slot_col2:
                    if st.button("Delete", key=f"delete_timetable_slot_{slot['id']}"):
                        DEFAULT_TEACHING_PLANNER_SERVICE.delete_timetable_slot(int(slot["id"]))
                        st.rerun()
        else:
            st.info("No timetable slots saved for this subject yet.")
        if all_timetable_slots:
            st.markdown("**All Imported Timetable Slots**")
            st.dataframe(
                [
                    {
                        "Subject": slot["subject"],
                        "Weekday": WEEKDAY_LABELS.get(int(slot["weekday"]), "Day"),
                        "Start": slot["start_time"],
                        "End": slot["end_time"],
                        "Auto Record": "Yes" if slot.get("auto_record_enabled") else "No",
                    }
                    for slot in all_timetable_slots
                ],
                use_container_width=True,
                hide_index=True,
            )

    with planner_coverage_tab:
        st.caption(
            "Use the current subject workspace or timetable slot as the subject context. "
            "Raw audio is processed and then discarded; only transcript and coverage summary are kept."
        )
        planner_local_audio_bytes_key = f"planner_local_audio_bytes_{class_id}_{selected_subject_name}"
        planner_local_audio_mime_key = f"planner_local_audio_mime_{class_id}_{selected_subject_name}"
        planner_auto_capture_key = f"planner_auto_capture_{class_id}_{selected_subject_name}"
        recorder_status = DEFAULT_TEACHING_PLANNER_SERVICE.get_local_recorder_status()
        recorder_session = recorder_status.get("session") or {}

        if (
            recorder_status.get("active")
            and recorder_session.get("class_id") == class_id
            and recorder_session.get("subject") == selected_subject_name
            and recorder_session.get("scheduled_end")
        ):
            try:
                stop_deadline = datetime.strptime(
                    f"{datetime.now().date().isoformat()} {recorder_session['scheduled_end']}",
                    "%Y-%m-%d %H:%M",
                )
                if datetime.now() >= stop_deadline + timedelta(minutes=5):
                    finished_capture = DEFAULT_TEACHING_PLANNER_SERVICE.stop_local_microphone_recording()
                    st.session_state[planner_local_audio_bytes_key] = finished_capture["audio_bytes"]
                    st.session_state[planner_local_audio_mime_key] = finished_capture["audio_mime_type"]
                    st.info("Local microphone recording auto-stopped 5 minutes after the scheduled end time.")
                    st.rerun()
            except Exception:
                pass

        if capture_support_status.get("automatic_capture_available"):
            auto_capture_enabled = st.checkbox(
                "Use sounddevice local microphone recording for active timetable slots",
                value=st.session_state.get(planner_auto_capture_key, True),
                key=planner_auto_capture_key,
            )
            if (
                auto_capture_enabled
                and active_timetable_slot
                and active_timetable_slot.get("auto_record_enabled")
                and not recorder_status.get("active")
            ):
                try:
                    DEFAULT_TEACHING_PLANNER_SERVICE.start_local_microphone_recording(
                        class_id=class_id,
                        subject=selected_subject_name,
                        timetable_slot_id=int(active_timetable_slot["id"]),
                        scheduled_end=str(active_timetable_slot.get("end_time") or ""),
                    )
                    st.info("Started sounddevice local microphone recording for the active timetable slot.")
                    st.rerun()
                except Exception as exc:
                    st.warning(f"Could not auto-start local microphone recording: {exc}")

            local_record_col1, local_record_col2, local_record_col3 = st.columns(3)
            with local_record_col1:
                if st.button(
                    "Start Local Mic Recording",
                    key=f"planner_start_local_mic_{class_id}_{selected_subject_name}",
                    disabled=bool(recorder_status.get("active")),
                ):
                    try:
                        DEFAULT_TEACHING_PLANNER_SERVICE.start_local_microphone_recording(
                            class_id=class_id,
                            subject=selected_subject_name,
                            timetable_slot_id=int(active_timetable_slot["id"]) if active_timetable_slot else None,
                            scheduled_end=str((active_timetable_slot or {}).get("end_time") or ""),
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to start local microphone recording: {exc}")
            with local_record_col2:
                if st.button(
                    "Stop Local Mic Recording",
                    key=f"planner_stop_local_mic_{class_id}_{selected_subject_name}",
                    disabled=not bool(recorder_status.get("active")),
                ):
                    try:
                        finished_capture = DEFAULT_TEACHING_PLANNER_SERVICE.stop_local_microphone_recording()
                        st.session_state[planner_local_audio_bytes_key] = finished_capture["audio_bytes"]
                        st.session_state[planner_local_audio_mime_key] = finished_capture["audio_mime_type"]
                        st.success("Local microphone recording saved to the current class session.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to stop local microphone recording: {exc}")
            with local_record_col3:
                if st.button(
                    "Clear Local Mic Clip",
                    key=f"planner_clear_local_mic_{class_id}_{selected_subject_name}",
                    disabled=planner_local_audio_bytes_key not in st.session_state,
                ):
                    st.session_state.pop(planner_local_audio_bytes_key, None)
                    st.session_state.pop(planner_local_audio_mime_key, None)
                    st.rerun()

            if recorder_status.get("active") and recorder_session.get("class_id") == class_id:
                st.caption(
                    f"Local microphone recording is active for {recorder_session.get('subject', selected_subject_name)} "
                    f"since {recorder_session.get('started_at', '')}."
                )

            if st.session_state.get(planner_local_audio_bytes_key):
                st.audio(st.session_state[planner_local_audio_bytes_key], format="audio/wav")
                st.caption(
                    f"Stored local microphone clip: {len(st.session_state[planner_local_audio_bytes_key])} bytes | audio/wav"
                )

        with st.form(f"planner_coverage_form_{class_id}_{selected_subject_name}"):
            coverage_date = st.date_input(
                "Session Date",
                value=datetime.now().date(),
                key=f"planner_coverage_date_{class_id}_{selected_subject_name}",
            )
            coverage_col1, coverage_col2 = st.columns(2)
            with coverage_col1:
                scheduled_start_input = st.text_input(
                    "Scheduled Start",
                    value=(active_timetable_slot or {}).get("start_time", ""),
                    key=f"planner_scheduled_start_{class_id}_{selected_subject_name}",
                )
                actual_start_input = st.text_input(
                    "Actual Start",
                    value="",
                    key=f"planner_actual_start_{class_id}_{selected_subject_name}",
                )
            with coverage_col2:
                scheduled_end_input = st.text_input(
                    "Scheduled End",
                    value=(active_timetable_slot or {}).get("end_time", ""),
                    key=f"planner_scheduled_end_{class_id}_{selected_subject_name}",
                )
                actual_end_input = st.text_input(
                    "Actual End",
                    value="",
                    key=f"planner_actual_end_{class_id}_{selected_subject_name}",
                )
            teacher_note = st.text_area(
                "Optional teacher note",
                height=120,
                placeholder="Add a short recap if the recording is noisy or if you want to mention unfinished parts of the lesson.",
            )
            recorded_class_audio = st.audio_input(
                "Record classroom audio",
                key=f"planner_audio_input_{class_id}_{selected_subject_name}",
            )
            uploaded_class_audio = st.file_uploader(
                "Or upload recorded classroom audio",
                type=["wav", "mp3", "ogg", "m4a", "webm"],
                key=f"planner_audio_upload_{class_id}_{selected_subject_name}",
            )
            process_session = st.form_submit_button("Process Class Session", type="primary")
        if process_session:
            try:
                audio_bytes = b""
                audio_mime_type = "audio/wav"
                if st.session_state.get(planner_local_audio_bytes_key):
                    audio_bytes = st.session_state[planner_local_audio_bytes_key]
                    audio_mime_type = st.session_state.get(planner_local_audio_mime_key, "audio/wav")
                elif recorded_class_audio:
                    audio_bytes = recorded_class_audio.getvalue()
                    audio_mime_type = recorded_class_audio.type or "audio/wav"
                elif uploaded_class_audio:
                    audio_bytes = uploaded_class_audio.getvalue()
                    audio_mime_type = uploaded_class_audio.type or "audio/wav"
                result = DEFAULT_TEACHING_PLANNER_SERVICE.process_class_session(
                    class_id=class_id,
                    teacher_id=teacher["id"],
                    subject=selected_subject_name,
                    academic_year=selected_class.get("academic_year", ""),
                    session_date=coverage_date.isoformat(),
                    teacher_note=teacher_note,
                    audio_bytes=audio_bytes,
                    audio_mime_type=audio_mime_type,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    use_llama_server=use_llama_server,
                    scheduled_start=scheduled_start_input,
                    scheduled_end=scheduled_end_input,
                    actual_start=actual_start_input,
                    actual_end=actual_end_input,
                    timetable_slot_id=int(active_timetable_slot["id"]) if active_timetable_slot else None,
                )
                st.session_state[f"planner_result_{class_id}_{selected_subject_name}"] = result
                st.session_state.pop(planner_local_audio_bytes_key, None)
                st.session_state.pop(planner_local_audio_mime_key, None)
                st.success("Class session processed and teaching plan updated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to process class session: {exc}")

        latest_planner_result = st.session_state.get(f"planner_result_{class_id}_{selected_subject_name}")
        if latest_planner_result:
            st.markdown("**Latest Coverage Update**")
            st.caption(
                f"Updated chapter: {latest_planner_result['updated_unit']['chapter_name']} | "
                f"confidence {round(latest_planner_result['coverage_confidence'] * 100, 1)}%"
            )
            if latest_planner_result.get("coverage_summary"):
                st.write(latest_planner_result["coverage_summary"])
            if latest_planner_result.get("teacher_transcript"):
                with st.expander("Teacher Transcript", expanded=False):
                    st.write(latest_planner_result["teacher_transcript"])

        if coverage_sessions:
            st.markdown("**Recent Processed Sessions**")
            for session in coverage_sessions[:5]:
                with st.expander(
                    f"{session['session_date']} | {session.get('coverage_summary') or session.get('subject', selected_subject_name)}",
                    expanded=False,
                ):
                    st.caption(
                        f"Confidence: {round(float(session.get('confidence_score') or 0.0) * 100, 1)}% | "
                        f"Source: {session.get('source', '')}"
                    )
                    if session.get("coverage_summary"):
                        st.write(session["coverage_summary"])
                    coverage_payload = session.get("coverage", {})
                    if coverage_payload.get("covered_subtopics"):
                        st.write(f"Covered: {', '.join(coverage_payload['covered_subtopics'])}")
                    if session.get("transcript_text"):
                        st.caption("Stored Transcript")
                        st.text_area(
                            "Stored Transcript",
                            value=session["transcript_text"],
                            height=140,
                            disabled=True,
                            key=f"coverage_transcript_{session['id']}",
                            label_visibility="collapsed",
                        )

    with planner_next_tab:
        refreshed_snapshot = DEFAULT_TEACHING_PLANNER_SERVICE.get_plan_snapshot(
            class_id=class_id,
            subject=selected_subject_name,
            academic_year=selected_class.get("academic_year", ""),
        )
        upcoming_units = refreshed_snapshot.get("upcoming_units", [])
        if upcoming_units:
            st.markdown("**Recommended Upcoming Teaching Sequence**")
            for index, unit in enumerate(upcoming_units, start=1):
                st.write(
                    f"{index}. {unit['chapter_name']} | completion {unit['completion_percent']}% | "
                    f"next subtopics: {', '.join(unit.get('subtopics', [])[:4]) or 'No subtopics recorded'}"
                )
        else:
            st.info("No upcoming units found yet. Create a year plan first or all units are already marked complete.")

        if coverage_sessions:
            latest_session = coverage_sessions[0]
            latest_coverage = latest_session.get("coverage", {})
            st.markdown("**Last Class Outcome**")
            st.write(latest_session.get("coverage_summary") or "No summary recorded.")
            if latest_coverage.get("mentioned_not_taught"):
                st.write(f"Mentioned but not fully taught: {', '.join(latest_coverage['mentioned_not_taught'])}")
            if latest_coverage.get("homework_or_next_class"):
                st.write(f"Homework or next class cues: {', '.join(latest_coverage['homework_or_next_class'])}")

with tab_chat:
    st.markdown("**Chat With Gemma**")
    st.caption("Ask Gemma about this class, weak concepts, remediation, quiz design, or lesson planning.")
    st.caption(f"Agent tool-call budget per run: {MODEL_SAMPLING.max_agent_iterations}")

    chat_history_key = f"teacher_chat_history_{class_id}"
    chat_history = st.session_state.setdefault(chat_history_key, [])
    pending_agent_key = f"teacher_chat_pending_agent_{class_id}"
    pending_agent = st.session_state.get(pending_agent_key)
    pending_user_prompt_key = f"teacher_chat_pending_user_prompt_{class_id}"
    pending_user_prompt = st.session_state.pop(pending_user_prompt_key, None)
    running_key = f"teacher_chat_running_{class_id}"
    stop_requested_key = f"teacher_chat_stop_requested_{class_id}"
    is_chat_running = bool(st.session_state.get(running_key, False))

    if pending_user_prompt:
        st.session_state.pop(pending_agent_key, None)
        st.session_state[stop_requested_key] = False
        append_conversation_log("USER_INPUT", pending_user_prompt)
        chat_history.append({"role": "user", "content": pending_user_prompt})
        st.session_state[chat_history_key] = chat_history

    top_bar_left, top_bar_right = st.columns([4, 1])
    with top_bar_left:
        st.write(
            f"Current context: Grade {overview.get('grade', '')}-{overview.get('section', '')} | "
            f"{selected_subject_name}"
        )
    with top_bar_right:
        if st.button("Clear Chat", key=f"clear_teacher_chat_{class_id}"):
            st.session_state[chat_history_key] = []
            st.rerun()

    if chat_history:
        for index, message in enumerate(chat_history):
            role = "assistant" if message["role"] == "assistant" else "user"
            with st.chat_message(role):
                st.markdown(message["content"])
                reasoning_trace = message.get("reasoning_trace") or ""
                if MODEL_SAMPLING.show_reasoning and reasoning_trace:
                    with st.expander("Reasoning", expanded=False):
                        st.code(reasoning_trace)
                tool_trace = message.get("tool_trace") or []
                if tool_trace:
                    with st.expander("Tools Used", expanded=False):
                        for tool_item in tool_trace:
                            st.code(tool_item)
    else:
        st.info("No chat messages yet. Ask Gemma for a teaching strategy, a remediation plan, or a quiz idea.")

    teacher_chat_rag_context = build_retrieval_context(
        grade=selected_class["grade"],
        subject=selected_subject_name,
        query="classroom teaching strategy remediation lesson planning quiz design chapter explanation",
        top_k=MODEL_SAMPLING.rag_top_k,
    )
    teacher_chat_roster = student_repository.list_class_roster(class_id)
    teacher_chat_chapters = curriculum_repository.list_chapters_for_class(class_id, selected_subject_name)

    if pending_user_prompt:
        try:
            st.session_state[running_key] = True
            with st.status("Gemma is reviewing class context and preparing a response.", expanded=False) as chat_status:
                result = run_teacher_chat_tool_loop(
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    selected_class=selected_class,
                    overview=overview,
                    concept_gaps=concept_gaps,
                    roster=teacher_chat_roster,
                    chapters=teacher_chat_chapters,
                    history=chat_history,
                    max_agent_iterations=MODEL_SAMPLING.max_agent_iterations,
                    subject_name=selected_subject_name,
                    retrieval_context=teacher_chat_rag_context,
                    daily_brief=brief_payload,
                )
                if result.get("tool_trace"):
                    chat_status.update(
                        label=f"Gemma used {len(result['tool_trace'])} tool call(s). Streaming final answer next.",
                        state="running",
                    )
                else:
                    chat_status.update(
                        label="Gemma prepared a direct answer. Streaming final answer next.",
                        state="running",
                    )
            tool_trace = result["tool_trace"]
            reasoning_trace = ""
            if result["status"] == "ready_for_final_answer":
                with st.chat_message("assistant"):
                    streamed_text, reasoning_trace = render_streamed_teacher_final_answer(
                        llama_base_url=llama_base_url,
                        llama_model_name=llama_model_name,
                        messages=result["messages"],
                    )
                assistant_text = streamed_text or result["assistant_text"] or "Gemma returned an empty response."
            elif result["status"] == "limit_reached":
                st.session_state[pending_agent_key] = {
                    "messages": result["messages"],
                    "tool_trace": tool_trace,
                }
                assistant_text = (
                    "Gemma used the current tool-call budget and needs approval to continue. "
                    "Review the tool trace and choose whether to continue or stop."
                )
        except (RequestException, ValueError, json.JSONDecodeError) as exc:
            assistant_text = f"Could not reach llama-server: {exc}"
            reasoning_trace = ""
            tool_trace = []
        except Exception as exc:
            assistant_text = f"Chat tool loop failed: {exc}"
            reasoning_trace = ""
            tool_trace = []
        finally:
            st.session_state[running_key] = False

        chat_history.append(
            {
                "role": "assistant",
                "content": assistant_text,
                "reasoning_trace": reasoning_trace,
                "tool_trace": tool_trace,
            }
        )
        st.session_state[chat_history_key] = chat_history
        st.rerun()

    if pending_agent:
        st.warning(
            "Gemma reached the maximum continuous tool-call budget before finishing this answer. "
            f"Allow another {MODEL_SAMPLING.max_agent_iterations} tool calls or stop this agent run."
        )
        pending_col1, pending_col2 = st.columns(2)
        with pending_col1:
            if st.button(
                f"Continue For Another {MODEL_SAMPLING.max_agent_iterations} Tool Calls",
                key=f"continue_teacher_agent_{class_id}",
                type="primary",
            ):
                try:
                    st.session_state[running_key] = True
                    st.session_state[stop_requested_key] = False
                    with st.status("Gemma is continuing the previous run.", expanded=False) as chat_status:
                        result = run_teacher_chat_tool_loop(
                            llama_base_url=llama_base_url,
                            llama_model_name=llama_model_name,
                            selected_class=selected_class,
                            overview=overview,
                            concept_gaps=concept_gaps,
                            roster=teacher_chat_roster,
                            chapters=teacher_chat_chapters,
                            history=[],
                            max_agent_iterations=MODEL_SAMPLING.max_agent_iterations,
                            subject_name=selected_subject_name,
                            retrieval_context=teacher_chat_rag_context,
                            daily_brief=brief_payload,
                            base_messages=pending_agent.get("messages"),
                            prior_tool_trace=pending_agent.get("tool_trace"),
                        )
                        if result.get("tool_trace"):
                            chat_status.update(
                                label=f"Gemma used {len(result['tool_trace'])} total tool call(s). Streaming final answer next.",
                                state="running",
                            )
                    if result["status"] == "ready_for_final_answer":
                        with st.chat_message("assistant"):
                            streamed_text, reasoning_trace = render_streamed_teacher_final_answer(
                                llama_base_url=llama_base_url,
                                llama_model_name=llama_model_name,
                                messages=result["messages"],
                            )
                        chat_history.append(
                            {
                                "role": "assistant",
                                "content": streamed_text or result["assistant_text"] or "Gemma returned an empty response.",
                                "reasoning_trace": reasoning_trace,
                                "tool_trace": result["tool_trace"],
                            }
                        )
                        st.session_state.pop(pending_agent_key, None)
                    else:
                        st.session_state[pending_agent_key] = {
                            "messages": result["messages"],
                            "tool_trace": result["tool_trace"],
                        }
                    st.session_state[chat_history_key] = chat_history
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to continue agent run: {exc}")
                finally:
                    st.session_state[running_key] = False
        with pending_col2:
            if st.button("Stop Agent Run", key=f"stop_teacher_agent_{class_id}"):
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": "Agent run stopped before a final answer was produced.",
                        "reasoning_trace": "",
                        "tool_trace": pending_agent.get("tool_trace", []),
                    }
                )
                st.session_state.pop(pending_agent_key, None)
                st.session_state[chat_history_key] = chat_history
                st.rerun()

    composer_col, send_col, stop_col = st.columns([8, 1.2, 1.2])
    with composer_col:
        st.text_input(
            "Ask Gemma about this class",
            value="",
            key=f"teacher_chat_input_{class_id}",
            placeholder="Ask Gemma about this class",
            label_visibility="collapsed",
        )
        teacher_chat_audio = st.audio_input(
            "Speak to Gemma",
            key=f"teacher_chat_audio_{class_id}",
        )
        teacher_chat_files = st.file_uploader(
            "Attach PDF or image for Gemma",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=f"teacher_chat_files_{class_id}",
        )
        if teacher_chat_files:
            st.caption(
                "Attached: " + ", ".join(file.name for file in teacher_chat_files)
            )
    with send_col:
        send_prompt = st.button(
            "Send",
            key=f"teacher_chat_send_{class_id}",
            type="primary",
            use_container_width=True,
            disabled=is_chat_running,
        )
    with stop_col:
        stop_prompt = st.button(
            "Stop",
            key=f"teacher_chat_stop_inline_{class_id}",
            use_container_width=True,
            disabled=not (is_chat_running or pending_agent),
        )

    if send_prompt:
        try:
            compiled_prompt = build_teacher_chat_prompt_from_inputs(
                class_id=class_id,
                subject_name=selected_subject_name,
                uploaded_chat_files=teacher_chat_files,
                recorded_chat_audio=teacher_chat_audio,
                llama_base_url=llama_base_url,
                llama_model_name=llama_model_name,
                use_llama_server=use_llama_server,
            )
            if compiled_prompt:
                queue_teacher_chat_prompt(class_id, compiled_prompt)
                st.rerun()
            else:
                st.warning("Enter a prompt, record a voice note, or attach a PDF/image first.")
        except Exception as exc:
            st.error(f"Failed to prepare chat input: {exc}")

    if stop_prompt:
        stop_teacher_chat_run(class_id)
        chat_history.append(
            {
                "role": "assistant",
                "content": "Gemma run stopped.",
                "reasoning_trace": "",
                "tool_trace": [],
            }
        )
        st.session_state[chat_history_key] = chat_history
        st.rerun()
