from __future__ import annotations

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
from requests import RequestException

ROOT = Path(__file__).resolve().parent.parent
CONVERSATION_LOG_PATH = ROOT / "conversation.txt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import ensure_database
from app.demo_seed import ensure_demo_data
from app.assessment_sync import sync_google_form_assessment
from app.attendance import mark_attendance_from_identifiers, parse_absent_students_from_audio, resolve_absent_students
from app.gemma_adaptation_profile import generate_student_adaptation_profile_with_gemma
from app.generator import build_lesson_pack, build_quiz_questions
from app.llama_client import LlamaServerClient, LlamaServerConfig
from app.material_ingestion import ingest_reading_material
from app.model_control import choose_quiz_generation_mode, load_model_sampling_config
from app.pdf_export import build_quiz_pdf_bytes
from app.rag import build_retrieval_context, search_subject_materials
from mcp_servers.llama_MCP_bridge import cleanup as cleanup_mcp_bridge
from mcp_servers.llama_MCP_bridge import execute_tool as execute_mcp_tool
from mcp_servers.llama_MCP_bridge import get_all_mcp_tools
from mcp_servers.llama_MCP_bridge import start_servers as start_mcp_servers
from app.repository import (
    add_subject_to_class,
    add_student_to_class,
    clear_class_attendance,
    clear_student_attendance,
    create_chapter_for_class,
    create_class_for_teacher,
    create_assessment,
    deactivate_student,
    delete_chapter_if_unused,
    get_class_attendance_stats,
    get_student_adaptation_profile,
    get_student_adaptation_profile_context,
    get_class_overview,
    get_attendance_overview,
    get_curriculum_subject,
    get_student_assessment_review,
    get_student_detail,
    get_teacher,
    list_class_subjects,
    list_grade_curriculum_subjects,
    list_curriculum_chapters,
    list_inactive_class_students,
    list_assessments_for_sync,
    list_attendance_for_date,
    list_attempted_students_for_assessment,
    list_chapters_for_class,
    list_class_assessments,
    list_class_concept_gaps,
    list_class_roster,
    list_class_students,
    list_teacher_classes,
    list_queue_items,
    list_recent_ingestion_runs,
    list_subject_materials,
    reactivate_student,
    update_chapter_details,
    update_class_details,
    update_class_subject_details,
    update_student_details,
    update_student_attendance_status,
    update_assessment_google_form_info,
    upsert_student_adaptation_profile,
)


st.set_page_config(page_title="Pathshala Play", page_icon="PP", layout="wide")

ensure_database()
ensure_demo_data()
MODEL_SAMPLING = load_model_sampling_config()


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


def build_teacher_chat_messages(
    *,
    class_row: dict,
    overview: dict,
    concept_gaps: list[dict],
    history: list[dict],
    subject_name: str,
    retrieval_context: str = "",
) -> list[dict[str, str]]:
    top_concepts = concept_gaps[:5]
    concept_summary = "\n".join(
        f"- {item['concept_name']}: {item['mastery_percent']}% mastery, {item['students_lagging']} students lagging"
        for item in top_concepts
    ) or "- No concept gap data available yet."
    system_prompt = (
        "You are Gemma, acting as a classroom copilot for an Indian teacher. "
        "Give concise, practical, teacher-facing answers. "
        "Use the class context below. If you make recommendations, tie them to the subject, chapter misconceptions, "
        "assessment planning, or remediation strategy.\n\n"
        f"Class: Grade {overview.get('grade', class_row.get('grade', ''))}-{overview.get('section', class_row.get('section', ''))}\n"
        f"Subject: {subject_name}\n"
        f"Medium: {overview.get('medium', class_row.get('medium', 'Not set'))}\n"
        f"Student count: {overview.get('student_count', class_row.get('student_count', 0))}\n"
        f"Assessment count: {overview.get('assessment_count', 0)}\n"
        "Top concept gaps:\n"
        f"{concept_summary}"
    )
    if retrieval_context:
        system_prompt += f"\n\nRetrieved subject material:\n{retrieval_context}"
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
    history: list[dict],
    max_agent_iterations: int,
    subject_name: str,
    retrieval_context: str = "",
    base_messages: list[dict] | None = None,
    prior_tool_trace: list[str] | None = None,
) -> dict:
    client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
    messages = list(base_messages) if base_messages else build_teacher_chat_messages(
        class_row=selected_class,
        overview=overview,
        concept_gaps=concept_gaps,
        history=history,
        subject_name=subject_name,
        retrieval_context=retrieval_context,
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
                    "status": "completed_without_tools",
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
    history: list[dict],
    max_agent_iterations: int,
    subject_name: str,
    retrieval_context: str = "",
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
            history=history,
            max_agent_iterations=max_agent_iterations,
            subject_name=subject_name,
            retrieval_context=retrieval_context,
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


teacher = get_teacher()

st.title("Pathshala Play")
st.caption("Teacher dashboard for class diagnostics, quiz creation, and remedial planning")

if not teacher:
    st.error("No teacher record found in the database.")
    st.stop()

with st.sidebar:
    st.subheader("Teacher")
    st.write(teacher["full_name"])
    st.caption(f"{teacher['school_name']} | {teacher['google_account_email']}")

    st.subheader("Model")
    llama_base_url = st.text_input("llama-server URL", value="http://127.0.0.1:8080")
    llama_model_name = st.text_input("Model Name", value="Gemma-4-E4B-Q4_K_M")
    use_llama_server = st.toggle("Use llama-server for quiz generation", value=True)
    auto_generation_strategy, auto_generation_reason = choose_quiz_generation_mode(
        llama_model_name,
        MODEL_SAMPLING.quiz_question_generation_mode,
    )
    st.caption(
        f"Sampling from model_control.env: temp={MODEL_SAMPLING.temperature}, "
        f"top_p={MODEL_SAMPLING.top_p}, top_k={MODEL_SAMPLING.top_k}"
    )
    st.caption(f"Auto grading poll interval: {MODEL_SAMPLING.auto_grade_poll_interval_seconds}s")
    st.caption(
        "Quiz generation mode: "
        f"{'One by one' if auto_generation_strategy == 'one_by_one' else 'One shot'}"
    )
    st.caption(auto_generation_reason)

    classes = list_teacher_classes(teacher["id"])
    if not classes:
        st.warning("No classes found for this teacher.")
        st.stop()

    selected_class = st.selectbox(
        "Select Class",
        options=classes,
        format_func=format_class_label,
    )

class_id = selected_class["id"]
overview = get_class_overview(class_id)
students = list_class_students(class_id)
attendance_stats = get_class_attendance_stats(class_id)
assessments = list_class_assessments(class_id)
concept_gaps = list_class_concept_gaps(class_id)
latest_sync_result = st.session_state.get("latest_sync_result")
class_subject_options = list_class_subjects(class_id)
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

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Students", overview.get("student_count", 0))
metric_2.metric("Assessments", overview.get("assessment_count", 0))
metric_3.metric("Average Score", f"{overview.get('avg_percentage') or 0}%")
metric_4.metric("Concepts To Reteach", sum(1 for row in concept_gaps if row["mastery_percent"] < 60))

st.subheader(f"Grade {overview.get('grade', '')}-{overview.get('section', '')}")
st.caption(
    f"Academic year {overview.get('academic_year', '')} | "
    f"Subjects: {overview.get('subjects_csv') or overview.get('subject', '')} | "
    f"Active subject workspace: {selected_subject_name or 'Not set'} | "
    f"Subject medium: {current_subject_medium or 'Not set'}"
)

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

tab_overview, tab_quiz, tab_students, tab_class_gaps, tab_assessments, tab_attendance, tab_management, tab_materials, tab_chat = st.tabs(
    [
        "Class Overview",
        "Create Quiz",
        "Student Learning",
        "Class Misconceptions",
        "Assessments",
        "Attendance",
        "Class Management",
        "Reading Materials",
        "Chat With Gemma",
    ]
)

with tab_overview:
    overview_attendance_date_value = st.date_input(
        "Overview Attendance Date",
        value=datetime.now().date(),
        key=f"overview_attendance_date_{class_id}",
    )
    overview_attendance_date = format_attendance_date(overview_attendance_date_value)
    overview_attendance_rows = list_attendance_for_date(class_id, overview_attendance_date)
    attendance_by_student_id = {
        row["student_id"]: row
        for row in overview_attendance_rows
    }
    selected_overview_student_id = st.session_state.get("overview_selected_student_id")
    if selected_overview_student_id:
        student_detail = get_student_detail(selected_overview_student_id)
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
                            context = get_student_adaptation_profile_context(student["id"], selected_profile_subject or student["subject"])
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
                            upsert_student_adaptation_profile(
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
                    learner_profile = (
                        ((adaptation_profile or {}).get("summary"))
                        or (get_subject_blueprint(student_detail, selected_profile_subject) or {}).get("narrative")
                        or "Target weak concepts and reinforce understanding."
                    )
                    personalized_questions = None
                    personalized_note = "Generated using local mock logic."
                    personalized_rag_context = build_retrieval_context(
                        grade=student["grade"],
                        subject=selected_profile_subject or student["subject"],
                        query=f"{topic} {learner_profile}",
                        top_k=MODEL_SAMPLING.rag_top_k,
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
                                source_material=(
                                    "Generate a personalized remedial quiz from this student's strengths, "
                                    "weaknesses, and misconceptions.\n\n"
                                    f"Retrieved subject material:\n{personalized_rag_context}"
                                ),
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

                        student_chapters = list_chapters_for_class(class_id)
                        chapter_id = student_chapters[0]["id"] if student_chapters else preview["chapter_id"] if "preview" in locals() else 1
                        assessment_id = create_assessment(
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
                        update_assessment_google_form_info(
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
            st.markdown("**Reteach Priorities**")
            if concept_gaps:
                for concept in concept_gaps[:3]:
                    st.write(
                        f"- {concept['concept_name']}: {concept['mastery_percent']}% mastery, "
                        f"{concept['students_lagging']} students lagging"
                    )
            else:
                st.info("No class mastery data yet.")

            st.markdown("**Teacher Actions**")
            st.write("- Create a quiz from the next chapter or reteach concept.")
            st.write("- Review lagging students before the next lesson.")
            st.write("- Use misconception trends to decide tomorrow's board explanation.")

with tab_quiz:
    st.markdown("**Draft a quiz for the selected class**")

    chapter_options = list_chapters_for_class(class_id, selected_subject_name)
    if not chapter_options:
        st.warning("No chapters found for the selected subject and grade.")
    else:
        quiz_col1, quiz_col2 = st.columns(2)
        with quiz_col1:
            selected_chapter = st.selectbox(
                "Chapter",
                options=chapter_options,
                format_func=lambda chapter: chapter["chapter_name"],
                key="quiz_chapter",
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
            concept_names = [row["concept_name"] for row in concept_gaps] or [selected_chapter["chapter_name"]]
            lesson_pack = build_lesson_pack(
                request=type(
                    "LessonRequestProxy",
                    (),
                    {
                        "subject": selected_subject_name,
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
                subject=selected_subject_name,
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
                        subject=selected_subject_name,
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
                "subject": selected_subject_name,
                "retrieval_context": rag_context,
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
                    assessment_id = create_assessment(
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
                            assessment_id = create_assessment(
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
                        update_assessment_google_form_info(
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
        student_detail = get_student_detail(selected_student["id"])
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
    syncable_assessments = list_assessments_for_sync(class_id)
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

        attempted_students = list_attempted_students_for_assessment(selected_sync_assessment["id"])
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
            review = get_student_assessment_review(
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
    queue_items = list_queue_items(limit=10)
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
    attendance_date_value = st.date_input(
        "Attendance Date",
        value=datetime.now().date(),
        key=f"attendance_date_{class_id}",
    )
    attendance_date = format_attendance_date(attendance_date_value)
    roster = list_class_roster(class_id)
    attendance_overview = get_attendance_overview(class_id, attendance_date)
    attendance_rows = list_attendance_for_date(class_id, attendance_date)

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
                update_student_attendance_status(
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
                    "present_count": get_attendance_overview(class_id, attendance_date)["present_count"],
                    "absent_count": get_attendance_overview(class_id, attendance_date)["absent_count"],
                }
                st.rerun()
    with edit_col2:
        clear_student_confirm = st.checkbox(
            "Confirm clearing this student's full attendance history",
            key=f"attendance_clear_student_confirm_{class_id}",
        )
        if st.button("Clear Selected Student Attendance", key=f"attendance_clear_student_{class_id}"):
            if selected_attendance_student and clear_student_confirm:
                clear_student_attendance(selected_attendance_student["id"], class_id)
                st.session_state.pop(attendance_result_key, None)
                st.rerun()
        clear_class_confirm = st.checkbox(
            "Confirm clearing attendance of all students in this class",
            key=f"attendance_clear_class_confirm_{class_id}",
        )
        if st.button("Clear All Class Attendance", key=f"attendance_clear_class_{class_id}"):
            if clear_class_confirm:
                clear_class_attendance(class_id)
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
                st.session_state.pop(attendance_proposal_key, None)
                st.rerun()
        with discard_col:
            if st.button("Discard Detection", key=f"attendance_discard_{class_id}"):
                st.session_state.pop(attendance_proposal_key, None)
                st.rerun()

    latest_attendance_result = st.session_state.get(attendance_result_key)
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
    current_rows = list_attendance_for_date(class_id, attendance_date)
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
    class_subjects = list_class_subjects(managed_class_id)
    selected_management_subject = st.selectbox(
        "Subject Workspace",
        options=class_subjects,
        format_func=lambda item: item["subject"],
        key=f"management_workspace_{teacher['id']}",
    ) if class_subjects else None
    management_roster = list_class_roster(managed_class_id)
    inactive_students = list_inactive_class_students(managed_class_id)
    current_management_chapters = list_chapters_for_class(
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
                add_subject_to_class(class_id=managed_class_id, subject=new_subject, medium=new_medium)
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
                    update_class_subject_details(
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
                    add_student_to_class(
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
                selected_edit_student_detail = get_student_detail(selected_edit_student["id"]).get("student") if selected_edit_student else None
                with st.form(f"edit_student_form_{managed_class_id}_{selected_edit_student['id']}"):
                    edit_student_roll = st.text_input("Roll Number", value=selected_edit_student_detail["roll_number"] if selected_edit_student_detail else "")
                    edit_student_name = st.text_input("Student Name", value=selected_edit_student_detail["full_name"] if selected_edit_student_detail else "")
                    edit_student_email = st.text_input("Student Email", value=selected_edit_student_detail.get("email", "") if selected_edit_student_detail else "")
                    edit_student_language = st.text_input("Preferred Language", value=selected_edit_student_detail.get("preferred_language", "") if selected_edit_student_detail else "")
                    edit_student_accessibility = st.text_area("Accessibility Notes", value=selected_edit_student_detail.get("accessibility_notes", "") if selected_edit_student_detail else "", height=80)
                    edit_student_submitted = st.form_submit_button("Save Student Changes")
                if edit_student_submitted:
                    try:
                        update_student_details(
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
                            deactivate_student(selected_manage_student["id"])
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
                        reactivate_student(selected_inactive_student["id"])
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
                    create_chapter_for_class(
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
                        update_chapter_details(
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
                        deleted, message = delete_chapter_if_unused(selected_delete_chapter["id"])
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
    grade_subjects = list_grade_curriculum_subjects(selected_class["grade"])
    current_curriculum_subject = get_curriculum_subject(grade=selected_class["grade"], subject=selected_subject_name)
    current_curriculum_chapters = (
        list_curriculum_chapters(current_curriculum_subject["id"])
        if current_curriculum_subject
        else []
    )
    materials = list_subject_materials(grade=selected_class["grade"], subject=selected_subject_name) if selected_subject_name else []
    ingestion_runs = list_recent_ingestion_runs(limit=10)

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

with tab_chat:
    st.markdown("**Teacher Chat**")
    st.caption("Ask Gemma about this class, weak concepts, remediation, quiz design, or lesson planning.")
    st.caption(f"Agent tool-call budget per run: {MODEL_SAMPLING.max_agent_iterations}")

    chat_history_key = f"teacher_chat_history_{class_id}"
    chat_history = st.session_state.setdefault(chat_history_key, [])
    pending_agent_key = f"teacher_chat_pending_agent_{class_id}"
    pending_agent = st.session_state.get(pending_agent_key)
    pending_user_prompt_key = f"teacher_chat_pending_user_prompt_{class_id}"
    pending_user_prompt = st.session_state.pop(pending_user_prompt_key, None)

    if pending_user_prompt:
        st.session_state.pop(pending_agent_key, None)
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

    if pending_user_prompt:
        try:
            result = run_teacher_chat_tool_loop(
                llama_base_url=llama_base_url,
                llama_model_name=llama_model_name,
                selected_class=selected_class,
                overview=overview,
                concept_gaps=concept_gaps,
                history=chat_history,
                max_agent_iterations=MODEL_SAMPLING.max_agent_iterations,
                subject_name=selected_subject_name,
                retrieval_context=teacher_chat_rag_context,
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
            elif result["status"] == "completed_without_tools":
                assistant_text = result["assistant_text"] or "Gemma returned an empty response."
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
                    result = run_teacher_chat_tool_loop(
                        llama_base_url=llama_base_url,
                        llama_model_name=llama_model_name,
                        selected_class=selected_class,
                        overview=overview,
                    concept_gaps=concept_gaps,
                    history=[],
                    max_agent_iterations=MODEL_SAMPLING.max_agent_iterations,
                    subject_name=selected_subject_name,
                    retrieval_context=teacher_chat_rag_context,
                    base_messages=pending_agent.get("messages"),
                    prior_tool_trace=pending_agent.get("tool_trace"),
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
                    elif result["status"] == "completed_without_tools":
                        chat_history.append(
                            {
                                "role": "assistant",
                                "content": result["assistant_text"] or "Gemma returned an empty response.",
                                "reasoning_trace": "",
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

    with st.form(key=f"teacher_chat_form_{class_id}", clear_on_submit=True):
        user_prompt = st.text_input(
            "Ask Gemma about this class",
            value="",
            key=f"teacher_chat_input_{class_id}",
            placeholder="Ask Gemma about this class",
            label_visibility="collapsed",
        )
        send_prompt = st.form_submit_button("Send", type="primary")

    if send_prompt and user_prompt.strip():
        st.session_state[pending_user_prompt_key] = user_prompt.strip()
        st.rerun()
