from __future__ import annotations

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
from requests import RequestException

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import ensure_database
from app.demo_seed import ensure_demo_data
from app.assessment_sync import sync_google_form_assessment
from app.generator import build_lesson_pack, build_quiz_questions
from app.llama_client import LlamaServerClient, LlamaServerConfig
from app.model_control import choose_quiz_generation_mode, load_model_sampling_config
from mcp_servers.llama_MCP_bridge import cleanup as cleanup_mcp_bridge
from mcp_servers.llama_MCP_bridge import execute_tool as execute_mcp_tool
from mcp_servers.llama_MCP_bridge import get_all_mcp_tools
from mcp_servers.llama_MCP_bridge import start_servers as start_mcp_servers
from app.repository import (
    create_assessment,
    get_class_overview,
    get_student_assessment_review,
    get_student_detail,
    get_teacher,
    list_assessments_for_sync,
    list_attempted_students_for_assessment,
    list_chapters_for_class,
    list_class_assessments,
    list_class_concept_gaps,
    list_class_students,
    list_teacher_classes,
    list_queue_items,
    update_assessment_google_form_info,
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
    return (
        f"Grade {class_row['grade']}-{class_row['section']} | "
        f"{class_row['subject']} | {class_row['student_count']} students"
    )


def build_student_quiz_topic(student_detail: dict, subject: str | None = None) -> str:
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


def get_available_student_subjects(student_detail: dict) -> list[str]:
    subjects = [item.get("subject") for item in (student_detail.get("blueprints") or []) if item.get("subject")]
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


def build_teacher_chat_messages(
    *,
    class_row: dict,
    overview: dict,
    concept_gaps: list[dict],
    history: list[dict],
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
        f"Subject: {overview.get('subject', class_row.get('subject', ''))}\n"
        f"Medium: {overview.get('medium', class_row.get('medium', 'Not set'))}\n"
        f"Student count: {overview.get('student_count', class_row.get('student_count', 0))}\n"
        f"Assessment count: {overview.get('assessment_count', 0)}\n"
        "Top concept gaps:\n"
        f"{concept_summary}"
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
    history: list[dict],
    max_agent_iterations: int,
    base_messages: list[dict] | None = None,
    prior_tool_trace: list[str] | None = None,
) -> dict:
    client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
    messages = list(base_messages) if base_messages else build_teacher_chat_messages(
        class_row=selected_class,
        overview=overview,
        concept_gaps=concept_gaps,
        history=history,
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
assessments = list_class_assessments(class_id)
concept_gaps = list_class_concept_gaps(class_id)
latest_sync_result = st.session_state.get("latest_sync_result")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Students", overview.get("student_count", 0))
metric_2.metric("Assessments", overview.get("assessment_count", 0))
metric_3.metric("Average Score", f"{overview.get('avg_percentage') or 0}%")
metric_4.metric("Concepts To Reteach", sum(1 for row in concept_gaps if row["mastery_percent"] < 60))

st.subheader(f"Grade {overview.get('grade', '')}-{overview.get('section', '')} {overview.get('subject', '')}")
st.caption(f"Academic year {overview.get('academic_year', '')} | Medium: {overview.get('medium', 'Not set')}")

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

tab_overview, tab_quiz, tab_students, tab_class_gaps, tab_assessments, tab_chat = st.tabs(
    [
        "Class Overview",
        "Create Quiz",
        "Student Learning",
        "Class Misconceptions",
        "Assessments",
        "Chat With Gemma",
    ]
)

with tab_overview:
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

            profile_col1, profile_col2 = st.columns([1.1, 1])
            with profile_col1:
                st.write(f"Preferred language: {student['preferred_language'] or 'Not set'}")
                st.write(f"Accessibility notes: {student['accessibility_notes'] or 'None'}")
                st.markdown("**Concept Mastery**")
                st.dataframe(student_detail["mastery"], use_container_width=True, hide_index=True)
                st.markdown("**Assessment History**")
                st.dataframe(student_detail["assessments"], use_container_width=True, hide_index=True)

            with profile_col2:
                available_subjects = get_available_student_subjects(student_detail)
                selected_profile_subject = st.selectbox(
                    "Subject",
                    options=available_subjects,
                    index=available_subjects.index(student["subject"]) if student["subject"] in available_subjects else 0,
                    key=f"overview_subject_{student['id']}",
                ) if available_subjects else student["subject"]
                blueprint = get_subject_blueprint(student_detail, selected_profile_subject)
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
                        (get_subject_blueprint(student_detail, selected_profile_subject) or {}).get("narrative")
                        or "Target weak concepts and reinforce understanding."
                    )
                    personalized_questions = None
                    personalized_note = "Generated using local mock logic."
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
                                    "weaknesses, and misconceptions."
                                ),
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
            for student in students:
                row_col1, row_col2, row_col3, row_col4 = st.columns([2.2, 1, 1, 1])
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

    chapter_options = list_chapters_for_class(class_id)
    if not chapter_options:
        st.warning("No chapters found for this class subject and grade.")
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
                        "subject": selected_class["subject"],
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

            if use_llama_server:
                client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
                try:
                    questions, generation_note, raw_llama_outputs = generate_quiz_with_llama(
                        client=client,
                        base_url=llama_base_url,
                        model_name=llama_model_name,
                        generation_strategy=auto_generation_strategy,
                        subject=selected_class["subject"],
                        grade=selected_class["grade"],
                        chapter_name=selected_chapter["chapter_name"],
                        learner_profile=learner_profile,
                        source_material=source_material,
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
                "generation_mode": generation_mode,
                "generation_note": generation_note,
                "generation_strategy": auto_generation_strategy,
                "raw_llama_outputs": raw_llama_outputs,
                "due_at": general_due_at,
                "topic": selected_chapter["chapter_name"],
                "subject": selected_class["subject"],
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

            if st.button("Save Quiz Draft To Database"):
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

            if st.button("Create Google Form Draft"):
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
                        subject=preview.get("subject", selected_class["subject"]),
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
        chat_history.append({"role": "user", "content": pending_user_prompt})
        st.session_state[chat_history_key] = chat_history

    top_bar_left, top_bar_right = st.columns([4, 1])
    with top_bar_left:
        st.write(
            f"Current context: Grade {overview.get('grade', '')}-{overview.get('section', '')} | "
            f"{overview.get('subject', selected_class['subject'])}"
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
