from __future__ import annotations

import json

from app.llama_client import LlamaServerClient
from app.model_control import load_model_sampling_config


MODEL_SAMPLING = load_model_sampling_config()


def infer_question_type_constraint(
    *,
    learner_profile: str,
    source_material: str,
    teacher_instructions: str = "",
) -> str | None:
    instruction_text = f"{learner_profile}\n{source_material}\n{teacher_instructions}".lower()
    short_answer_signals = [
        "no mcq",
        "no mcqs",
        "no multiple choice",
        "no multiple-choice",
        "all short answer",
        "all short answers",
        "all descriptive",
        "all detailed question answer",
        "all detailed questions",
        "only short answer",
        "only short answers",
        "only descriptive",
        "detailed question answer",
        "detailed questions answers",
    ]
    mcq_signals = [
        "all mcq",
        "all mcqs",
        "only mcq",
        "only mcqs",
        "only multiple choice",
        "only multiple-choice",
        "all multiple choice",
        "all multiple-choice",
    ]
    if any(signal in instruction_text for signal in short_answer_signals):
        return "short_answer"
    if any(signal in instruction_text for signal in mcq_signals):
        return "mcq"
    return None


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
    question_type_constraint = infer_question_type_constraint(
        learner_profile=learner_profile,
        source_material=source_material,
        teacher_instructions=teacher_instructions,
    )
    question_type_requirement = (
        "- Every question must use question_type short_answer. Do not generate any mcq questions.\n"
        if question_type_constraint == "short_answer"
        else "- Every question must use question_type mcq. Do not generate any short_answer questions.\n"
        if question_type_constraint == "mcq"
        else "- Use only these question_type values: mcq, short_answer.\n"
    )
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
                "Strictly follow the teacher instruction on whether to use mcq or short_answer question types or mix of both if no clear instruction is given. "
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
                f"- Create exactly {question_count} questions.\n"
                f"strictly follow the teacher instruction on whether to use mcq or short_answer question types or mix of both if no clear instruction is given.\n"
                f"Source material: {source_material}\n\n"
                "Requirements:\n"
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
    question_type_constraint = infer_question_type_constraint(
        learner_profile=learner_profile,
        source_material=source_material,
        teacher_instructions=teacher_instructions,
    )
    question_type_requirement = (
        "- This question must use question_type short_answer. Do not generate any mcq question.\n"
        if question_type_constraint == "short_answer"
        else "- This question must use question_type mcq. Do not generate any short_answer question.\n"
        if question_type_constraint == "mcq"
        else "- Use only these question_type values: mcq, short_answer.\n"
    )
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
                f"{question_type_requirement}"
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


def apply_question_type_constraint(questions: list[dict], question_type_constraint: str | None) -> list[dict]:
    if question_type_constraint not in {"mcq", "short_answer"}:
        return questions
    constrained_questions: list[dict] = []
    for question in questions:
        constrained_question = dict(question)
        constrained_question["question_type"] = question_type_constraint
        if question_type_constraint == "short_answer":
            constrained_question["options"] = {}
        else:
            options = dict(constrained_question.get("options", {}))
            constrained_question["options"] = {
                "A": str(options.get("A", "")).strip(),
                "B": str(options.get("B", "")).strip(),
                "C": str(options.get("C", "")).strip(),
                "D": str(options.get("D", "")).strip(),
            }
        constrained_questions.append(constrained_question)
    return constrained_questions


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
    question_type_constraint = infer_question_type_constraint(
        learner_profile=learner_profile,
        source_material=source_material,
        teacher_instructions=teacher_instructions,
    )
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
            parsed_question = apply_question_type_constraint([parsed_question], question_type_constraint)[0]
            if parsed_question["question_type"] == "mcq" and not all(parsed_question["options"].values()):
                return None, (
                    f"llama-server generated question {question_number}, but it did not satisfy the required MCQ option structure."
                ), raw_outputs
            questions.append(parsed_question)
        return questions, (
            f"Generated via llama-server at {base_url} using model {model_name} in one-by-one mode."
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
    parsed_questions = apply_question_type_constraint(parsed_questions, question_type_constraint)
    if question_type_constraint == "mcq" and any(not all(question["options"].values()) for question in parsed_questions):
        return None, "llama-server returned MCQ questions without four valid options.", raw_outputs
    return parsed_questions, (
        f"Generated via llama-server at {base_url} using model {model_name} in one-shot mode."
    ), raw_outputs
