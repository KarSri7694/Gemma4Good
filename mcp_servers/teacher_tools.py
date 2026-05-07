from __future__ import annotations

import os
import sys
from pathlib import Path
from math import prod

from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llama_client import LlamaServerClient, LlamaServerConfig
from app.model_control import load_model_sampling_config


mcp = FastMCP("Science Teacher Tools")


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


if __name__ == "__main__":
    mcp.run(transport="stdio")
