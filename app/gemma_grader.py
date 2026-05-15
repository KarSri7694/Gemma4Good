from __future__ import annotations

import json
from typing import Any

from app.llama_client import LlamaServerClient


class GemmaGrader:
    def grade_answer_with_gemma(
        self,
        *,
        client: LlamaServerClient,
        model_name: str,
        temperature: float,
        top_p: float,
        top_k: int,
        question: dict[str, Any],
        student_answer: str,
    ) -> dict[str, Any]:
        response = client.chat_completion(
            messages=self._build_grading_messages(question=question, student_answer=student_answer),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_format={"type": "json_object"},
            extra_payload={"model": model_name},
        )
        parsed = self._parse_grading_response(response)
        if parsed:
            return parsed
        return self._fallback_grade(question=question, student_answer=student_answer)

    def _build_grading_messages(self, *, question: dict[str, Any], student_answer: str) -> list[dict[str, str]]:
        schema_hint = {
            "score_awarded": 2,
            "is_correct": True,
            "feedback": "string",
            "reasoning": "string",
            "error_type": "concept_misunderstanding or incomplete_reasoning or careless_mistake or language_issue or none",
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict but fair Indian school teacher. "
                    "Grade exactly one student answer. "
                    "Return strict JSON only with this exact shape: "
                    f"{json.dumps(schema_hint)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question['question_text']}\n"
                    f"Question type: {question['question_type']}\n"
                    f"Maximum marks: {question['marks']}\n"
                    f"Correct answer or rubric: {question.get('correct_answer', '')}\n"
                    f"Teacher explanation: {question.get('explanation', '')}\n"
                    f"Student answer: {student_answer}\n\n"
                    "Requirements:\n"
                    "- Award a numeric score between 0 and maximum marks.\n"
                    "- Set is_correct true only if the core concept is correct.\n"
                    "- Keep feedback short and specific.\n"
                    "- reasoning should explain briefly why this score was given.\n"
                    "- Use error_type 'none' when the answer is correct.\n"
                    "- Output JSON only."
                ),
            },
        ]

    def _parse_grading_response(self, response: dict[str, Any]) -> dict[str, Any] | None:
        content = ""
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content", "") or choices[0].get("text", "")
        elif "content" in response:
            content = response.get("content", "")
        elif "text" in response:
            content = response.get("text", "")

        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                parsed = json.loads(content[start : end + 1].replace(",}", "}").replace(",]", "]"))
            except json.JSONDecodeError:
                return None

        score_awarded = float(parsed.get("score_awarded", 0))
        is_correct = bool(parsed.get("is_correct", False))
        feedback = str(parsed.get("feedback", "")).strip()
        reasoning = str(parsed.get("reasoning", "")).strip()
        error_type = str(parsed.get("error_type", "none")).strip().lower()
        return {
            "score_awarded": score_awarded,
            "is_correct": 1 if is_correct else 0,
            "feedback": feedback,
            "grading_reasoning": reasoning,
            "error_type": None if error_type == "none" else error_type,
        }

    def _fallback_grade(self, *, question: dict[str, Any], student_answer: str) -> dict[str, Any]:
        normalized_expected = str(question.get("correct_answer", "")).strip().lower()
        normalized_answer = student_answer.strip().lower()
        is_correct = normalized_answer == normalized_expected and bool(normalized_answer)
        return {
            "score_awarded": float(question["marks"]) if is_correct else 0.0,
            "is_correct": 1 if is_correct else 0,
            "feedback": "Auto fallback grading applied.",
            "grading_reasoning": "Fallback exact-match grading was used because structured Gemma grading was unavailable.",
            "error_type": None if is_correct else "concept_misunderstanding",
        }


DEFAULT_GEMMA_GRADER = GemmaGrader()


def grade_answer_with_gemma(
    *,
    client: LlamaServerClient,
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    question: dict[str, Any],
    student_answer: str,
) -> dict[str, Any]:
    return DEFAULT_GEMMA_GRADER.grade_answer_with_gemma(
        client=client,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        question=question,
        student_answer=student_answer,
    )
