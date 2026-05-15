from __future__ import annotations

import json
from typing import Any

from app.llama_client import LlamaServerClient


class StudentBlueprintGenerator:
    def generate_student_blueprint_with_gemma(
        self,
        *,
        client: LlamaServerClient,
        model_name: str,
        temperature: float,
        top_p: float,
        top_k: int,
        student_context: dict[str, Any],
    ) -> dict[str, Any]:
        response = client.chat_completion(
            messages=self._build_blueprint_messages(student_context),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_format={"type": "json_object"},
            extra_payload={"model": model_name},
        )
        parsed = self._parse_blueprint_response(response)
        if parsed:
            return parsed
        return self._fallback_blueprint(student_context)

    def _build_blueprint_messages(self, student_context: dict[str, Any]) -> list[dict[str, str]]:
        schema_hint = {
            "strengths": ["string"],
            "weaknesses": ["string"],
            "opportunities": ["string"],
            "threats": ["string"],
            "recommendations": ["string"],
            "narrative": "string",
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are an educational analyst for Indian classrooms. "
                    "Create a cumulative SWOT-style student blueprint from assessment history. "
                    "Return strict JSON only with this exact shape: "
                    f"{json.dumps(schema_hint)}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(student_context, ensure_ascii=False),
            },
        ]

    def _parse_blueprint_response(self, response: dict[str, Any]) -> dict[str, Any] | None:
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

        return {
            "strengths": self._ensure_list(parsed.get("strengths")),
            "weaknesses": self._ensure_list(parsed.get("weaknesses")),
            "opportunities": self._ensure_list(parsed.get("opportunities")),
            "threats": self._ensure_list(parsed.get("threats")),
            "recommendations": self._ensure_list(parsed.get("recommendations")),
            "narrative": str(parsed.get("narrative", "")).strip(),
        }

    def _ensure_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _fallback_blueprint(self, student_context: dict[str, Any]) -> dict[str, Any]:
        weak_concepts = student_context.get("lagging_concepts", [])
        strong_concepts = student_context.get("strong_concepts", [])
        return {
            "strengths": strong_concepts[:3] or ["Has at least one demonstrated area of understanding."],
            "weaknesses": weak_concepts[:3] or ["Needs more evidence across recent assessments."],
            "opportunities": [
                "Use targeted remedial quizzes after each chapter.",
                "Reinforce stronger concepts through peer teaching.",
            ],
            "threats": [
                "Recurring misconceptions may compound if not corrected early.",
                "Low confidence can reduce performance on later written responses.",
            ],
            "recommendations": [
                "Review one weak concept at a time with worked examples.",
                "Track progress after each new form submission.",
            ],
            "narrative": "Fallback blueprint generated from local analytics because structured Gemma output was unavailable.",
        }


DEFAULT_BLUEPRINT_GENERATOR = StudentBlueprintGenerator()


def generate_student_blueprint_with_gemma(
    *,
    client: LlamaServerClient,
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    student_context: dict[str, Any],
) -> dict[str, Any]:
    return DEFAULT_BLUEPRINT_GENERATOR.generate_student_blueprint_with_gemma(
        client=client,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        student_context=student_context,
    )
