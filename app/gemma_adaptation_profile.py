from __future__ import annotations

import json
from typing import Any

from app.llama_client import LlamaServerClient


class StudentAdaptationProfileGenerator:
    def generate_student_adaptation_profile_with_gemma(
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
            messages=self._build_adaptation_messages(student_context),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            response_format={"type": "json_object"},
            extra_payload={"model": model_name},
        )
        parsed = self._parse_adaptation_response(response)
        if parsed:
            return parsed
        return self._fallback_adaptation_profile(student_context)

    def _build_adaptation_messages(self, student_context: dict[str, Any]) -> list[dict[str, str]]:
        schema_hint = {
            "support_preferences": {
                "preferred_language": "string",
                "accessibility_support": ["string"],
                "pace_support": "string",
                "explanation_style": ["string"],
                "response_support": ["string"],
            },
            "priority_targets": ["string"],
            "misconception_map": [
                {
                    "concept": "string",
                    "issue": "string",
                    "evidence": ["string"],
                }
            ],
            "response_style": {
                "best_formats": ["string"],
                "needs_more_support_in": ["string"],
                "confidence_signal": "string",
                "pacing_signal": "string",
            },
            "recommended_interventions": ["string"],
            "summary": "string",
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are an educational remediation analyst for Indian classrooms. "
                    "Create a structured student adaptation profile for a remedial tutoring system. "
                    "Use only the evidence in the provided context. "
                    "Do not infer personality traits. "
                    "Return strict JSON only with this exact shape: "
                    f"{json.dumps(schema_hint)}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(student_context, ensure_ascii=False),
            },
        ]

    def _parse_adaptation_response(self, response: dict[str, Any]) -> dict[str, Any] | None:
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

        support_preferences = parsed.get("support_preferences", {}) if isinstance(parsed.get("support_preferences"), dict) else {}
        response_style = parsed.get("response_style", {}) if isinstance(parsed.get("response_style"), dict) else {}
        misconception_map = parsed.get("misconception_map", [])
        normalized_misconceptions = []
        if isinstance(misconception_map, list):
            for item in misconception_map:
                if not isinstance(item, dict):
                    continue
                concept = str(item.get("concept", "")).strip()
                issue = str(item.get("issue", "")).strip()
                evidence = self._ensure_list(item.get("evidence"))
                if concept or issue:
                    normalized_misconceptions.append(
                        {"concept": concept, "issue": issue, "evidence": evidence}
                    )

        return {
            "support_preferences": {
                "preferred_language": str(support_preferences.get("preferred_language", "")).strip(),
                "accessibility_support": self._ensure_list(support_preferences.get("accessibility_support")),
                "pace_support": str(support_preferences.get("pace_support", "")).strip(),
                "explanation_style": self._ensure_list(support_preferences.get("explanation_style")),
                "response_support": self._ensure_list(support_preferences.get("response_support")),
            },
            "priority_targets": self._ensure_list(parsed.get("priority_targets")),
            "misconception_map": normalized_misconceptions,
            "response_style": {
                "best_formats": self._ensure_list(response_style.get("best_formats")),
                "needs_more_support_in": self._ensure_list(response_style.get("needs_more_support_in")),
                "confidence_signal": str(response_style.get("confidence_signal", "")).strip(),
                "pacing_signal": str(response_style.get("pacing_signal", "")).strip(),
            },
            "recommended_interventions": self._ensure_list(parsed.get("recommended_interventions")),
            "summary": str(parsed.get("summary", "")).strip(),
        }

    def _ensure_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _fallback_adaptation_profile(self, student_context: dict[str, Any]) -> dict[str, Any]:
        support_preferences = student_context.get("support_preferences", {})
        lagging = student_context.get("lagging_concepts", [])
        answer_formats = student_context.get("answer_format_summary", {})
        return {
            "support_preferences": {
                "preferred_language": support_preferences.get("preferred_language", ""),
                "accessibility_support": support_preferences.get("accessibility_support", []),
                "pace_support": support_preferences.get("pace_support", "Needs more evidence."),
                "explanation_style": support_preferences.get("explanation_style", ["simple explanation", "worked examples"]),
                "response_support": support_preferences.get("response_support", ["step-by-step prompting"]),
            },
            "priority_targets": lagging[:5] or ["Needs more concept evidence."],
            "misconception_map": [],
            "response_style": {
                "best_formats": answer_formats.get("best_formats", []),
                "needs_more_support_in": answer_formats.get("needs_more_support_in", []),
                "confidence_signal": "Fallback profile generated from local analytics.",
                "pacing_signal": "Use gradual remedial progression.",
            },
            "recommended_interventions": [
                "Focus on one weak concept at a time.",
                "Use short practice followed by immediate correction.",
            ],
            "summary": "Fallback adaptation profile generated from local analytics because structured Gemma output was unavailable.",
        }


DEFAULT_ADAPTATION_PROFILE_GENERATOR = StudentAdaptationProfileGenerator()


def generate_student_adaptation_profile_with_gemma(
    *,
    client: LlamaServerClient,
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    student_context: dict[str, Any],
) -> dict[str, Any]:
    return DEFAULT_ADAPTATION_PROFILE_GENERATOR.generate_student_adaptation_profile_with_gemma(
        client=client,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        student_context=student_context,
    )
