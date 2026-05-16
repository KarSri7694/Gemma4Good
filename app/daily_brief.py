from __future__ import annotations

"""Daily Gemma analysis generation and cache management.

The dashboard uses this service to show a saved analysis immediately on startup
and refresh it with a model-backed summary only when needed.
"""

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any

from requests import RequestException

from app.llama_client import LlamaServerClient, LlamaServerConfig
from app.teaching_progress import DEFAULT_TEACHING_PLANNER_SERVICE

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
DAILY_BRIEF_CACHE_PATH = CACHE_DIR / "daily_loop_briefs.json"


@dataclass
class DailyLoopService:
    def _extract_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            return str(
                message.get("content")
                or choices[0].get("content")
                or choices[0].get("text")
                or response.get("content")
                or ""
            ).strip()
        return str(response.get("content") or response.get("text") or "").strip()

    def _parse_json_content(self, content: str) -> dict[str, Any] | None:
        if not content:
            return None
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None

    def _normalize_list_field(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                text = str(item or "").strip()
                if text:
                    items.append(text)
            return items
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if "\n" in text:
                parts = [part.strip(" -0123456789.\t") for part in text.splitlines()]
                cleaned = [part.strip() for part in parts if part.strip()]
                if cleaned:
                    return cleaned
            if "," in text:
                parts = [part.strip() for part in text.split(",") if part.strip()]
                if len(parts) > 1:
                    return parts
            return [text]
        return [str(value).strip()] if str(value).strip() else []

    def _normalize_brief_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Force model output into the list/string structure expected by the UI."""
        normalized = dict(payload or {})
        normalized["daily_summary"] = str(normalized.get("daily_summary") or "").strip()
        normalized["next_concept_to_teach"] = str(normalized.get("next_concept_to_teach") or "").strip()
        normalized["reteach_concepts"] = self._normalize_list_field(normalized.get("reteach_concepts"))
        normalized["speed_up_areas"] = self._normalize_list_field(normalized.get("speed_up_areas"))
        normalized["students_to_watch"] = self._normalize_list_field(normalized.get("students_to_watch"))
        normalized["teacher_actions"] = self._normalize_list_field(normalized.get("teacher_actions"))
        return normalized

    def build_daily_context(
        self,
        *,
        selected_class: dict,
        overview: dict,
        subject_name: str,
        students: list[dict],
        attendance_rows: list[dict],
        concept_gaps: list[dict],
        assessments: list[dict],
        coverage_sessions: list[dict],
        plan_snapshot: dict[str, Any],
        for_date: str,
    ) -> dict[str, Any]:
        """Assemble a compact classroom snapshot for daily analysis."""
        attendance_by_student_id = {int(item["student_id"]): item for item in attendance_rows if item.get("student_id") is not None}
        absent_students = [
            {
                "student_id": int(student["id"]),
                "full_name": student["full_name"],
                "roll_number": student["roll_number"],
            }
            for student in students
            if attendance_by_student_id.get(int(student["id"]), {}).get("status") == "absent"
        ]
        support_students = sorted(
            students,
            key=lambda item: (
                -(item.get("lagging_concepts") or 0),
                item["avg_percentage"] if item.get("avg_percentage") is not None else 100,
            ),
        )[:5]
        latest_assessments = assessments[:3]
        latest_coverage = next(
            (item for item in coverage_sessions if str(item.get("session_date", "")) == for_date),
            coverage_sessions[0] if coverage_sessions else None,
        )
        return {
            "date": for_date,
            "class": {
                "grade": overview.get("grade", selected_class.get("grade", "")),
                "section": overview.get("section", selected_class.get("section", "")),
                "subject": subject_name,
                "student_count": overview.get("student_count", 0),
                "assessment_count": overview.get("assessment_count", 0),
                "avg_percentage": overview.get("avg_percentage"),
            },
            "attendance": {
                "present_count": sum(1 for item in attendance_rows if item.get("status") == "present"),
                "absent_count": sum(1 for item in attendance_rows if item.get("status") == "absent"),
                "absent_students": absent_students,
            },
            "coverage": latest_coverage or {},
            "plan": {
                "completion_percent": plan_snapshot.get("completion_percent", 0.0),
                "upcoming_units": plan_snapshot.get("upcoming_units", []),
            },
            "concept_gaps": concept_gaps[:5],
            "recent_assessments": latest_assessments,
            "support_students": [
                {
                    "full_name": item["full_name"],
                    "avg_percentage": item.get("avg_percentage"),
                    "lagging_concepts": item.get("lagging_concepts", 0),
                    "attendance_percentage": item.get("attendance_percentage"),
                }
                for item in support_students
            ],
        }

    def _fallback_brief(self, context: dict[str, Any]) -> dict[str, Any]:
        """Deterministic summary used when a model call is skipped or fails."""
        concept_gaps = context.get("concept_gaps", [])
        upcoming_units = context.get("plan", {}).get("upcoming_units", [])
        latest_coverage = context.get("coverage") or {}
        top_gap_names = [item.get("concept_name", "") for item in concept_gaps[:3] if item.get("concept_name")]
        next_concept = ""
        if upcoming_units:
            unit = upcoming_units[0]
            next_concept = ", ".join(unit.get("subtopics", [])[:2]) or unit.get("chapter_name", "")
        elif top_gap_names:
            next_concept = top_gap_names[0]
        pace_status = ((latest_coverage.get("coverage") or {}).get("pace_status") if isinstance(latest_coverage, dict) else "") or "unknown"
        speed_up_areas = []
        if pace_status == "ahead" and upcoming_units:
            speed_up_areas.append(f"Move faster through {upcoming_units[0].get('chapter_name', 'the next unit')}.")
        summary = (
            f"Attendance today: {context.get('attendance', {}).get('present_count', 0)} present, "
            f"{context.get('attendance', {}).get('absent_count', 0)} absent. "
            f"Plan completion for {context.get('class', {}).get('subject', '')}: "
            f"{context.get('plan', {}).get('completion_percent', 0)}%."
        )
        return {
            "daily_summary": summary,
            "next_concept_to_teach": next_concept or "Continue the next planned concept.",
            "reteach_concepts": top_gap_names,
            "speed_up_areas": speed_up_areas,
            "students_to_watch": [item.get("full_name", "") for item in context.get("support_students", [])[:3] if item.get("full_name")],
            "teacher_actions": [
                "Use the next planned unit as tomorrow's starting point.",
                *([f"Reteach {name} with a short check-for-understanding." for name in top_gap_names[:2]]),
            ],
        }

    def _cache_key(self, *, class_id: int, subject_name: str) -> str:
        return f"{class_id}:{subject_name.strip().lower()}"

    def _load_cache_store(self) -> dict[str, Any]:
        if not DAILY_BRIEF_CACHE_PATH.exists():
            return {}
        try:
            return json.loads(DAILY_BRIEF_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache_store(self, payload: dict[str, Any]) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        DAILY_BRIEF_CACHE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_cached_daily_dashboard_brief(
        self,
        *,
        class_id: int,
        subject_name: str,
    ) -> dict[str, Any] | None:
        # Cache by class and subject so the dashboard can restore the last shown
        # analysis quickly on the next app launch.
        cache_store = self._load_cache_store()
        cache_item = cache_store.get(self._cache_key(class_id=class_id, subject_name=subject_name))
        if not isinstance(cache_item, dict):
            return None
        payload = dict(cache_item)
        payload["brief"] = self._normalize_brief_payload(payload.get("brief") or {})
        return payload

    def save_cached_daily_dashboard_brief(
        self,
        *,
        class_id: int,
        subject_name: str,
        brief_payload: dict[str, Any],
    ) -> None:
        cache_store = self._load_cache_store()
        payload = dict(brief_payload)
        payload["brief"] = self._normalize_brief_payload(payload.get("brief") or {})
        payload["cache_metadata"] = {
            "class_id": class_id,
            "subject_name": subject_name,
            "saved_at": date.today().isoformat(),
        }
        cache_store[self._cache_key(class_id=class_id, subject_name=subject_name)] = payload
        self._save_cache_store(cache_store)

    def build_local_daily_dashboard_brief(
        self,
        *,
        selected_class: dict,
        overview: dict,
        subject_name: str,
        students: list[dict],
        attendance_rows: list[dict],
        concept_gaps: list[dict],
        assessments: list[dict],
        coverage_sessions: list[dict],
        plan_snapshot: dict[str, Any],
        for_date: str | None = None,
    ) -> dict[str, Any]:
        target_date = for_date or date.today().isoformat()
        context = self.build_daily_context(
            selected_class=selected_class,
            overview=overview,
            subject_name=subject_name,
            students=students,
            attendance_rows=attendance_rows,
            concept_gaps=concept_gaps,
            assessments=assessments,
            coverage_sessions=coverage_sessions,
            plan_snapshot=plan_snapshot,
            for_date=target_date,
        )
        return {
            "context": context,
            "brief": self._normalize_brief_payload(self._fallback_brief(context)),
            "generated_by_model": "fallback",
        }

    def generate_daily_dashboard_brief(
        self,
        *,
        selected_class: dict,
        overview: dict,
        subject_name: str,
        students: list[dict],
        attendance_rows: list[dict],
        concept_gaps: list[dict],
        assessments: list[dict],
        coverage_sessions: list[dict],
        plan_snapshot: dict[str, Any],
        llama_base_url: str,
        llama_model_name: str,
        use_llama_server: bool,
        for_date: str | None = None,
    ) -> dict[str, Any]:
        local_payload = self.build_local_daily_dashboard_brief(
            selected_class=selected_class,
            overview=overview,
            subject_name=subject_name,
            students=students,
            attendance_rows=attendance_rows,
            concept_gaps=concept_gaps,
            assessments=assessments,
            coverage_sessions=coverage_sessions,
            plan_snapshot=plan_snapshot,
            for_date=for_date,
        )
        context = local_payload["context"]
        if not use_llama_server:
            return local_payload

        client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
        try:
            response = client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Gemma, the teacher's closed-loop classroom copilot. "
                            "Synthesize the day's attendance, teaching coverage, plan progress, and assessment signals. "
                            "Return strict JSON only with this shape: "
                            "{\"daily_summary\":\"string\",\"next_concept_to_teach\":\"string\","
                            "\"reteach_concepts\":[\"string\"],\"speed_up_areas\":[\"string\"],"
                            "\"students_to_watch\":[\"string\"],\"teacher_actions\":[\"string\"]}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Create the teacher's end-of-day dashboard brief.\n"
                            "Decide what to teach next, what to reteach, where to speed up, and which students need attention.\n\n"
                            f"Daily context:\n{json.dumps(context, indent=2, ensure_ascii=False)}"
                        ),
                    },
                ],
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                response_format={"type": "json_object"},
                extra_payload={"model": llama_model_name},
            )
            parsed = self._parse_json_content(self._extract_text(response))
            if not parsed:
                raise ValueError("Model daily brief response was not valid JSON.")
            return {
                "context": context,
                "brief": self._normalize_brief_payload(parsed),
                "generated_by_model": llama_model_name,
            }
        except (RequestException, ValueError, json.JSONDecodeError):
            return local_payload


DEFAULT_DAILY_LOOP_SERVICE = DailyLoopService()
