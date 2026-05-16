from __future__ import annotations

import base64
from datetime import date
import json
import re
from typing import Any

from app.llama_client import LlamaServerClient
from app.model_control import load_model_sampling_config
from app.repository import attendance_repository, student_repository


class AttendanceService:
    def extract_text(self, response: dict[str, Any]) -> str:
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

    def parse_json_content(self, content: str) -> dict[str, Any] | None:
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

    def normalize_name(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()

    def normalize_roll_reference(self, value: str) -> str:
        cleaned = str(value).strip().lower()
        if not cleaned:
            return ""
        digit_groups = re.findall(r"\d+", cleaned)
        if digit_groups:
            normalized = digit_groups[-1].lstrip("0")
            return normalized or "0"
        return cleaned

    def split_identifiers(self, value: str) -> list[str]:
        if not value.strip():
            return []
        return [item.strip() for item in re.split(r"[,\n;]+", value) if item.strip()]

    def _audio_format_from_mime(self, audio_mime_type: str) -> str:
        normalized_mime = (audio_mime_type or "").lower()
        if "mpeg" in normalized_mime or "mp3" in normalized_mime:
            return "mp3"
        if "ogg" in normalized_mime:
            return "ogg"
        if "webm" in normalized_mime:
            return "webm"
        if "m4a" in normalized_mime or "mp4" in normalized_mime:
            return "m4a"
        return "wav"

    def parse_absent_students_from_audio(
        self,
        *,
        client: LlamaServerClient,
        model_name: str,
        class_label: str,
        students: list[dict[str, Any]],
        audio_bytes: bytes,
        audio_mime_type: str = "audio/wav",
    ) -> dict[str, Any]:
        if not audio_bytes:
            raise ValueError("Recorded audio is empty.")
        sampling = load_model_sampling_config()
        roster_lines = "\n".join(
            f"- Roll {student['roll_number']}: {student['full_name']}" for student in students
        ) or "- No students available."
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        prompt_text = (
            "You are an attendance assistant for an Indian classroom. "
            "Listen to the teacher audio and identify only the students who are absent. "
            "The teacher may mention roll numbers, names, or both. "
            "Use the class roster to map spoken references accurately. "
            "If the teacher says no one is absent, return empty arrays. "
            "Return strict JSON only with this shape:\n"
            "{"
            "\"absent_roll_numbers\": [\"string\"], "
            "\"absent_student_names\": [\"string\"], "
            "\"uncertain_mentions\": [\"string\"], "
            "\"spoken_summary\": \"string\""
            "}\n\n"
            f"Class: {class_label}\n"
            "Roster:\n"
            f"{roster_lines}\n"
        )
        audio_format = self._audio_format_from_mime(audio_mime_type)

        if client.provider == "OPENROUTER":
            transcription = client.transcriptions(
                input_audio={"data": audio_b64, "format": audio_format},
                extra_payload={"model": sampling.openrouter_transcription_model},
            )
            transcript = str(transcription.get("text") or "").strip()
            if not transcript:
                raise ValueError("Audio transcription returned an empty result.")
            response = client.chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"{prompt_text}\n"
                            "Teacher transcript:\n"
                            f"{transcript}"
                        ),
                    }
                ],
                temperature=min(sampling.temperature, 0.2),
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                extra_payload={
                    "model": model_name,
                    "max_tokens": -1,
                    "reasoning_format": "auto",
                },
            )
        else:
            response = client.chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_b64,
                                    "format": audio_format,
                                },
                            },
                        ],
                    }
                ],
                temperature=min(sampling.temperature, 0.2),
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                extra_payload={
                    "model": model_name,
                    "max_tokens": -1,
                    "reasoning_format": "auto",
                },
            )
        content = self.extract_text(response)
        parsed = self.parse_json_content(content)
        if not parsed:
            raise ValueError("Gemma returned attendance audio output that was not valid JSON.")
        return {
            "absent_roll_numbers": [str(item).strip() for item in parsed.get("absent_roll_numbers", []) if str(item).strip()],
            "absent_student_names": [str(item).strip() for item in parsed.get("absent_student_names", []) if str(item).strip()],
            "uncertain_mentions": [str(item).strip() for item in parsed.get("uncertain_mentions", []) if str(item).strip()],
            "spoken_summary": str(parsed.get("spoken_summary", "")).strip(),
            "raw_model_output": content or json.dumps(response, ensure_ascii=False),
        }

    def resolve_absent_students(
        self,
        *,
        students: list[dict[str, Any]],
        absent_roll_numbers: list[str],
        absent_student_names: list[str],
    ) -> dict[str, Any]:
        roll_map = {str(student["roll_number"]).strip().lower(): student for student in students}
        shorthand_roll_map: dict[str, dict[str, Any]] = {}
        for student in students:
            shorthand_key = self.normalize_roll_reference(student["roll_number"])
            if shorthand_key and shorthand_key not in shorthand_roll_map:
                shorthand_roll_map[shorthand_key] = student
        name_map = {self.normalize_name(str(student["full_name"])): student for student in students}

        absent_ids: set[int] = set()
        matched_roll_numbers: list[str] = []
        matched_names: list[str] = []
        unresolved: list[str] = []

        for item in absent_roll_numbers:
            normalized_item = item.strip().lower()
            student = roll_map.get(normalized_item)
            if not student:
                student = shorthand_roll_map.get(self.normalize_roll_reference(item))
            if student:
                absent_ids.add(int(student["id"]))
                matched_roll_numbers.append(student["roll_number"])
            else:
                unresolved.append(item)

        for item in absent_student_names:
            normalized = self.normalize_name(item)
            student = name_map.get(normalized)
            if not student and normalized:
                student = next(
                    (
                        candidate
                        for candidate in students
                        if normalized in self.normalize_name(str(candidate["full_name"]))
                        or self.normalize_name(str(candidate["full_name"])) in normalized
                    ),
                    None,
                )
            if student:
                absent_ids.add(int(student["id"]))
                matched_names.append(student["full_name"])
            else:
                unresolved.append(item)

        return {
            "absent_student_ids": sorted(absent_ids),
            "matched_roll_numbers": sorted(set(matched_roll_numbers)),
            "matched_names": sorted(set(matched_names)),
            "unresolved_mentions": unresolved,
        }

    def mark_attendance_from_identifiers(
        self,
        *,
        class_id: int,
        teacher_id: int,
        attendance_date: str | None,
        absent_roll_numbers: str = "",
        absent_student_names: str = "",
        source: str,
        raw_model_output: str = "",
    ) -> dict[str, Any]:
        students = student_repository.list_class_roster(class_id)
        resolved = self.resolve_absent_students(
            students=students,
            absent_roll_numbers=self.split_identifiers(absent_roll_numbers),
            absent_student_names=self.split_identifiers(absent_student_names),
        )
        marked = attendance_repository.upsert_class_attendance(
            class_id=class_id,
            teacher_id=teacher_id,
            attendance_date=attendance_date or date.today().isoformat(),
            absent_student_ids=resolved["absent_student_ids"],
            source=source,
            raw_model_output=raw_model_output,
        )
        return {
            **marked,
            **resolved,
        }


DEFAULT_ATTENDANCE_SERVICE = AttendanceService()


def parse_absent_students_from_audio(
    *,
    client: LlamaServerClient,
    model_name: str,
    class_label: str,
    students: list[dict[str, Any]],
    audio_bytes: bytes,
    audio_mime_type: str = "audio/wav",
) -> dict[str, Any]:
    return DEFAULT_ATTENDANCE_SERVICE.parse_absent_students_from_audio(
        client=client,
        model_name=model_name,
        class_label=class_label,
        students=students,
        audio_bytes=audio_bytes,
        audio_mime_type=audio_mime_type,
    )


def resolve_absent_students(
    *,
    students: list[dict[str, Any]],
    absent_roll_numbers: list[str],
    absent_student_names: list[str],
) -> dict[str, Any]:
    return DEFAULT_ATTENDANCE_SERVICE.resolve_absent_students(
        students=students,
        absent_roll_numbers=absent_roll_numbers,
        absent_student_names=absent_student_names,
    )


def mark_attendance_from_identifiers(
    *,
    class_id: int,
    teacher_id: int,
    attendance_date: str | None,
    absent_roll_numbers: str = "",
    absent_student_names: str = "",
    source: str,
    raw_model_output: str = "",
) -> dict[str, Any]:
    return DEFAULT_ATTENDANCE_SERVICE.mark_attendance_from_identifiers(
        class_id=class_id,
        teacher_id=teacher_id,
        attendance_date=attendance_date,
        absent_roll_numbers=absent_roll_numbers,
        absent_student_names=absent_student_names,
        source=source,
        raw_model_output=raw_model_output,
    )
