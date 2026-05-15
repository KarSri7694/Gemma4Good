from __future__ import annotations

import base64
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime
import io
import importlib.util
import json
import re
import threading
import wave
from typing import Any

from requests import RequestException

from app.llama_client import LlamaServerClient, LlamaServerConfig
from app.repository import (
    coverage_repository,
    curriculum_repository,
    planning_repository,
    teacher_class_repository,
    timetable_repository,
)


WEEKDAY_OPTIONS = [
    (0, "Monday"),
    (1, "Tuesday"),
    (2, "Wednesday"),
    (3, "Thursday"),
    (4, "Friday"),
    (5, "Saturday"),
    (6, "Sunday"),
]
WEEKDAY_LABELS = {key: value for key, value in WEEKDAY_OPTIONS}


class SoundDeviceRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stream = None
        self._chunks: list[bytes] = []
        self._sample_rate = 16_000
        self._channels = 1
        self._dtype = "int16"
        self._session: dict[str, Any] | None = None

    def is_supported(self) -> bool:
        return importlib.util.find_spec("sounddevice") is not None

    def start(
        self,
        *,
        class_id: int,
        subject: str,
        timetable_slot_id: int | None,
        scheduled_end: str = "",
        sample_rate: int = 16_000,
        channels: int = 1,
    ) -> dict[str, Any]:
        if not self.is_supported():
            raise RuntimeError("sounddevice is not installed.")
        import sounddevice as sd

        with self._lock:
            if self._stream is not None:
                raise RuntimeError("A local microphone recording is already running.")

            self._chunks = []
            self._sample_rate = sample_rate
            self._channels = channels
            self._dtype = "int16"

            def callback(indata, frames, callback_time, status) -> None:
                del frames, callback_time
                if status:
                    # Best-effort recorder; keep collecting audio even if the device reports transient issues.
                    pass
                with self._lock:
                    self._chunks.append(indata.copy().tobytes())

            stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype=self._dtype,
                callback=callback,
            )
            stream.start()
            self._stream = stream
            self._session = {
                "class_id": class_id,
                "subject": subject,
                "timetable_slot_id": timetable_slot_id,
                "scheduled_end": scheduled_end.strip(),
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
            return dict(self._session)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._stream is None or self._session is None:
                raise RuntimeError("No local microphone recording is currently running.")
            stream = self._stream
            self._stream = None
            session = dict(self._session)
            self._session = None
            chunks = list(self._chunks)
            self._chunks = []

        stream.stop()
        stream.close()

        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self._channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(b"".join(chunks))
        session["audio_bytes"] = wav_buffer.getvalue()
        session["audio_mime_type"] = "audio/wav"
        session["stopped_at"] = datetime.now().isoformat(timespec="seconds")
        session["duration_seconds_estimate"] = round(len(session["audio_bytes"]) / max(1, self._sample_rate * 2), 1)
        return session

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = self._stream is not None and self._session is not None
            return {
                "supported": self.is_supported(),
                "active": active,
                "session": dict(self._session) if self._session else None,
            }


SOUNDDEVICE_RECORDER = SoundDeviceRecorder()


@dataclass
class TeachingPlannerService:
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

    def _normalize_name(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

    def get_capture_support_status(self) -> dict[str, Any]:
        sounddevice_available = SOUNDDEVICE_RECORDER.is_supported()
        return {
            "automatic_capture_available": sounddevice_available,
            "reason": (
                "Automatic local microphone capture can be added on this machine."
                if sounddevice_available
                else "Automatic local microphone capture is unavailable because the optional 'sounddevice' package is not installed."
            ),
        }

    def get_local_recorder_status(self) -> dict[str, Any]:
        return SOUNDDEVICE_RECORDER.status()

    def start_local_microphone_recording(
        self,
        *,
        class_id: int,
        subject: str,
        timetable_slot_id: int | None,
        scheduled_end: str = "",
    ) -> dict[str, Any]:
        return SOUNDDEVICE_RECORDER.start(
            class_id=class_id,
            subject=subject,
            timetable_slot_id=timetable_slot_id,
            scheduled_end=scheduled_end,
        )

    def stop_local_microphone_recording(self) -> dict[str, Any]:
        return SOUNDDEVICE_RECORDER.stop()

    def transcribe_audio_to_text(
        self,
        *,
        audio_bytes: bytes,
        audio_mime_type: str,
        llama_base_url: str,
        llama_model_name: str,
        prompt_hint: str = "",
    ) -> str:
        if not audio_bytes:
            raise ValueError("Audio input is empty.")
        audio_format = "wav"
        normalized_mime = (audio_mime_type or "").lower()
        if "mpeg" in normalized_mime or "mp3" in normalized_mime:
            audio_format = "mp3"
        elif "ogg" in normalized_mime:
            audio_format = "ogg"
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
        response = client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Transcribe this teacher audio into plain text. "
                                "Keep the wording faithful, remove filler when it does not change meaning, "
                                "and return only the transcript."
                                + (f" Context hint: {prompt_hint.strip()}" if prompt_hint.strip() else "")
                            ),
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": audio_format},
                        },
                    ],
                }
            ],
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            extra_payload={"model": llama_model_name, "max_tokens": -1},
        )
        transcript = self._extract_text(response).strip()
        if not transcript:
            raise ValueError("Audio transcription returned an empty result.")
        return transcript

    def _extract_text_from_image_bytes(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        llama_base_url: str,
        llama_model_name: str,
    ) -> str:
        client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Read this syllabus or school timetable image. "
                                "Return plain text only. Preserve headings, weekdays, time ranges, subjects, and chapter lists."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            extra_payload={"model": llama_model_name},
        )
        return self._extract_text(response)

    def _read_pdf_text(self, pdf_bytes: bytes) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise RuntimeError("pypdf is not installed. Add the 'pypdf' dependency before uploading PDFs.") from exc

        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_texts: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            extracted = (page.extract_text() or "").strip()
            if extracted:
                page_texts.append(f"Page {page_number}\n{extracted}")
        return "\n\n".join(page_texts).strip()

    def extract_uploaded_text(
        self,
        *,
        content_bytes: bytes,
        mime_type: str,
        original_filename: str,
        llama_base_url: str,
        llama_model_name: str,
        use_llama_server: bool,
    ) -> str:
        if not content_bytes:
            raise ValueError("Uploaded file is empty.")
        normalized_name = (original_filename or "").lower()
        normalized_mime = (mime_type or "").lower()
        is_pdf = normalized_name.endswith(".pdf") or "pdf" in normalized_mime
        is_image = (
            normalized_name.endswith((".png", ".jpg", ".jpeg", ".webp"))
            or normalized_mime.startswith("image/")
        )

        extracted_text = ""
        if is_pdf:
            extracted_text = self._read_pdf_text(content_bytes)
            if extracted_text:
                return extracted_text
            if not use_llama_server:
                raise ValueError("This PDF appears image-based. Enable llama-server to extract text from scanned PDFs.")
            raise ValueError("Scanned PDF OCR is not available yet in the planner without a text layer.")

        if is_image:
            if not use_llama_server:
                raise ValueError("Enable llama-server to extract text from uploaded images.")
            extracted_text = self._extract_text_from_image_bytes(
                image_bytes=content_bytes,
                mime_type=mime_type or "image/png",
                llama_base_url=llama_base_url,
                llama_model_name=llama_model_name,
            )
            if extracted_text.strip():
                return extracted_text.strip()
            raise ValueError("Could not extract readable text from the uploaded image.")

        return content_bytes.decode("utf-8", errors="ignore").strip()

    def _fallback_plan_units(self, syllabus_text: str) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        lines = [line.strip("-• \t") for line in syllabus_text.splitlines() if line.strip()]
        for index, line in enumerate(lines, start=1):
            chapter_name = line
            subtopics: list[str] = []
            if ":" in line:
                left, right = line.split(":", 1)
                chapter_name = left.strip() or line
                subtopics = [item.strip() for item in re.split(r"[,;/]", right) if item.strip()]
            units.append(
                {
                    "chapter_code": f"UNIT-{index:02d}",
                    "chapter_name": chapter_name,
                    "subtopics": subtopics,
                    "recommended_sessions": max(1, len(subtopics) or 2),
                    "target_month": "",
                    "term": "",
                    "sequence_order": index,
                    "completed_subtopics": [],
                    "completion_percent": 0.0,
                    "status": "not_started",
                }
            )
        if not units:
            units.append(
                {
                    "chapter_code": "UNIT-01",
                    "chapter_name": "Syllabus Overview",
                    "subtopics": [],
                    "recommended_sessions": 1,
                    "target_month": "",
                    "term": "",
                    "sequence_order": 1,
                    "completed_subtopics": [],
                    "completion_percent": 0.0,
                    "status": "not_started",
                }
            )
        return units

    def _normalize_plan_units(self, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, unit in enumerate(units, start=1):
            chapter_name = str(unit.get("chapter_name") or f"Unit {index}").strip()
            if not chapter_name:
                continue
            dedupe_key = self._normalize_name(chapter_name)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            subtopics = [
                str(item).strip()
                for item in (unit.get("subtopics") or [])
                if str(item).strip()
            ]
            normalized.append(
                {
                    "chapter_code": str(unit.get("chapter_code") or f"UNIT-{index:02d}").strip(),
                    "chapter_name": chapter_name,
                    "subtopics": subtopics,
                    "recommended_sessions": max(1, int(unit.get("recommended_sessions") or len(subtopics) or 2)),
                    "target_month": str(unit.get("target_month") or "").strip(),
                    "term": str(unit.get("term") or "").strip(),
                    "sequence_order": int(unit.get("sequence_order") or index),
                    "completed_subtopics": [],
                    "completion_percent": float(unit.get("completion_percent") or 0.0),
                    "status": str(unit.get("status") or "not_started").strip(),
                }
            )
        return normalized or self._fallback_plan_units("Syllabus Overview")

    def _generate_plan_with_model(
        self,
        *,
        grade: str,
        subject: str,
        academic_year: str,
        syllabus_text: str,
        llama_base_url: str,
        llama_model_name: str,
    ) -> dict[str, Any]:
        client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an academic year planning assistant for Indian school teachers. "
                        "Convert the syllabus into a realistic year outline. "
                        "Return strict JSON only with this shape: "
                        "{\"plan_title\":\"string\",\"units\":[{\"chapter_code\":\"string\",\"chapter_name\":\"string\","
                        "\"subtopics\":[\"string\"],\"recommended_sessions\":1,\"target_month\":\"string\","
                        "\"term\":\"string\",\"sequence_order\":1}]}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Grade: {grade}\n"
                        f"Subject: {subject}\n"
                        f"Academic year: {academic_year}\n\n"
                        "Create a practical teaching outline with chapters and teachable subtopics.\n"
                        "Keep subtopics concise. Do not invent exam dates.\n\n"
                        f"Syllabus:\n{syllabus_text}"
                    ),
                },
            ],
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            response_format={"type": "json_object"},
            extra_payload={"model": llama_model_name},
        )
        content = self._extract_text(response)
        parsed = self._parse_json_content(content)
        if not parsed:
            raise ValueError("Model year-plan response was not valid JSON.")
        return parsed

    def _generate_grade_syllabus_with_model(
        self,
        *,
        grade: str,
        academic_year: str,
        syllabus_text: str,
        llama_base_url: str,
        llama_model_name: str,
    ) -> dict[str, Any]:
        client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract a full grade syllabus into subject-wise academic plans for an Indian school. "
                        "Return strict JSON only with this shape: "
                        "{\"subjects\":[{\"subject\":\"string\",\"plan_title\":\"string\","
                        "\"units\":[{\"chapter_code\":\"string\",\"chapter_name\":\"string\","
                        "\"subtopics\":[\"string\"],\"recommended_sessions\":1,\"target_month\":\"string\","
                        "\"term\":\"string\",\"sequence_order\":1}]}]}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Grade: {grade}\n"
                        f"Academic year: {academic_year}\n\n"
                        "Separate this full-grade syllabus into subjects. "
                        "For each subject, create a practical chapter and subtopic teaching outline.\n\n"
                        f"Syllabus document:\n{syllabus_text}"
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
            raise ValueError("Model grade syllabus response was not valid JSON.")
        return parsed

    def _fallback_parse_grade_syllabus(self, syllabus_text: str) -> list[dict[str, Any]]:
        subjects: list[dict[str, Any]] = []
        current_subject = ""
        buffer: list[str] = []

        def flush_subject() -> None:
            nonlocal current_subject, buffer
            if not current_subject or not buffer:
                buffer = []
                return
            subjects.append(
                {
                    "subject": current_subject,
                    "plan_title": f"{current_subject} Teaching Plan",
                    "units": self._fallback_plan_units("\n".join(buffer)),
                }
            )
            buffer = []

        for raw_line in syllabus_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            heading_match = re.match(r"^(?:subject\s*[:\-]\s*)?([A-Za-z][A-Za-z &/()\-]{2,})$", line)
            if heading_match and len(line.split()) <= 4:
                flush_subject()
                current_subject = heading_match.group(1).strip()
                continue
            if not current_subject and ":" in line:
                possible_subject = line.split(":", 1)[0].strip()
                if 1 <= len(possible_subject.split()) <= 4:
                    flush_subject()
                    current_subject = possible_subject
                    remainder = line.split(":", 1)[1].strip()
                    if remainder:
                        buffer.append(remainder)
                    continue
            buffer.append(line)
        flush_subject()
        return subjects

    def _persist_subject_plan(
        self,
        *,
        teacher_id: int,
        class_id: int,
        academic_year: str,
        grade: str,
        board_type: str,
        subject_name: str,
        raw_syllabus_text: str,
        units: list[dict[str, Any]],
        generated_by_model: str,
        medium: str = "",
    ) -> dict[str, Any]:
        normalized_units = self._normalize_plan_units(units)
        if not normalized_units:
            raise ValueError(f"No chapters were parsed for subject '{subject_name}'.")
        curriculum_subject_id = curriculum_repository.ensure_curriculum_subject(
            board_type=board_type,
            grade=grade,
            subject=subject_name,
            default_medium=medium,
        )
        curriculum_repository.upsert_curriculum_chapters(
            curriculum_subject_id=curriculum_subject_id,
            board_type=board_type,
            grade=grade,
            subject=subject_name,
            chapters=normalized_units,
        )
        plan_title = f"{subject_name} {academic_year} Teaching Plan"
        plan_id = planning_repository.upsert_academic_year_plan(
            teacher_id=teacher_id,
            class_id=class_id,
            subject=subject_name,
            academic_year=academic_year,
            raw_syllabus_text=raw_syllabus_text,
            plan_title=plan_title,
            planning={"plan_title": plan_title, "units_count": len(normalized_units)},
            generated_by_model=generated_by_model,
            status="active",
        )
        planning_repository.replace_academic_year_plan_units(plan_id, normalized_units)
        return {
            "subject": subject_name,
            "plan_id": plan_id,
            "units_count": len(normalized_units),
            "chapter_names": [item["chapter_name"] for item in normalized_units],
        }

    def _parse_timetable_with_model(
        self,
        *,
        timetable_text: str,
        subject: str,
        llama_base_url: str,
        llama_model_name: str,
    ) -> dict[str, Any]:
        client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract weekly class timetable slots for one subject from school timetable text. "
                        "Return strict JSON only with this shape: "
                        "{\"slots\":[{\"weekday\":0,\"subject\":\"string\",\"start_time\":\"HH:MM\",\"end_time\":\"HH:MM\"}]}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Target subject: {subject}\n"
                        "Extract only the timetable periods for this subject. "
                        "Use weekday numbers Monday=0 through Sunday=6. "
                        "Normalize times to 24-hour HH:MM.\n\n"
                        f"Timetable text:\n{timetable_text}"
                    ),
                },
            ],
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            response_format={"type": "json_object"},
            extra_payload={"model": llama_model_name},
        )
        parsed = self._parse_json_content(self._extract_text(response))
        if not parsed:
            raise ValueError("Model timetable response was not valid JSON.")
        return parsed

    def _to_24h_time(self, value: str) -> str:
        normalized = str(value or "").strip().lower().replace(".", "")
        if not normalized:
            return ""
        match = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", normalized)
        if not match:
            return value.strip()
        hour = int(match.group(1))
        minute = int(match.group(2) or "00")
        meridiem = match.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    def _fallback_parse_timetable(self, *, timetable_text: str, subject: str) -> list[dict[str, Any]]:
        slots: list[dict[str, Any]] = []
        subject_key = self._normalize_name(subject)
        weekday_names = {label.lower(): key for key, label in WEEKDAY_OPTIONS}
        for raw_line in timetable_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized_line = self._normalize_name(line)
            if subject_key and subject_key not in normalized_line:
                continue
            weekday_value = next(
                (key for label, key in weekday_names.items() if label in normalized_line),
                None,
            )
            time_match = re.search(
                r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:to|-)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
                line,
                re.IGNORECASE,
            )
            if weekday_value is None or not time_match:
                continue
            slots.append(
                {
                    "weekday": weekday_value,
                    "subject": subject,
                    "start_time": self._to_24h_time(time_match.group(1)),
                    "end_time": self._to_24h_time(time_match.group(2)),
                }
            )
        return slots

    def import_timetable_from_text(
        self,
        *,
        class_id: int,
        subject: str,
        timetable_text: str,
        llama_base_url: str,
        llama_model_name: str,
        use_llama_server: bool,
    ) -> list[dict[str, Any]]:
        timetable_text = timetable_text.strip()
        if not timetable_text:
            raise ValueError("Timetable text is required.")

        slots: list[dict[str, Any]]
        if use_llama_server:
            try:
                parsed = self._parse_timetable_with_model(
                    timetable_text=timetable_text,
                    subject=subject,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                )
                slots = parsed.get("slots", [])
            except (RequestException, ValueError, json.JSONDecodeError):
                slots = self._fallback_parse_timetable(timetable_text=timetable_text, subject=subject)
        else:
            slots = self._fallback_parse_timetable(timetable_text=timetable_text, subject=subject)

        normalized_slots: list[dict[str, Any]] = []
        for slot in slots:
            try:
                weekday = int(slot.get("weekday"))
            except (TypeError, ValueError):
                continue
            if weekday < 0 or weekday > 6:
                continue
            start_time = self._to_24h_time(slot.get("start_time", ""))
            end_time = self._to_24h_time(slot.get("end_time", ""))
            if not start_time or not end_time:
                continue
            self.add_timetable_slot(
                class_id=class_id,
                subject=subject,
                weekday=weekday,
                start_time=start_time,
                end_time=end_time,
                auto_record_enabled=True,
            )
            normalized_slots.append(
                {
                    "weekday": weekday,
                    "subject": subject,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
        if not normalized_slots:
            raise ValueError("No timetable slots could be extracted for the current subject.")
        return normalized_slots

    def _parse_timetable_grid_with_model(
        self,
        *,
        timetable_text: str,
        llama_base_url: str,
        llama_model_name: str,
    ) -> dict[str, Any]:
        client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract a weekly school timetable grid into subject slots. "
                        "The timetable has weekday rows and time-slot columns. "
                        "Return strict JSON only with this shape: "
                        "{\"slots\":[{\"weekday\":0,\"subject\":\"string\",\"start_time\":\"HH:MM\",\"end_time\":\"HH:MM\"}]}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Extract every weekday, time range, and subject slot from this timetable. "
                        "Use weekday numbers Monday=0 through Sunday=6. "
                        "Normalize times to 24-hour HH:MM. Ignore blank cells.\n\n"
                        f"Timetable text:\n{timetable_text}"
                    ),
                },
            ],
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            response_format={"type": "json_object"},
            extra_payload={"model": llama_model_name},
        )
        parsed = self._parse_json_content(self._extract_text(response))
        if not parsed:
            raise ValueError("Model timetable-grid response was not valid JSON.")
        return parsed

    def generate_year_plan(
        self,
        *,
        teacher_id: int,
        class_id: int,
        academic_year: str,
        grade: str,
        subject: str,
        board_type: str,
        syllabus_text: str,
        llama_base_url: str,
        llama_model_name: str,
        use_llama_server: bool,
    ) -> dict[str, Any]:
        syllabus_text = syllabus_text.strip()
        if not syllabus_text:
            raise ValueError("Syllabus text is required.")

        planning: dict[str, Any]
        generated_by_model = "fallback-parser"
        if use_llama_server:
            try:
                planning = self._generate_plan_with_model(
                    grade=grade,
                    subject=subject,
                    academic_year=academic_year,
                    syllabus_text=syllabus_text,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                )
                generated_by_model = llama_model_name.strip() or "llama-server"
            except (RequestException, ValueError, json.JSONDecodeError):
                planning = {
                    "plan_title": f"{subject} {academic_year} Teaching Plan",
                    "units": self._fallback_plan_units(syllabus_text),
                }
        else:
            planning = {
                "plan_title": f"{subject} {academic_year} Teaching Plan",
                "units": self._fallback_plan_units(syllabus_text),
            }

        units = self._normalize_plan_units(planning.get("units") or [])
        plan_title = str(planning.get("plan_title") or f"{subject} {academic_year} Teaching Plan").strip()
        plan_id = planning_repository.upsert_academic_year_plan(
            teacher_id=teacher_id,
            class_id=class_id,
            subject=subject,
            academic_year=academic_year,
            raw_syllabus_text=syllabus_text,
            plan_title=plan_title,
            planning={"plan_title": plan_title, "units_count": len(units)},
            generated_by_model=generated_by_model,
            status="active",
        )
        planning_repository.replace_academic_year_plan_units(plan_id, units)

        curriculum_subject_id = curriculum_repository.ensure_curriculum_subject(
            board_type=board_type,
            grade=grade,
            subject=subject,
            default_medium="",
        )
        curriculum_repository.upsert_curriculum_chapters(
            curriculum_subject_id=curriculum_subject_id,
            board_type=board_type,
            grade=grade,
            subject=subject,
            chapters=units,
        )

        return self.get_plan_snapshot(class_id=class_id, subject=subject, academic_year=academic_year)

    def import_grade_syllabus_document(
        self,
        *,
        teacher_id: int,
        class_id: int,
        academic_year: str,
        grade: str,
        board_type: str,
        syllabus_text: str,
        llama_base_url: str,
        llama_model_name: str,
        use_llama_server: bool,
    ) -> dict[str, Any]:
        syllabus_text = syllabus_text.strip()
        if not syllabus_text:
            raise ValueError("Syllabus text is required.")

        class_overview = teacher_class_repository.get_class_overview(class_id)
        class_medium = str(class_overview.get("medium") or "").strip()
        parsed_subjects: list[dict[str, Any]]
        generated_by_model = "fallback-parser"
        if use_llama_server:
            try:
                parsed = self._generate_grade_syllabus_with_model(
                    grade=grade,
                    academic_year=academic_year,
                    syllabus_text=syllabus_text,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                )
                parsed_subjects = [item for item in parsed.get("subjects", []) if isinstance(item, dict)]
                generated_by_model = llama_model_name.strip() or "llama-server"
            except (RequestException, ValueError, json.JSONDecodeError):
                parsed_subjects = self._fallback_parse_grade_syllabus(syllabus_text)
        else:
            parsed_subjects = self._fallback_parse_grade_syllabus(syllabus_text)

        if not parsed_subjects:
            raise ValueError("No subject-wise syllabus structure could be extracted from the uploaded document.")

        imported_subjects: list[dict[str, Any]] = []
        for item in parsed_subjects:
            subject_name = str(item.get("subject") or "").strip()
            if not subject_name:
                continue
            imported_subjects.append(
                self._persist_subject_plan(
                    teacher_id=teacher_id,
                    class_id=class_id,
                    academic_year=academic_year,
                    grade=grade,
                    board_type=board_type,
                    subject_name=subject_name,
                    raw_syllabus_text=syllabus_text,
                    units=item.get("units") or [],
                    generated_by_model=generated_by_model,
                    medium=class_medium,
                )
            )

        if not imported_subjects:
            raise ValueError("The syllabus document was parsed, but no valid subjects were found to import.")
        return {
            "subjects_imported": imported_subjects,
            "subject_count": len(imported_subjects),
            "generated_by_model": generated_by_model,
        }

    def get_plan_snapshot(
        self,
        *,
        class_id: int,
        subject: str,
        academic_year: str = "",
    ) -> dict[str, Any]:
        plan = planning_repository.get_active_academic_year_plan(
            class_id=class_id,
            subject=subject,
            academic_year=academic_year,
        )
        if not plan:
            return {"plan": None, "units": [], "upcoming_units": [], "completion_percent": 0.0}
        units = planning_repository.list_academic_year_plan_units(int(plan["id"]))
        completion_percent = round(
            sum(float(item.get("completion_percent") or 0.0) for item in units) / max(1, len(units)),
            1,
        )
        upcoming_units = [item for item in units if float(item.get("completion_percent") or 0.0) < 100][:3]
        return {
            "plan": plan,
            "units": units,
            "upcoming_units": upcoming_units,
            "completion_percent": completion_percent,
        }

    def add_timetable_slot(
        self,
        *,
        class_id: int,
        subject: str,
        weekday: int,
        start_time: str,
        end_time: str,
        auto_record_enabled: bool = True,
    ) -> int:
        return timetable_repository.upsert_class_timetable_slot(
            class_id=class_id,
            subject=subject,
            weekday=weekday,
            start_time=start_time,
            end_time=end_time,
            auto_record_enabled=auto_record_enabled,
        )

    def list_timetable_slots(self, *, class_id: int, subject: str) -> list[dict[str, Any]]:
        return timetable_repository.list_class_timetable_slots(class_id, subject)

    def delete_timetable_slot(self, slot_id: int) -> None:
        timetable_repository.delete_class_timetable_slot(slot_id)

    def import_timetable_grid_document(
        self,
        *,
        class_id: int,
        grade: str,
        timetable_text: str,
        llama_base_url: str,
        llama_model_name: str,
        use_llama_server: bool,
    ) -> list[dict[str, Any]]:
        timetable_text = timetable_text.strip()
        if not timetable_text:
            raise ValueError("Timetable text is required.")

        slots: list[dict[str, Any]] = []
        if use_llama_server:
            try:
                parsed = self._parse_timetable_grid_with_model(
                    timetable_text=timetable_text,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                )
                slots = parsed.get("slots", [])
            except (RequestException, ValueError, json.JSONDecodeError):
                slots = []

        normalized_slots: list[dict[str, Any]] = []
        seen_keys: set[tuple[int, str, str, str]] = set()
        class_subjects = {self._normalize_name(item["subject"]) for item in teacher_class_repository.list_class_subjects(class_id)}
        for slot in slots:
            subject_name = str(slot.get("subject") or "").strip()
            if not subject_name:
                continue
            try:
                weekday = int(slot.get("weekday"))
            except (TypeError, ValueError):
                continue
            if weekday < 0 or weekday > 6:
                continue
            start_time = self._to_24h_time(slot.get("start_time", ""))
            end_time = self._to_24h_time(slot.get("end_time", ""))
            if not start_time or not end_time:
                continue
            dedupe_key = (weekday, start_time, end_time, self._normalize_name(subject_name))
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            if self._normalize_name(subject_name) not in class_subjects:
                curriculum_repository.ensure_curriculum_subject(
                    board_type="CBSE",
                    grade=grade,
                    subject=subject_name,
                    default_medium="",
                )
                class_subjects.add(self._normalize_name(subject_name))
            self.add_timetable_slot(
                class_id=class_id,
                subject=subject_name,
                weekday=weekday,
                start_time=start_time,
                end_time=end_time,
                auto_record_enabled=True,
            )
            normalized_slots.append(
                {
                    "weekday": weekday,
                    "subject": subject_name,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

        if not normalized_slots:
            raise ValueError("No timetable slots could be extracted from the uploaded timetable grid.")
        return normalized_slots

    def find_active_timetable_slot(
        self,
        *,
        class_id: int,
        subject: str,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        now = now or datetime.now()
        weekday = now.weekday()
        current_time = now.strftime("%H:%M")
        for slot in self.list_timetable_slots(class_id=class_id, subject=subject):
            if int(slot.get("weekday") or -1) != weekday:
                continue
            if str(slot.get("start_time") or "") <= current_time <= str(slot.get("end_time") or ""):
                return slot
        return None

    def _fallback_coverage_from_note(
        self,
        *,
        teacher_note: str,
        units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        transcript = teacher_note.strip()
        target_unit = next((item for item in units if float(item.get("completion_percent") or 0.0) < 100), units[0] if units else {})
        chapter_name = str(target_unit.get("chapter_name") or "Unknown chapter").strip()
        subtopics = [item for item in target_unit.get("subtopics", [])[:3]]
        mentioned: list[str] = []
        lowered_note = self._normalize_name(transcript)
        for unit in units:
            for subtopic in unit.get("subtopics", []):
                if self._normalize_name(subtopic) in lowered_note:
                    mentioned.append(subtopic)
        return {
            "teacher_transcript": transcript,
            "chapter_name": chapter_name,
            "chapter_code": str(target_unit.get("chapter_code") or "").strip(),
            "covered_subtopics": mentioned or subtopics,
            "mentioned_not_taught": [],
            "homework_or_next_class": [],
            "coverage_summary": transcript or f"Coverage was inferred for {chapter_name}.",
            "completion_signal": "partial",
            "pace_status": "unknown",
            "coverage_confidence": 0.35 if transcript else 0.1,
        }

    def _infer_coverage_with_model(
        self,
        *,
        audio_bytes: bytes,
        audio_mime_type: str,
        teacher_note: str,
        units: list[dict[str, Any]],
        llama_base_url: str,
        llama_model_name: str,
    ) -> dict[str, Any]:
        client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
        plan_summary = "\n".join(
            f"- {item.get('chapter_name', '')}: {', '.join(item.get('subtopics', [])[:8]) or 'No subtopics'}"
            for item in units[:20]
        ) or "- No plan units available."
        audio_format = "wav"
        if "mpeg" in audio_mime_type or "mp3" in audio_mime_type:
            audio_format = "mp3"
        elif "ogg" in audio_mime_type:
            audio_format = "ogg"
        elif "webm" in audio_mime_type:
            audio_format = "wav"
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a classroom teaching-progress tracker. "
                        "Focus on the teacher's instructional speech and ignore student noise unless it changes what was taught. "
                        "Return strict JSON only with this shape: "
                        "{\"teacher_transcript\":\"string\",\"chapter_name\":\"string\",\"chapter_code\":\"string\","
                        "\"covered_subtopics\":[\"string\"],\"mentioned_not_taught\":[\"string\"],"
                        "\"homework_or_next_class\":[\"string\"],\"coverage_summary\":\"string\","
                        "\"completion_signal\":\"partial|mostly_complete|complete\","
                        "\"pace_status\":\"behind|on_track|ahead|unknown\",\"coverage_confidence\":0.0}"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this classroom recording.\n"
                                "Extract only the teacher-side transcript relevant to what was taught.\n"
                                "Map the taught material to the year-plan units below.\n"
                                "Do not mark a subtopic complete if it was only mentioned briefly.\n"
                                f"Teacher note: {teacher_note.strip() or 'None'}\n\n"
                                "Year plan units:\n"
                                f"{plan_summary}"
                            ),
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": audio_format},
                        },
                    ],
                },
            ],
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            response_format={"type": "json_object"},
            extra_payload={"model": llama_model_name, "max_tokens": -1},
        )
        content = self._extract_text(response)
        parsed = self._parse_json_content(content)
        if not parsed:
            raise ValueError("Coverage model response was not valid JSON.")
        return parsed

    def _select_target_unit(
        self,
        *,
        units: list[dict[str, Any]],
        chapter_name: str,
        chapter_code: str,
    ) -> dict[str, Any] | None:
        normalized_name = self._normalize_name(chapter_name)
        normalized_code = self._normalize_name(chapter_code)
        for unit in units:
            if normalized_code and self._normalize_name(unit.get("chapter_code", "")) == normalized_code:
                return unit
        for unit in units:
            unit_name = self._normalize_name(unit.get("chapter_name", ""))
            if normalized_name and (unit_name == normalized_name or normalized_name in unit_name or unit_name in normalized_name):
                return unit
        return next((item for item in units if float(item.get("completion_percent") or 0.0) < 100), units[0] if units else None)

    def _apply_coverage_to_unit(
        self,
        *,
        unit: dict[str, Any],
        covered_subtopics: list[str],
        completion_signal: str,
    ) -> dict[str, Any]:
        existing_completed = [str(item).strip() for item in unit.get("completed_subtopics", []) if str(item).strip()]
        all_subtopics = [str(item).strip() for item in unit.get("subtopics", []) if str(item).strip()]
        normalized_existing = {self._normalize_name(item): item for item in existing_completed}
        normalized_all = {self._normalize_name(item): item for item in all_subtopics}
        for item in covered_subtopics:
            normalized = self._normalize_name(item)
            if normalized in normalized_all:
                normalized_existing[normalized] = normalized_all[normalized]
            elif item.strip():
                normalized_existing[normalized] = item.strip()
        completed_subtopics = list(normalized_existing.values())
        if all_subtopics:
            completion_percent = (len({self._normalize_name(item) for item in completed_subtopics} & set(normalized_all.keys())) / len(all_subtopics)) * 100
        else:
            completion_percent = float(unit.get("completion_percent") or 0.0)
            if completion_signal == "partial":
                completion_percent = max(completion_percent, 35.0)
            elif completion_signal == "mostly_complete":
                completion_percent = max(completion_percent, 75.0)
            elif completion_signal == "complete":
                completion_percent = 100.0
        if completion_signal == "complete":
            completion_percent = 100.0
        elif completion_signal == "mostly_complete":
            completion_percent = max(completion_percent, 75.0)
        status = "completed" if completion_percent >= 99.9 else "in_progress" if completion_percent > 0 else "not_started"
        planning_repository.update_academic_year_plan_unit_progress(
            plan_unit_id=int(unit["id"]),
            completed_subtopics=completed_subtopics,
            completion_percent=completion_percent,
            status=status,
        )
        updated_unit = dict(unit)
        updated_unit["completed_subtopics"] = completed_subtopics
        updated_unit["completion_percent"] = round(min(100.0, completion_percent), 1)
        updated_unit["status"] = status
        return updated_unit

    def process_class_session(
        self,
        *,
        class_id: int,
        teacher_id: int,
        subject: str,
        academic_year: str,
        session_date: str,
        teacher_note: str,
        audio_bytes: bytes,
        audio_mime_type: str,
        llama_base_url: str,
        llama_model_name: str,
        use_llama_server: bool,
        scheduled_start: str = "",
        scheduled_end: str = "",
        actual_start: str = "",
        actual_end: str = "",
        timetable_slot_id: int | None = None,
    ) -> dict[str, Any]:
        snapshot = self.get_plan_snapshot(class_id=class_id, subject=subject, academic_year=academic_year)
        plan = snapshot.get("plan")
        units = snapshot.get("units", [])
        if not plan or not units:
            raise ValueError("Create an academic year plan for this subject before processing classroom coverage.")
        if not audio_bytes and not teacher_note.strip():
            raise ValueError("Provide classroom audio or a teacher note.")

        if audio_bytes and use_llama_server:
            try:
                coverage = self._infer_coverage_with_model(
                    audio_bytes=audio_bytes,
                    audio_mime_type=audio_mime_type,
                    teacher_note=teacher_note,
                    units=units,
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                )
            except (RequestException, ValueError, json.JSONDecodeError):
                coverage = self._fallback_coverage_from_note(teacher_note=teacher_note, units=units)
        else:
            coverage = self._fallback_coverage_from_note(teacher_note=teacher_note, units=units)

        target_unit = self._select_target_unit(
            units=units,
            chapter_name=str(coverage.get("chapter_name") or ""),
            chapter_code=str(coverage.get("chapter_code") or ""),
        )
        if not target_unit:
            raise ValueError("No plan units were available to update.")
        updated_unit = self._apply_coverage_to_unit(
            unit=target_unit,
            covered_subtopics=[
                str(item).strip()
                for item in (coverage.get("covered_subtopics") or [])
                if str(item).strip()
            ],
            completion_signal=str(coverage.get("completion_signal") or "partial").strip(),
        )
        refreshed_snapshot = self.get_plan_snapshot(class_id=class_id, subject=subject, academic_year=academic_year)
        coverage_payload = {
            "chapter_name": updated_unit.get("chapter_name", ""),
            "chapter_code": updated_unit.get("chapter_code", ""),
            "covered_subtopics": coverage.get("covered_subtopics", []),
            "mentioned_not_taught": coverage.get("mentioned_not_taught", []),
            "homework_or_next_class": coverage.get("homework_or_next_class", []),
            "completion_signal": coverage.get("completion_signal", "partial"),
            "pace_status": coverage.get("pace_status", "unknown"),
        }
        session_id = coverage_repository.create_class_coverage_session(
            class_id=class_id,
            teacher_id=teacher_id,
            subject=subject,
            plan_id=int(plan["id"]),
            timetable_slot_id=timetable_slot_id,
            session_date=session_date,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            actual_start=actual_start,
            actual_end=actual_end,
            source="audio_plus_note" if audio_bytes and teacher_note.strip() else "audio" if audio_bytes else "manual_note",
            transcript_text=str(coverage.get("teacher_transcript") or teacher_note).strip(),
            coverage=coverage_payload,
            confidence_score=float(coverage.get("coverage_confidence") or 0.0),
            coverage_summary=str(coverage.get("coverage_summary") or "").strip(),
            processing_status="completed",
            processing_notes="Raw audio is not retained after processing.",
        )
        return {
            "session_id": session_id,
            "plan": refreshed_snapshot.get("plan"),
            "updated_unit": updated_unit,
            "upcoming_units": refreshed_snapshot.get("upcoming_units", []),
            "completion_percent": refreshed_snapshot.get("completion_percent", 0.0),
            "coverage": coverage_payload,
            "teacher_transcript": str(coverage.get("teacher_transcript") or teacher_note).strip(),
            "coverage_summary": str(coverage.get("coverage_summary") or "").strip(),
            "coverage_confidence": float(coverage.get("coverage_confidence") or 0.0),
        }


DEFAULT_TEACHING_PLANNER_SERVICE = TeachingPlannerService()
