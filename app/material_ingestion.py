from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Callable

from requests import RequestException

from app.llama_client import LlamaServerClient, LlamaServerConfig
from app.model_control import load_model_sampling_config
from app.rag import index_material_chunks
from app.repository import (
    curriculum_repository,
    material_repository,
)


ROOT = Path(__file__).resolve().parent.parent
MATERIALS_DIR = ROOT / "data" / "materials"
ProgressCallback = Callable[[float, str, str], None]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "material"


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def _emit_progress(callback: ProgressCallback | None, ratio: float, stage: str, detail: str = "") -> None:
    if callback:
        callback(max(0.0, min(1.0, ratio)), stage, detail)


def _guess_image_mime_type(image_name: str, fallback: str = "image/png") -> str:
    normalized = (image_name or "").strip().lower()
    if normalized.endswith(".jpg") or normalized.endswith(".jpeg"):
        return "image/jpeg"
    if normalized.endswith(".webp"):
        return "image/webp"
    if normalized.endswith(".gif"):
        return "image/gif"
    if normalized.endswith(".bmp"):
        return "image/bmp"
    if normalized.endswith(".tif") or normalized.endswith(".tiff"):
        return "image/tiff"
    return fallback


def _normalize_image_payload(
    image_bytes: bytes,
    image_name: str,
) -> tuple[bytes, str]:
    try:
        from PIL import Image
    except ImportError:
        return image_bytes, _guess_image_mime_type(image_name)

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            converted = io.BytesIO()
            image.convert("RGB").save(converted, format="PNG")
            return converted.getvalue(), "image/png"
    except Exception:
        return image_bytes, _guess_image_mime_type(image_name)


def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        return 0, 0
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.size
    except Exception:
        return 0, 0


def _split_into_semantic_sections(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_heading = "General"
    buffer: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_heading = (
            len(line) < 120
            and (line.isupper() or re.match(r"^(chapter|unit|lesson|section)\b", line, re.IGNORECASE))
        )
        if is_heading and buffer:
            sections.append({"section_heading": current_heading, "text": "\n".join(buffer).strip()})
            buffer = []
            current_heading = line
            continue
        if is_heading:
            current_heading = line
            continue
        buffer.append(line)
    if buffer:
        sections.append({"section_heading": current_heading, "text": "\n".join(buffer).strip()})
    return sections or [{"section_heading": "General", "text": text.strip()}]


def _clean_heading_text(value: str) -> str:
    cleaned = str(value or "").strip()
    if "|" in cleaned:
        parts = [part.strip() for part in cleaned.split("|", 1)]
        if len(parts) == 2 and re.match(r"^\d+(?:\.\d+)+$", parts[0]):
            return parts[1]
    return re.sub(r"^(?:chapter|unit|lesson)\s+\d+[:.\- ]*", "", cleaned, flags=re.IGNORECASE).strip() or cleaned


def _is_subtopic_heading(code: str, name: str) -> bool:
    normalized_code = str(code or "").strip()
    normalized_name = str(name or "").strip()
    if re.fullmatch(r"\d+\.\d+(?:\.\d+)*", normalized_code):
        return True
    if re.match(r"^\d+\.\d+(?:\.\d+)*\s*[|:-]?\s*", normalized_name):
        return True
    return False


def _normalize_extracted_chapters(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, chapter in enumerate(chapters, start=1):
        raw_code = str(chapter.get("chapter_code") or "").strip()
        raw_name = str(chapter.get("chapter_name") or "").strip()
        cleaned_name = _clean_heading_text(raw_name)
        if _is_subtopic_heading(raw_code, raw_name):
            continue
        if not cleaned_name:
            continue
        dedupe_key = cleaned_name.lower()
        if dedupe_key in seen_names:
            continue
        seen_names.add(dedupe_key)
        normalized.append(
            {
                "chapter_code": raw_code or f"CH-{index:02d}",
                "chapter_name": cleaned_name,
                "chapter_order": len(normalized) + 1,
                "term": str(chapter.get("term") or "").strip(),
            }
        )
    return normalized


def chunk_text(
    *,
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    default_content_type: str = "text",
    source_title: str = "",
    chapter_name: str = "",
) -> list[dict[str, Any]]:
    sections = _split_into_semantic_sections(text)
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for section in sections:
        words = section["text"].split()
        if not words:
            continue
        approx_words_per_chunk = max(80, int(target_tokens / 1.3))
        overlap_words = max(20, int(overlap_tokens / 1.3))
        start = 0
        while start < len(words):
            end = min(len(words), start + approx_words_per_chunk)
            chunk_words = words[start:end]
            chunk_text_value = " ".join(chunk_words).strip()
            if chunk_text_value:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "chunk_text": chunk_text_value,
                        "section_heading": section["section_heading"],
                        "content_type": default_content_type,
                        "metadata": {
                            "source_title": source_title,
                            "chapter_name": chapter_name,
                            "estimated_tokens": _approx_tokens(chunk_text_value),
                        },
                    }
                )
                chunk_index += 1
            if end >= len(words):
                break
            start = max(end - overlap_words, start + 1)
    return chunks


def _read_pdf_text_and_images(
    pdf_bytes: bytes,
    progress_callback: ProgressCallback | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise RuntimeError("pypdf is not installed. Add the 'pypdf' dependency before uploading PDFs.") from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_texts: list[str] = []
    images: list[dict[str, Any]] = []
    total_pages = max(1, len(reader.pages))
    weak_pages: list[int] = []
    for page_number, page in enumerate(reader.pages, start=1):
        _emit_progress(
            progress_callback,
            0.05 + (0.25 * (page_number / total_pages)),
            "Reading PDF",
            f"Scanning page {page_number} of {total_pages}",
        )
        extracted = (page.extract_text() or "").strip()
        if extracted:
            page_texts.append(f"Page {page_number}\n{extracted}")
        if len(extracted) < 200:
            weak_pages.append(page_number)

    if weak_pages:
        rendered_pages = _render_pdf_pages_as_images(
            pdf_bytes=pdf_bytes,
            page_numbers=weak_pages,
            progress_callback=progress_callback,
        )
        if rendered_pages:
            images.extend(rendered_pages)
        else:
            images.extend(
                _collect_filtered_embedded_images(
                    reader=reader,
                    page_numbers=weak_pages,
                    progress_callback=progress_callback,
                )
            )
    return "\n\n".join(page_texts).strip(), images


def _render_pdf_pages_as_images(
    *,
    pdf_bytes: bytes,
    page_numbers: list[int],
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError:
        return []

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[dict[str, Any]] = []
    total_pages = max(1, len(page_numbers))
    try:
        for index, page_number in enumerate(page_numbers, start=1):
            _emit_progress(
                progress_callback,
                0.30 + (0.16 * (index / total_pages)),
                "Rendering PDF pages",
                f"Rendering page {page_number} of {len(document)} as one image",
            )
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            images.append(
                {
                    "page_number": page_number,
                    "image_name": f"page-{page_number}.png",
                    "bytes": pixmap.tobytes("png"),
                    "mime_type": "image/png",
                }
            )
    finally:
        document.close()
    return images


def _collect_filtered_embedded_images(
    *,
    reader,
    page_numbers: list[int],
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    total_pages = max(1, len(page_numbers))
    for page_index, page_number in enumerate(page_numbers, start=1):
        _emit_progress(
            progress_callback,
            0.30 + (0.16 * (page_index / total_pages)),
            "Filtering embedded images",
            f"Inspecting embedded images on page {page_number}",
        )
        page = reader.pages[page_number - 1]
        best_candidate: dict[str, Any] | None = None
        best_area = 0
        page_images = getattr(page, "images", [])
        for image_index, image in enumerate(page_images, start=1):
            image_name = getattr(image, "name", f"page-{page_number}-image-{image_index}")
            image_data = getattr(image, "data", b"")
            if len(image_data) < 10_000:
                continue
            normalized_bytes, normalized_mime_type = _normalize_image_payload(image_data, image_name)
            width, height = _image_dimensions(normalized_bytes)
            area = width * height
            if area < 250_000:
                continue
            if area > best_area:
                best_area = area
                best_candidate = {
                    "page_number": page_number,
                    "image_name": image_name,
                    "bytes": normalized_bytes,
                    "mime_type": normalized_mime_type,
                }
        if best_candidate:
            images.append(best_candidate)
    return images


def _extract_text_from_image_bytes(
    *,
    image_bytes: bytes,
    mime_type: str,
    llama_base_url: str,
    llama_model_name: str,
    progress_callback: ProgressCallback | None = None,
    progress_ratio: float | None = None,
    progress_stage: str = "Analyzing image",
    progress_detail: str = "",
) -> str:
    if progress_ratio is not None:
        _emit_progress(progress_callback, progress_ratio, progress_stage, progress_detail)
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
                            "Read this educational image or textbook page. "
                            "Return plain text only. First transcribe visible text faithfully. "
                            "Then add a short note about any diagram labels or educational meaning."
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
    choices = response.get("choices", [])
    if choices:
        return str(choices[0].get("message", {}).get("content") or "").strip()
    return ""


def _extract_structure_with_gemma(
    *,
    text: str,
    grade: str,
    llama_base_url: str,
    llama_model_name: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _emit_progress(progress_callback, 0.62, "Inferring structure", f"Gemma is identifying the subject and chapters for Grade {grade}")
    preview_text = text[:24000]
    client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
    messages = [
        {
            "role": "system",
            "content": (
                "You extract textbook structure for an Indian school platform. "
                "Return strict JSON only with this shape: "
                "{\"subject\":\"string\",\"chapters\":[{\"chapter_code\":\"string\",\"chapter_name\":\"string\",\"chapter_order\":1,\"term\":\"string\"}],\"summary\":\"string\"}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Target grade: {grade}\n"
                "Infer the subject from the book content and list only the top-level chapter names in order.\n"
                "Do not return section headings or subtopics such as 7.1, 7.2, 7.4.1, or decimal-numbered subsections.\n"
                "If the content shows subtopics, map them back to their parent chapter instead of listing the subtopics.\n\n"
                f"Book content preview:\n{preview_text}"
            ),
        },
    ]
    try:
        response = client.chat_completion(
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            response_format={"type": "json_object"},
            extra_payload={"model": llama_model_name},
        )
    except RequestException:
        response = client.chat_completion(
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            extra_payload={"model": llama_model_name},
        )
    choices = response.get("choices", [])
    content = str(choices[0].get("message", {}).get("content") or "").strip() if choices else ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content[start : end + 1])
        raise


def ingest_reading_material(
    *,
    teacher_id: int,
    board_type: str,
    grade: str,
    title: str,
    source_type: str,
    content_bytes: bytes | None = None,
    text_content: str = "",
    original_filename: str = "",
    mime_type: str = "",
    llama_base_url: str,
    llama_model_name: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    config = load_model_sampling_config()
    subject = ""
    material_title = title.strip() or original_filename.strip() or "Reading Material"
    raw_text = ""
    image_chunks: list[dict[str, Any]] = []
    _emit_progress(progress_callback, 0.02, "Preparing material", material_title)

    if source_type == "text":
        _emit_progress(progress_callback, 0.10, "Reading text", "Using pasted reading material")
        raw_text = text_content.strip()
    elif source_type == "image":
        if not content_bytes:
            raise ValueError("No image bytes were provided.")
        _emit_progress(progress_callback, 0.10, "Reading image", original_filename or material_title)
        try:
            raw_text = _extract_text_from_image_bytes(
                image_bytes=content_bytes,
                mime_type=mime_type or "image/png",
                llama_base_url=llama_base_url,
                llama_model_name=llama_model_name,
                progress_callback=progress_callback,
                progress_ratio=0.22,
                progress_stage="Analyzing image",
                progress_detail=original_filename or material_title,
            )
        except Exception as exc:
            raise RuntimeError(f"Image understanding failed for '{original_filename or material_title}': {exc}") from exc
    elif source_type == "pdf":
        if not content_bytes:
            raise ValueError("No PDF bytes were provided.")
        _emit_progress(progress_callback, 0.08, "Opening PDF", original_filename or material_title)
        pdf_text, embedded_images = _read_pdf_text_and_images(content_bytes, progress_callback=progress_callback)
        raw_text = pdf_text
        total_images = max(1, len(embedded_images))
        for image_index, image_item in enumerate(embedded_images, start=1):
            try:
                image_text = _extract_text_from_image_bytes(
                    image_bytes=image_item["bytes"],
                    mime_type=image_item.get("mime_type", "image/png"),
                    llama_base_url=llama_base_url,
                    llama_model_name=llama_model_name,
                    progress_callback=progress_callback,
                    progress_ratio=0.32 + (0.20 * (image_index / total_images)),
                    progress_stage="Analyzing embedded images",
                    progress_detail=f"Image {image_index} of {total_images} from page {image_item['page_number']}",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Embedded image understanding failed for page {image_item['page_number']} in '{original_filename or material_title}': {exc}"
                ) from exc
            if image_text:
                image_chunks.append(
                    {
                        "chunk_index": 0,
                        "chunk_text": image_text,
                        "page_start": image_item["page_number"],
                        "page_end": image_item["page_number"],
                        "section_heading": f"Embedded image on page {image_item['page_number']}",
                        "content_type": "image_ocr",
                        "metadata": {
                            "source_title": material_title,
                            "image_name": image_item["image_name"],
                            "estimated_tokens": _approx_tokens(image_text),
                        },
                    }
                )
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    if not raw_text and not image_chunks:
        raise ValueError("No readable content could be extracted from the uploaded material.")

    try:
        structure = _extract_structure_with_gemma(
            text=(raw_text or "\n\n".join(chunk["chunk_text"] for chunk in image_chunks)),
            grade=grade,
            llama_base_url=llama_base_url,
            llama_model_name=llama_model_name,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        raise RuntimeError(f"Subject and chapter extraction failed for '{original_filename or material_title}': {exc}") from exc
    structure["chapters"] = _normalize_extracted_chapters(structure.get("chapters") or [])
    subject = str(structure.get("subject") or "").strip()
    if not subject:
        raise ValueError("Gemma could not infer the subject from the uploaded material.")

    curriculum_subject_id = curriculum_repository.ensure_curriculum_subject(
        board_type=board_type,
        grade=grade,
        subject=subject,
        default_medium="",
    )
    material_storage_dir = MATERIALS_DIR / grade / _slugify(subject)
    material_storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = ""
    if content_bytes:
        _emit_progress(progress_callback, 0.68, "Saving material", "Persisting the uploaded source file")
        filename = original_filename.strip() or f"{_slugify(material_title)}.{('pdf' if source_type == 'pdf' else 'bin')}"
        file_path = material_storage_dir / filename
        file_path.write_bytes(content_bytes)
        storage_path = str(file_path)

    material_id = material_repository.create_source_material(
        curriculum_subject_id=curriculum_subject_id,
        uploaded_by_teacher_id=teacher_id,
        title=material_title,
        source_type=source_type,
        original_filename=original_filename,
        mime_type=mime_type,
        storage_path=storage_path,
        raw_text=raw_text,
        extraction_summary=str(structure.get("summary") or "").strip(),
    )
    run_id = material_repository.create_ingestion_run(
        source_material_id=material_id,
        status="processing",
        extraction_summary=str(structure.get("summary") or "").strip(),
        raw_structure_json=json.dumps(structure, ensure_ascii=False),
    )

    try:
        _emit_progress(progress_callback, 0.72, "Updating curriculum", f"Creating subject {subject} and its chapters")
        chapters = curriculum_repository.upsert_curriculum_chapters(
            curriculum_subject_id=curriculum_subject_id,
            board_type=board_type,
            grade=grade,
            subject=subject,
            chapters=structure.get("chapters") or [],
        )
        _emit_progress(progress_callback, 0.80, "Chunking material", "Splitting content into retrieval chunks")
        text_chunks = chunk_text(
            text=raw_text,
            target_tokens=config.rag_chunk_target_tokens,
            overlap_tokens=config.rag_chunk_overlap_tokens,
            default_content_type="text",
            source_title=material_title,
        ) if raw_text else []
        for offset, chunk in enumerate(image_chunks, start=len(text_chunks)):
            chunk["chunk_index"] = offset
            chunk["metadata"]["source_title"] = material_title
        all_chunks = [*text_chunks, *image_chunks]
        material_repository.replace_material_chunks(
            source_material_id=material_id,
            curriculum_subject_id=curriculum_subject_id,
            chunks=all_chunks,
        )
        _emit_progress(progress_callback, 0.90, "Creating embeddings", f"Embedding {len(all_chunks)} chunks on the embedding server")
        indexed_count = index_material_chunks(
            source_material_id=material_id,
            grade=grade,
            subject=subject,
            chunks=all_chunks,
        )
        embedding_warning = ""
        if all_chunks and indexed_count == 0:
            embedding_warning = (
                "Embedding server unavailable. Material was saved and chunked, but retrieval indexing was skipped."
            )
        summary = str(structure.get("summary") or "").strip()
        material_repository.update_source_material(material_id=material_id, extraction_summary=summary, raw_text=raw_text)
        material_repository.update_ingestion_run(
            run_id=run_id,
            status="completed",
            extraction_summary=summary,
            raw_structure_json=json.dumps(structure, ensure_ascii=False),
        )
        completion_detail = (
            embedding_warning
            or f"Indexed {indexed_count} chunks for {subject}"
        )
        _emit_progress(progress_callback, 1.0, "Completed", completion_detail)
        return {
            "material_id": material_id,
            "curriculum_subject_id": curriculum_subject_id,
            "subject": subject,
            "grade": grade,
            "chapters": chapters,
            "chunk_count": len(all_chunks),
            "indexed_count": indexed_count,
            "embedding_warning": embedding_warning,
            "summary": summary,
        }
    except Exception as exc:
        material_repository.update_ingestion_run(
            run_id=run_id,
            status="failed",
            extraction_summary=str(structure.get("summary") or "").strip(),
            raw_structure_json=json.dumps(structure, ensure_ascii=False),
            error_text=str(exc),
        )
        raise
