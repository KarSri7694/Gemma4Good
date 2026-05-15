from __future__ import annotations

import io
from datetime import datetime
from typing import Any


def build_quiz_pdf_bytes(
    *,
    title: str,
    subject: str,
    chapter_name: str,
    language: str,
    questions: list[dict[str, Any]],
    due_at: str | None = None,
) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise RuntimeError("reportlab is not installed. Add the 'reportlab' dependency before exporting quiz PDFs.") from exc

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_width, page_height = A4
    left_margin = 18 * mm
    right_margin = page_width - (18 * mm)
    top_margin = page_height - (18 * mm)
    bottom_margin = 18 * mm
    line_height = 6 * mm
    y = top_margin

    def ensure_space(required_lines: int = 1) -> None:
        nonlocal y
        if y - (required_lines * line_height) < bottom_margin:
            pdf.showPage()
            y = top_margin
            pdf.setFont("Helvetica", 11)

    def draw_wrapped_text(text: str, *, font_name: str = "Helvetica", font_size: int = 11, indent: float = 0) -> None:
        nonlocal y
        pdf.setFont(font_name, font_size)
        available_width = right_margin - (left_margin + indent)
        words = str(text).split()
        if not words:
            ensure_space(1)
            y -= line_height
            return
        current_line = words[0]
        for word in words[1:]:
            candidate = f"{current_line} {word}"
            if stringWidth(candidate, font_name, font_size) <= available_width:
                current_line = candidate
            else:
                ensure_space(1)
                pdf.drawString(left_margin + indent, y, current_line)
                y -= line_height
                current_line = word
        ensure_space(1)
        pdf.drawString(left_margin + indent, y, current_line)
        y -= line_height

    pdf.setTitle(title)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(left_margin, y, title)
    y -= 9 * mm

    pdf.setFont("Helvetica", 11)
    metadata_lines = [
        f"Subject: {subject}",
        f"Topic: {chapter_name}",
        f"Language: {language}",
    ]
    if due_at:
        try:
            due_label = datetime.strptime(due_at, "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y at %I:%M %p")
        except ValueError:
            due_label = due_at
        metadata_lines.append(f"Submit before: {due_label}")
    for item in metadata_lines:
        draw_wrapped_text(item)

    y -= 2 * mm
    total_marks = sum(float(question.get("marks", 0)) for question in questions)
    draw_wrapped_text(f"Total questions: {len(questions)} | Total marks: {int(total_marks) if total_marks.is_integer() else total_marks}")
    y -= 2 * mm

    for index, question in enumerate(questions, start=1):
        ensure_space(3)
        marks = float(question.get("marks", 0))
        marks_label = int(marks) if marks.is_integer() else marks
        draw_wrapped_text(
            f"Q{index}. {question.get('question_text', '').strip()} ({marks_label} marks)",
            font_name="Helvetica-Bold",
            font_size=11,
        )
        if question.get("question_type") == "mcq":
            for option_key in ["A", "B", "C", "D"]:
                option_value = str((question.get("options") or {}).get(option_key, "")).strip()
                if option_value:
                    draw_wrapped_text(f"{option_key}. {option_value}", indent=8 * mm)
        y -= 2 * mm

    pdf.save()
    return buffer.getvalue()
