from __future__ import annotations

from typing import Any


def build_student_retrieval_query(
    *,
    subject: str,
    topic_hint: str,
    adaptation_profile: dict[str, Any] | None,
    student_context: dict[str, Any],
) -> str:
    profile_payload = (adaptation_profile or {}).get("profile", {}) if adaptation_profile else {}
    priority_targets = profile_payload.get("priority_targets", [])[:4]
    lagging_concepts = student_context.get("lagging_concepts", [])[:4]
    misconceptions = [
        item.get("concept", "")
        for item in profile_payload.get("misconception_map", [])[:4]
        if item.get("concept")
    ]
    strong_concepts = student_context.get("strong_concepts", [])[:2]
    best_formats = (profile_payload.get("response_style", {}) or {}).get("best_formats", [])[:2]
    terms = [
        subject,
        topic_hint,
        " ".join(priority_targets),
        " ".join(lagging_concepts),
        " ".join(misconceptions),
        " ".join(best_formats),
        "definition examples misconception remediation worked example",
    ]
    if strong_concepts:
        terms.append(f"avoid over-focusing on {' '.join(strong_concepts)}")
    return " ".join(part.strip() for part in terms if str(part).strip())


def build_student_generation_context(
    *,
    subject: str,
    topic_hint: str,
    adaptation_profile: dict[str, Any] | None,
    student_context: dict[str, Any],
    retrieval_context: str = "",
) -> tuple[str, str]:
    student = student_context.get("student") or {}
    profile_payload = (adaptation_profile or {}).get("profile", {}) if adaptation_profile else {}
    response_style = profile_payload.get("response_style", {}) or {}
    support_preferences = profile_payload.get("support_preferences", {}) or student_context.get("support_preferences", {})
    priority_targets = profile_payload.get("priority_targets", [])[:5]
    misconception_lines = [
        f"- {item.get('concept', 'Concept')}: {item.get('issue', '')}".strip()
        for item in profile_payload.get("misconception_map", [])[:5]
        if item.get("issue")
    ]
    intervention_history = student_context.get("intervention_history", [])[:4]
    recent_answers = student_context.get("recent_answer_samples", [])[:6]
    assessment_history = student_context.get("assessment_history", [])[:4]
    attendance_signal = student_context.get("attendance_signal", {}) or {}

    learner_profile = (
        f"Student: {student.get('full_name', 'Student')} | Subject: {subject}\n"
        f"Preferred language: {support_preferences.get('preferred_language') or student.get('preferred_language') or 'English'}\n"
        f"Priority targets: {', '.join(priority_targets) or topic_hint or 'concept reinforcement'}\n"
        f"Lagging concepts: {', '.join(student_context.get('lagging_concepts', [])[:5]) or 'Not enough data'}\n"
        f"Developing concepts: {', '.join(student_context.get('developing_concepts', [])[:4]) or 'Not enough data'}\n"
        f"Strong concepts: {', '.join(student_context.get('strong_concepts', [])[:3]) or 'Not enough data'}\n"
        f"Best formats: {', '.join(response_style.get('best_formats', [])[:3]) or 'Not enough data'}\n"
        f"Needs support in: {', '.join(response_style.get('needs_more_support_in', [])[:3]) or 'Not enough data'}\n"
        f"Pace support: {support_preferences.get('pace_support') or 'Use normal pacing with checkpoints'}\n"
        f"Accessibility support: {', '.join(support_preferences.get('accessibility_support', [])[:3]) or 'None recorded'}"
    )

    answer_lines = []
    for row in recent_answers:
        answer_lines.append(
            f"- Q: {row.get('question_text', '')}\n"
            f"  Student answer: {row.get('raw_answer', '') or '(blank)'}\n"
            f"  Score: {row.get('score_awarded', 0)} | Error type: {row.get('error_type', '') or 'Not tagged'}\n"
            f"  Reasoning: {row.get('grading_reasoning', '') or row.get('feedback', '') or 'No reasoning available'}"
        )

    intervention_lines = [
        f"- {item.get('concept_name', 'Concept')} | {item.get('recommendation_type', '')}: {item.get('recommendation_text', '')}"
        for item in intervention_history
    ]
    assessment_lines = [
        f"- {item.get('title', '')} | {item.get('chapter_name', '')} | {item.get('percentage', 0)}%"
        for item in assessment_history
    ]

    source_material = (
        "Generate a personalized remedial quiz grounded in this student's evidence-backed profile.\n"
        f"Topic focus: {topic_hint or 'concept reinforcement'}\n\n"
        "Recurring misconceptions:\n"
        f"{chr(10).join(misconception_lines) if misconception_lines else '- No misconception map available yet.'}\n\n"
        "Recent assessment history:\n"
        f"{chr(10).join(assessment_lines) if assessment_lines else '- No assessment history available.'}\n\n"
        "Recent answer evidence:\n"
        f"{chr(10).join(answer_lines) if answer_lines else '- No recent answer samples available.'}\n\n"
        "Interventions already tried:\n"
        f"{chr(10).join(intervention_lines) if intervention_lines else '- No prior interventions recorded.'}\n\n"
        "Attendance signal:\n"
        f"- Attendance percentage: {attendance_signal.get('attendance_percentage') if attendance_signal.get('attendance_percentage') is not None else 'Not enough data'}\n\n"
        "Retrieved subject material:\n"
        f"{retrieval_context or 'No retrieved textbook context available.'}"
    )
    return learner_profile, source_material
