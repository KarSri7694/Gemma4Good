from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

from app.gemma_adaptation_profile import generate_student_adaptation_profile_with_gemma
from app.gemma_blueprint import generate_student_blueprint_with_gemma
from app.gemma_grader import grade_answer_with_gemma
from app.google_forms import list_google_form_responses
from app.llama_client import LlamaServerClient, LlamaServerConfig
from app.model_control import load_model_sampling_config
from app.repository import (
    claim_next_queued_response,
    enqueue_response_for_processing,
    get_assessment_sync_bundle,
    get_student_adaptation_profile_context,
    get_student_blueprint_context,
    list_google_linked_assessments,
    mark_queue_item_completed,
    mark_queue_item_failed,
    replace_mastery_snapshots,
    upsert_student_adaptation_profile,
    upsert_student_blueprint,
    upsert_google_form_response_sync,
)

MODEL_SAMPLING = load_model_sampling_config()


def sync_google_form_assessment(
    assessment_id: int,
    *,
    llama_base_url: str,
    llama_model_name: str,
) -> dict[str, Any]:
    responses_seen = enqueue_new_google_form_responses(assessment_id=assessment_id)
    processed = 0
    while True:
        result = process_next_queued_response(
            llama_base_url=llama_base_url,
            llama_model_name=llama_model_name,
            assessment_id_filter=assessment_id,
        )
        if not result:
            break
        processed += 1

    bundle = get_assessment_sync_bundle(assessment_id)
    student_mastery_rows, class_mastery_rows = recompute_assessment_mastery(bundle)
    replace_mastery_snapshots(
        class_id=bundle["assessment"]["class_id"],
        student_mastery_rows=student_mastery_rows,
        class_mastery_rows=class_mastery_rows,
    )

    gemma_graded_answers = sum(1 for row in student_mastery_rows if row["questions_attempted"] > 0)
    return {
        "responses_seen": responses_seen,
        "students_synced": processed,
        "responses_skipped": max(0, responses_seen - processed),
        "gemma_graded_answers": gemma_graded_answers,
        "student_mastery_rows": len(student_mastery_rows),
        "class_mastery_rows": len(class_mastery_rows),
    }


def enqueue_new_google_form_responses(*, assessment_id: int | None = None) -> int:
    assessments = (
        [get_assessment_sync_bundle(assessment_id)["assessment"]]
        if assessment_id
        else list_google_linked_assessments()
    )
    enqueued_count = 0
    for assessment in assessments:
        if not assessment or not assessment.get("google_form_id"):
            continue
        responses = list_google_form_responses(form_id=assessment["google_form_id"])
        for response in responses:
            created = enqueue_response_for_processing(
                assessment_id=assessment["id"],
                response_id=response.get("responseId", ""),
                respondent_email=(response.get("respondentEmail") or "").strip().lower(),
                submitted_at=response.get("lastSubmittedTime"),
                raw_response_json=json.dumps(response),
            )
            if created:
                enqueued_count += 1
    return enqueued_count


def process_next_queued_response(
    *,
    llama_base_url: str,
    llama_model_name: str,
    assessment_id_filter: int | None = None,
) -> dict[str, Any] | None:
    queue_item = claim_next_queued_response()
    if not queue_item:
        return None

    if assessment_id_filter is not None and queue_item["assessment_id"] != assessment_id_filter:
        mark_queue_item_failed(queue_item["id"], "Queue item belonged to a different assessment filter.")
        return None

    try:
        response = json.loads(queue_item["raw_response_json"])
        bundle = get_assessment_sync_bundle(queue_item["assessment_id"])
        if not bundle:
            raise ValueError("Assessment sync bundle not found.")

        respondent_email = (response.get("respondentEmail") or "").strip().lower()
        student = bundle["students_by_email"].get(respondent_email)
        if not student:
            raise ValueError(f"No student match found for respondent email {respondent_email or 'blank'}.")

        llama_client = LlamaServerClient(LlamaServerConfig(base_url=llama_base_url))
        answers_payload = _extract_answer_rows(
            response=response,
            questions_by_google_id=bundle["questions_by_google_id"],
            llama_client=llama_client,
            llama_model_name=llama_model_name,
        )
        if not answers_payload:
            raise ValueError("No gradable answers found in the response.")

        total_marks = sum(question["marks"] for question in bundle["questions_by_google_id"].values())
        score_obtained = sum(answer["score_awarded"] for answer in answers_payload)
        percentage = round((score_obtained / total_marks) * 100, 1) if total_marks else 0.0

        upsert_google_form_response_sync(
            assessment_id=queue_item["assessment_id"],
            student_id=student["id"],
            response_id=response.get("responseId", ""),
            submitted_at=response.get("lastSubmittedTime"),
            score_obtained=score_obtained,
            percentage=percentage,
            answers=answers_payload,
        )

        student_mastery_rows, class_mastery_rows = recompute_assessment_mastery(bundle)
        replace_mastery_snapshots(
            class_id=bundle["assessment"]["class_id"],
            student_mastery_rows=student_mastery_rows,
            class_mastery_rows=class_mastery_rows,
        )
        _refresh_student_blueprint(
            student_id=student["id"],
            class_id=bundle["assessment"]["class_id"],
            subject=bundle["assessment"]["subject"],
            last_submission_at=response.get("lastSubmittedTime"),
            llama_client=llama_client,
            llama_model_name=llama_model_name,
        )
        _refresh_student_adaptation_profile(
            student_id=student["id"],
            class_id=bundle["assessment"]["class_id"],
            subject=bundle["assessment"]["subject"],
            last_submission_at=response.get("lastSubmittedTime"),
            llama_client=llama_client,
            llama_model_name=llama_model_name,
        )

        mark_queue_item_completed(queue_item["id"])
        return {"assessment_id": queue_item["assessment_id"], "student_id": student["id"]}
    except Exception as exc:
        mark_queue_item_failed(queue_item["id"], str(exc))
        return None


def recompute_assessment_mastery(bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from app.db import get_connection

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT sat.student_id, sa.assessment_question_id, sa.is_correct
            FROM student_answers sa
            JOIN student_assessments sat ON sat.id = sa.student_assessment_id
            WHERE sat.assessment_id = ?
            """,
            (bundle["assessment"]["id"],),
        ).fetchall()

    mastery_input_rows: list[dict[str, Any]] = []
    concepts_by_question: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for link in bundle["question_concepts"]:
        concepts_by_question[link["assessment_question_id"]].append(link)

    for row in rows:
        linked_concepts = concepts_by_question.get(row["assessment_question_id"], [])
        for link in linked_concepts:
            mastery_input_rows.append(
                {
                    "student_id": row["student_id"],
                    "concept_id": link["concept_id"],
                    "is_correct": bool(row["is_correct"]),
                    "weightage": float(link["weightage"]),
                }
            )

    return _aggregate_mastery_rows(
        class_id=bundle["assessment"]["class_id"],
        mastery_input_rows=mastery_input_rows,
    )


def _extract_answer_rows(
    *,
    response: dict[str, Any],
    questions_by_google_id: dict[str, dict[str, Any]],
    llama_client: LlamaServerClient,
    llama_model_name: str,
) -> list[dict[str, Any]]:
    answers = response.get("answers", {})
    results: list[dict[str, Any]] = []

    for google_question_id, answer in answers.items():
        question = questions_by_google_id.get(google_question_id)
        if not question:
            continue

        raw_answer = _extract_text_answer(answer)
        grading = _grade_response(
            question=question,
            raw_answer=raw_answer,
            llama_client=llama_client,
            llama_model_name=llama_model_name,
        )

        results.append(
            {
                "assessment_question_id": question["id"],
                "raw_answer": raw_answer,
                "normalized_answer": raw_answer.strip().lower(),
                "is_correct": grading["is_correct"],
                "score_awarded": min(float(question["marks"]), max(0.0, grading["score_awarded"])),
                "feedback": grading["feedback"],
                "grading_reasoning": grading.get("grading_reasoning", ""),
                "error_type": grading["error_type"],
                "graded_by": grading["graded_by"],
            }
        )
    return results


def _extract_text_answer(answer: dict[str, Any]) -> str:
    text_answers = answer.get("textAnswers", {}).get("answers", [])
    if text_answers:
        return " | ".join((item.get("value") or "").strip() for item in text_answers if item.get("value"))
    file_answers = answer.get("fileUploadAnswers", {}).get("answers", [])
    if file_answers:
        return " | ".join((item.get("fileId") or "").strip() for item in file_answers if item.get("fileId"))
    return ""


def _fallback_correctness(question: dict[str, Any], raw_answer: str) -> bool:
    if question["question_type"] == "mcq":
        options = question.get("options", {})
        correct_answer = str(question.get("correct_answer", "")).strip()
        expected = f"{correct_answer}. {options.get(correct_answer, '')}".strip().lower()
        return raw_answer.strip().lower() == expected or raw_answer.strip().lower() == correct_answer.lower()
    return raw_answer.strip().lower() == str(question.get("correct_answer", "")).strip().lower()


def _grade_response(
    *,
    question: dict[str, Any],
    raw_answer: str,
    llama_client: LlamaServerClient,
    llama_model_name: str,
) -> dict[str, Any]:
    if question["question_type"] == "mcq":
        is_correct = _fallback_correctness(question, raw_answer)
        return {
            "score_awarded": float(question["marks"]) if is_correct else 0.0,
            "is_correct": 1 if is_correct else 0,
            "feedback": "Correct." if is_correct else question.get("explanation", ""),
            "grading_reasoning": (
                "Rule-based MCQ grading matched the submitted option to the correct option."
                if is_correct
                else "Rule-based MCQ grading found that the submitted option did not match the correct option."
            ),
            "error_type": None if is_correct else "concept_misunderstanding",
            "graded_by": "rule",
        }

    graded = grade_answer_with_gemma(
        client=llama_client,
        model_name=llama_model_name,
        temperature=MODEL_SAMPLING.temperature,
        top_p=MODEL_SAMPLING.top_p,
        top_k=MODEL_SAMPLING.top_k,
        question=question,
        student_answer=raw_answer,
    )
    graded["graded_by"] = "gemma"
    return graded


def _refresh_student_blueprint(
    *,
    student_id: int,
    class_id: int,
    subject: str,
    last_submission_at: str | None,
    llama_client: LlamaServerClient,
    llama_model_name: str,
) -> None:
    context = get_student_blueprint_context(student_id, subject)
    assessment_count = len(context.get("assessment_history", []))
    blueprint = generate_student_blueprint_with_gemma(
        client=llama_client,
        model_name=llama_model_name,
        temperature=MODEL_SAMPLING.temperature,
        top_p=MODEL_SAMPLING.top_p,
        top_k=MODEL_SAMPLING.top_k,
        student_context=context,
    )
    upsert_student_blueprint(
        student_id=student_id,
        class_id=class_id,
        subject=subject,
        blueprint=blueprint,
        generated_by_model=llama_model_name,
        based_on_assessments=assessment_count,
        last_submission_at=last_submission_at,
    )


def _refresh_student_adaptation_profile(
    *,
    student_id: int,
    class_id: int,
    subject: str,
    last_submission_at: str | None,
    llama_client: LlamaServerClient,
    llama_model_name: str,
) -> None:
    context = get_student_adaptation_profile_context(student_id, subject)
    assessment_count = len(context.get("assessment_history", []))
    generated = generate_student_adaptation_profile_with_gemma(
        client=llama_client,
        model_name=llama_model_name,
        temperature=MODEL_SAMPLING.temperature,
        top_p=MODEL_SAMPLING.top_p,
        top_k=MODEL_SAMPLING.top_k,
        student_context=context,
    )
    final_profile = {
        "mastery_map": context.get("mastery_map", []),
        "attendance_signal": context.get("attendance_signal", {}),
        "intervention_history": context.get("intervention_history", []),
        **generated,
    }
    upsert_student_adaptation_profile(
        student_id=student_id,
        class_id=class_id,
        subject=subject,
        profile=final_profile,
        summary=generated.get("summary", ""),
        generated_by_model=llama_model_name,
        based_on_assessments=assessment_count,
        last_submission_at=last_submission_at,
    )


def _aggregate_mastery_rows(
    *,
    class_id: int,
    mastery_input_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    student_buckets: dict[tuple[int, int], dict[str, Any]] = {}
    for row in mastery_input_rows:
        key = (row["student_id"], row["concept_id"])
        bucket = student_buckets.setdefault(
            key,
            {
                "student_id": row["student_id"],
                "concept_id": row["concept_id"],
                "total_weight": 0.0,
                "correct_weight": 0.0,
                "questions_attempted": 0,
                "questions_correct": 0,
            },
        )
        bucket["total_weight"] += row["weightage"]
        bucket["correct_weight"] += row["weightage"] if row["is_correct"] else 0.0
        bucket["questions_attempted"] += 1
        bucket["questions_correct"] += 1 if row["is_correct"] else 0

    student_mastery_rows: list[dict[str, Any]] = []
    class_buckets: dict[int, dict[str, Any]] = {}
    for bucket in student_buckets.values():
        mastery_score = bucket["correct_weight"] / bucket["total_weight"] if bucket["total_weight"] else 0.0
        status = "strong" if mastery_score >= 0.8 else "developing" if mastery_score >= 0.5 else "lagging"
        student_mastery_rows.append(
            {
                "student_id": bucket["student_id"],
                "concept_id": bucket["concept_id"],
                "mastery_score": mastery_score,
                "confidence_score": min(1.0, bucket["questions_attempted"] / 3),
                "questions_attempted": bucket["questions_attempted"],
                "questions_correct": bucket["questions_correct"],
                "status": status,
            }
        )
        class_bucket = class_buckets.setdefault(
            bucket["concept_id"],
            {"total_mastery": 0.0, "students_assessed": 0, "students_lagging": 0},
        )
        class_bucket["total_mastery"] += mastery_score
        class_bucket["students_assessed"] += 1
        class_bucket["students_lagging"] += 1 if status == "lagging" else 0

    class_mastery_rows = [
        {
            "concept_id": concept_id,
            "average_mastery_score": (
                bucket["total_mastery"] / bucket["students_assessed"] if bucket["students_assessed"] else 0.0
            ),
            "students_assessed": bucket["students_assessed"],
            "students_lagging": bucket["students_lagging"],
        }
        for concept_id, bucket in class_buckets.items()
    ]
    return student_mastery_rows, class_mastery_rows
