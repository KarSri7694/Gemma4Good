from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLIENT_SECRET_PATH = ROOT / "credentials.json"
DEFAULT_TOKEN_PATH = ROOT / "token.json"
SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/forms.responses.readonly",
    "https://www.googleapis.com/auth/drive.file",
]


@dataclass
class GoogleFormsConfig:
    client_secret_path: Path = DEFAULT_CLIENT_SECRET_PATH
    token_path: Path = DEFAULT_TOKEN_PATH


def get_forms_service(config: GoogleFormsConfig | None = None):
    config = config or GoogleFormsConfig()
    credentials = _get_credentials(config)
    return build("forms", "v1", credentials=credentials)


def create_google_form_quiz(
    *,
    title: str,
    description: str,
    questions: list[dict[str, Any]],
    config: GoogleFormsConfig | None = None,
) -> dict[str, Any]:
    service = get_forms_service(config)

    form = service.forms().create(body={"info": {"title": title}}).execute()
    form_id = form["formId"]

    service.forms().batchUpdate(
        formId=form_id,
        body={
            "requests": [
                {
                    "updateSettings": {
                        "settings": {
                            "quizSettings": {"isQuiz": True},
                            "emailCollectionType": "RESPONDER_INPUT",
                        },
                        "updateMask": "quizSettings.isQuiz,emailCollectionType",
                    }
                }
            ]
        },
    ).execute()

    requests = []
    if description.strip():
        requests.append(
            {
                "updateFormInfo": {
                    "info": {"description": description},
                    "updateMask": "description",
                }
            }
        )

    for index, question in enumerate(questions):
        requests.append(
            {
                "createItem": {
                    "location": {"index": index},
                    "item": _build_form_item(question),
                }
            }
        )

    if requests:
        update_result = service.forms().batchUpdate(
            formId=form_id,
            body={"requests": requests},
        ).execute()
    else:
        update_result = {"replies": []}

    created_form = service.forms().get(formId=form_id).execute()
    responder_uri = created_form.get("responderUri")
    edit_uri = f"https://docs.google.com/forms/d/{form_id}/edit"
    question_id_map = []
    create_item_index = 0
    for reply in update_result.get("replies", []):
        create_item = reply.get("createItem", {})
        question_ids = create_item.get("questionId", [])
        if question_ids:
            create_item_index += 1
            question_id_map.append(
                {
                    "question_number": create_item_index,
                    "google_question_id": question_ids[0],
                }
            )
    return {
        "form_id": form_id,
        "responder_uri": responder_uri,
        "edit_uri": edit_uri,
        "question_id_map": question_id_map,
        "form": created_form,
    }


def list_google_form_responses(
    *,
    form_id: str,
    config: GoogleFormsConfig | None = None,
) -> list[dict[str, Any]]:
    service = get_forms_service(config)
    response = service.forms().responses().list(formId=form_id).execute()
    return response.get("responses", [])


def _get_credentials(config: GoogleFormsConfig) -> Credentials:
    credentials = None
    if config.token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(config.token_path), SCOPES)

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        config.token_path.write_text(credentials.to_json(), encoding="utf-8")
        return credentials

    if credentials and credentials.valid:
        return credentials

    if not config.client_secret_path.exists():
        raise FileNotFoundError(
            f"Google OAuth client secrets not found at {config.client_secret_path}"
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(config.client_secret_path), SCOPES)
    credentials = flow.run_local_server(port=0)
    config.token_path.write_text(credentials.to_json(), encoding="utf-8")
    return credentials


def _build_form_item(question: dict[str, Any]) -> dict[str, Any]:
    question_type = question["question_type"]
    if question_type == "mcq":
        return _build_choice_item(question)
    return _build_text_item(question)


def _build_choice_item(question: dict[str, Any]) -> dict[str, Any]:
    options = question.get("options", {})
    option_values = [
        {"value": f"A. {options.get('A', '')}"},
        {"value": f"B. {options.get('B', '')}"},
        {"value": f"C. {options.get('C', '')}"},
        {"value": f"D. {options.get('D', '')}"},
    ]

    correct_answer = str(question.get("correct_answer", "")).strip()
    correct_value = correct_answer
    if correct_answer in options:
        correct_value = f"{correct_answer}. {options[correct_answer]}"

    return {
        "title": question["question_text"],
        "questionItem": {
            "question": {
                "required": True,
                "grading": {
                    "pointValue": int(question["marks"]),
                    "correctAnswers": {"answers": [{"value": correct_value}]},
                    "whenRight": {"text": "Correct."},
                    "whenWrong": {"text": question.get("explanation", "")},
                },
                "choiceQuestion": {
                    "type": "RADIO",
                    "options": option_values,
                    "shuffle": False,
                },
            }
        },
    }


def _build_text_item(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": question["question_text"],
        "questionItem": {
            "question": {
                "required": True,
                "grading": {
                    "pointValue": int(question["marks"]),
                    "correctAnswers": {"answers": [{"value": question.get("correct_answer", "")}]},
                    "generalFeedback": {"text": question.get("explanation", "")},
                },
                "textQuestion": {"paragraph": False},
            }
        },
    }
