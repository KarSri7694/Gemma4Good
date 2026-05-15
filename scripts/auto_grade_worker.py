from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.assessment_sync import AssessmentSyncService
from app.db import ensure_database
from app.model_control import load_model_sampling_config


def main() -> None:
    ensure_database()
    config = load_model_sampling_config()
    sync_service = AssessmentSyncService()

    llama_base_url = config.llama_base_url
    llama_model_name = config.llama_model_name
    poll_interval = max(5, config.auto_grade_poll_interval_seconds)

    print(f"Auto grading worker started. Poll interval: {poll_interval}s")
    while True:
        enqueued = sync_service.enqueue_new_google_form_responses()
        processed = 0
        while True:
            result = sync_service.process_next_queued_response(
                llama_base_url=llama_base_url,
                llama_model_name=llama_model_name,
            )
            if not result:
                break
            processed += 1

        if enqueued or processed:
            print(f"Queued {enqueued} new responses, processed {processed} responses.")

        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
