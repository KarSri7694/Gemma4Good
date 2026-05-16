# Pathshala Play

`Pathshala Play` is a Streamlit-based classroom copilot built around Gemma-style workflows for teachers.

The application combines:

- class dashboard and daily Gemma analysis
- syllabus and timetable ingestion from text, PDF, and images
- quiz generation and assessment review
- attendance capture, including audio-assisted flows
- teaching progress tracking and post-class analysis
- chat with Gemma using text, uploads, and recorded audio

## Current Architecture

The app is intentionally split into three layers:

- `app/main.py`: Streamlit UI orchestration
- `app/repository.py`: database access and persistence helpers
- service modules such as `app/teaching_progress.py`, `app/daily_brief.py`, and `app/material_ingestion.py`: AI-assisted extraction and workflow logic

Gemma is used as an analysis and extraction layer, not as the system of record. Persistent data still lives in SQLite and is accessed through repository modules.

## Key Flows

1. Teacher selects a class and subject workspace.
2. The app loads canonical state from SQLite repositories.
3. Uploaded or recorded inputs are converted to text or structured data through service modules.
4. Gemma-assisted workflows generate recommendations, plans, quiz targets, or extracted records.
5. Normalized outputs are written back to storage and reused across tabs.

## Important Files

- [app/main.py](/D:/Projects/Gemma4Good/app/main.py:1): main Streamlit app
- [app/repository.py](/D:/Projects/Gemma4Good/app/repository.py:1): persistence layer
- [app/teaching_progress.py](/D:/Projects/Gemma4Good/app/teaching_progress.py:1): timetable, syllabus, and recording workflows
- [app/daily_brief.py](/D:/Projects/Gemma4Good/app/daily_brief.py:1): daily Gemma analysis and cache
- [db/schema.sql](/D:/Projects/Gemma4Good/db/schema.sql:1): database schema
- [scripts/run_streamlit_app.py](/D:/Projects/Gemma4Good/scripts/run_streamlit_app.py:1): recommended launcher

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Initialize the database if needed:

```powershell
python scripts/init_db.py
```

Run the app:

```powershell
python scripts/run_streamlit_app.py
```

## Runtime Configuration

Machine-specific model and endpoint values live in `model_control.env`.

Typical settings include:

- `LLAMA_BASE_URL`
- `LLAMA_MODEL_NAME`
- provider-specific model names for transcription or grading

Environment variables can override those values on another machine without changing the repository.

## AI Design Notes

- The embedding server is optional. The app should continue working without it.
- Daily dashboard analysis is cached so startup is fast after restart.
- Uploaded PDFs prefer local text extraction first; images use model-based interpretation.

## Documentation

- [docs/architecture.md](/D:/Projects/Gemma4Good/docs/architecture.md:1)
- [docs/codebase-guide.md](/D:/Projects/Gemma4Good/docs/codebase-guide.md:1)
