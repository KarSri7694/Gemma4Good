# Codebase Guide

## Entry Points

- [scripts/run_streamlit_app.py](/D:/Projects/Gemma4Good/scripts/run_streamlit_app.py:1): starts Streamlit with the app-specific wrapper
- [scripts/init_db.py](/D:/Projects/Gemma4Good/scripts/init_db.py:1): creates the SQLite schema
- [scripts/auto_grade_worker.py](/D:/Projects/Gemma4Good/scripts/auto_grade_worker.py:1): background grading/sync worker

## Core Modules

### `app/main.py`

The top-level UI composition file.

Use this file when you need to:

- add or rearrange tabs
- change widget flow
- connect a button or upload action to an existing service
- update dashboard rendering

Avoid putting durable business rules or direct SQL here.

### `app/repository.py`

The persistence layer.

Use this file when you need to:

- add a new query
- change how academic records are written
- keep derived values consistent across screens

### `app/teaching_progress.py`

The main AI-assisted planning and recording service.

Use this file when you need to:

- import syllabus or timetable data
- change local recording behavior
- alter model prompts for structured extraction
- persist yearly plans or timetable slots

### `app/daily_brief.py`

The dashboard analysis service.

Use this file when you need to:

- change how daily Gemma analysis context is built
- update cache behavior
- adjust fallback dashboard summaries

## Working Rules

### When changing AI behavior

- prefer service-layer changes over UI-layer changes
- keep prompts task-specific
- normalize model outputs before persistence
- add a fallback path if the feature must still work without the model

### When changing storage behavior

- prefer repository-layer changes
- avoid calculating the same derived field differently in multiple tabs
- keep persistent records out of `st.session_state`

### When changing UI behavior

- let the UI read canonical data from repositories after a rerun
- avoid nested Streamlit containers that are known to break, such as nested expanders
- keep visual changes conservative unless they are tested across the main workflows

## Suggested Reading Order

If you are new to the repo, read files in this order:

1. [README.md](/D:/Projects/Gemma4Good/README.md:1)
2. [docs/architecture.md](/D:/Projects/Gemma4Good/docs/architecture.md:1)
3. [app/main.py](/D:/Projects/Gemma4Good/app/main.py:1)
4. [app/repository.py](/D:/Projects/Gemma4Good/app/repository.py:1)
5. [app/teaching_progress.py](/D:/Projects/Gemma4Good/app/teaching_progress.py:1)
6. [app/daily_brief.py](/D:/Projects/Gemma4Good/app/daily_brief.py:1)
