# Architecture

## Overview

Pathshala Play is structured around a closed teaching loop:

1. Capture classroom inputs.
2. Convert them into structured academic records.
3. Let Gemma analyze or recommend the next action.
4. Persist the result.
5. Surface the result back on the dashboard and related tabs.

This avoids isolated AI demos. A timetable upload changes subject scheduling. A syllabus upload changes chapters and yearly plans. Quiz history influences future quiz targeting. Class recordings influence progress tracking and daily analysis.

## Layers

### UI layer

[app/main.py](/D:/Projects/Gemma4Good/app/main.py:1) contains the Streamlit application.

Responsibilities:

- render workspace tabs and controls
- collect teacher inputs
- trigger service workflows
- read canonical state from repositories
- coordinate reruns and session state

The UI should not contain business rules that need to be shared elsewhere.

### Repository layer

[app/repository.py](/D:/Projects/Gemma4Good/app/repository.py:1) is the persistence boundary.

Responsibilities:

- execute SQL
- shape query results into app-friendly dictionaries
- centralize writes for attendance, classes, curriculum, plans, timetable, assessments, and analytics

This layer keeps the rest of the app from scattering SQL and duplicating persistence logic.

### Service layer

The main service modules are:

- [app/teaching_progress.py](/D:/Projects/Gemma4Good/app/teaching_progress.py:1)
- [app/daily_brief.py](/D:/Projects/Gemma4Good/app/daily_brief.py:1)
- [app/material_ingestion.py](/D:/Projects/Gemma4Good/app/material_ingestion.py:1)
- [app/assessment_sync.py](/D:/Projects/Gemma4Good/app/assessment_sync.py:1)

Responsibilities:

- call Gemma or other model endpoints
- normalize model output
- provide fallback behavior when AI is unavailable
- turn unstructured teacher inputs into structured records

## AI Integration Decisions

### AI is not the system of record

Gemma extracts, summarizes, recommends, and generates. SQLite stores the authoritative state. This makes the app auditable and stable across restarts.

### Optional AI dependencies

Some AI-backed capabilities are enhancements, not hard requirements. For example, embeddings improve retrieval, but the app should still run if the embedding server is missing.

### Cached daily analysis

Daily Gemma analysis is cached in `data/cache/daily_loop_briefs.json`. This allows the dashboard to render the last known state immediately on startup, then refresh on demand.

### Targeted context instead of full-database prompts

Each AI workflow gets a compact context payload instead of the entire class history. This reduces latency and makes outputs more predictable.

## State Model

There are two kinds of state in the app:

- durable state in SQLite and cache files
- transient interaction state in `st.session_state`

`st.session_state` should only control the current UI flow, pending uploads, and similar short-lived interaction details. Attendance, plans, chapters, quiz history, and daily analysis must be persisted.

## Audio and Shutdown

Local recording is handled through the optional `sounddevice` path inside [app/teaching_progress.py](/D:/Projects/Gemma4Good/app/teaching_progress.py:1). Shutdown hooks in [app/main.py](/D:/Projects/Gemma4Good/app/main.py:1) stop active recording and clean up bridge resources so desktop runs can be terminated cleanly with `Ctrl+C`.
