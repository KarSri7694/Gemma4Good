# Pathshala Play

`Pathshala Play` is an offline-first AI classroom co-pilot for mixed-ability learners.

It helps a teacher turn one lesson into:

- age-appropriate explanations for slow, average, and fast learners
- quick visual quiz/game activities that keep attention high
- recurring doubt summaries across the class
- accessible alternatives for differently abled learners

The project is aimed at the `Future of Education` track of the `Gemma 4 Good Hackathon`.

## Why This Idea

The strongest signal in the initial notes is a real classroom problem:

- one teacher serves students with different learning speeds
- class time is limited
- recurring doubts are hard to track manually
- engagement drops when content is too static
- accessibility needs are often handled late or poorly

This is a better hackathon project than a generic chatbot because it solves a concrete workflow for a specific user: the teacher.

## Product Pitch

Given a lesson topic, grade level, and optional worksheet/image, the app generates:

1. three difficulty-calibrated explanations
2. a short game or quiz activity
3. likely misconceptions and teacher follow-up prompts
4. an accessibility-aware version of the same content
5. a class doubt digest after student interactions

## Why Gemma 4 Fits

Gemma 4 is relevant here because the hackathon explicitly emphasizes:

- local or edge deployment
- multimodal understanding
- native tool use / agentic workflows
- meaningful real-world impact

This project can use those capabilities directly:

- `multimodal`: understand textbook images, worksheets, diagrams
- `tool use`: call quiz generators, rubric scorers, local storage, analytics
- `edge/local`: preserve privacy in schools with weak connectivity
- `multilingual`: adapt content for local language support

## MVP

The first working version should do only this:

1. Teacher enters topic, grade, subject, and class profile.
2. Teacher optionally uploads lesson text or an image.
3. System generates:
   - simple / standard / advanced explanation
   - 5-question quiz
   - 1 short game prompt
   - misconception checklist
   - accessibility adaptations
4. Teacher reviews and exports the plan.
5. Student answers are summarized into recurring doubts.

## Demo Story

The best demo is:

1. A teacher in a low-resource classroom has one science lesson and mixed-ability students.
2. The teacher uploads a worksheet/photo and picks class level.
3. The app instantly creates differentiated teaching material.
4. Students answer a short game/quiz.
5. The app clusters doubts and suggests what to reteach tomorrow.

That gives a clean impact story, visible utility, and a real end-to-end workflow.

## Suggested Stack

- `Frontend`: Streamlit or Gradio for fast demo velocity
- `Backend`: Python
- `Model runtime`:
  - local: Ollama / llama.cpp / Kaggle model access
  - larger hosted prototype: Kaggle / Hugging Face / Vertex if needed
- `Storage`: local JSON or SQLite
- `Optional`: simple retrieval layer for lesson plans and curriculum snippets

## Repo Plan

- `docs/idea-brief.md`: product framing, judging angle, and scope
- `app/`: application code
- `db/schema.sql`: SQLite schema for classroom, quiz, and mastery analytics
- `data/`: sample lessons and test inputs
- `prompts/`: generation prompts for teacher flows
- `notebooks/`: experiments and evaluation

## Database Setup

For the demo, the project uses `SQLite`.

Create the local database with:

```powershell
python scripts/init_db.py
```

That creates `data/pathshala_play.db` using [schema.sql](db/schema.sql).

## Runtime Configuration

Machine-specific runtime values now live in `model_control.env` instead of being hardcoded in the Python source.

- Set `LLAMA_BASE_URL` and `LLAMA_MODEL_NAME` for the llama.cpp server available on the current machine.
- Set `TAVILY_API_KEY` in your shell environment if you want the optional Tavily MCP server to connect.
- Keep model files under `models/Google/` if you want to use the checked-in `model_presets.ini` as-is, or update that file to point at your local model directory.
- Environment variables override `model_control.env`, so CI or another machine can supply different values without editing the repo.

## Automatic Grading Worker

To automatically poll linked Google Forms and queue new submissions for Gemma grading, run:

```powershell
python scripts/auto_grade_worker.py
```

## Run The App

To start the Streamlit app with a shutdown wrapper that handles `Ctrl+C` cleanly, run:

```powershell
python scripts/run_streamlit_app.py
```

## FastMCP Teacher Tool Servers

Teacher-facing MCP servers are defined in [mcp.json](mcp.json).

The repo now follows the same pattern as your other project:

- `mcp.json` contains an `mcpServers` object
- each server entry declares its own launch command
- the bridge starts and connects to every configured server automatically
- relative script paths in `mcp.json` are resolved from the repo config file, so the setup is portable across machines

Use your bridge to start and connect to all configured MCP servers with:

```powershell
python scripts/run_mcp_bridge.py
```

Optionally execute a tool after connecting:

```powershell
python scripts/run_mcp_bridge.py --tool answer_subject_question --tool-args "{\"question\":\"Explain photosynthesis simply\"}"
```

The local teacher server is now a directly runnable FastMCP module at [teacher_tools.py](mcp_servers/teacher_tools.py).

## Immediate Next Step

Build the smallest demo around one subject and one age band.

Recommended default:

- subject: `Science`
- grade: `6-8`
- lesson type: `worksheet + concept explanation`

That is narrow enough to finish before the deadline and broad enough to demonstrate impact.
