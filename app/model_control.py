from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
MODEL_CONTROL_PATH = ROOT / "model_control.env"


@dataclass
class ModelSamplingConfig:
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 40
    llama_base_url: str = "http://127.0.0.1:8080"
    llama_model_name: str = "Gemma-4-E4B-Q4_K_M"
    quiz_question_generation_mode: str = "AUTO"
    auto_grade_poll_interval_seconds: int = 15
    max_agent_iterations: int = 10
    show_reasoning: bool = True
    embedding_model_server: str = "http://127.0.0.1:8081"
    embedding_model_name: str = "embedding-model"
    rag_chunk_target_tokens: int = 750
    rag_chunk_overlap_tokens: int = 150
    rag_top_k: int = 5


def load_model_sampling_config(path: Path = MODEL_CONTROL_PATH) -> ModelSamplingConfig:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    def get_value(key: str, default: str) -> str:
        return os.getenv(key, values.get(key, default)).strip()

    return ModelSamplingConfig(
        temperature=float(get_value("LLAMA_TEMPERATURE", "0.2")),
        top_p=float(get_value("LLAMA_TOP_P", "0.95")),
        top_k=int(get_value("LLAMA_TOP_K", "40")),
        llama_base_url=get_value("LLAMA_BASE_URL", "http://127.0.0.1:8080"),
        llama_model_name=get_value("LLAMA_MODEL_NAME", "Gemma-4-E4B-Q4_K_M"),
        quiz_question_generation_mode=get_value("QUIZ_QUESTION_GENERATION_MODE", "AUTO").upper(),
        auto_grade_poll_interval_seconds=int(get_value("AUTO_GRADE_POLL_INTERVAL_SECONDS", "15")),
        max_agent_iterations=int(get_value("MAX_AGENT_ITERATIONS", "10")),
        show_reasoning=get_value("SHOW_REASONING", "true").lower() in {"1", "true", "yes", "on"},
        embedding_model_server=get_value("EMBEDDING_MODEL_SERVER", "http://127.0.0.1:8081"),
        embedding_model_name=get_value("EMBEDDING_MODEL_NAME", "embedding-model"),
        rag_chunk_target_tokens=int(get_value("RAG_CHUNK_TARGET_TOKENS", "750")),
        rag_chunk_overlap_tokens=int(get_value("RAG_CHUNK_OVERLAP_TOKENS", "150")),
        rag_top_k=int(get_value("RAG_TOP_K", "5")),
    )


def choose_quiz_generation_mode(model_name: str, override: str) -> tuple[str, str]:
    normalized_override = (override or "AUTO").strip().upper()
    if normalized_override == "ONE_BY_ONE":
        return "one_by_one", "Forced by QUIZ_QUESTION_GENERATION_MODE=ONE_BY_ONE"
    if normalized_override == "ONE_SHOT":
        return "one_shot", "Forced by QUIZ_QUESTION_GENERATION_MODE=ONE_SHOT"

    normalized_model_name = (model_name or "").strip().upper()
    size_match = re.search(r"GEMMA-4-([A-Z0-9_]+)-", normalized_model_name)
    model_size = size_match.group(1) if size_match else ""

    if model_size.startswith("E2B") or model_size.startswith("E4B"):
        return "one_by_one", f"Auto-selected one_by_one for smaller model size {model_size}"

    numeric_match = re.search(r"(\d+)", model_size)
    if numeric_match and int(numeric_match.group(1)) >= 20:
        return "one_shot", f"Auto-selected one_shot for larger model size {model_size}"

    if "A4B" in model_size:
        return "one_shot", f"Auto-selected one_shot for MoE-style model size {model_size}"

    return "one_by_one", f"Auto-selected one_by_one as safe default for model size {model_size or 'unknown'}"
