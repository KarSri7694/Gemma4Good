from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent.parent
MODEL_CONTROL_PATH = ROOT / "model_control.env"


@dataclass
class ModelSamplingConfig:
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 40
    quiz_question_generation_mode: str = "AUTO"
    auto_grade_poll_interval_seconds: int = 15
    max_agent_iterations: int = 10
    show_reasoning: bool = True


def load_model_sampling_config(path: Path = MODEL_CONTROL_PATH) -> ModelSamplingConfig:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    return ModelSamplingConfig(
        temperature=float(values.get("LLAMA_TEMPERATURE", "0.2")),
        top_p=float(values.get("LLAMA_TOP_P", "0.95")),
        top_k=int(values.get("LLAMA_TOP_K", "40")),
        quiz_question_generation_mode=values.get("QUIZ_QUESTION_GENERATION_MODE", "AUTO").strip().upper(),
        auto_grade_poll_interval_seconds=int(values.get("AUTO_GRADE_POLL_INTERVAL_SECONDS", "15")),
        max_agent_iterations=int(values.get("MAX_AGENT_ITERATIONS", "10")),
        show_reasoning=values.get("SHOW_REASONING", "true").strip().lower() in {"1", "true", "yes", "on"},
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
