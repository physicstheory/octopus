"""Settings, paths, base models and system prompts for Octopus."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
HOME = Path(os.environ.get("OCTOPUS_HOME", ROOT / "workspace"))
DATA_DIR = HOME / "datasets"      # datasets you upload in the Train section
ADAPTERS_DIR = HOME / "brains"    # LoRA adapters ("brains") produced by training
RUN_DIR = HOME / "runs"           # scratch folder for code run from Coding mode
INCOMING_DIR = HOME / "incoming"  # files being imported (brains, 3D models)
for _d in (DATA_DIR, ADAPTERS_DIR, RUN_DIR, INCOMING_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Open-weights base models Octopus can start from. All are Apache-2.0 licensed.
BASE_MODELS: dict[str, dict] = {
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": {
        "label": "Hatchling",
        "blurb": "Tiny and very fast, for quick questions and older computers. About 1 to 2 GB.",
        "vram": 2,
    },
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": {
        "label": "Reef",
        "blurb": "Everyday default. Quick and capable at chat and code. About 2 to 4 GB of memory.",
        "vram": 4,
    },
    "Qwen/Qwen2.5-Coder-3B-Instruct": {
        "label": "Abyss",
        "blurb": "Thinks deeper. Noticeably smarter answers. About 4 to 8 GB.",
        "vram": 8,
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "label": "Kraken",
        "blurb": "The strongest coder here. Slow without a strong GPU. About 8 to 16 GB.",
        "vram": 16,
    },
}
DEFAULT_BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

COLAB_URL = "https://colab.research.google.com/github/physicstheory/octopus/blob/main/colab/Octopus_Cloud.ipynb"

SYSTEM_PROMPTS = {
    "chat": (
        "You are Octopus, a personal AI assistant. You are warm, direct and genuinely helpful. "
        "Lead with the answer, keep replies as short as the question allows while staying complete, "
        "and use Markdown (lists, tables, fenced code blocks with a language tag) when it helps."
    ),
    "code": (
        "You are Octopus in Coding mode, an expert software engineer. "
        "Write complete, working, idiomatic code with no placeholders. "
        "Put every program in a fenced code block tagged with its language (python, javascript, bash, html...). "
        "When the user wants something runnable, prefer a single self-contained Python script that prints its result. "
        "After the code, explain the key decisions briefly. "
        "If the user shares an error or output from running your code, find the cause and give the corrected full code."
    ),
}


def _bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    host: str = os.environ.get("OCTOPUS_HOST", "127.0.0.1")
    port: int = int(os.environ.get("OCTOPUS_PORT", "8000"))
    password: str = os.environ.get("OCTOPUS_PASSWORD", "")
    base_model: str = os.environ.get("OCTOPUS_BASE_MODEL", DEFAULT_BASE)
    load_in_4bit: bool = _bool("OCTOPUS_4BIT")
    max_context: int = int(os.environ.get("OCTOPUS_MAX_CONTEXT", "8192"))
    allow_code_run: bool = _bool("OCTOPUS_ALLOW_RUN", False)  # opt-in, see README
    run_timeout: int = int(os.environ.get("OCTOPUS_RUN_TIMEOUT", "30"))
    mock: bool = _bool("OCTOPUS_MOCK")
    backend: str = os.environ.get("OCTOPUS_BACKEND", "auto").lower()  # auto | ollama | transformers
    extra: dict = field(default_factory=dict)


settings = Settings()
