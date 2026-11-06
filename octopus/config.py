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
# Open models Octopus can chat with and train. "ollama" is the matching Ollama library tag (same weights,
# quantized) used on computers without an NVIDIA GPU. "think" is sent to Ollama for models with a
# thinking switch (None = the model has no switch). "colab" says how the free cloud notebook copes:
# "fast", "ok" (needs 4-bit training) or "slow" (fits, but can hit the free session's time limit).
BASE_MODELS: dict[str, dict] = {
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": {
        "label": "Hatchling", "size": "0.5B", "maker": "Qwen 2.5 Coder", "ollama": "qwen2.5-coder:0.5b", "colab": "fast",
        "blurb": "Tiny and very fast, for quick questions and older computers. About 1 to 2 GB.",
        "vram": 2,
    },
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": {
        "label": "Reef", "size": "1.5B", "maker": "Qwen 2.5 Coder", "ollama": "qwen2.5-coder:1.5b", "colab": "fast",
        "blurb": "Everyday default. Quick and capable at chat and code. About 2 to 4 GB of memory.",
        "vram": 4,
    },
    "Qwen/Qwen2.5-Coder-3B-Instruct": {
        "label": "Abyss", "size": "3B", "maker": "Qwen 2.5 Coder", "ollama": "qwen2.5-coder:3b", "colab": "fast",
        "blurb": "Thinks deeper. Noticeably smarter answers. About 4 to 8 GB.",
        "vram": 8,
    },
    "microsoft/Phi-4-mini-instruct": {
        "label": "Cuttlefish", "size": "3.8B", "maker": "Microsoft Phi-4 mini", "ollama": "phi4-mini:3.8b", "colab": "fast",
        "blurb": "Small but sharp at maths and logic puzzles. About 3 to 6 GB.",
        "vram": 6,
    },
    "Qwen/Qwen3-4B-Instruct-2507": {
        "label": "Nautilus", "size": "4B", "maker": "Qwen 3", "ollama": "qwen3:4b-instruct-2507-q4_K_M", "colab": "fast",
        "blurb": "The best everyday pick for a basic GPU. About as smart as Kraken at half the size. About 3 to 6 GB.",
        "vram": 6,
    },
    "unsloth/gemma-3-4b-it": {
        "label": "Coral", "size": "4B", "maker": "Google Gemma 3", "ollama": "gemma3:4b", "colab": "ok",
        "blurb": "Friendly, polished writing and good with many languages. About 4 to 6 GB.",
        "vram": 6,
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "label": "Kraken", "size": "7B", "maker": "Qwen 2.5 Coder", "ollama": "qwen2.5-coder:7b", "colab": "ok",
        "blurb": "A strong coder. Slow without a strong GPU. About 6 to 10 GB.",
        "vram": 10,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "label": "Trench", "size": "7B", "maker": "DeepSeek R1", "ollama": "deepseek-r1:7b", "colab": "ok", "think": True,
        "blurb": "Thinks step by step before answering. Slower, but better at tricky problems. About 6 to 10 GB.",
        "vram": 10,
    },
    "Qwen/Qwen3-8B": {
        "label": "Leviathan", "size": "8B", "maker": "Qwen 3", "ollama": "qwen3:8b", "colab": "ok", "think": False,
        "blurb": "The smartest all-rounder here for chat, reasoning and code. About 6 to 10 GB.",
        "vram": 10,
    },
    "unsloth/Llama-3.1-8B-Instruct": {
        "label": "Orca", "size": "8B", "maker": "Meta Llama 3.1", "ollama": "llama3.1:8b", "colab": "ok",
        "blurb": "Meta's popular all-rounder, with lots of community know-how. About 6 to 10 GB.",
        "vram": 10,
    },
    "unsloth/gemma-3-12b-it": {
        "label": "Manta", "size": "12B", "maker": "Google Gemma 3", "ollama": "gemma3:12b", "colab": "slow",
        "blurb": "Big, thoughtful writer. Needs a lot of memory and is slow on a basic GPU. About 10 to 14 GB.",
        "vram": 14,
    },
    "Qwen/Qwen2.5-Coder-14B-Instruct": {
        "label": "Megalodon", "size": "14B", "maker": "Qwen 2.5 Coder", "ollama": "qwen2.5-coder:14b", "colab": "slow",
        "blurb": "The strongest coder here. Needs a lot of memory and is slow on a basic GPU. About 10 to 16 GB.",
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
