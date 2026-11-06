"""Chat engine backed by a local Ollama server.

Ollama runs models on NVIDIA, AMD, Intel and Apple GPUs (or the CPU), so this is
the default engine on computers without an NVIDIA card. Brains trained in the
cloud notebook are imported as GGUF files and become Ollama models named
"octopus-<name>".
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

from .config import SYSTEM_PROMPTS, settings

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL

# Hugging Face base model -> Ollama library tag (same Qwen 2.5 Coder weights, quantized).
OLLAMA_TAGS = {
    "Qwen/Qwen2.5-Coder-0.5B-Instruct": "qwen2.5-coder:0.5b",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": "qwen2.5-coder:1.5b",
    "Qwen/Qwen2.5-Coder-3B-Instruct": "qwen2.5-coder:3b",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "qwen2.5-coder:7b",
}
BRAIN_PREFIX = "octopus-"
CHATML = ("{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}"
          "<|im_start|>assistant\n")


class OllamaError(RuntimeError):
    pass


def _req(path: str, body: dict | None = None, method: str | None = None, timeout: float = 30, data: bytes | None = None,
         headers: dict | None = None):
    payload = data if data is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(OLLAMA_URL + path, data=payload, method=method or ("POST" if payload is not None else "GET"),
                                 headers=headers or ({"Content-Type": "application/json"} if body is not None else {}))
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except Exception:
            pass
        raise OllamaError(f"Ollama: {detail}") from None
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        raise OllamaError("Ollama isn't running. Install it from ollama.com/download and open it, then try again.") from None


def _lines(resp) -> Iterator[dict]:
    for raw in resp:
        raw = raw.strip()
        if raw:
            obj = json.loads(raw)
            if obj.get("error"):
                raise OllamaError("Ollama: " + obj["error"])
            yield obj


def available() -> bool:
    try:
        _req("/api/version", timeout=2).read()
        return True
    except OllamaError:
        return False


def installed_models() -> list[dict]:
    return json.loads(_req("/api/tags", timeout=5).read()).get("models", [])


def list_brains() -> list[dict]:
    try:
        models = installed_models()
    except OllamaError:
        return []
    out = []
    for m in models:
        name = m.get("name", "").split(":")[0]
        if name.startswith(BRAIN_PREFIX):
            out.append({"id": name[len(BRAIN_PREFIX):], "base": None, "created": (m.get("modified_at") or "")[:16].replace("T", " "),
                        "examples": None, "final_loss": None, "size": m.get("details", {}).get("parameter_size")})
    return out


def delete_brain(brain: str) -> None:
    _req("/api/delete", {"model": BRAIN_PREFIX + brain}, method="DELETE").read()


def import_gguf(path: Path, name: str) -> Iterator[str]:
    """Register a GGUF file with Ollama as octopus-<name>; yields progress text."""
    yield "Checking the file…"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    digest = "sha256:" + h.hexdigest()
    try:
        _req(f"/api/blobs/{digest}", method="HEAD", timeout=10).read()
    except OllamaError:
        yield "Copying it into Ollama…"
        with path.open("rb") as f:
            req = urllib.request.Request(OLLAMA_URL + f"/api/blobs/{digest}", data=f, method="POST",
                                         headers={"Content-Length": str(path.stat().st_size)})
            try:
                urllib.request.urlopen(req, timeout=600).read()
            except urllib.error.HTTPError as exc:
                raise OllamaError("Ollama rejected the file: " + exc.read().decode(errors="replace")) from None
    yield "Creating the model…"
    body = {"model": BRAIN_PREFIX + name, "files": {path.name: digest}, "template": CHATML,
            "parameters": {"stop": ["<|im_end|>", "<|endoftext|>"], "temperature": 0.7}, "stream": True}
    for ev in _lines(_req("/api/create", body, timeout=600)):
        if ev.get("status"):
            yield ev["status"]


class OllamaEngine:
    """Same interface as engine.Engine, backed by Ollama."""

    backend = "ollama"

    def __init__(self) -> None:
        self.base_id: str | None = None
        self.adapter: str | None = None
        self.status = "idle"
        self.error: str | None = None

    def info(self) -> dict:
        return {"status": self.status, "error": self.error, "base": self.base_id, "adapter": self.adapter,
                "device": "Ollama", "backend": "ollama"}

    def model_name(self, base: str | None, brain: str | None) -> str:
        if brain:
            return BRAIN_PREFIX + brain
        return OLLAMA_TAGS.get(base or settings.base_model, "qwen2.5-coder:1.5b")

    def prepare(self, base: str | None, brain: str | None) -> Iterator[str]:
        """Make sure the model is downloaded; yields progress messages while pulling."""
        name = self.model_name(base, brain)
        have = {m["name"] for m in installed_models()}
        if name not in have and f"{name}:latest" not in have:
            if brain:
                raise OllamaError(f"The brain '{brain}' isn't imported. Import its .gguf file on the Train page.")
            yield f"Downloading {name} (first time only)…"
            last = -1
            for ev in _lines(_req("/api/pull", {"model": name, "stream": True}, timeout=3600)):
                if ev.get("total") and ev.get("completed"):
                    pct = int(100 * ev["completed"] / ev["total"])
                    if pct != last and pct % 5 == 0:
                        last = pct
                        yield f"Downloading {name}… {pct}%"
        self.base_id, self.adapter, self.status = base, brain, "ready"

    def ensure_loaded(self, base=None, adapter=None) -> None:
        for _ in self.prepare(base, adapter):
            pass

    def unload(self) -> None:
        # Ask Ollama to free the model's memory right away (keep_alive 0).
        if self.status == "ready":
            try:
                _req("/api/generate", {"model": self.model_name(self.base_id, self.adapter), "keep_alive": 0}, timeout=10).read()
            except OllamaError:
                pass
        self.status = "idle"

    def stream(self, messages: list[dict], mode: str = "chat", stop=None, temperature: float = 0.7,
               max_new_tokens: int = 2048) -> Iterator[str]:
        convo = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}] + [
            {"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in ("user", "assistant")]
        body = {"model": self.model_name(self.base_id, self.adapter), "messages": convo, "stream": True,
                "options": {"temperature": temperature, "num_ctx": settings.max_context, "num_predict": max_new_tokens}}
        resp = _req("/api/chat", body, timeout=600)
        try:
            for ev in _lines(resp):
                if stop is not None and stop.event.is_set():
                    break
                piece = ev.get("message", {}).get("content")
                if piece:
                    yield piece
                if ev.get("done"):
                    break
        finally:
            resp.close()
