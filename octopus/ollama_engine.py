"""Chat engine backed by a local Ollama server.

Ollama runs models on NVIDIA, AMD, Intel and Apple GPUs (or the CPU), so this is
the default engine on computers without an NVIDIA card. Brains trained in the
cloud notebook are imported as GGUF files and become Ollama models named
"octopus-<name>". They borrow the chat template and stop words of their base
model's Ollama tag, so every model family (Qwen, Llama, Gemma, Phi, DeepSeek)
formats conversations the way it was trained to.
"""

from __future__ import annotations

import hashlib
import math
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

from .config import BASE_MODELS, DEFAULT_BASE, HOME, SYSTEM_PROMPTS, settings

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL

# Hugging Face base model -> Ollama library tag (same weights, quantized).
OLLAMA_TAGS = {k: v["ollama"] for k, v in BASE_MODELS.items()}
BRAIN_PREFIX = "octopus-"
CHATML = ("{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}"
          "<|im_start|>assistant\n")


BRAINS_FILE = HOME / "brains.json"  # brain name -> base model id, for brains imported into Ollama


class OllamaError(RuntimeError):
    pass


def _brain_bases() -> dict:
    try:
        return json.loads(BRAINS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _remember_base(brain: str, base: str | None) -> None:
    bases = _brain_bases()
    if base:
        bases[brain] = base
    else:
        bases.pop(brain, None)
    BRAINS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BRAINS_FILE.write_text(json.dumps(bases, indent=1))


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
    out, bases = [], _brain_bases()
    for m in models:
        name = m.get("name", "").split(":")[0]
        if name.startswith(BRAIN_PREFIX):
            out.append({"id": name[len(BRAIN_PREFIX):], "base": bases.get(name[len(BRAIN_PREFIX):]), "created": (m.get("modified_at") or "")[:16].replace("T", " "),
                        "examples": None, "final_loss": None, "size": m.get("details", {}).get("parameter_size")})
    return out


def delete_brain(brain: str) -> None:
    _req("/api/delete", {"model": BRAIN_PREFIX + brain}, method="DELETE").read()
    _remember_base(brain, None)


def _upload_blob(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    digest = "sha256:" + h.hexdigest()
    try:
        _req(f"/api/blobs/{digest}", method="HEAD", timeout=10).read()
    except OllamaError:
        with path.open("rb") as f:
            req = urllib.request.Request(OLLAMA_URL + f"/api/blobs/{digest}", data=f, method="POST",
                                         headers={"Content-Length": str(path.stat().st_size)})
            try:
                urllib.request.urlopen(req, timeout=600).read()
            except urllib.error.HTTPError as exc:
                raise OllamaError("Ollama rejected the file: " + exc.read().decode(errors="replace")) from None
    return digest


def _pull(tag: str) -> Iterator[str]:
    have = {m["name"] for m in installed_models()}
    if tag in have or f"{tag}:latest" in have:
        return
    yield f"Downloading {tag} (first time only)…"
    last = -1
    for ev in _lines(_req("/api/pull", {"model": tag, "stream": True}, timeout=3600)):
        if ev.get("total") and ev.get("completed"):
            pct = int(100 * ev["completed"] / ev["total"])
            if pct != last and pct % 5 == 0:
                last = pct
                yield f"Downloading {tag}… {pct}%"


def _total_memory_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return 0.0


def _check_memory(base: str) -> None:
    """Refuse a model that can't fit in this computer's memory, instead of letting it crash everything."""
    info = BASE_MODELS.get(base, {})
    need = info.get("vram", 0) * 0.6  # 4-bit weights plus room to think
    have = _total_memory_gb()
    if have and need > have:
        fits = [v["label"] for v in BASE_MODELS.values() if v.get("vram", 0) * 0.6 <= have]
        raise OllamaError(f"{info.get('label', base)} needs about {math.ceil(need)} GB of memory, but this computer only has {int(have)} GB. "
                          f"Pick a smaller model, such as {fits[-1] if fits else 'Hatchling'}.")


REGISTRY = "https://registry.ollama.ai/v2/library"


def _base_recipe(tag: str) -> dict:
    """The chat template and parameters of an Ollama library model, read locally if it's installed,
    otherwise from the Ollama registry (a few KB, without downloading the model)."""
    try:
        info = json.loads(_req("/api/show", {"model": tag}, timeout=20).read())
        params: dict = {}
        for line in (info.get("parameters") or "").splitlines():
            k, _, v = line.strip().partition(" ")
            v = v.strip().strip('"')
            if k == "stop":
                params.setdefault("stop", []).append(v)
            elif k:
                try:
                    params[k] = float(v) if "." in v else int(v)
                except ValueError:
                    params[k] = v
        return {"template": info.get("template"), "parameters": params}
    except OllamaError:
        pass
    name, _, ref = tag.partition(":")
    accept = {"Accept": "application/vnd.docker.distribution.manifest.v2+json"}
    with urllib.request.urlopen(urllib.request.Request(f"{REGISTRY}/{name}/manifests/{ref or 'latest'}", headers=accept), timeout=20) as r:
        layers = json.loads(r.read())["layers"]
    out: dict = {}
    for layer in layers:
        kind = layer["mediaType"].rsplit(".", 1)[-1]
        if kind in ("template", "params"):
            with urllib.request.urlopen(f"{REGISTRY}/{name}/blobs/{layer['digest']}", timeout=20) as r:
                text = r.read().decode()
            out["template" if kind == "template" else "parameters"] = text if kind == "template" else json.loads(text)
    return out


def import_gguf(path: Path, name: str, base: str | None = None) -> Iterator[str]:
    """Register a GGUF file with Ollama as octopus-<name>; yields progress text.

    With the base model it was trained from, the brain gets that model's chat template and stop words.
    Without one, Ollama reads the template stored inside the GGUF file."""
    recipe: dict = {}
    if base in OLLAMA_TAGS:
        yield "Looking up how this model formats chats…"
        try:
            recipe = _base_recipe(OLLAMA_TAGS[base])
        except Exception:
            recipe = {}
    elif base is None and "qwen" in path.name.lower():
        recipe = {"template": CHATML, "parameters": {"stop": ["<|im_end|>", "<|endoftext|>"]}}
    yield "Copying it into Ollama…"
    digest = _upload_blob(path)
    yield "Creating the model…"
    body = {"model": BRAIN_PREFIX + name, "files": {path.name: digest}, "stream": True}
    if recipe.get("template"):
        body["template"] = recipe["template"]
    params = {k: v for k, v in (recipe.get("parameters") or {}).items() if k in ("stop", "temperature", "top_p", "top_k", "min_p", "repeat_penalty")}
    params.setdefault("temperature", 0.7)
    body["parameters"] = params
    for ev in _lines(_req("/api/create", body, timeout=600)):
        if ev.get("status"):
            yield ev["status"]
    _remember_base(name, base if base in BASE_MODELS else None)


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
        return OLLAMA_TAGS.get(base or settings.base_model, OLLAMA_TAGS[DEFAULT_BASE])

    def prepare(self, base: str | None, brain: str | None) -> Iterator[str]:
        """Make sure the model is downloaded; yields progress messages while pulling."""
        name = self.model_name(base, brain)
        if brain:
            have = {m["name"] for m in installed_models()}
            if name not in have and f"{name}:latest" not in have:
                raise OllamaError(f"The brain '{brain}' isn't imported. Import it on the Train page.")
            base = _brain_bases().get(brain, base)
        else:
            _check_memory(base or settings.base_model)
            yield from _pull(name)
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
        think = BASE_MODELS.get(self.base_id or "", {}).get("think")
        if think is not None:
            body["think"] = think
        resp = _req("/api/chat", body, timeout=600)
        try:
            for ev in _lines(resp):
                if stop is not None and stop.event.is_set():
                    break
                msg = ev.get("message", {})
                if msg.get("thinking"):
                    yield ("think", msg["thinking"])
                if msg.get("content"):
                    yield msg["content"]
                if ev.get("done"):
                    break
        finally:
            resp.close()
