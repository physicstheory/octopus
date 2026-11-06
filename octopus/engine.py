"""The Octopus engine: loads an open model (plus any LoRA "brain" you trained) and streams replies."""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import ADAPTERS_DIR, BASE_MODELS, DEFAULT_BASE, SYSTEM_PROMPTS, settings


@dataclass
class Device:
    name: str  # "cuda", "mps" or "cpu"
    label: str
    dtype: object


def detect_device() -> Device:
    import torch

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return Device("cuda", f"{name} ({mem:.0f} GB)", dtype)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return Device("mps", "Apple GPU (Metal)", torch.float16)
    return Device("cpu", "CPU (no GPU found, this will be slow)", torch.float32)


class StopFlag:
    """A StoppingCriteria-compatible flag so the browser's Stop button works."""

    def __init__(self) -> None:
        self.event = threading.Event()

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ARG002
        return self.event.is_set()


class Engine:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.model = None
        self.tokenizer = None
        self.base_id: str | None = None
        self.adapter: str | None = None
        self.device: Device | None = None
        self.status = "idle"  # idle | loading | ready | error
        self.error: str | None = None

    # ---------- loading ----------

    def info(self) -> dict:
        return {
            "status": self.status,
            "error": self.error,
            "base": self.base_id,
            "adapter": self.adapter,
            "device": self.device.label if self.device else None,
        }

    def ensure_loaded(self, base_id: str | None = None, adapter: str | None = None) -> None:
        base_id = base_id or self.base_id or settings.base_model or DEFAULT_BASE
        if base_id not in BASE_MODELS:
            raise ValueError(f"Unknown base model {base_id}")
        with self.lock:
            if self.model is not None and self.base_id == base_id and self.adapter == adapter:
                return
            self._load(base_id, adapter)

    def _load(self, base_id: str, adapter: str | None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.status, self.error = "loading", None
        try:
            self.unload()
            self.device = detect_device()
            kwargs = {"dtype": self.device.dtype, "low_cpu_mem_usage": True}
            if self.device.name == "cuda":
                kwargs["device_map"] = "auto"
            if settings.load_in_4bit and self.device.name == "cuda":
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=self.device.dtype, bnb_4bit_quant_type="nf4"
                )
            tok = AutoTokenizer.from_pretrained(base_id)
            model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
            if self.device.name != "cuda":
                model.to(self.device.name)
            if adapter:
                from peft import PeftModel

                path = ADAPTERS_DIR / adapter
                if not (path / "adapter_config.json").exists():
                    raise FileNotFoundError(f"Brain '{adapter}' not found in {ADAPTERS_DIR}")
                model = PeftModel.from_pretrained(model, str(path))
            model.eval()
            torch.set_grad_enabled(False)
            self.model, self.tokenizer = model, tok
            self.base_id, self.adapter = base_id, adapter
            self.status = "ready"
        except Exception as exc:  # surfaced in the UI
            self.status, self.error = "error", str(exc)
            raise

    def unload(self) -> None:
        with self.lock:
            self.model = None
            self.tokenizer = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self.status = "idle"

    # ---------- generation ----------

    def stream(
        self,
        messages: list[dict],
        mode: str = "chat",
        stop: StopFlag | None = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> Iterator[str]:
        from transformers import StoppingCriteriaList, TextIteratorStreamer

        with self.lock:
            model, tok = self.model, self.tokenizer
        if model is None:
            raise RuntimeError("No model loaded")

        convo = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}] + [
            {"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in ("user", "assistant")
        ]
        prompt = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        # Keep the prompt inside the model's context window by dropping old turns.
        limit = settings.max_context - max_new_tokens
        while inputs["input_ids"].shape[1] > limit and len(convo) > 2:
            convo.pop(1)
            prompt = tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=300)
        stop = stop or StopFlag()
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            stopping_criteria=StoppingCriteriaList([stop]),
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            repetition_penalty=1.05,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)

        errors: list[BaseException] = []

        def run() -> None:
            try:
                with self.lock:
                    model.generate(**gen_kwargs)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
                streamer.end()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        for piece in streamer:
            if piece:
                yield piece
        thread.join()
        if errors:
            raise errors[0]


def list_brains() -> list[dict]:
    """Trained LoRA adapters saved by the Train section."""
    out = []
    if ADAPTERS_DIR.exists():
        for p in sorted(ADAPTERS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            cfg = p / "adapter_config.json"
            if cfg.exists():
                meta = _read_json(p / "octopus.json")
                out.append({
                    "id": p.name,
                    "base": meta.get("base") or _read_json(cfg).get("base_model_name_or_path"),
                    "created": meta.get("created"),
                    "examples": meta.get("examples"),
                    "final_loss": meta.get("final_loss"),
                })
    return out


def _read_json(path: Path) -> dict:
    import json

    try:
        return json.loads(path.read_text())
    except Exception:
        return {}
