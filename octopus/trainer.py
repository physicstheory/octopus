"""The Train section's engine: LoRA fine-tuning on your own GPU, with live progress for the browser."""

from __future__ import annotations

import json
import math
import random
import re
import threading
import time
import traceback
from dataclasses import asdict, dataclass

from .config import ADAPTERS_DIR, BASE_MODELS, DEFAULT_BASE, SYSTEM_PROMPTS, settings


@dataclass
class TrainConfig:
    name: str = "octopus-brain"
    base: str = DEFAULT_BASE
    datasets: list = None
    epochs: float = 2.0
    learning_rate: float = 2e-4
    lora_rank: int = 16
    batch_size: int = 4
    grad_accum: int = 4
    max_len: int = 1024
    builtin_limit: int = 2000
    test_prompt: str = "Introduce yourself in two sentences."


class Trainer:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.cond = threading.Condition()
        self.events: list[dict] = []
        self.state: dict = {"status": "idle"}

    # ---------- event log shared with the browser ----------

    def emit(self, kind: str, **data) -> None:
        with self.cond:
            ev = {"seq": len(self.events), "t": kind, "time": time.time(), **data}
            self.events.append(ev)
            if kind == "progress":
                self.state.update(data)
            self.cond.notify_all()

    def log(self, msg: str) -> None:
        self.emit("log", msg=msg)

    def wait_events(self, after: int, timeout: float = 15.0) -> list[dict]:
        with self.cond:
            if len(self.events) <= after:
                self.cond.wait(timeout)
            return self.events[after:]

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    # ---------- control ----------

    def start(self, cfg: TrainConfig) -> None:
        if self.running:
            raise RuntimeError("Training is already running")
        if not cfg.datasets:
            raise ValueError("Pick at least one dataset to train on")
        if cfg.base not in BASE_MODELS:
            raise ValueError("Unknown base model")
        cfg.name = re.sub(r"[^A-Za-z0-9._-]+", "-", cfg.name).strip("-") or "octopus-brain"
        self.stop_event.clear()
        with self.cond:
            self.events = []
            self.state = {"status": "starting", "config": asdict(cfg)}
        target = self._run_mock if settings.mock else self._run
        self.thread = threading.Thread(target=target, args=(cfg,), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _finish(self, status: str, **data) -> None:
        self.state["status"] = status
        self.emit("status", status=status, **data)

    # ---------- the real thing ----------

    def _run(self, cfg: TrainConfig) -> None:
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

            from .data import load_examples
            from .engine import detect_device

            self._finish("preparing")
            self.log("Pausing chat and freeing GPU memory for training…")
            self.engine.unload()

            device = detect_device()
            self.log(f"Training on {device.label}")

            examples = load_examples(cfg.datasets, cfg.builtin_limit, log=self.log)
            if not examples:
                raise ValueError("The chosen datasets had no usable examples")

            self.log(f"Loading base model {cfg.base}…")
            tok = AutoTokenizer.from_pretrained(cfg.base)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = device.dtype
            if "gemma" in cfg.base.lower() and dtype == torch.float16:
                dtype = torch.float32  # Gemma overflows in float16 (e.g. on a free Colab T4), so compute in float32
            four_bit = device.name == "cuda" and (settings.load_in_4bit or needs_4bit(cfg.base, torch.cuda.get_device_properties(0).total_memory))
            kwargs = {"dtype": dtype, "low_cpu_mem_usage": True}
            if four_bit:
                from transformers import BitsAndBytesConfig

                self.log("Loading in 4-bit (QLoRA) so the model fits in GPU memory")
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")
                kwargs["device_map"] = {"": 0}
            model = AutoModelForCausalLM.from_pretrained(cfg.base, **kwargs)
            if "device_map" not in kwargs:
                model.to(device.name)
            if four_bit:
                from peft import prepare_model_for_kbit_training

                model = prepare_model_for_kbit_training(model)

            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False
            lora = LoraConfig(
                r=cfg.lora_rank, lora_alpha=cfg.lora_rank * 2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                # Every linear layer of the language model (names differ between Qwen, Llama, Gemma and Phi),
                # but not the picture encoder of models that can also see images.
                target_modules="all-linear", exclude_modules=r".*(vision_tower|vision_model|multi_modal_projector).*",
            )
            model = get_peft_model(model, lora)
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            self.log(f"Training {trainable / 1e6:.1f}M of {total / 1e9:.2f}B parameters (LoRA rank {cfg.lora_rank})")

            self.log("Tokenizing examples…")
            data = [x for x in (encode(tok, c, cfg.max_len) for c in examples) if x]
            if not data:
                raise ValueError("No examples fit after tokenizing. Try a larger max length.")
            self.log(f"{len(data)} training examples ready")

            steps_per_epoch = math.ceil(len(data) / (cfg.batch_size * cfg.grad_accum))
            total_steps = max(1, int(steps_per_epoch * cfg.epochs))
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
            sched = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * 0.03)), total_steps)
            use_amp = device.name == "cuda" and dtype != torch.float32
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)

            self._finish("training", total_steps=total_steps, examples=len(data))
            model.train()
            step, micro, tokens_seen, t0 = 0, 0, 0, time.time()
            ema = None
            order: list[int] = []
            rng = random.Random(0)

            while step < total_steps and not self.stop_event.is_set():
                if not order:
                    order = list(range(len(data)))
                    rng.shuffle(order)
                batch = [data[order.pop()] for _ in range(min(cfg.batch_size, len(order)))]
                ids, labels, attn = collate(batch, tok.pad_token_id)
                ids, labels, attn = ids.to(model.device), labels.to(model.device), attn.to(model.device)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                    loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss / cfg.grad_accum
                scaler.scale(loss).backward()
                micro += 1
                tokens_seen += int(attn.sum())
                if micro % cfg.grad_accum:
                    continue

                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1

                value = loss.item() * cfg.grad_accum
                ema = value if ema is None else 0.9 * ema + 0.1 * value
                elapsed = time.time() - t0
                self.emit("progress", step=step, total_steps=total_steps, loss=round(value, 4), smooth=round(ema, 4),
                          lr=sched.get_last_lr()[0], epoch=round(step / steps_per_epoch, 2),
                          tok_per_s=int(tokens_seen / max(elapsed, 1e-6)),
                          eta=int(elapsed / step * (total_steps - step)))
                if step % 10 == 0 or step == total_steps:
                    self.log(f"step {step}/{total_steps}  loss {ema:.3f}")

            stopped = self.stop_event.is_set()
            if step == 0:
                self._finish("stopped", msg="Stopped before the first step, nothing saved")
                return

            out = ADAPTERS_DIR / cfg.name
            i = 2
            while out.exists():
                out = ADAPTERS_DIR / f"{cfg.name}-{i}"
                i += 1
            model.save_pretrained(str(out))
            tok.save_pretrained(str(out))
            (out / "octopus.json").write_text(json.dumps({
                "base": cfg.base, "created": time.strftime("%Y-%m-%d %H:%M"), "examples": len(data),
                "steps": step, "final_loss": round(ema or 0, 4), "config": asdict(cfg), "partial": stopped,
            }, indent=2))
            self.log(f"Saved brain to {out}")

            sample = ""
            if cfg.test_prompt:
                model.eval()
                model.config.use_cache = True
                convo = [{"role": "system", "content": SYSTEM_PROMPTS["chat"]}, {"role": "user", "content": cfg.test_prompt}]
                enc = tok(tok.apply_chat_template(convo, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9,
                                         pad_token_id=tok.pad_token_id)
                sample = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            del model, opt
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._finish("stopped" if stopped else "done", brain=out.name, sample=sample)
        except Exception as exc:
            traceback.print_exc()
            self.log("Error: " + str(exc))
            self._finish("error", error=str(exc))

    # ---------- mock (for UI work without a GPU) ----------

    def _run_mock(self, cfg: TrainConfig) -> None:
        self._finish("preparing")
        self.log("Mock mode: simulating training")
        total = 60
        self._finish("training", total_steps=total, examples=480)
        ema = None
        for step in range(1, total + 1):
            if self.stop_event.is_set():
                break
            time.sleep(0.08)
            loss = 2.4 * math.exp(-step / 22) + 0.55 + random.random() * 0.12
            ema = loss if ema is None else 0.9 * ema + 0.1 * loss
            self.emit("progress", step=step, total_steps=total, loss=round(loss, 4), smooth=round(ema, 4),
                      lr=cfg.learning_rate, epoch=round(step / 30, 2), tok_per_s=5400, eta=int((total - step) * 0.08))
            if step % 10 == 0:
                self.log(f"step {step}/{total}  loss {ema:.3f}")
        out = ADAPTERS_DIR / cfg.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": cfg.base}))
        (out / "octopus.json").write_text(json.dumps({"base": cfg.base, "created": time.strftime("%Y-%m-%d %H:%M"),
                                                      "examples": 480, "final_loss": round(ema or 0, 4)}))
        self._finish("done", brain=out.name, sample="Hi, I'm Octopus, your own AI assistant. I can chat, explain things and write code for you.")


def needs_4bit(base: str, gpu_bytes: int) -> bool:
    """Train in 4-bit when the 16-bit weights would take more than about half the GPU's memory."""
    try:
        billions = float(str(BASE_MODELS.get(base, {}).get("size", "0")).rstrip("Bb"))
    except ValueError:
        return False
    return billions * 2e9 > gpu_bytes * 0.5


# ---------- tokenization helpers ----------

def encode(tok, convo: list[dict], max_len: int):
    """Tokenize a conversation, training only on the assistant's replies."""
    if convo[0]["role"] != "system":
        convo = [{"role": "system", "content": SYSTEM_PROMPTS["chat"]}] + convo
    ids: list[int] = []
    labels: list[int] = []
    for k, msg in enumerate(convo):
        if msg["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(convo[:k], tokenize=False, add_generation_prompt=True)
        upto = tok.apply_chat_template(convo[: k + 1], tokenize=False)
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        if upto.startswith(prefix):
            new_answer = tok(upto, add_special_tokens=False)["input_ids"][len(p_ids):]
        else:
            # Some templates (DeepSeek R1) open a <think> block in the prompt but drop it from past replies,
            # so build the reply by hand: an empty thought, then the answer.
            text = ("</think>\n\n" if prefix.rstrip().endswith("<think>") else "") + msg["content"] + (tok.eos_token or "")
            new_answer = tok(text, add_special_tokens=False)["input_ids"]
        new_prompt = p_ids[len(ids):]
        ids += new_prompt + new_answer
        labels += [-100] * len(new_prompt) + new_answer
    if not any(x != -100 for x in labels):
        return None
    ids, labels = ids[:max_len], labels[:max_len]
    if all(x == -100 for x in labels):
        return None
    return ids, labels


def collate(batch, pad_id: int):
    import torch

    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, : len(a)] = torch.tensor(a)
        labels[i, : len(b)] = torch.tensor(b)
        attn[i, : len(a)] = 1
    return ids, labels, attn
