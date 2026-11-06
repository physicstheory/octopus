"""Training data: built-in open datasets, your uploads, and conversion to chat examples."""

from __future__ import annotations

import csv
import io
import json
import random
import re
import time
from pathlib import Path

from .config import DATA_DIR

# Free, openly licensed datasets that can be mixed into training with one click.
BUILTIN_DATASETS = {
    "no_robots": {
        "hf": "HuggingFaceH4/no_robots", "split": "train",
        "label": "Helpful chat (No Robots)",
        "blurb": "10k conversations written by people, covering Q&A, writing, brainstorming and more.",
        "license": "CC BY-NC 4.0",
    },
    "dolly": {
        "hf": "databricks/databricks-dolly-15k", "split": "train",
        "label": "General knowledge (Dolly 15k)",
        "blurb": "15k human-written instructions and answers across many topics.",
        "license": "CC BY-SA 3.0",
    },
    "code_alpaca": {
        "hf": "sahil2801/CodeAlpaca-20k", "split": "train",
        "label": "Coding tasks (CodeAlpaca 20k)",
        "blurb": "20k short programming tasks with solutions.",
        "license": "CC BY 4.0",
    },
    "python_18k": {
        "hf": "iamtarun/python_code_instructions_18k_alpaca", "split": "train",
        "label": "Python coding (18k)",
        "blurb": "18k Python problems with complete solutions.",
        "license": "Apache 2.0",
    },
}

ROLE_ALIASES = {"human": "user", "user": "user", "prompter": "user", "gpt": "assistant",
                "assistant": "assistant", "bot": "assistant", "model": "assistant", "system": "system"}


def to_messages(rec: dict) -> list[dict] | None:
    """Turn one record in any common format into a chat conversation."""
    if not isinstance(rec, dict):
        return None
    # {"messages": [{"role", "content"}]} or ShareGPT {"conversations": [{"from", "value"}]}
    for key in ("messages", "conversations", "conversation", "dialog"):
        if isinstance(rec.get(key), list):
            msgs = []
            for m in rec[key]:
                role = ROLE_ALIASES.get(str(m.get("role") or m.get("from") or "").lower())
                content = m.get("content") if "content" in m else m.get("value")
                if role and isinstance(content, str) and content.strip():
                    msgs.append({"role": role, "content": content.strip()})
            return msgs if any(m["role"] == "assistant" for m in msgs) else None

    def pick(*names):
        for n in names:
            v = rec.get(n)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    prompt = pick("prompt", "instruction", "question", "input_text", "query", "user", "human")
    extra = pick("input", "context")
    answer = pick("response", "output", "answer", "completion", "assistant", "gpt", "chosen")
    if prompt and answer:
        if extra and extra != prompt:
            prompt = f"{prompt}\n\n{extra}"
        msgs = []
        if pick("system"):
            msgs.append({"role": "system", "content": pick("system")})
        return msgs + [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    return None


def parse_text(text: str, filename: str = "") -> list[list[dict]]:
    """Parse an uploaded file (.jsonl, .json, .csv or .txt) into conversations."""
    name = filename.lower()
    out: list[list[dict]] = []
    stripped = text.lstrip()

    if name.endswith(".csv"):
        for row in csv.DictReader(io.StringIO(text)):
            m = to_messages({k.strip().lower(): v for k, v in row.items() if k})
            if m:
                out.append(m)
        return out

    if name.endswith(".json") or (stripped.startswith("[") and not name.endswith(".txt")):
        try:
            data = json.loads(text)
            items = data if isinstance(data, list) else data.get("data") or data.get("examples") or [data]
            return [m for m in (to_messages(r) for r in items) if m]
        except json.JSONDecodeError:
            pass

    if name.endswith(".jsonl") or stripped.startswith("{"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                m = to_messages(json.loads(line))
            except json.JSONDecodeError:
                continue
            if m:
                out.append(m)
        if out:
            return out

    return parse_plain(text)


_TURN = re.compile(r"^\s*(?:#{1,3}\s*)?(user|human|you|me|prompt|q|assistant|ai|bot|octopus|response|answer|a)\s*[:：]\s*", re.I | re.M)


def parse_plain(text: str) -> list[list[dict]]:
    """Plain text: blocks separated by blank lines / '---', with 'User:' and 'Assistant:' turns."""
    convos = []
    for block in re.split(r"\n\s*(?:---+|===+)\s*\n|\n{3,}", text):
        parts = _TURN.split(block)
        msgs = []
        for i in range(1, len(parts) - 1, 2):
            who = parts[i].lower()
            role = "user" if who in ("user", "human", "you", "me", "prompt", "q") else "assistant"
            content = parts[i + 1].strip()
            if not content:
                continue
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += "\n\n" + content
            else:
                msgs.append({"role": role, "content": content})
        if any(m["role"] == "assistant" for m in msgs):
            convos.append(msgs)
    return convos


# ---------- stored datasets ----------

def _safe_name(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._ -]+", "", Path(name).stem).strip() or "dataset"
    return base[:60]


def save_upload(filename: str, text: str) -> dict:
    convos = parse_text(text, filename)
    if not convos:
        raise ValueError("No prompt and response pairs found. See the Train page for supported formats.")
    stem = _safe_name(filename)
    path = DATA_DIR / f"{stem}.jsonl"
    i = 2
    while path.exists():
        path = DATA_DIR / f"{stem}-{i}.jsonl"
        i += 1
    _write(path, convos)
    return describe(path)


def save_pairs(pairs: list[dict], name: str = "My examples") -> dict:
    """Pairs typed into the Train page are appended to one dataset file."""
    convos = [[{"role": "user", "content": p["prompt"].strip()}, {"role": "assistant", "content": p["response"].strip()}]
              for p in pairs if p.get("prompt", "").strip() and p.get("response", "").strip()]
    if not convos:
        raise ValueError("Add at least one prompt with a response.")
    path = DATA_DIR / f"{_safe_name(name)}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for c in convos:
            f.write(json.dumps({"messages": c}, ensure_ascii=False) + "\n")
    return describe(path)


def _write(path: Path, convos: list[list[dict]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in convos:
            f.write(json.dumps({"messages": c}, ensure_ascii=False) + "\n")


def describe(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    return {"id": f"file:{path.name}", "label": path.stem, "count": count,
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime)), "kind": "file"}


def list_datasets() -> list[dict]:
    files = [describe(p) for p in sorted(DATA_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)]
    builtins = [{"id": f"builtin:{k}", "label": v["label"], "blurb": v["blurb"], "license": v["license"], "kind": "builtin"}
                for k, v in BUILTIN_DATASETS.items()]
    return files + builtins


def delete_dataset(ds_id: str) -> None:
    if not ds_id.startswith("file:"):
        raise ValueError("Only your own datasets can be deleted")
    path = (DATA_DIR / ds_id[5:]).resolve()
    if path.parent != DATA_DIR.resolve() or not path.exists():
        raise FileNotFoundError(ds_id)
    path.unlink()


def load_examples(ds_ids: list[str], builtin_limit: int = 2000, log=print) -> list[list[dict]]:
    """Load and merge the chosen datasets into one shuffled list of conversations."""
    out: list[list[dict]] = []
    for ds_id in ds_ids:
        if ds_id.startswith("file:"):
            path = (DATA_DIR / ds_id[5:]).resolve()
            if path.parent != DATA_DIR.resolve():
                continue
            items = parse_text(path.read_text(encoding="utf-8"), path.name)
            log(f"Loaded {len(items)} examples from {path.stem}")
            out += items
        elif ds_id.startswith("builtin:"):
            meta = BUILTIN_DATASETS.get(ds_id[8:])
            if not meta:
                continue
            from datasets import load_dataset

            log(f"Downloading {meta['label']} (first time only)…")
            ds = load_dataset(meta["hf"], split=meta["split"])
            ds = ds.shuffle(seed=42).select(range(min(builtin_limit, len(ds))))
            items = [m for m in (to_messages(r) for r in ds) if m]
            log(f"Loaded {len(items)} examples from {meta['label']}")
            out += items
    random.Random(0).shuffle(out)
    return out
