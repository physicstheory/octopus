"""Fold a trained LoRA brain into its base model's weights, one file at a time.

peft's merge_and_unload() loads the whole model into memory, which an 8B model doesn't fit
on a free Colab (about 12 GB of RAM). This streams the base model's safetensors shards
instead, so it only ever holds one shard plus the small adapter.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

_SKIP = {"model", "language_model"}


def _norm(key: str) -> str:
    """'model.language_model.layers.0.mlp.up_proj' and 'language_model.model.layers.0.mlp.up_proj' -> 'layers.0.mlp.up_proj'."""
    return ".".join(p for p in key.split(".") if p not in _SKIP)


def merge(base_id: str, adapter_dir: str | Path, out_dir: str | Path, log=print) -> Path:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / (rank ** 0.5 if cfg.get("use_rslora") else rank)

    lora = load_file(str(adapter_dir / "adapter_model.safetensors"))
    pairs: dict[str, dict] = {}
    for k, v in lora.items():
        m = re.match(r"(?:base_model\.model\.)?(.+)\.lora_([AB])(?:\.default)?\.weight$", k)
        if m:
            pairs.setdefault(_norm(m.group(1)) + ".weight", {})[m.group(2)] = v.float()
    log(f"Adapter has {len(pairs)} layers to fold in (rank {rank})")

    base = Path(snapshot_download(base_id, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.tiktoken"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.suffix != ".safetensors":
            shutil.copy(f, out_dir / f.name)

    done = set()
    shards = sorted(base.glob("*.safetensors"))
    for i, shard in enumerate(shards, 1):
        tensors = load_file(str(shard))
        for key, w in tensors.items():
            ab = pairs.get(_norm(key))
            if ab and "A" in ab and "B" in ab:
                if _norm(key) in done:
                    raise RuntimeError(f"Two base weights match {key}")
                delta = (ab["B"] @ ab["A"]) * scale
                tensors[key] = (w.float() + delta).to(w.dtype)
                done.add(_norm(key))
        save_file(tensors, str(out_dir / shard.name), metadata={"format": "pt"})
        log(f"Merged shard {i}/{len(shards)}")
        del tensors
    missing = set(pairs) - done
    if missing:
        raise RuntimeError(f"{len(missing)} adapter layers didn't match the base model, e.g. {sorted(missing)[0]}")
    return out_dir
