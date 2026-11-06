"""Small settings saved on this computer (workspace/settings.json), changed from the Settings dialog."""

from __future__ import annotations

import json
import os
import re
import threading

from .config import HOME

PATH = HOME / "settings.json"
_lock = threading.Lock()
DEFAULTS = {"hf_token": "", "studio_where": "auto"}
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,80}")


def load() -> dict:
    try:
        return {**DEFAULTS, **json.loads(PATH.read_text())}
    except Exception:
        return dict(DEFAULTS)


def save(**changes) -> dict:
    with _lock:
        data = {**load(), **{k: v for k, v in changes.items() if k in DEFAULTS}}
        PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(PATH)
        return data


def hf_token() -> str:
    return load().get("hf_token") or os.environ.get("HF_TOKEN", "")


def public() -> dict:
    """Settings safe to send to the browser (the token is never sent back)."""
    data = load()
    tok = hf_token()
    return {"hasToken": bool(tok), "tokenHint": (tok[:5] + "…" + tok[-3:]) if tok else "",
            "tokenFromEnv": bool(os.environ.get("HF_TOKEN")) and not data.get("hf_token"),
            "studioWhere": data.get("studio_where", "auto")}
