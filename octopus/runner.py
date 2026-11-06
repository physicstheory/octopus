"""Coding mode's Run button: executes a snippet on this computer in a scratch folder with a time limit."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time

from .config import RUN_DIR, settings

LANGS = {
    "python": ([sys.executable, "-u"], ".py"),
    "py": ([sys.executable, "-u"], ".py"),
    "javascript": (["node"], ".js"),
    "js": (["node"], ".js"),
    "bash": (["bash"], ".sh"),
    "sh": (["bash"], ".sh"),
    "shell": (["bash"], ".sh"),
    "powershell": (["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"], ".ps1"),
}

MAX_OUTPUT = 60_000


def available_languages() -> list[str]:
    out = []
    for lang, (cmd, _) in LANGS.items():
        if shutil.which(cmd[0]) or cmd[0] == sys.executable:
            out.append(lang)
    return out


def run_code(language: str, code: str) -> dict:
    lang = (language or "").lower()
    if lang not in LANGS:
        return {"ok": False, "stdout": "", "stderr": f"Running {language or 'this'} code isn't supported. Try Python, JavaScript or Bash.", "code": -1, "ms": 0}
    cmd, ext = LANGS[lang]
    if not (shutil.which(cmd[0]) or cmd[0] == sys.executable):
        return {"ok": False, "stdout": "", "stderr": f"{cmd[0]} isn't installed on this computer.", "code": -1, "ms": 0}

    workdir = tempfile.mkdtemp(prefix="run-", dir=RUN_DIR)
    path = os.path.join(workdir, "main" + ext)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"}
    start = time.time()
    try:
        proc = subprocess.run(cmd + [path], cwd=workdir, capture_output=True, text=True, timeout=settings.run_timeout,
                              env=env, stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
        out, err, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = (exc.stderr or "") + f"\nStopped after {settings.run_timeout} seconds."
        out = out.decode() if isinstance(out, bytes) else out
        err = err.decode() if isinstance(err, bytes) else err
        rc = -9
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {"ok": rc == 0, "stdout": out[-MAX_OUTPUT:], "stderr": err[-MAX_OUTPUT:], "code": rc,
            "ms": int((time.time() - start) * 1000)}
