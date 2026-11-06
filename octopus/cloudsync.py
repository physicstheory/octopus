"""Keep Octopus's data safe when it runs on a free Hugging Face Space.

A free Space's disk is wiped whenever it restarts or wakes from sleep. With a write token in the
Space's HF_TOKEN secret, Octopus keeps its workspace (accounts, chats, datasets, 3D models, images,
imported brains and settings) in a private Hugging Face dataset, restores it at start-up and saves
changes every minute and when the Space shuts down.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .config import HOME

SKIP_DIRS = {"incoming", "runs", ".cache", ".huggingface"}
SKIP_FILES = {"octopus.log"}
INTERVAL = 60

_lock = threading.Lock()
_seen: dict[str, tuple[int, int]] = {}
_status = {"enabled": False, "repo": None, "last": None, "error": None}


def repo_id() -> str | None:
    """OCTOPUS_SYNC_REPO, or "<owner>/octopus-data" when running on a Space with a token."""
    explicit = os.environ.get("OCTOPUS_SYNC_REPO", "").strip()
    if explicit:
        return explicit
    space = os.environ.get("SPACE_ID", "")
    if space and os.environ.get("HF_TOKEN"):
        return space.split("/")[0] + "/octopus-data"
    return None


def enabled() -> bool:
    return repo_id() is not None


def status() -> dict:
    return dict(_status)


def _files() -> dict[str, tuple[int, int]]:
    out = {}
    for root, dirs, files in os.walk(HOME):
        rel_root = Path(root).relative_to(HOME)
        dirs[:] = [d for d in dirs if not (rel_root == Path(".") and d in SKIP_DIRS) and not d.startswith(".")]
        for f in files:
            if f in SKIP_FILES or f.endswith((".tmp", ".part", ".lock")) or f.startswith("."):
                continue
            p = Path(root) / f
            try:
                st = p.stat()
            except OSError:
                continue
            out[p.relative_to(HOME).as_posix()] = (st.st_size, int(st.st_mtime))
    return out


def restore(log=print) -> None:
    """Create the private dataset if needed and copy its files into the workspace."""
    repo = repo_id()
    if not repo:
        return
    from huggingface_hub import HfApi, snapshot_download

    _status.update(enabled=True, repo=repo)
    api = HfApi()
    try:
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        snapshot_download(repo, repo_type="dataset", local_dir=str(HOME))
        _seen.update(_files())
        log(f"     Restored your Octopus data from {repo} ({len(_seen)} files)")
    except Exception as exc:  # keep running; saving will retry
        _status["error"] = f"Couldn't restore saved data: {exc}"
        log(f"     {_status['error']}")


def save(log=print) -> None:
    """Upload what changed since the last save, and remove what was deleted."""
    repo = repo_id()
    if not repo:
        return
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    with _lock:
        now = _files()
        changed = [k for k, v in now.items() if _seen.get(k) != v]
        gone = [k for k in _seen if k not in now]
        if not changed and not gone:
            return
        ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(HOME / k)) for k in changed]
        ops += [CommitOperationDelete(path_in_repo=k) for k in gone]
        try:
            HfApi().create_commit(repo, ops, repo_type="dataset", commit_message=f"Octopus: {len(changed)} saved, {len(gone)} removed")
            for k in changed:
                _seen[k] = now[k]
            for k in gone:
                _seen.pop(k, None)
            _status.update(last=time.strftime("%Y-%m-%d %H:%M:%S"), error=None)
        except Exception as exc:
            _status["error"] = f"Couldn't save to {repo}: {exc}"
            log(f"     {_status['error']}")


def start(log=print) -> None:
    """Restore now, then save in the background."""
    if not enabled():
        return
    restore(log)

    def loop():
        while True:
            time.sleep(INTERVAL)
            save(log)

    threading.Thread(target=loop, daemon=True, name="cloudsync").start()
