"""Accounts on this Octopus: each person signs in and gets their own chats and creations.

Stored in workspace/accounts.json. Passwords are salted and hashed with PBKDF2-SHA256;
only a hash of each sign-in token is kept. The first account made is the owner, who can
turn new sign-ups on or off and remove accounts in Settings.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

from .config import HOME

PATH = HOME / "accounts.json"
USERS_DIR = HOME / "users"
SESSION_DAYS = 30
ITERATIONS = 240_000
USERNAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{1,31}")
MAX_CHATS_BYTES = 25_000_000
_lock = threading.RLock()


class AccountError(ValueError):
    pass


def _load() -> dict:
    try:
        data = json.loads(PATH.read_text())
    except Exception:
        data = {}
    data.setdefault("users", {})
    data.setdefault("sessions", {})
    data.setdefault("signups_open", True)
    return data


def _save(data: dict) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(PATH)


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS).hex()


def _token_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def public(user: dict) -> dict:
    return {"id": user["id"], "name": user["name"], "username": user["username"], "role": user["role"],
            "created": user["created"]}


def has_accounts() -> bool:
    return bool(_load()["users"])


def signups_open() -> bool:
    data = _load()
    return not data["users"] or bool(data["signups_open"])


def _check_password(password: str) -> None:
    if len(password) < 6:
        raise AccountError("Use a password of at least 6 characters.")
    if len(password) > 200:
        raise AccountError("That password is too long.")


def create(name: str, username: str, password: str) -> dict:
    name = " ".join(str(name or "").split())[:60]
    username = str(username or "").strip().lower()
    if not name:
        raise AccountError("Enter your name.")
    if not USERNAME_RE.fullmatch(username):
        raise AccountError("Usernames are 2 to 32 letters, numbers, dots, dashes or underscores.")
    _check_password(password)
    with _lock:
        data = _load()
        if data["users"] and not data["signups_open"]:
            raise AccountError("New accounts are turned off. Ask the owner of this Octopus.")
        if any(u["username"] == username for u in data["users"].values()):
            raise AccountError("That username is taken.")
        uid = secrets.token_hex(6)
        salt = secrets.token_hex(16)
        user = {"id": uid, "name": name, "username": username, "salt": salt, "hash": _hash(password, salt),
                "role": "owner" if not data["users"] else "member", "created": int(time.time())}
        data["users"][uid] = user
        _save(data)
    (USERS_DIR / uid).mkdir(parents=True, exist_ok=True)
    return user


def verify(username: str, password: str) -> dict | None:
    username = str(username or "").strip().lower()
    data = _load()
    user = next((u for u in data["users"].values() if u["username"] == username), None)
    if not user:
        _hash(password or "x", "00" * 16)  # same work either way, so timing doesn't reveal usernames
        return None
    return user if hmac.compare_digest(_hash(password or "", user["salt"]), user["hash"]) else None


def start_session(uid: str) -> str:
    token = secrets.token_urlsafe(32)
    with _lock:
        data = _load()
        now = time.time()
        data["sessions"] = {k: v for k, v in data["sessions"].items() if v["exp"] > now}
        data["sessions"][_token_key(token)] = {"uid": uid, "exp": now + SESSION_DAYS * 86400}
        _save(data)
    return token


def end_session(token: str) -> None:
    with _lock:
        data = _load()
        if data["sessions"].pop(_token_key(token or ""), None):
            _save(data)


def user_for(token: str) -> dict | None:
    if not token:
        return None
    data = _load()
    s = data["sessions"].get(_token_key(token))
    if not s or s["exp"] < time.time():
        return None
    return data["users"].get(s["uid"])


def update(uid: str, name: str | None = None, old_password: str | None = None, new_password: str | None = None) -> dict:
    with _lock:
        data = _load()
        user = data["users"].get(uid)
        if not user:
            raise AccountError("That account no longer exists.")
        if name is not None:
            name = " ".join(str(name).split())[:60]
            if not name:
                raise AccountError("Enter your name.")
            user["name"] = name
        if new_password:
            if not hmac.compare_digest(_hash(old_password or "", user["salt"]), user["hash"]):
                raise AccountError("Your current password isn't right.")
            _check_password(new_password)
            user["salt"] = secrets.token_hex(16)
            user["hash"] = _hash(new_password, user["salt"])
            # sign out everywhere else
            data["sessions"] = {k: v for k, v in data["sessions"].items() if v["uid"] != uid}
        _save(data)
        return user


def list_users() -> list[dict]:
    return [public(u) for u in sorted(_load()["users"].values(), key=lambda u: u["created"])]


def set_signups(open_: bool) -> None:
    with _lock:
        data = _load()
        data["signups_open"] = bool(open_)
        _save(data)


def remove(uid: str) -> None:
    with _lock:
        data = _load()
        user = data["users"].get(uid)
        if not user:
            raise AccountError("That account no longer exists.")
        if user["role"] == "owner":
            raise AccountError("The owner account can't be removed.")
        del data["users"][uid]
        data["sessions"] = {k: v for k, v in data["sessions"].items() if v["uid"] != uid}
        _save(data)
    shutil.rmtree(USERS_DIR / uid, ignore_errors=True)


# ---------- each person's chats ----------

def _chats_path(uid: str) -> Path:
    return USERS_DIR / uid / "chats.json"


def load_chats(uid: str) -> list:
    try:
        return json.loads(_chats_path(uid).read_text())
    except Exception:
        return []


def save_chats(uid: str, raw: bytes) -> None:
    if len(raw) > MAX_CHATS_BYTES:
        raise AccountError("Your chat history is too large to save. Delete some old chats.")
    chats = json.loads(raw)
    if not isinstance(chats, list):
        raise AccountError("Expected a list of chats.")
    path = _chats_path(uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(chats))
    tmp.replace(path)
