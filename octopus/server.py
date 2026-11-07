"""Octopus web server: serves the UI and the chat, training, studio and import APIs."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from dataclasses import fields
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import data as datasets
from . import accounts, cloudsync, meshkit, prefs
from . import studio as studio_mod
from .config import ADAPTERS_DIR, BASE_MODELS, COLAB_URL, DEFAULT_BASE, HOME, INCOMING_DIR, WEB_DIR, settings
from .engine import StopFlag
from .trainer import TrainConfig, Trainer

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


# ---------- choose the chat backend ----------

def has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pick_engine():
    if settings.mock:
        from .mock import MockEngine

        return MockEngine(), "mock"
    if settings.backend == "transformers" or (settings.backend == "auto" and has_cuda()):
        from .engine import Engine

        return Engine(), "transformers"
    from .ollama_engine import OllamaEngine

    return OllamaEngine(), "ollama"


engine, BACKEND = pick_engine()
LOCAL_GPU = settings.mock or has_cuda()  # local training and studio need an NVIDIA GPU
trainer = Trainer(engine)
studio = studio_mod.Studio(free_others=engine.unload)
app = FastAPI(title="Octopus", docs_url=None, redoc_url=None, openapi_url=None)
_device_label = None


def device_label() -> str:
    global _device_label
    if _device_label is None:
        if settings.mock:
            _device_label = "Mock mode (no GPU used)"
        elif BACKEND == "transformers":
            from .engine import detect_device

            _device_label = detect_device().label
        else:
            _device_label = "Ollama (your GPU or CPU)"
    return _device_label


def list_brains() -> list[dict]:
    if BACKEND == "ollama":
        from .ollama_engine import list_brains as lb

        return lb()
    from .engine import list_brains as lb

    return lb()


# ---------- auth & request hygiene ----------

COOKIE = "octopus_session"
_attempts: dict[str, list] = {}


def is_local(req: Request) -> bool:
    """True only for requests made directly from this computer (not via a tunnel or proxy)."""
    host = (urlparse("//" + req.headers.get("host", "")).hostname or "").lower()
    forwarded = any(h in req.headers for h in ("x-forwarded-for", "forwarded", "cf-connecting-ip", "x-real-ip"))
    return bool(req.client) and req.client.host in LOOPBACK and host in LOOPBACK and not forwarded


def reachable(req: Request) -> bool:
    """Without OCTOPUS_PASSWORD only this computer may use Octopus (blocks tunnels and DNS rebinding)."""
    return bool(settings.password) or is_local(req)


def current_user(req: Request) -> dict | None:
    if not reachable(req):
        return None
    return accounts.user_for(req.cookies.get(COOKIE, ""))


def authed(req: Request) -> bool:
    return current_user(req) is not None


def need_auth(req: Request) -> dict:
    if not reachable(req):
        raise HTTPException(403, "Set OCTOPUS_PASSWORD to use Octopus from another device.")
    user = current_user(req)
    if not user:
        raise HTTPException(401, "Not signed in")
    return user


def need_owner(req: Request) -> dict:
    user = need_auth(req)
    if user["role"] != "owner":
        raise HTTPException(403, "Only the owner of this Octopus can do that.")
    return user


RAW_UPLOADS = {"/api/import"}


@app.middleware("http")
async def guard(request: Request, call_next):
    # Block cross-site requests: other websites open in your browser must not be
    # able to drive the Octopus API on your machine.
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        host = request.headers.get("host", "")
        if origin and urlparse(origin).netloc != host:
            return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)
        ctype = request.headers.get("content-type", "")
        expected = "application/octet-stream" if request.url.path in RAW_UPLOADS else "application/json"
        if request.method != "DELETE" and expected not in ctype:
            return JSONResponse({"detail": f"Expected {expected}"}, status_code=415)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    if not request.url.path.startswith("/files/"):
        resp.headers["X-Frame-Options"] = "DENY"
    return resp


ON_SPACE = bool(os.environ.get("SPACE_ID"))  # running on a Hugging Face Space, behind its proxy
# In GitHub Codespaces a forwarded port is private to the codespace's owner unless they make it public,
# so the very first account (the owner's) can be created there without the invite code.
ON_CODESPACE = os.environ.get("CODESPACES") == "true"


def code_needed(req: Request) -> bool:
    if is_local(req):
        return False
    return not (ON_CODESPACE and not accounts.has_accounts())
OLLAMA_BRAINS = HOME / "ollama-brains"  # copies of imported .gguf brains, so a cloud restart can bring them back


def _throttle(req: Request) -> tuple[str, int, float]:
    ip = req.client.host if req.client else "?"
    if ON_SPACE and req.headers.get("x-forwarded-for"):
        ip = req.headers["x-forwarded-for"].split(",")[0].strip()
    n, t = _attempts.get(ip, [0, time.time()])
    if time.time() - t > 900:
        n, t = 0, time.time()
    if n >= 10:
        raise HTTPException(429, "Too many attempts. Try again in a few minutes.")
    return ip, n, t


def _signed_in(req: Request, user: dict) -> JSONResponse:
    resp = JSONResponse({"ok": True, "user": accounts.public(user)})
    # On a Space, Octopus can also be opened inside huggingface.co's page (an iframe), which needs SameSite=None.
    resp.set_cookie(COOKIE, accounts.start_session(user["id"]), httponly=True, samesite="none" if ON_SPACE else "strict",
                    max_age=accounts.SESSION_DAYS * 86400, secure=ON_SPACE or req.url.scheme == "https")
    return resp


@app.post("/api/signup")
async def signup(req: Request):
    if not reachable(req):
        raise HTTPException(403, "Set OCTOPUS_PASSWORD to use Octopus from another device.")
    ip, n, t = _throttle(req)
    body = await req.json()
    if code_needed(req) and not hmac.compare_digest(str(body.get("code", "")), settings.password):
        _attempts[ip] = [n + 1, t]
        raise HTTPException(401, "That invite code isn't right.")
    try:
        user = await _in_thread(accounts.create, body.get("name", ""), body.get("username", ""), str(body.get("password", "")))
    except accounts.AccountError as exc:
        raise HTTPException(400, str(exc))
    return _signed_in(req, user)


@app.post("/api/login")
async def login(req: Request):
    if not reachable(req):
        raise HTTPException(403, "Set OCTOPUS_PASSWORD to use Octopus from another device.")
    ip, n, t = _throttle(req)
    body = await req.json()
    user = await _in_thread(accounts.verify, body.get("username", ""), str(body.get("password", "")))
    if not user:
        _attempts[ip] = [n + 1, t]
        raise HTTPException(401, "Wrong username or password.")
    _attempts.pop(ip, None)
    return _signed_in(req, user)


@app.post("/api/logout")
async def logout(req: Request):
    accounts.end_session(req.cookies.get(COOKIE, ""))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp


@app.get("/api/me")
def me(req: Request):
    return accounts.public(need_auth(req))


@app.post("/api/me")
async def update_me(req: Request):
    user = need_auth(req)
    b = await req.json()
    try:
        u = await _in_thread(accounts.update, user["id"], b.get("name"), b.get("old_password"), b.get("new_password"))
    except accounts.AccountError as exc:
        raise HTTPException(400, str(exc))
    if b.get("new_password"):  # other sessions were signed out; keep this one
        return _signed_in(req, u)
    return {"ok": True, "user": accounts.public(u)}


@app.get("/api/me/chats")
def my_chats(req: Request):
    return accounts.load_chats(need_auth(req)["id"])


@app.put("/api/me/chats")
async def save_my_chats(req: Request):
    user = need_auth(req)
    raw = await req.body()
    try:
        await _in_thread(accounts.save_chats, user["id"], raw)
    except (accounts.AccountError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True}


@app.get("/api/accounts")
def list_accounts(req: Request):
    need_owner(req)
    return {"users": accounts.list_users(), "signupsOpen": accounts.signups_open()}


@app.post("/api/accounts")
async def account_settings(req: Request):
    need_owner(req)
    b = await req.json()
    if "signupsOpen" in b:
        accounts.set_signups(bool(b["signupsOpen"]))
    return {"users": accounts.list_users(), "signupsOpen": accounts.signups_open()}


@app.delete("/api/accounts/{uid}")
def delete_account(uid: str, req: Request):
    me_ = need_owner(req)
    if uid == me_["id"]:
        raise HTTPException(400, "You can't remove your own account.")
    try:
        accounts.remove(uid)
    except accounts.AccountError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True}


# ---------- config & models ----------

def run_allowed(req: Request) -> bool:
    return settings.allow_code_run and is_local(req)


@app.get("/api/config")
def config(req: Request):
    user = current_user(req)
    ok = user is not None
    out = {"authed": ok, "needsSetup": not accounts.has_accounts(), "signupsOpen": accounts.signups_open(),
           "codeRequired": code_needed(req), "mock": settings.mock,
           "user": accounts.public(user) if user else None}
    if ok:
        allow = run_allowed(req)
        langs = []
        if allow:
            from .runner import available_languages

            langs = available_languages()
        ollama_ok = None
        if BACKEND == "ollama":
            from .ollama_engine import available

            ollama_ok = available()
        out.update({
            "device": device_label(),
            "backend": BACKEND,
            "ollamaRunning": ollama_ok,
            "localGpu": LOCAL_GPU,
            "colabUrl": COLAB_URL,
            "bases": [{"id": k, **v} for k, v in BASE_MODELS.items()],
            "defaultBase": settings.base_model or DEFAULT_BASE,
            "brains": list_brains(),
            "engine": engine.info(),
            "training": trainer.running,
            "allowRun": allow,
            "languages": langs,
            "studio": studio_mod.capabilities(),
        })
    return out


@app.post("/api/load")
async def load(req: Request):
    need_auth(req)
    body = await req.json()
    if trainer.running:
        raise HTTPException(409, "Training is running. Chat comes back when it finishes.")
    try:
        await _in_thread(engine.ensure_loaded, body.get("base"), body.get("brain") or None)
    except Exception as exc:
        raise HTTPException(500, f"Couldn't load the model: {exc}")
    return engine.info()


@app.get("/api/brains")
def brains(req: Request):
    need_auth(req)
    return list_brains()


@app.delete("/api/brains/{brain_id}")
def delete_brain(brain_id: str, req: Request):
    need_auth(req)
    if engine.adapter == brain_id:
        engine.unload()
    if BACKEND == "ollama":
        from .ollama_engine import delete_brain as db

        db(brain_id)
        (OLLAMA_BRAINS / f"{brain_id}.gguf").unlink(missing_ok=True)
        return {"ok": True}
    path = (ADAPTERS_DIR / brain_id).resolve()
    if path.parent != ADAPTERS_DIR.resolve() or not path.exists():
        raise HTTPException(404, "Not found")
    shutil.rmtree(path)
    return {"ok": True}


# ---------- chat ----------

_streams: dict[str, StopFlag] = {}


def sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/api/chat")
async def chat(req: Request):
    need_auth(req)
    if trainer.running:
        raise HTTPException(409, "Octopus is training right now. Chat comes back when it finishes.")
    if studio.busy:
        raise HTTPException(409, "Octopus is making an image or 3D model. Chat comes back when it finishes.")
    body = await req.json()
    messages = body.get("messages") or []
    mode = "code" if body.get("mode") == "code" else "chat"
    base, brain = body.get("base"), body.get("brain") or None
    temperature = float(body.get("temperature", 0.7 if mode == "chat" else 0.3))
    stream_id = str(body.get("id") or uuid.uuid4().hex)
    if not messages:
        raise HTTPException(400, "messages is required")

    stop = StopFlag()
    _streams[stream_id] = stop

    def events():
        try:
            if engine.status != "ready" or (base and engine.base_id != base) or engine.adapter != brain:
                if hasattr(engine, "prepare"):
                    for status in engine.prepare(base, brain):
                        yield sse({"t": "status", "d": status})
                else:
                    yield sse({"t": "status", "d": "Loading the model onto your GPU (the first time downloads it)…"})
                    engine.ensure_loaded(base, brain)
                yield sse({"t": "engine", "d": engine.info()})
            started = time.time()
            for piece in engine.stream(messages, mode=mode, stop=stop, temperature=temperature):
                if isinstance(piece, tuple):  # ("think", text) from models that reason before answering
                    yield sse({"t": "think", "d": piece[1]})
                else:
                    yield sse({"t": "text", "d": piece})
            yield sse({"t": "done", "ms": int((time.time() - started) * 1000), "stopped": stop.event.is_set()})
        except Exception as exc:
            yield sse({"t": "error", "d": str(exc)})
        finally:
            _streams.pop(stream_id, None)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/chat/stop")
async def chat_stop(req: Request):
    need_auth(req)
    body = await req.json()
    flag = _streams.get(str(body.get("id", "")))
    if flag:
        flag.event.set()
    return {"ok": True}


# ---------- Coding mode "Run" (opt-in, this computer only) ----------

@app.post("/api/run")
async def run(req: Request):
    need_auth(req)
    if not run_allowed(req):
        raise HTTPException(403, "Running code is off. Start Octopus with OCTOPUS_ALLOW_RUN=1 on this computer to enable it.")
    from .runner import run_code

    body = await req.json()
    return await _in_thread(run_code, body.get("language", ""), body.get("code", ""))


# ---------- Studio: images and 3D models ----------

def _studio_stream(gen):
    def events():
        for ev in gen:
            yield ": ping\n\n" if ev.get("t") == "ping" else sse(ev)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _studio_ready():
    if trainer.running:
        raise HTTPException(409, "Octopus is training right now. Try again when it finishes.")


def _where(b: dict) -> str:
    w = b.get("where") or prefs.load().get("studio_where", "auto")
    return w if w in ("auto", "local", "cloud") else "auto"


@app.post("/api/generate/image")
async def gen_image(req: Request):
    user = need_auth(req)
    _studio_ready()
    if not studio_mod.capabilities()["image"]:
        raise HTTPException(409, "Image generation isn't set up. Run start again to finish installing.")
    b = await req.json()
    return _studio_stream(studio.make_image(str(b.get("prompt", ""))[:2000], b.get("model", "sdxl"), b.get("aspect", "1:1"),
                                            int(b.get("count", 1)), b.get("seed"), _where(b), owner=user["id"]))


@app.post("/api/generate/model")
async def gen_model(req: Request):
    user = need_auth(req)
    _studio_ready()
    if not studio_mod.capabilities()["model3d"]:
        raise HTTPException(409, "3D modelling isn't set up. Run start again to finish installing.")
    b = await req.json()
    finish = meshkit.FinishOptions.from_request(b) if b.get("finish", True) else None
    return _studio_stream(studio.make_model(str(b.get("prompt", ""))[:2000], b.get("image_id"), b.get("preset", "quality"),
                                            bool(b.get("texture", True)), finish, b.get("seed"), b.get("image_name"),
                                            _where(b), owner=user["id"]))


@app.post("/api/assets/{aid}/refinish")
async def refinish_asset(aid: str, req: Request):
    """Rebuild a model's game files (budget, texture size, height, LODs) without generating it again."""
    need_auth(req)
    if studio.busy:
        raise HTTPException(409, "Octopus is making something. Try again when it finishes.")
    b = await req.json()
    try:
        return await _in_thread(studio_mod.refinish, aid, meshkit.FinishOptions.from_request(b))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/settings")
def get_settings(req: Request):
    need_auth(req)
    return prefs.public()


@app.post("/api/settings")
async def set_settings(req: Request):
    need_owner(req)
    b = await req.json()
    changes = {}
    if "hf_token" in b:
        tok = str(b.get("hf_token") or "").strip()
        if tok and not prefs.TOKEN_RE.fullmatch(tok):
            raise HTTPException(400, "That doesn't look like a Hugging Face token. It starts with hf_.")
        changes["hf_token"] = tok
    if b.get("studio_where") in ("auto", "local", "cloud"):
        changes["studio_where"] = b["studio_where"]
    prefs.save(**changes)
    return prefs.public()


@app.post("/api/upload-image")
async def upload_image(req: Request):
    user = need_auth(req)
    b = await req.json()
    try:
        return await _in_thread(studio_mod.save_upload, str(b.get("name", "image")), b.get("data", ""), user["id"])
    except Exception as exc:
        raise HTTPException(400, f"Couldn't read that image: {exc}")


@app.get("/api/assets")
def assets(req: Request, kind: str | None = None):
    return studio_mod.list_assets(kind, need_auth(req)["id"])


@app.delete("/api/assets/{aid}")
def delete_asset(aid: str, req: Request):
    need_auth(req)
    try:
        studio_mod.delete_asset(aid)
    except FileNotFoundError:
        raise HTTPException(404, "Not found")
    return {"ok": True}


@app.get("/files/{aid}/{name}")
def asset_file(aid: str, name: str, req: Request):
    need_auth(req)
    folder = studio_mod.asset_dir(aid)
    if not folder or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name) or name == "meta.json":
        raise HTTPException(404, "Not found")
    path = folder / name
    if not path.is_file():
        raise HTTPException(404, "Not found")
    media = {".glb": "model/gltf-binary", ".obj": "text/plain", ".mtl": "text/plain", ".stl": "model/stl",
             ".png": "image/png", ".zip": "application/zip", ".txt": "text/plain"}.get(path.suffix)
    resp = FileResponse(path, media_type=media)
    # Lets the sandboxed HTML preview in Coding mode load your models.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# ---------- importing results from the cloud notebook ----------

MAX_IMPORT = 20 * 1024**3


@app.post("/api/import")
async def import_file(req: Request, kind: str, filename: str, name: str = ""):
    """Import a brain (.gguf or adapter .zip) or a 3D model (.glb/.obj/.stl/.zip) produced elsewhere."""
    user = need_auth(req)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "-", filename)[-120:] or "upload.bin"
    ext = filename.lower().rsplit(".", 1)[-1]
    allowed = {"brain": {"gguf", "zip"}, "model": {"glb", "obj", "stl", "zip"}}.get(kind)
    if not allowed or ext not in allowed:
        raise HTTPException(400, f"Can't import a .{ext} file here.")

    tmp = INCOMING_DIR / f"{uuid.uuid4().hex}-{filename}"
    size = 0
    with tmp.open("wb") as f:
        async for chunk in req.stream():
            size += len(chunk)
            if size > MAX_IMPORT:
                f.close()
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, "File is too large")
            f.write(chunk)

    try:
        if kind == "brain":
            brain = re.sub(r"[^a-z0-9._-]+", "-", (name or filename.rsplit(".", 1)[0]).lower()).strip("-")[:40] or "brain"
            return await _in_thread(_import_brain, tmp, ext, brain)
        return await _in_thread(_import_model, tmp, ext, name or filename.rsplit(".", 1)[0], user["id"])
    finally:
        tmp.unlink(missing_ok=True)


def _keep_brain(path, brain: str) -> None:
    """When Octopus saves to the cloud, keep the brain's file so it survives a restart."""
    if cloudsync.enabled():
        OLLAMA_BRAINS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, OLLAMA_BRAINS / f"{brain}.gguf")


def _restore_brains() -> None:
    """Re-register brains whose files were restored from the cloud but aren't in Ollama yet."""
    from .ollama_engine import BRAIN_PREFIX, _brain_bases, import_gguf, installed_models

    have = {m.get("name", "").split(":")[0] for m in installed_models()}
    bases = _brain_bases()
    for f in sorted(OLLAMA_BRAINS.glob("*.gguf")):
        if BRAIN_PREFIX + f.stem not in have:
            print(f"     Restoring brain {f.stem}…")
            try:
                for _ in import_gguf(f, f.stem, bases.get(f.stem)):
                    pass
            except Exception as exc:
                print(f"     Couldn't restore brain {f.stem}: {exc}")


def _import_brain(tmp, ext: str, brain: str) -> dict:
    if ext == "gguf":
        if BACKEND != "ollama":
            raise HTTPException(400, "GGUF brains are for the Ollama engine. On an NVIDIA setup, import the adapter .zip instead.")
        from .ollama_engine import OllamaError, import_gguf

        try:
            for _ in import_gguf(tmp, brain):
                pass
        except OllamaError as exc:
            raise HTTPException(400, str(exc))
        _keep_brain(tmp, brain)
        return {"ok": True, "brain": brain}
    with zipfile.ZipFile(tmp) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        if not any(m.endswith("adapter_config.json") for m in members):
            raise HTTPException(400, "That zip doesn't contain a trained brain (adapter_config.json is missing).")
        cfg = json.loads(z.read(next(m for m in members if m.endswith("adapter_config.json"))))
        base = cfg.get("base_model_name_or_path")
        gguf = next((m for m in members if m.endswith(".gguf")), None)
        if BACKEND == "ollama":
            if not gguf:
                raise HTTPException(400, "This zip is for NVIDIA setups. For this computer, import the brain's .gguf file, "
                                         "or package it again with the latest cloud notebook.")
            from .ollama_engine import OllamaError, import_gguf

            part = tmp.with_name(tmp.name + "-" + gguf.rsplit("/", 1)[-1])
            try:
                with z.open(gguf) as src, part.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 8 << 20)
                for _ in import_gguf(part, brain, base):
                    pass
                _keep_brain(part, brain)
            except OllamaError as exc:
                raise HTTPException(400, str(exc))
            finally:
                part.unlink(missing_ok=True)
            return {"ok": True, "brain": brain}
        out = ADAPTERS_DIR / brain
        if out.exists():
            out = ADAPTERS_DIR / f"{brain}-{int(time.time())}"
        out.mkdir(parents=True)
        for m in members:
            if m.endswith(".gguf"):
                continue
            target = (out / m.split("/")[-1]).resolve()
            if target.parent == out.resolve():
                target.write_bytes(z.read(m))
    return {"ok": True, "brain": out.name}


def _import_model(tmp, ext: str, label: str, owner: str | None = None) -> dict:
    """Import a model made elsewhere (cloud notebook, Blender, another tool) and finish it for games."""
    aid, folder = studio_mod.new_asset("model", prompt=label[:80], preset="imported", source="import", where="import", owner=owner)
    ok_name = re.compile(r"[A-Za-z0-9_.-]{1,80}")
    try:
        if ext == "zip":
            with zipfile.ZipFile(tmp) as z:
                names = {m.split("/")[-1]: m for m in z.namelist() if not m.endswith("/")}
                packed = None
                if "meta.json" in names:
                    try:
                        packed = json.loads(z.read(names["meta.json"]))
                    except Exception:
                        packed = None
                if packed and packed.get("kind") == "model" and packed.get("files"):
                    # Already finished by Octopus (for example in the cloud notebook): copy it as is.
                    for base, m in names.items():
                        if ok_name.fullmatch(base) and base != "meta.json" and \
                                base.lower().rsplit(".", 1)[-1] in ("glb", "obj", "mtl", "stl", "png", "zip", "txt"):
                            (folder / base).write_bytes(z.read(m))
                    keep = {k: packed.get(k) for k in ("files", "stats", "finish", "reference", "sources", "engine", "textured")}
                    meta = {**studio_mod.read_meta(aid), **keep}
                    studio_mod.write_meta(folder, meta)
                    return meta
                models = [b for b in names if b.lower().rsplit(".", 1)[-1] in ("glb", "obj", "stl")]
                if not models:
                    raise HTTPException(400, "No 3D model found in that zip.")
                textured = [b for b in models if "textured" in b.lower()]
                pick = (textured or sorted(models, key=lambda b: (not b.lower().endswith(".glb"), b)))[0]
                # OBJ files need their .mtl and textures next to them.
                for base, m in names.items():
                    if ok_name.fullmatch(base) and base.lower().rsplit(".", 1)[-1] in ("glb", "obj", "mtl", "stl", "png", "jpg", "jpeg"):
                        (folder / f"in-{base}").write_bytes(z.read(m))
                src = folder / f"in-{pick}"
        else:
            src = folder / f"source-shape.{ext}"
            shutil.copy(tmp, src)
        shape = folder / f"source-shape{src.suffix}"
        if src != shape:
            src.rename(shape)
        meta = {**studio_mod.read_meta(aid), "sources": {"shape": shape.name, "textured": None}, "engine": "Imported"}
        meta.update(studio_mod.finish_asset(folder, shape, None, meshkit.FinishOptions(), lambda *a, **k: None))
        studio_mod.write_meta(folder, meta)
        return meta
    except HTTPException:
        studio_mod.delete_asset(aid)
        raise
    except Exception as exc:
        studio_mod.delete_asset(aid)
        raise HTTPException(400, f"Couldn't read that 3D model: {exc}")


# ---------- datasets ----------

@app.get("/api/datasets")
def get_datasets(req: Request):
    need_auth(req)
    return datasets.list_datasets()


@app.post("/api/datasets/upload")
async def upload(req: Request):
    need_auth(req)
    body = await req.json()
    try:
        return datasets.save_upload(body.get("filename", "dataset.txt"), body.get("text", ""))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/datasets/pairs")
async def add_pairs(req: Request):
    need_auth(req)
    body = await req.json()
    try:
        return datasets.save_pairs(body.get("pairs", []), body.get("name") or "My examples")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/datasets/export")
async def export_datasets(req: Request):
    """Combine your own datasets into one JSONL file to upload to the cloud notebook."""
    need_auth(req)
    body = await req.json()
    ids = [i for i in body.get("datasets", []) if str(i).startswith("file:")]
    convos = datasets.load_examples(ids, log=lambda *_: None)
    if not convos:
        raise HTTPException(400, "Pick at least one of your own datasets. Open datasets can be ticked in the notebook.")
    text = "".join(json.dumps({"messages": c}, ensure_ascii=False) + "\n" for c in convos)
    return StreamingResponse(iter([text]), media_type="application/x-ndjson",
                             headers={"Content-Disposition": 'attachment; filename="octopus-training.jsonl"'})


@app.delete("/api/datasets/{ds_id}")
def remove_dataset(ds_id: str, req: Request):
    need_auth(req)
    try:
        datasets.delete_dataset(ds_id)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True}


# ---------- training ----------

@app.post("/api/train/start")
async def train_start(req: Request):
    need_auth(req)
    if not LOCAL_GPU:
        raise HTTPException(409, "Training on this computer needs an NVIDIA GPU. Use the free cloud notebook instead.")
    if studio.busy:
        raise HTTPException(409, "Octopus is making an image or 3D model. Try again when it finishes.")
    body = await req.json()
    allowed = {f.name for f in fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in body.items() if k in allowed})
    try:
        trainer.start(cfg)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True}


@app.post("/api/train/stop")
async def train_stop(req: Request):
    need_auth(req)
    trainer.stop()
    return {"ok": True}


@app.get("/api/train/state")
def train_state(req: Request):
    need_auth(req)
    return {"running": trainer.running, **trainer.state, "events": len(trainer.events)}


@app.get("/api/train/events")
def train_events(req: Request, after: int = 0):
    need_auth(req)

    def gen():
        seq, idle = after, 0
        while True:
            evs = trainer.wait_events(seq, timeout=10)
            if not evs:
                idle += 1
                yield ": ping\n\n"
                if not trainer.running and idle > 1:
                    return
                continue
            idle = 0
            for ev in evs:
                seq = ev["seq"] + 1
                yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------- static UI ----------

async def _in_thread(fn, *args):
    import asyncio

    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


@app.on_event("shutdown")
def _save_on_shutdown() -> None:
    cloudsync.save()


@app.get("/healthz")
def health():
    return {"ok": True, "busy": studio.busy, "studio": studio.last}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def _warm_up_ollama() -> None:
    """Download the default chat model in the background so the first chat starts quickly."""
    import threading

    def run():
        try:
            from .ollama_engine import OLLAMA_TAGS, available, installed_models

            for _ in range(30):  # Ollama may still be starting up
                if available():
                    break
                time.sleep(2)
            else:
                return
            if OLLAMA_BRAINS.exists():
                _restore_brains()
            tag = OLLAMA_TAGS.get(settings.base_model or DEFAULT_BASE)
            if tag and not any(m.get("name", "").split(":latest")[0] == tag for m in installed_models()):
                print(f"     Downloading the chat model ({tag}) in the background…")
                for _ in engine.prepare(settings.base_model or DEFAULT_BASE, None):
                    pass
                print("     Chat model ready.")
        except Exception as exc:
            print(f"     Couldn't pre-download the chat model: {exc}")

    threading.Thread(target=run, daemon=True).start()


def _log_to_file() -> None:
    """Copy everything Octopus prints, and native crashes, to workspace/octopus.log for troubleshooting."""
    import faulthandler
    import sys

    from .config import HOME

    try:
        log = open(HOME / "octopus.log", "a", encoding="utf-8", buffering=1)
    except OSError:
        return
    log.write(f"\n===== Octopus started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    faulthandler.enable(log)

    class Tee:
        def __init__(self, stream):
            self.stream = stream

        def write(self, text):
            try:
                log.write(text)
            except Exception:
                pass
            try:
                return self.stream.write(text) if self.stream else len(text)
            except Exception:  # a console that can't show a character must never stop Octopus
                return len(text)

        def flush(self):
            for s in (self.stream, log):
                try:
                    s.flush()
                except Exception:
                    pass

        def __getattr__(self, name):
            return getattr(self.stream, name)

    sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)


def main() -> None:
    import uvicorn

    _log_to_file()

    local = settings.host in LOOPBACK
    if not local and not settings.password:
        raise SystemExit("Set OCTOPUS_PASSWORD before sharing Octopus on your network (see README), "
                         "or keep OCTOPUS_HOST=127.0.0.1.")
    cloudsync.start()
    print(f"\n  🐙 Octopus is running at http://{'localhost' if local else settings.host}:{settings.port}")
    print(f"     Chat engine: {BACKEND}{'' if LOCAL_GPU else ' · 3D and images use free cloud GPUs, training uses the cloud notebook'}\n")
    if BACKEND == "ollama":
        _warm_up_ollama()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
