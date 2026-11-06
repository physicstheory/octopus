"""Octopus web server: serves the UI and the chat and training APIs."""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import time
import uuid
from dataclasses import fields
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import data as datasets
from .config import ADAPTERS_DIR, BASE_MODELS, DEFAULT_BASE, WEB_DIR, settings
from .trainer import TrainConfig, Trainer

if settings.mock:
    from .mock import MockEngine as _Engine

    def detect():
        return "Mock mode (no GPU used)"
else:
    from .engine import Engine as _Engine, detect_device

    def detect():
        try:
            return detect_device().label
        except Exception as exc:
            return f"Unknown ({exc})"

from .engine import StopFlag, list_brains

engine = _Engine()
trainer = Trainer(engine)
app = FastAPI(title="Octopus", docs_url=None, redoc_url=None, openapi_url=None)
_device_label = None
LOOPBACK = {"127.0.0.1", "::1", "localhost"}

# ---------- auth & request hygiene ----------

COOKIE = "octopus_session"
_secret = hashlib.sha256(("octopus:" + settings.password).encode()).digest()
_token = hmac.new(_secret, b"ok", hashlib.sha256).hexdigest()
_attempts: dict[str, list] = {}


def authed(req: Request) -> bool:
    if not settings.password:
        return is_local(req)
    return hmac.compare_digest(req.cookies.get(COOKIE, ""), _token)


def need_auth(req: Request) -> None:
    if not settings.password and not is_local(req):
        # Without a password, only this computer may use Octopus (blocks tunnels and DNS rebinding).
        raise HTTPException(403, "Set OCTOPUS_PASSWORD to use Octopus from another device.")
    if not authed(req):
        raise HTTPException(401, "Not signed in")


@app.middleware("http")
async def guard(request: Request, call_next):
    # Block cross-site requests: other websites open in your browser must not be
    # able to drive the Octopus API on your machine.
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        host = request.headers.get("host", "")
        if origin and urlparse(origin).netloc != host:
            return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)
        if "application/json" not in request.headers.get("content-type", ""):
            return JSONResponse({"detail": "Expected JSON"}, status_code=415)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    return resp


@app.post("/api/login")
async def login(req: Request):
    ip = req.client.host if req.client else "?"
    n, t = _attempts.get(ip, [0, time.time()])
    if time.time() - t > 900:
        n, t = 0, time.time()
    if n >= 10:
        raise HTTPException(429, "Too many attempts. Try again in a few minutes.")
    body = await req.json()
    if not settings.password or hmac.compare_digest(str(body.get("password", "")), settings.password):
        _attempts.pop(ip, None)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, _token, httponly=True, samesite="strict", max_age=60 * 60 * 24 * 30,
                        secure=req.url.scheme == "https")
        return resp
    _attempts[ip] = [n + 1, t]
    raise HTTPException(401, "Wrong password")


@app.post("/api/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp


# ---------- config & models ----------

def is_local(req: Request) -> bool:
    """True only for requests made directly from this computer (not via a tunnel or proxy)."""
    host = (urlparse("//" + req.headers.get("host", "")).hostname or "").lower()
    forwarded = any(h in req.headers for h in ("x-forwarded-for", "forwarded", "cf-connecting-ip", "x-real-ip"))
    return bool(req.client) and req.client.host in LOOPBACK and host in LOOPBACK and not forwarded


def run_allowed(req: Request) -> bool:
    return settings.allow_code_run and is_local(req)


@app.get("/api/config")
def config(req: Request):
    global _device_label
    ok = authed(req)
    out = {"authed": ok, "passwordRequired": bool(settings.password), "mock": settings.mock}
    if ok:
        if _device_label is None:
            _device_label = detect()
        allow = run_allowed(req)
        langs = []
        if allow:
            from .runner import available_languages

            langs = available_languages()
        out.update({
            "device": _device_label,
            "bases": [{"id": k, **v} for k, v in BASE_MODELS.items()],
            "defaultBase": settings.base_model or DEFAULT_BASE,
            "brains": list_brains(),
            "engine": engine.info(),
            "training": trainer.running,
            "allowRun": allow,
            "languages": langs,
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
    path = (ADAPTERS_DIR / brain_id).resolve()
    if path.parent != ADAPTERS_DIR.resolve() or not path.exists():
        raise HTTPException(404, "Not found")
    if engine.adapter == brain_id:
        engine.unload()
    shutil.rmtree(path)
    return {"ok": True}


# ---------- chat ----------

_streams: dict[str, StopFlag] = {}


@app.post("/api/chat")
async def chat(req: Request):
    need_auth(req)
    if trainer.running:
        raise HTTPException(409, "Octopus is training right now. Chat comes back when it finishes.")
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
        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n"

        try:
            if engine.status != "ready" or (base and engine.base_id != base) or engine.adapter != brain:
                yield sse({"t": "status", "d": "Loading the model onto your GPU (the first time downloads it)…"})
                engine.ensure_loaded(base, brain)
                yield sse({"t": "engine", "d": engine.info()})
            started, count = time.time(), 0
            for piece in engine.stream(messages, mode=mode, stop=stop, temperature=temperature):
                count += 1
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


@app.get("/healthz")
def health():
    return {"ok": True}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


def main() -> None:
    import uvicorn

    local = settings.host in LOOPBACK
    if not local and not settings.password:
        raise SystemExit("Set OCTOPUS_PASSWORD before sharing Octopus on your network (see README), "
                         "or keep OCTOPUS_HOST=127.0.0.1.")
    print(f"\n  🐙 Octopus is running at http://{'localhost' if local else settings.host}:{settings.port}\n")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
