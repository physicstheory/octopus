"""Free cloud generation through public Hugging Face Spaces (ZeroGPU).

Used for images and 3D models when this computer has no NVIDIA GPU, so nobody
has to open a notebook. These are free, open-source models (Hunyuan3D-2.1 and
FLUX.1 schnell) running on Hugging Face's shared GPUs. Your prompt and picture
are sent to that Space. Free use has a daily GPU allowance: anonymous use gets
the smallest share, and a free Hugging Face access token (Settings) gives more.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from . import prefs

SPACES_3D = [
    # PBR textures (albedo + metallic + roughness), the best open 3D model with free hosting.
    {"space": "tencent/Hunyuan3D-2.1", "label": "Hunyuan3D 2.1 (PBR)"},
    {"space": "tencent/Hunyuan3D-2", "label": "Hunyuan3D 2.0"},
]
SPACE_IMAGE = {"space": "black-forest-labs/FLUX.1-schnell", "label": "FLUX.1 schnell"}

# Hunyuan3D gradio parameters, by name. Unknown parameters keep the Space's own defaults.
PRESETS_3D = {
    "fast": {"steps": 10, "octree_resolution": 256},
    "quality": {"steps": 30, "octree_resolution": 384},
}


class CloudError(RuntimeError):
    pass


def available() -> bool:
    import importlib.util

    return importlib.util.find_spec("gradio_client") is not None


def _client(space: str):
    from gradio_client import Client

    token = prefs.hf_token() or None
    kw = {"verbose": False}
    try:
        return Client(space, token=token, analytics_enabled=False, **kw)
    except TypeError:  # older gradio_client
        return Client(space, hf_token=token, **kw)


def _friendly(exc: Exception, what: str) -> CloudError:
    msg = str(exc)
    low = msg.lower()
    if "quota" in low or ("zerogpu" in low and "exceed" in low):
        hint = ("You've used today's free cloud GPU time. Add a free Hugging Face token in Settings for more, "
                "wait until tomorrow, or use the free cloud notebook.") if not prefs.hf_token() else (
                "You've used today's free cloud GPU time on your Hugging Face account. It refills daily, "
                "or use the free cloud notebook.")
        return CloudError(hint)
    if "sleeping" in low or "building" in low or "paused" in low or "runtime_error" in low:
        return CloudError(f"The free {what} service is starting up or paused right now. Try again in a minute.")
    if "connect" in low or "resolve" in low or "timed out" in low or "403" in low:
        return CloudError(f"Couldn't reach the free {what} service. Check your internet connection.")
    return CloudError(f"The free {what} service had a problem: {msg[:300]}")


def _endpoint(client, prefer: list[str]) -> tuple[str, list[dict]]:
    """Find an API endpoint by name and return its parameter list."""
    api = client.view_api(print_info=False, return_format="dict") or {}
    named = api.get("named_endpoints") or {}
    for name in prefer:
        if name in named:
            return name, named[name].get("parameters", [])
    for name in named:
        if any(p.strip("/") in name for p in prefer):
            return name, named[name].get("parameters", [])
    raise CloudError("This cloud service changed its interface. Update Octopus, or use the free cloud notebook.")


def _args(params: list[dict], values: dict) -> list:
    out = []
    for p in params:
        name = p.get("parameter_name") or ""
        label = (p.get("label") or "").lower().strip()
        if name in values:
            out.append(values[name])
        elif label.replace(" ", "_") in values:
            out.append(values[label.replace(" ", "_")])
        elif label in ("image prompt", "image", "input image") and "image" in values:
            out.append(values["image"])
        elif p.get("parameter_has_default"):
            out.append(p.get("parameter_default"))
        else:
            out.append(None)
    return out


def _paths(result) -> list[str]:
    """Every local file path inside a gradio result."""
    found = []

    def walk(x):
        if isinstance(x, str) and os.path.isfile(x):
            found.append(x)
        elif isinstance(x, dict):
            for k in ("path", "value", "name"):
                if isinstance(x.get(k), str) and os.path.isfile(x[k]):
                    found.append(x[k])
                    return
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(result)
    return found


def _wait(job, emit, stage: str, start: float, end: float, expect_s: float):
    """Wait for a gradio job, reporting queue position and a time-based progress estimate."""
    t0 = None
    while not job.done():
        try:
            st = job.status()
            code = str(getattr(st, "code", "")).split(".")[-1]
            if code in ("IN_QUEUE", "JOINING_QUEUE", "STARTING") and getattr(st, "rank", None) is not None:
                emit(f"{stage} (waiting in line, {st.rank + 1} ahead)", start)
            else:
                t0 = t0 or time.time()
                frac = min(0.95, (time.time() - t0) / expect_s)
                emit(stage, start + (end - start) * frac)
        except Exception:
            pass
        time.sleep(1.0)
    return job.result()


def make_image(prompt: str, width: int, height: int, seed: int | None, emit, start=0.0, end=95.0) -> Path:
    """FLUX.1 schnell on Hugging Face. Returns the path of the downloaded image."""
    try:
        client = _client(SPACE_IMAGE["space"])
        name, params = _endpoint(client, ["/infer"])
        values = {"prompt": prompt, "seed": seed or 0, "randomize_seed": seed is None,
                  "width": width, "height": height, "num_inference_steps": 4}
        job = client.submit(*_args(params, values), api_name=name)
        result = _wait(job, emit, "Painting in the free cloud…", start, end, 20)
    except CloudError:
        raise
    except Exception as exc:
        raise _friendly(exc, "image") from exc
    files = _paths(result)
    if not files:
        raise CloudError("The cloud image service returned nothing. Try again.")
    return Path(files[0])


def _too_long(exc: Exception) -> bool:
    """The Space asks for more GPU time per job than free accounts may use at all."""
    return "larger than the maximum allowed" in str(exc).lower()


def make_model(image_path: Path, preset: str, texture: bool, seed: int | None, emit, start=0.0, end=80.0) -> dict:
    """Hunyuan3D on Hugging Face. Returns {'shape': path, 'textured': path or None, 'space': label, 'note': str}.

    Textured jobs ask the Space for far more GPU time than shape-only ones (Hunyuan3D 2.1 asks
    for 270 s, more than a free account may use in one job; 2.0 asks for 135 s). So the order is:
    textured on 2.1, textured on 2.0, then the shape alone (about 10 s of GPU time), which still
    gives a full game-ready mesh with UVs, just without painted colour.
    """
    from gradio_client import handle_file

    spec = PRESETS_3D.get(preset, PRESETS_3D["quality"])
    plan = [(s, "/generation_all") for s in SPACES_3D] if texture else []
    plan += [(s, "/shape_generation") for s in SPACES_3D]
    clients: dict[str, object] = {}
    errors: list[CloudError] = []
    quota_hit = False
    for space, want in plan:
        textured_job = want == "/generation_all"
        try:
            emit(f"Connecting to {space['label']} in the free cloud…", start)
            if space["space"] not in clients:
                clients[space["space"]] = _client(space["space"])
            client = clients[space["space"]]
            name, params = _endpoint(client, [want])
            values = {
                "caption": None, "image": handle_file(str(image_path)),
                "mv_image_front": None, "mv_image_back": None, "mv_image_left": None, "mv_image_right": None,
                "steps": spec["steps"], "num_steps": spec["steps"], "num_inference_steps": spec["steps"],
                "guidance_scale": 5.0, "cfg_scale": 5.0, "seed": seed if seed is not None else 1234,
                "octree_resolution": spec["octree_resolution"], "check_box_rembg": True, "num_chunks": 8000,
                "randomize_seed": seed is None,
            }
            job = client.submit(*_args(params, values), api_name=name)
            stage = f"Sculpting{' and texturing' if textured_job else ''} with {space['label']}…"
            result = _wait(job, emit, stage, start, end, 150 if textured_job else 30)
            files = [f for f in _paths(result) if f.lower().endswith((".glb", ".obj", ".ply", ".gltf"))]
            if not files:
                raise CloudError("The 3D service returned no model.")
            textured = None
            if textured_job and len(files) > 1:
                textured = next((f for f in files if "textured" in os.path.basename(f).lower()), files[-1])
            shape = next((f for f in files if f != textured), files[0])
            note = ""
            if texture and not textured:
                note = ("Made without painted colour: the free cloud's textured jobs need more GPU time than is "
                        "left today. Add a free Hugging Face token in Settings for more, or press Try again tomorrow.")
            return {"shape": Path(shape), "textured": Path(textured) if textured else None,
                    "space": space["label"], "note": note}
        except CloudError as exc:
            errors.append(exc)
        except Exception as exc:
            if _too_long(exc):
                continue  # this Space's textured job is beyond free limits; try the next option quietly
            err = _friendly(exc, "3D")
            quota_hit = quota_hit or "GPU time" in str(err)
            errors.append(err)
        nxt = plan.index((space, want)) + 1
        if nxt < len(plan) and not (quota_hit and textured_job):
            emit(f"{space['label']} is unavailable, trying another…", start)
        elif nxt < len(plan):
            emit("Not enough free GPU time left for textures today. Making the shape alone…", start)
    quota = next((e for e in errors if "GPU time" in str(e)), None)
    raise quota or (errors[-1] if errors else CloudError("The free 3D services are unavailable right now."))


def keep(src: Path, folder: Path, name: str) -> Path:
    dst = folder / name
    shutil.copy(src, dst)
    return dst
