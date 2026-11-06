"""Octopus Studio: local image generation and image/text-to-3D modelling.

Images use open diffusion models (SDXL, FLUX.1 schnell) and 3D models use Tencent's
Hunyuan3D. They run on your NVIDIA GPU when you have one (after install_studio),
otherwise on free cloud GPUs through Hugging Face (see cloud.py). Every 3D model is
then finished on your own computer by meshkit.py: cleaned, reduced to a game
budget, UV unwrapped, PBR-baked and split into LODs. Results go to workspace/assets.
"""

from __future__ import annotations

import base64
import gc
import importlib.util
import io
import json
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable, Iterator

from . import cloud, meshkit
from .config import HOME, settings

ASSETS_DIR = HOME / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_MODELS = {
    "sdxl": {
        "hf": "stabilityai/stable-diffusion-xl-base-1.0", "label": "Standard · SDXL",
        "blurb": "Good quality, about 8 GB VRAM.", "steps": 30, "guidance": 6.0, "vram": 8,
        "license": "CreativeML Open RAIL++-M (commercial use allowed)",
    },
    "flux": {
        "hf": "black-forest-labs/FLUX.1-schnell", "label": "Best · FLUX.1 schnell",
        "blurb": "Sharpest images, 4 steps. 12 GB+ VRAM (uses CPU offload).", "steps": 4, "guidance": 0.0, "vram": 12,
        "license": "Apache 2.0",
    },
}

MODEL3D = {
    "fast": {
        "label": "Fast · Hunyuan3D-2 mini turbo", "blurb": "Shape in under a minute. About 6 GB VRAM.",
        "repo": "tencent/Hunyuan3D-2mini", "subfolder": "hunyuan3d-dit-v2-mini-turbo", "steps": 5, "octree": 256, "vram": 6,
    },
    "quality": {
        "label": "Quality · Hunyuan3D-2", "blurb": "Most detailed shape. About 10 GB, 16 GB with textures.",
        "repo": "tencent/Hunyuan3D-2", "subfolder": "hunyuan3d-dit-v2-0", "steps": 30, "octree": 380, "vram": 10,
    },
}
PAINT_REPO = "tencent/Hunyuan3D-2"
LICENSE_3D = ("Tencent Hunyuan 3D Community License: free to use outside the EU, UK and South Korea; "
              "products with over 1M monthly users need a licence from Tencent.")

TEXT_TO_3D_SUFFIX = (", a single complete object, centered, full view, isolated on a plain white background, "
                     "3D game asset render, soft studio lighting, no shadows, no text")


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


_CUDA: bool | None = None


def has_cuda() -> bool:
    global _CUDA
    if _CUDA is None:
        try:
            import torch

            _CUDA = bool(torch.cuda.is_available())
        except Exception:
            _CUDA = False
    return _CUDA


def local_ok(kind: str) -> bool:
    if not has_cuda():
        return False
    return has("diffusers") if kind == "image" else has("hy3dgen")


def resolve_where(kind: str, requested: str | None) -> str:
    """Where a job runs: 'mock', 'local' (your NVIDIA GPU) or 'cloud' (free Hugging Face GPUs)."""
    if settings.mock:
        return "mock"
    requested = requested or "auto"
    if requested == "local" and local_ok(kind):
        return "local"
    if requested == "cloud" and cloud.available():
        return "cloud"
    if local_ok(kind):
        return "local"
    if cloud.available():
        return "cloud"
    raise RuntimeError("Can't make this here: there's no NVIDIA GPU and the cloud helper (gradio_client) "
                       "isn't installed. Run start again to finish installing.")


def capabilities() -> dict:
    texture = False
    if has("hy3dgen"):
        texture = has("custom_rasterizer")
    cloud_ok = cloud.available()
    return {
        "image": settings.mock or local_ok("image") or cloud_ok,
        "model3d": settings.mock or local_ok("model3d") or cloud_ok,
        "texture": settings.mock or texture or cloud_ok,
        "local": {"image": local_ok("image"), "model3d": local_ok("model3d")},
        "cloud": cloud_ok,
        "cloud3d": [s["label"] for s in cloud.SPACES_3D],
        "imageModels": [{"id": k, **{x: v[x] for x in ("label", "blurb", "vram", "license")}} for k, v in IMAGE_MODELS.items()],
        "modelPresets": [{"id": k, "label": v["label"], "blurb": v["blurb"], "vram": v["vram"]} for k, v in MODEL3D.items()],
        "budgets": [{"id": k, "label": v["label"], "tris": v["tris"]} for k, v in meshkit.BUDGETS.items()],
        "textureSizes": meshkit.TEXTURE_SIZES,
        "license3d": LICENSE_3D,
    }


# ---------- assets ----------

def new_asset(kind: str, **meta) -> tuple[str, Path]:
    aid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    folder = ASSETS_DIR / aid
    folder.mkdir(parents=True)
    write_meta(folder, {"id": aid, "kind": kind, "created": time.strftime("%Y-%m-%d %H:%M"), **meta})
    return aid, folder


def write_meta(folder: Path, meta: dict) -> None:
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))


def read_meta(aid: str) -> dict | None:
    folder = asset_dir(aid)
    if not folder:
        return None
    try:
        return json.loads((folder / "meta.json").read_text())
    except Exception:
        return None


def asset_dir(aid: str) -> Path | None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", aid or ""):
        return None
    folder = (ASSETS_DIR / aid).resolve()
    return folder if folder.parent == ASSETS_DIR.resolve() and folder.is_dir() else None


def list_assets(kind: str | None = None, owner: str | None = None) -> list[dict]:
    out = []
    for folder in sorted(ASSETS_DIR.iterdir(), reverse=True):
        meta = read_meta(folder.name)
        if (meta and (kind is None or meta.get("kind") == kind) and meta.get("files")
                and (owner is None or meta.get("owner") in (owner, None))):
            out.append(meta)
    return out


def delete_asset(aid: str) -> None:
    folder = asset_dir(aid)
    if not folder:
        raise FileNotFoundError(aid)
    shutil.rmtree(folder)


def save_upload(name: str, data_url: str, owner: str | None = None) -> dict:
    """Save an image the user attached (data: URL) as an 'upload' asset."""
    from PIL import Image

    m = re.match(r"data:image/[a-zA-Z+.-]+;base64,(.*)", data_url or "", re.S)
    if not m:
        raise ValueError("Expected a PNG, JPG or WebP image")
    raw = base64.b64decode(m.group(1))
    if len(raw) > 20_000_000:
        raise ValueError("Image is over 20 MB")
    img = Image.open(io.BytesIO(raw))
    img.load()
    img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
    img.thumbnail((1536, 1536))
    aid, folder = new_asset("upload", name=name[:80], owner=owner)
    img.save(folder / "image.png")
    meta = {**read_meta(aid), "files": {"png": "image.png"}}
    write_meta(folder, meta)
    return meta


# ---------- GPU sharing ----------

class Studio:
    """Holds the image and 3D pipelines; only one heavy job runs at a time."""

    def __init__(self, free_others: Callable[[], None]) -> None:
        self.lock = threading.Lock()
        self.free_others = free_others
        self.image_pipe = None
        self.image_model = None
        self.shape_pipe = None
        self.shape_preset = None
        self.paint_pipe = None
        self.rembg = None
        self.last: dict | None = None  # outcome of the latest job, for a browser that lost its connection

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    def unload(self) -> None:
        self.image_pipe = self.shape_pipe = self.paint_pipe = self.rembg = None
        self.image_model = self.shape_preset = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def run(self, job: Callable[[Callable[..., None]], dict]) -> Iterator[dict]:
        """Run a job in a worker thread and yield its progress events."""
        if not self.lock.acquire(blocking=False):
            yield {"t": "error", "d": "Octopus is already making something. Wait for it to finish."}
            return
        q: queue.Queue = queue.Queue()

        def emit(stage: str, pct: float | None = None, **extra) -> None:
            q.put({"t": "progress", "stage": stage, "pct": pct, **extra})

        def worker() -> None:
            try:
                asset = job(emit)
                self.last = {"asset": asset, "ended": time.time()}
                q.put({"t": "done", "asset": asset})
            except BaseException as exc:
                traceback.print_exc()
                msg = str(exc) or exc.__class__.__name__
                if isinstance(exc, MemoryError):
                    msg = "Your computer ran out of memory. Try a smaller triangle budget or texture size."
                self.last = {"error": msg, "ended": time.time()}
                q.put({"t": "error", "d": msg})
            finally:
                self.lock.release()
                q.put(None)

        # The job owns the lock until it really ends, so it keeps going (and nothing else starts)
        # even if the browser's connection drops; the page then picks up the result from /healthz.
        self.last = {"running": True, "started": time.time()}
        threading.Thread(target=worker, daemon=True).start()
        while True:
            try:
                ev = q.get(timeout=15)
            except queue.Empty:
                yield {"t": "ping"}
                continue
            if ev is None:
                break
            yield ev

    # ---------- images ----------

    def _image_pipeline(self, key: str, emit):
        if self.image_pipe is not None and self.image_model == key:
            return self.image_pipe
        import torch
        from diffusers import AutoPipelineForText2Image

        from .engine import detect_device

        self.unload()
        spec = IMAGE_MODELS[key]
        dev = detect_device()
        emit(f"Loading {spec['label']} (first time downloads it)…", 2)
        kwargs = {"torch_dtype": torch.bfloat16 if key == "flux" and dev.name == "cuda" else (torch.float16 if dev.name != "cpu" else torch.float32)}
        if key == "sdxl" and dev.name != "cpu":
            kwargs.update(variant="fp16", use_safetensors=True)
        pipe = AutoPipelineForText2Image.from_pretrained(spec["hf"], **kwargs)
        if dev.name == "cuda" and (key == "flux" or settings.extra.get("offload")):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(dev.name)
        self.image_pipe, self.image_model = pipe, key
        return pipe

    def _generate_images(self, prompt: str, model: str, width: int, height: int, count: int, seed: int | None, emit,
                         negative: str = "") -> list:
        import torch

        spec = IMAGE_MODELS[model]
        pipe = self._image_pipeline(model, emit)
        steps = spec["steps"]

        def on_step(p, step, timestep, kw):
            emit("Painting…", 5 + 90 * (step + 1) / steps)
            return kw

        gen = torch.Generator("cpu").manual_seed(seed if seed is not None else int(time.time()) % 2**31)
        args = dict(prompt=prompt, width=width, height=height, num_inference_steps=steps, guidance_scale=spec["guidance"],
                    num_images_per_prompt=count, generator=gen, callback_on_step_end=on_step)
        if model == "sdxl" and negative:
            args["negative_prompt"] = negative
        return pipe(**args).images

    def _cloud_images(self, prompt: str, w: int, h: int, count: int, seed, emit, start=5.0, end=95.0) -> list:
        from PIL import Image

        out = []
        for i in range(count):
            a = start + (end - start) * i / count
            b = start + (end - start) * (i + 1) / count
            path = cloud.make_image(prompt, w, h, None if seed is None else seed + i, emit, a, b)
            img = Image.open(path)
            img.load()
            out.append(img.convert("RGB"))
        return out

    def make_image(self, prompt: str, model: str = "sdxl", aspect: str = "1:1", count: int = 1, seed: int | None = None,
                   where: str = "auto", owner: str | None = None):
        model = model if model in IMAGE_MODELS else "sdxl"
        w, h = {"1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344), "4:3": (1152, 896)}.get(aspect, (1024, 1024))
        count = max(1, min(int(count), 4))

        def job(emit):
            place = resolve_where("image", where)
            if place == "local":
                self.free_others()
            used = "flux" if place == "cloud" else model
            aid, folder = new_asset("image", prompt=prompt, model=used, aspect=aspect, where=place, owner=owner)
            try:
                if place == "mock":
                    images = _mock_images(prompt, w, h, count, emit)
                elif place == "cloud":
                    images = self._cloud_images(prompt, w, h, count, seed, emit)
                else:
                    images = self._generate_images(prompt, model, w, h, count, seed, emit,
                                                   negative="blurry, low quality, watermark, text")
            except Exception:
                shutil.rmtree(folder, ignore_errors=True)
                raise
            files = []
            for i, img in enumerate(images):
                name = f"image-{i + 1}.png"
                img.save(folder / name)
                files.append(name)
            meta = {**read_meta(aid), "files": {"png": files[0]}, "images": files}
            write_meta(folder, meta)
            emit("Done", 100)
            return meta

        return self.run(job)

    # ---------- 3D ----------

    def make_model(self, prompt: str = "", image_id: str | None = None, preset: str = "quality", texture: bool = True,
                   finish: "meshkit.FinishOptions | None" = None, seed: int | None = None, image_name: str | None = None,
                   where: str = "auto", owner: str | None = None):
        preset = preset if preset in MODEL3D else "quality"

        def job(emit):
            from PIL import Image

            place = resolve_where("model3d", where)
            if place == "local":
                self.free_others()
            aid, folder = new_asset("model", prompt=prompt, preset=preset, source="image" if image_id else "text", where=place, owner=owner)
            try:
                meta = self._make_model(aid, folder, place, prompt, image_id, image_name, preset, texture, finish, seed, emit)
            except Exception:
                shutil.rmtree(folder, ignore_errors=True)
                raise
            emit("Done", 100)
            return meta

        return self.run(job)

    def _make_model(self, aid, folder: Path, place, prompt, image_id, image_name, preset, texture, finish, seed, emit) -> dict:
        from PIL import Image

        # 1. Reference picture: the user's upload, or one painted from the prompt.
        if image_id:
            src = asset_dir(image_id)
            if not src:
                raise FileNotFoundError("That image is gone. Attach it again.")
            meta_src = read_meta(image_id) or {}
            names = meta_src.get("images") or [meta_src.get("files", {}).get("png") or "image.png"]
            image = Image.open(src / (image_name if image_name in names else names[0]))
        else:
            if not prompt.strip():
                raise ValueError("Describe the model you want, or attach a picture of it.")
            emit("Drawing a reference picture…", 1)
            ref_prompt = prompt + TEXT_TO_3D_SUFFIX
            if place == "mock":
                image = _mock_images(prompt, 768, 768, 1, emit)[0]
            elif place == "cloud":
                image = self._cloud_images(ref_prompt, 1024, 1024, 1, seed, emit, 1, 15)[0]
            else:
                key = "flux" if settings.extra.get("ref_model") == "flux" else "sdxl"
                image = self._generate_images(ref_prompt, key, 1024, 1024, 1, seed,
                                              lambda s, p=None, **k: emit("Drawing a reference picture…", (p or 0) * 0.3),
                                              negative="multiple objects, cropped, cut off, background scenery, text")[0]
                self.image_pipe = None  # free VRAM for the 3D stage
                gc.collect()
        image.save(folder / "reference.png")

        # 2. Raw shape (and texture) from Hunyuan3D.
        engine = {"mock": "Preview mode", "local": MODEL3D[preset]["label"]}.get(place)
        note = ""
        if place == "mock":
            shape, painted = _mock_model(folder, emit)
        elif place == "cloud":
            out = cloud.make_model(folder / "reference.png", preset, texture, seed, emit, 16, 75)
            engine = out["space"]
            note = out.get("note") or ""
            shape = cloud.keep(out["shape"], folder, "source-shape" + out["shape"].suffix)
            painted = cloud.keep(out["textured"], folder, "source-textured" + out["textured"].suffix) if out["textured"] else None
        else:
            shape, painted = self._shape_and_paint(image, folder, preset, texture, seed, emit)

        meta = {**read_meta(aid), "reference": "reference.png", "engine": engine,
                "sources": {"shape": shape.name, "textured": painted.name if painted else None},
                "textured": bool(painted), "note": note}
        # 3. Game-ready finishing on this computer.
        meta.update(finish_asset(folder, shape, painted, finish, emit, start=76 if place != "local" else 80))
        write_meta(folder, meta)
        return meta

    def _shape_and_paint(self, image, folder: Path, preset: str, texture: bool, seed, emit) -> tuple[Path, Path | None]:
        import torch
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline

        from .engine import detect_device

        spec = MODEL3D[preset]
        dev = detect_device()
        device = "cuda" if dev.name == "cuda" else dev.name

        emit("Removing the background…", 18)
        if image.mode != "RGBA" or image.getextrema()[3][0] == 255:
            from hy3dgen.rembg import BackgroundRemover

            self.rembg = self.rembg or BackgroundRemover()
            image = self.rembg(image.convert("RGB"))
        image.save(folder / "cutout.png")

        if self.shape_pipe is None or self.shape_preset != preset:
            emit("Loading the 3D shape model (first time downloads it)…", 22)
            self.shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                spec["repo"], subfolder=spec["subfolder"], use_safetensors=True, device=device)
            if "turbo" in spec["subfolder"]:
                try:
                    self.shape_pipe.enable_flashvdm()
                except Exception:
                    pass
            self.shape_preset = preset

        emit("Sculpting the 3D shape…", 30)
        mesh = self.shape_pipe(
            image=image, num_inference_steps=spec["steps"], octree_resolution=spec["octree"], num_chunks=20000,
            generator=torch.manual_seed(seed if seed is not None else 1234), output_type="trimesh",
        )[0]
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        shape = folder / "source-shape.glb"
        mesh.export(shape)  # full detail, used for the normal map bake

        if texture:
            try:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
            except Exception:
                emit("Textures need the extra install step (see README). Making an untextured model.", 60)
                return shape, None
            self.shape_pipe = None  # painting needs most of the VRAM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.paint_pipe is None:
                emit("Loading the texture painter (first time downloads it)…", 55)
                self.paint_pipe = Hunyuan3DPaintPipeline.from_pretrained(PAINT_REPO)
            emit("Painting textures…", 60)
            mesh = FaceReducer()(mesh, max_facenum=60000)
            mesh = self.paint_pipe(mesh, image=image)
            painted = folder / "source-textured.glb"
            mesh.export(painted)
            return shape, painted
        return shape, None


def finish_asset(folder: Path, shape: Path, painted: Path | None, opts, emit, start: float = 80) -> dict:
    """Turn the raw source files of a model asset into its downloadable files."""
    for old in folder.iterdir():  # remove a previous finish
        if old.name.startswith("model") and old.is_file():
            old.unlink()
    if opts is None:  # finishing off: hand over the raw files
        import trimesh

        emit("Exporting GLB, OBJ and STL…", 95)
        src = painted or shape
        shutil.copy(src, folder / "model.glb") if src.suffix == ".glb" else trimesh.load(src).export(folder / "model.glb")
        mesh = trimesh.load(folder / "model.glb", force="mesh")
        mesh.export(folder / "model.obj")
        mesh.export(folder / "model.stl")
        return {"files": {"glb": "model.glb", "obj": "model.obj", "stl": "model.stl"}, "finish": None,
                "stats": {"faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices))}}
    opts.name = "model"
    out = meshkit.finish(painted or shape, folder, opts, emit, detail_src=shape if painted else None, start=start, end=99)
    out["updated"] = int(time.time())
    return out


def refinish(aid: str, opts) -> dict:
    """Rebuild a model's game files with a different budget, texture size or height (no GPU needed)."""
    folder = asset_dir(aid)
    meta = read_meta(aid) if folder else None
    if not meta or meta.get("kind") != "model" or not meta.get("sources"):
        raise FileNotFoundError("This model has no source mesh to rebuild from.")
    shape = folder / meta["sources"]["shape"]
    painted = folder / meta["sources"]["textured"] if meta["sources"].get("textured") else None
    meta.update(finish_asset(folder, shape, painted, opts, lambda *a, **k: None))
    write_meta(folder, meta)
    return meta


# ---------- mock outputs for UI testing ----------

def _mock_images(prompt: str, w: int, h: int, count: int, emit) -> list:
    from PIL import Image, ImageDraw, ImageFilter

    out = []
    for i in range(count):
        for s in range(5):
            time.sleep(0.12)
            emit("Painting…", 10 + 18 * s)
        img = Image.new("RGB", (w, h), (18, 20, 40))
        d = ImageDraw.Draw(img)
        hue = (hash(prompt) + i * 70) % 255
        for r in range(min(w, h) // 2, 0, -8):
            c = (60 + (hue + r) % 150, 80 + r % 120, 200 - r % 120)
            d.ellipse([w / 2 - r, h / 2 - r * 0.9, w / 2 + r, h / 2 + r * 0.9], fill=c)
        out.append(img.filter(ImageFilter.GaussianBlur(2)))
    return out


def _mock_model(folder: Path, emit) -> tuple[Path, Path]:
    """A detailed octopus-like test mesh with colours, so the finishing pipeline runs for real."""
    import math

    import numpy as np
    import trimesh

    for stage, pct in [("Removing the background…", 20), ("Sculpting the 3D shape…", 40), ("Painting textures…", 60)]:
        emit(stage, pct)
        time.sleep(0.3)
    body = trimesh.creation.icosphere(subdivisions=6, radius=1.0)
    v = body.vertices
    body.vertices = v * (1 + 0.025 * np.sin(v[:, 0] * 30) * np.sin(v[:, 1] * 30) * np.sin(v[:, 2] * 30))[:, None]
    body.apply_scale([1, 1.15, 1])
    parts = [body]
    for k in range(8):
        a = k / 8 * 2 * math.pi
        arm = trimesh.creation.capsule(height=1.5, radius=0.17, count=[48, 48])
        arm.apply_transform(trimesh.transformations.rotation_matrix(0.55, [math.sin(a), 0, -math.cos(a)]))
        arm.apply_translation([math.cos(a) * 0.75, -1.35, math.sin(a) * 0.75])
        parts.append(arm)
    mesh = trimesh.util.concatenate(parts)
    y = (mesh.vertices[:, 1] - mesh.vertices[:, 1].min()) / np.ptp(mesh.vertices[:, 1])
    spots = (np.sin(mesh.vertices[:, 0] * 14) * np.sin(mesh.vertices[:, 2] * 14) > 0.6)
    rgb = np.stack([120 + 100 * y, 70 + 60 * y, 230 - 40 * y], 1)
    rgb[spots] *= 0.7
    mesh.visual.vertex_colors = np.c_[rgb, np.full(len(rgb), 255)].astype(np.uint8)
    path = folder / "source-shape.glb"
    mesh.export(path)
    return path, None
