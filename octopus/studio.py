"""Octopus Studio: local image generation and image/text-to-3D modelling.

Images use open diffusion models through `diffusers`. 3D models use Tencent's
Hunyuan3D-2 (`hy3dgen`), installed separately by install_3d.bat / install_3d.sh.
Everything runs on your own GPU and every result is saved to workspace/assets.
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
LICENSE_3D = ("Tencent Hunyuan 3D 2.0 Community License: free to use outside the EU, UK and South Korea; "
              "products with over 1M monthly users need a licence from Tencent.")

TEXT_TO_3D_SUFFIX = (", a single complete object, centered, full view, isolated on a plain white background, "
                     "3D game asset render, soft studio lighting, no shadows, no text")


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def capabilities() -> dict:
    texture = False
    if has("hy3dgen"):
        texture = has("custom_rasterizer")
    return {
        "image": settings.mock or has("diffusers"),
        "model3d": settings.mock or has("hy3dgen"),
        "texture": settings.mock or texture,
        "imageModels": [{"id": k, **{x: v[x] for x in ("label", "blurb", "vram", "license")}} for k, v in IMAGE_MODELS.items()],
        "modelPresets": [{"id": k, "label": v["label"], "blurb": v["blurb"], "vram": v["vram"]} for k, v in MODEL3D.items()],
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


def list_assets(kind: str | None = None) -> list[dict]:
    out = []
    for folder in sorted(ASSETS_DIR.iterdir(), reverse=True):
        meta = read_meta(folder.name)
        if meta and (kind is None or meta.get("kind") == kind) and meta.get("files"):
            out.append(meta)
    return out


def delete_asset(aid: str) -> None:
    folder = asset_dir(aid)
    if not folder:
        raise FileNotFoundError(aid)
    shutil.rmtree(folder)


def save_upload(name: str, data_url: str) -> dict:
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
    aid, folder = new_asset("upload", name=name[:80])
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
                self.free_others()
                q.put({"t": "done", "asset": job(emit)})
            except Exception as exc:
                traceback.print_exc()
                q.put({"t": "error", "d": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    yield {"t": "ping"}
                    continue
                if ev is None:
                    break
                yield ev
        finally:
            self.lock.release()

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

    def make_image(self, prompt: str, model: str = "sdxl", aspect: str = "1:1", count: int = 1, seed: int | None = None):
        model = model if model in IMAGE_MODELS else "sdxl"
        w, h = {"1:1": (1024, 1024), "16:9": (1344, 768), "9:16": (768, 1344), "4:3": (1152, 896)}.get(aspect, (1024, 1024))
        count = max(1, min(int(count), 4))

        def job(emit):
            aid, folder = new_asset("image", prompt=prompt, model=model, aspect=aspect)
            if settings.mock:
                images = _mock_images(prompt, w, h, count, emit)
            else:
                images = self._generate_images(prompt, model, w, h, count, seed, emit, negative="blurry, low quality, watermark, text")
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

    def make_model(self, prompt: str = "", image_id: str | None = None, preset: str = "fast", texture: bool = True,
                   game_ready: bool = True, seed: int | None = None, image_name: str | None = None):
        preset = preset if preset in MODEL3D else "fast"

        def job(emit):
            from PIL import Image

            aid, folder = new_asset("model", prompt=prompt, preset=preset, source="image" if image_id else "text")
            # 1. Reference image: the user's upload, or one we paint from the prompt.
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
                if settings.mock:
                    image = _mock_images(prompt, 768, 768, 1, emit)[0]
                else:
                    key = "flux" if has("diffusers") and settings.extra.get("ref_model") == "flux" else "sdxl"
                    image = self._generate_images(prompt + TEXT_TO_3D_SUFFIX, key, 1024, 1024, 1, seed,
                                                  lambda s, p=None, **k: emit("Drawing a reference picture…", (p or 0) * 0.3),
                                                  negative="multiple objects, cropped, cut off, background scenery, text")[0]
                    self.image_pipe = None  # free VRAM for the 3D stage
                    gc.collect()
            image.save(folder / "reference.png")

            if settings.mock:
                files = _mock_model(folder, emit)
            else:
                files = self._shape_and_paint(image, folder, preset, texture, game_ready, seed, emit)
            meta = {**read_meta(aid), "files": files, "reference": "reference.png", "textured": "textured" in files.get("glb", "")}
            meta["stats"] = files.pop("stats", None)
            write_meta(folder, meta)
            emit("Done", 100)
            return meta

        return self.run(job)

    def _shape_and_paint(self, image, folder: Path, preset: str, texture: bool, game_ready: bool, seed, emit) -> dict:
        import torch
        import trimesh
        from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover, Hunyuan3DDiTFlowMatchingPipeline

        from .engine import detect_device

        spec = MODEL3D[preset]
        dev = detect_device()
        device = "cuda" if dev.name == "cuda" else dev.name

        emit("Removing the background…", 32)
        if image.mode != "RGBA" or image.getextrema()[3][0] == 255:
            from hy3dgen.rembg import BackgroundRemover

            self.rembg = self.rembg or BackgroundRemover()
            image = self.rembg(image.convert("RGB"))
        image.save(folder / "cutout.png")

        if self.shape_pipe is None or self.shape_preset != preset:
            emit("Loading the 3D shape model (first time downloads it)…", 36)
            self.shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                spec["repo"], subfolder=spec["subfolder"], use_safetensors=True, device=device)
            if "turbo" in spec["subfolder"]:
                try:
                    self.shape_pipe.enable_flashvdm()
                except Exception:
                    pass
            self.shape_preset = preset

        emit("Sculpting the 3D shape…", 45)
        mesh = self.shape_pipe(
            image=image, num_inference_steps=spec["steps"], octree_resolution=spec["octree"], num_chunks=20000,
            generator=torch.manual_seed(seed if seed is not None else 1234), output_type="trimesh",
        )[0]

        emit("Cleaning up the mesh…", 65)
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=20000 if game_ready else 120000)
        files = {"stats": {"faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices))}}
        mesh.export(folder / "model.glb")
        files["glb"] = "model.glb"

        if texture:
            try:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
            except Exception:
                emit("Textures need the extra install step (see README). Saved an untextured model.", 90)
                texture = False
        if texture:
            # Free the shape model first, painting needs most of the VRAM.
            self.shape_pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.paint_pipe is None:
                emit("Loading the texture painter (first time downloads it)…", 70)
                self.paint_pipe = Hunyuan3DPaintPipeline.from_pretrained(PAINT_REPO)
            emit("Painting textures…", 75)
            mesh = self.paint_pipe(mesh, image=image)
            mesh.export(folder / "model-textured.glb")
            files["glb"] = "model-textured.glb"

        emit("Exporting OBJ and STL…", 95)
        scene = trimesh.load(folder / files["glb"], force="mesh")
        scene.export(folder / "model.obj")
        scene.export(folder / "model.stl")
        files.update(obj="model.obj", stl="model.stl")
        return files


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


def _mock_model(folder: Path, emit) -> dict:
    import trimesh

    for stage, pct in [("Removing the background…", 32), ("Sculpting the 3D shape…", 50), ("Painting textures…", 80)]:
        emit(stage, pct)
        time.sleep(0.4)
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    body.apply_scale([1, 1.15, 1])
    parts = [body]
    for k in range(6):
        import math

        a = k / 6 * 2 * math.pi
        arm = trimesh.creation.capsule(height=1.4, radius=0.18)
        arm.apply_transform(trimesh.transformations.rotation_matrix(0.5, [math.sin(a), 0, -math.cos(a)]))
        arm.apply_translation([math.cos(a) * 0.7, -1.3, math.sin(a) * 0.7])
        parts.append(arm)
    mesh = trimesh.util.concatenate(parts)
    mesh.visual.vertex_colors = [130, 110, 255, 255]
    mesh.export(folder / "model.glb")
    mesh.export(folder / "model.obj")
    mesh.export(folder / "model.stl")
    return {"glb": "model.glb", "obj": "model.obj", "stl": "model.stl",
            "stats": {"faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices))}}
