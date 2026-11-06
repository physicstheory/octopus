"""Octopus Mesh Kit: turns raw AI-generated meshes into game-ready assets.

AI 3D models come out as dense, messy triangle soup with a texture painted on.
This module finishes them the way a game artist would, on the CPU:

1. Clean: weld vertices, remove degenerate faces, small floating bits and holes.
2. Pivot at the bottom centre and scale to a real-world height (Y up, metres).
3. Reduce to a triangle budget (quadric decimation, like Blender's Decimate).
4. Unwrap clean UVs (xatlas) with padding for mipmaps.
5. Bake PBR textures from the detailed mesh onto the game mesh:
   Base Colour, ORM (occlusion/roughness/metallic) and a tangent-space Normal map
   (OpenGL / glTF convention), so the low-poly mesh keeps the high-poly detail.
6. Build LOD0, LOD1 and LOD2 (named _LOD0/_LOD1/_LOD2 for Unity, Unreal and Godot).
7. Export GLB (PBR), OBJ + MTL + textures, STL, and a zipped game pack.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

BUDGETS = {
    "mobile": {"label": "Mobile · 5k triangles", "tris": 5_000},
    "game": {"label": "Game · 20k triangles", "tris": 20_000},
    "hero": {"label": "Hero · 60k triangles", "tris": 60_000},
    "cinematic": {"label": "Cinematic · 150k triangles", "tris": 150_000},
}
TEXTURE_SIZES = [1024, 2048, 4096]
LOD_RATIOS = [1.0, 0.4, 0.15]  # share of the LOD0 budget for LOD0, LOD1, LOD2


@dataclass
class FinishOptions:
    budget: int = 20_000
    texture_size: int = 2048
    height: float | None = 1.0  # metres; None keeps the original scale
    lods: bool = True
    name: str = "model"

    @classmethod
    def from_request(cls, b: dict) -> "FinishOptions":
        budget = b.get("budget", "game")
        tris = BUDGETS[budget]["tris"] if budget in BUDGETS else int(budget or 20_000)
        size = int(b.get("texture_size") or 2048)
        size = min(TEXTURE_SIZES, key=lambda s: abs(s - size))
        try:
            height = float(b.get("height")) if b.get("height") not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            height = 1.0
        if height is not None:
            height = min(max(height, 0.01), 1000.0)
        return cls(budget=max(500, min(tris, 500_000)), texture_size=size, height=height, lods=bool(b.get("lods", True)))


Emit = Callable[..., None]


def _noop(*_a, **_k) -> None:
    pass


# ---------- loading and cleaning ----------

def load(src) -> "trimesh.Trimesh":
    import trimesh

    if isinstance(src, trimesh.Trimesh):
        return src
    if isinstance(src, trimesh.Scene):
        return src.to_geometry() if hasattr(src, "to_geometry") else src.dump(concatenate=True)
    mesh = trimesh.load(str(src), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("That file has no triangles in it.")
    return mesh


def drop_floaters(mesh, min_share: float = 0.004):
    """Remove disconnected bits whose surface area is under min_share of the whole (keeps visuals)."""
    import trimesh

    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    area = np.bincount(labels, weights=mesh.area_faces)
    keep = area >= max(min_share * area.sum(), 1e-12)
    keep[np.argmax(area)] = True
    mask = keep[labels]
    if not mask.all():
        mesh = mesh.copy()
        mesh.update_faces(mask)
        mesh.remove_unreferenced_vertices()
    return mesh


def geometry(mesh):
    """A welded, clean, untextured copy of the shape for decimation."""
    import trimesh

    g = trimesh.Trimesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces), process=True)
    g.merge_vertices()
    g.update_faces(g.nondegenerate_faces())
    g.update_faces(g.unique_faces())
    g.remove_unreferenced_vertices()
    return drop_floaters(g)


def _same_frame(a, b, tol: float = 0.06) -> bool:
    """True when two meshes occupy the same space (same model, same coordinates)."""
    ext = max(float(np.max(a.extents)), 1e-9)
    return bool(np.all(np.abs(np.asarray(a.bounds) - np.asarray(b.bounds)) < tol * ext))


def normalize_transform(mesh, height: float | None) -> np.ndarray:
    """4x4 matrix: pivot at the bottom centre, scaled so the model is `height` metres tall."""
    lo, hi = mesh.bounds
    m = np.eye(4)
    m[:3, 3] = [-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2]
    if height:
        s = height / max(hi[1] - lo[1], 1e-9)
        m = np.diag([s, s, s, 1.0]) @ m
    return m


def decimate(g, target: int):
    import fast_simplification
    import trimesh

    if len(g.faces) > target:
        v, f = fast_simplification.simplify(np.asarray(g.vertices, dtype=np.float32), np.asarray(g.faces, dtype=np.int64),
                                            target_reduction=1 - target / len(g.faces), agg=6)
        g = trimesh.Trimesh(v, f, process=True)
    g.update_faces(g.nondegenerate_faces())
    g.remove_unreferenced_vertices()
    try:
        trimesh.repair.fill_holes(g)
    except Exception:
        pass
    trimesh.repair.fix_normals(g)
    return g


def unwrap(g, size: int):
    """xatlas UVs. Returns (vmapping, faces, uvs) with uvs in [0, 1], v pointing up."""
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(np.asarray(g.vertices, dtype=np.float32), np.asarray(g.faces, dtype=np.uint32),
                   np.asarray(g.vertex_normals, dtype=np.float32))
    co = xatlas.ChartOptions()
    co.max_iterations = 2
    po = xatlas.PackOptions()
    po.resolution = size
    po.padding = max(2, size // 256)
    po.bilinear = True
    po.rotate_charts = True
    atlas.generate(co, po)
    vmap, faces, uvs = atlas[0]
    return np.asarray(vmap, dtype=np.int64), np.asarray(faces, dtype=np.int64), np.asarray(uvs, dtype=np.float64)


# ---------- sampling the detailed source ----------

def _sample_image(img, uv: np.ndarray) -> np.ndarray:
    """Bilinear lookup of a PIL image at OpenGL-style uv (v up). Returns float RGBA in 0..1."""
    a = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    x = (uv[:, 0] % 1.0) * w - 0.5
    y = (1.0 - (uv[:, 1] % 1.0)) * h - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, x1 = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0, y1 = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x1] * fx * (1 - fy) + a[y1, x0] * (1 - fx) * fy + a[y1, x1] * fx * fy)


def _bary(tris: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of p projected onto each triangle, clamped inside it."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bc = np.stack([1 - v - w, v, w], 1)
    bc = np.clip(bc, 0, None)
    return bc / np.maximum(bc.sum(1, keepdims=True), 1e-12)


class Surface:
    """Fast nearest-surface lookups on a detailed mesh, with its colours and materials."""

    def __init__(self, mesh, samples: int = 2_000_000) -> None:
        import trimesh
        from scipy.spatial import cKDTree

        self.mesh = mesh
        self.faces = np.asarray(mesh.faces)
        self.tris = np.asarray(mesh.triangles)
        self.fn = np.asarray(mesh.face_normals)
        self.vn = np.asarray(mesh.vertex_normals)
        n = int(min(samples, max(200_000, len(self.faces) * 4)))
        pts, fid = trimesh.sample.sample_surface(mesh, n, seed=7)
        pts = np.concatenate([pts, self.tris.mean(1)])
        self.fid = np.concatenate([fid, np.arange(len(self.faces))])
        self.tree = cKDTree(pts)
        self._material()

    def _material(self) -> None:
        vis = self.mesh.visual
        self.kind, self.base_img, self.mr_img = "flat", None, None
        self.base_factor = np.array([0.8, 0.8, 0.8, 1.0])
        self.metal, self.rough = 0.0, 0.75
        if getattr(vis, "kind", None) == "texture" and getattr(vis, "uv", None) is not None:
            mat = vis.material
            self.uv = np.asarray(vis.uv, dtype=np.float64)
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if img is not None:
                self.kind, self.base_img = "texture", img
            fac = getattr(mat, "baseColorFactor", None)
            if fac is not None:
                fac = np.asarray(fac, dtype=np.float64)
                self.base_factor = fac / 255.0 if fac.max() > 1.0 else fac
            elif img is not None:
                self.base_factor = np.ones(4)
            self.mr_img = getattr(mat, "metallicRoughnessTexture", None)
            if getattr(mat, "metallicFactor", None) is not None:
                self.metal = float(mat.metallicFactor)
            if getattr(mat, "roughnessFactor", None) is not None:
                self.rough = float(mat.roughnessFactor)
            if self.mr_img is None and getattr(mat, "metallicFactor", None) is None:
                self.metal, self.rough = 0.0, 0.75
        elif getattr(vis, "kind", None) == "vertex":
            self.kind = "vertex"
            self.vc = np.asarray(vis.vertex_colors, dtype=np.float64) / 255.0
        elif getattr(vis, "kind", None) == "face":
            self.kind = "face"
            self.fc = np.asarray(vis.face_colors, dtype=np.float64) / 255.0

    def locate(self, p: np.ndarray, ref_n: np.ndarray, k: int = 4, chunk: int = 400_000):
        """For each point, the face and barycentrics of the closest surface point facing the same way."""
        F = np.empty(len(p), dtype=np.int64)
        B = np.empty((len(p), 3))
        for s in range(0, len(p), chunk):
            q, rn = p[s:s + chunk], ref_n[s:s + chunk]
            _, idx = self.tree.query(q, k=k, workers=-1)
            cand = self.fid[idx]  # (n, k)
            best_f = cand[:, 0].copy()
            best_b = None
            best_score = np.full(len(q), np.inf)
            for j in range(k):
                f = cand[:, j]
                bc = _bary(self.tris[f], q)
                hit = (self.tris[f] * bc[:, :, None]).sum(1)
                d = np.linalg.norm(hit - q, axis=1)
                facing = (self.fn[f] * rn).sum(1)
                score = d * np.where(facing < 0, 4.0, 1.0)
                better = score < best_score
                best_score = np.where(better, score, best_score)
                best_f = np.where(better, f, best_f)
                best_b = bc if best_b is None else np.where(better[:, None], bc, best_b)
            F[s:s + chunk], B[s:s + chunk] = best_f, best_b
        return F, B

    def normal(self, f: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = (self.vn[self.faces[f]] * b[:, :, None]).sum(1)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def material(self, f: np.ndarray, b: np.ndarray):
        """Base colour (RGBA 0..1), roughness and metallic at the given surface points."""
        n = len(f)
        rough = np.full(n, self.rough)
        metal = np.full(n, self.metal)
        if self.kind == "texture":
            uv = (self.uv[self.faces[f]] * b[:, :, None]).sum(1)
            base = _sample_image(self.base_img, uv) * self.base_factor
            if self.mr_img is not None:
                mr = _sample_image(self.mr_img, uv)
                rough, metal = mr[:, 1] * self.rough, mr[:, 2] * self.metal
        elif self.kind == "vertex":
            base = (self.vc[self.faces[f]] * b[:, :, None]).sum(1)
        elif self.kind == "face":
            base = self.fc[f]
        else:
            base = np.tile(self.base_factor, (n, 1))
        return base, rough, metal


# ---------- baking ----------

def _tangents(v, f, uv, vn):
    """Per-vertex tangents (xyz) and handedness (w) from the UV layout."""
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    t0, t1, t2 = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    du1, dv1 = (t1 - t0)[:, 0:1], (t1 - t0)[:, 1:2]
    du2, dv2 = (t2 - t0)[:, 0:1], (t2 - t0)[:, 1:2]
    r = du1 * dv2 - du2 * dv1
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    tan = (e1 * dv2 - e2 * dv1) / r
    bit = (e2 * du1 - e1 * du2) / r
    T = np.zeros_like(v)
    Bv = np.zeros_like(v)
    for k in range(3):
        np.add.at(T, f[:, k], tan)
        np.add.at(Bv, f[:, k], bit)
    T = T - vn * (T * vn).sum(1, keepdims=True)
    bad = np.linalg.norm(T, axis=1) < 1e-12
    if bad.any():  # pick any perpendicular direction
        alt = np.where(np.abs(vn[bad, 0:1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        T[bad] = np.cross(vn[bad], alt)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    w = np.where((np.cross(vn, T) * Bv).sum(1) < 0, -1.0, 1.0)
    return T, w


def _rasterize(uv, f, size):
    """Which face covers each texel. Returns an int image (face index + 1, 0 = empty)."""
    from PIL import Image, ImageDraw

    img = Image.new("I", (size, size), 0)
    draw = ImageDraw.Draw(img)
    px = np.stack([uv[:, 0] * size, (1.0 - uv[:, 1]) * size], 1)
    tri = px[f]
    for i in range(len(f)):
        t = tri[i]
        draw.polygon([(t[0, 0], t[0, 1]), (t[1, 0], t[1, 1]), (t[2, 0], t[2, 1])], fill=i + 1)
    return np.asarray(img, dtype=np.int64)


def _dilate(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill every empty texel with its nearest baked neighbour so mipmaps and seams don't bleed."""
    from scipy import ndimage

    if mask.all() or not mask.any():
        return img
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return img[iy, ix]


def bake(v, f, uv, vn, color: Surface, detail: Surface, size: int):
    """Bake base colour, ORM and normal maps for a low-poly mesh. Returns three PIL images."""
    from PIL import Image

    fid = _rasterize(uv, f, size)
    mask = fid > 0
    ys, xs = np.nonzero(mask)
    face = fid[ys, xs] - 1
    puv = np.stack([(xs + 0.5) / size, 1.0 - (ys + 0.5) / size], 1)

    # Barycentrics of each texel in its UV triangle.
    a, b, c = uv[f[face, 0]], uv[f[face, 1]], uv[f[face, 2]]
    v0, v1, v2 = b - a, c - a, puv - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    bc = np.clip(np.stack([1 - l1 - l2, l1, l2], 1), 0, None)
    bc /= np.maximum(bc.sum(1, keepdims=True), 1e-12)

    fv = f[face]
    P = (v[fv] * bc[:, :, None]).sum(1)
    N = (vn[fv] * bc[:, :, None]).sum(1)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    T4, W = _tangents(v, f, uv, vn)
    T = (T4[fv] * bc[:, :, None]).sum(1)
    T -= N * (T * N).sum(1, keepdims=True)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    B = np.cross(N, T) * W[fv[:, 0]][:, None]

    # Normal map: detailed surface normal expressed in the low-poly tangent frame.
    df, db = detail.locate(P, N)
    hn = detail.normal(df, db)
    hn = np.where(((hn * N).sum(1) < 0.05)[:, None], N, hn)  # ignore lookups that land on the far side
    tn = np.stack([(hn * T).sum(1), (hn * B).sum(1), (hn * N).sum(1)], 1)
    tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-12)

    if color is detail:
        cf, cb = df, db
    else:
        cf, cb = color.locate(P, N)
    base, rough, metal = color.material(cf, cb)

    out_base = np.zeros((size, size, 3), dtype=np.uint8)
    out_orm = np.zeros((size, size, 3), dtype=np.uint8)
    out_nrm = np.zeros((size, size, 3), dtype=np.uint8)
    out_base[ys, xs] = np.clip(base[:, :3] * 255 + 0.5, 0, 255)
    out_orm[ys, xs, 0] = 255
    out_orm[ys, xs, 1] = np.clip(rough * 255 + 0.5, 0, 255)
    out_orm[ys, xs, 2] = np.clip(metal * 255 + 0.5, 0, 255)
    out_nrm[ys, xs] = np.clip((tn * 0.5 + 0.5) * 255 + 0.5, 0, 255)
    imgs = [Image.fromarray(_dilate(x, mask)) for x in (out_base, out_orm, out_nrm)]
    coverage = float(mask.mean())
    return imgs, coverage


# ---------- export ----------

def _write_obj(path: Path, v, f, uv, vn, mtl_name: str, material: str) -> None:
    lines = [f"# Made with Octopus", f"mtllib {mtl_name}", f"o {path.stem}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    lines += [f"vt {a:.6f} {b:.6f}" for a, b in uv]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in vn]
    lines.append(f"usemtl {material}")
    lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in (f + 1)]
    path.write_text("\n".join(lines) + "\n")


def _write_mtl(path: Path, material: str, base: str, orm: str, normal: str) -> None:
    path.write_text(
        f"newmtl {material}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 10\nillum 2\n"
        f"map_Kd {base}\nmap_Bump -bm 1 {normal}\nnorm {normal}\n"
        f"# PBR (ORM: R occlusion, G roughness, B metallic)\nmap_Pr {orm}\nmap_Pm {orm}\n")


def _glb(path: Path, v, f, uv, vn, base, orm, normal, name: str) -> None:
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    mat = PBRMaterial(name=name, baseColorTexture=base, metallicRoughnessTexture=orm, normalTexture=normal,
                      metallicFactor=1.0, roughnessFactor=1.0, baseColorFactor=[255, 255, 255, 255], doubleSided=False)
    mesh = trimesh.Trimesh(v, f, vertex_normals=vn, visual=TextureVisuals(uv=uv, material=mat), process=False)
    mesh.metadata["name"] = name
    path.write_bytes(trimesh.Scene({name: mesh}).export(file_type="glb", include_normals=True))


# ---------- the whole pipeline ----------

def finish(color_src, folder: Path, opts: FinishOptions | None = None, emit: Emit = _noop, detail_src=None,
           start: float = 0.0, end: float = 100.0) -> dict:
    """Finish a raw AI mesh into a game-ready asset in `folder`. Returns the files and stats for meta.json."""
    import trimesh

    opts = opts or FinishOptions()
    folder = Path(folder)
    span = end - start

    def step(msg: str, frac: float) -> None:
        emit(msg, start + span * frac)

    step("Cleaning the mesh…", 0.0)
    color_mesh = drop_floaters(load(color_src))
    detail_mesh = drop_floaters(load(detail_src)) if detail_src is not None else None
    if detail_mesh is not None and not _same_frame(color_mesh, detail_mesh):
        detail_mesh = None  # the two files don't line up, so bake from the textured one only
    ref = detail_mesh if detail_mesh is not None else color_mesh
    source_faces = int(len(ref.faces))

    xf = normalize_transform(ref, opts.height)
    color_mesh = color_mesh.copy()
    color_mesh.apply_transform(xf)
    if detail_mesh is not None:
        detail_mesh = detail_mesh.copy()
        detail_mesh.apply_transform(xf)
    else:
        detail_mesh = color_mesh

    g = geometry(detail_mesh)
    watertight = bool(g.is_watertight)

    step("Preparing to bake detail…", 0.08)
    detail = Surface(detail_mesh)
    color = detail if detail_mesh is color_mesh else Surface(color_mesh)

    n = opts.name
    files: dict = {"lods": [], "textures": []}
    lod_stats = []
    ratios = LOD_RATIOS if opts.lods else LOD_RATIOS[:1]
    sizes = [opts.texture_size, max(512, opts.texture_size // 2), max(256, opts.texture_size // 4)]
    pack: list[str] = []
    for lod, ratio in enumerate(ratios):
        base_frac = 0.12 + 0.8 * lod / len(ratios)
        part = 0.8 / len(ratios)
        target = max(300, int(opts.budget * ratio))
        step(f"Building LOD{lod}: reducing to {target:,} triangles…", base_frac)
        low = decimate(g, target)
        step(f"LOD{lod}: unwrapping UVs…", base_frac + part * 0.2)
        vmap, faces, uvs = unwrap(low, sizes[lod])
        v = np.asarray(low.vertices)[vmap]
        vn = np.asarray(low.vertex_normals)[vmap]
        step(f"LOD{lod}: baking {sizes[lod]}px colour, normal and roughness maps…", base_frac + part * 0.45)
        (img_base, img_orm, img_nrm), coverage = bake(v, faces, uvs, vn, color, detail, sizes[lod])

        stem = f"{n}_LOD{lod}"
        tex = {k: f"{stem}_{k}.png" for k in ("BaseColor", "ORM", "Normal")}
        img_base.save(folder / tex["BaseColor"], optimize=True)
        img_orm.save(folder / tex["ORM"], optimize=True)
        img_nrm.save(folder / tex["Normal"], optimize=True)
        glb_name = f"{stem}.glb"
        _glb(folder / glb_name, v, faces, uvs, vn, img_base, img_orm, img_nrm, stem)
        _write_obj(folder / f"{stem}.obj", v, faces, uvs, vn, f"{stem}.mtl", stem)
        _write_mtl(folder / f"{stem}.mtl", stem, tex["BaseColor"], tex["ORM"], tex["Normal"])
        files["lods"].append(glb_name)
        files["textures"] += list(tex.values())
        pack += [glb_name, f"{stem}.obj", f"{stem}.mtl", *tex.values()]
        lod_stats.append({"lod": lod, "triangles": int(len(faces)), "vertices": int(len(v)), "texture": sizes[lod],
                          "uv_coverage": round(coverage, 3)})

    step("Exporting STL for 3D printing…", 0.94)
    stl_mesh = decimate(g.copy(), 300_000) if len(g.faces) > 300_000 else g
    stl_mesh.export(folder / f"{n}.stl")
    pack.append(f"{n}.stl")

    ext = g.extents
    stats = {
        "faces": lod_stats[0]["triangles"], "vertices": lod_stats[0]["vertices"], "lods": lod_stats,
        "source_faces": source_faces, "texture": opts.texture_size, "watertight": watertight,
        "size_m": [round(float(x), 3) for x in ext], "pbr": True,
    }
    readme = (
        f"{n} (made with Octopus)\n\n"
        f"Size: {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m, Y up, pivot at the bottom centre.\n"
        + "".join(f"LOD{s['lod']}: {s['triangles']:,} triangles, {s['texture']}px textures\n" for s in lod_stats)
        + "\nTextures (PBR metal/rough):\n"
          "  _BaseColor  sRGB colour\n"
          "  _Normal     tangent-space normal map, OpenGL style (green up). In Unreal tick 'Flip Green Channel'.\n"
          "  _ORM        linear: R ambient occlusion, G roughness, B metallic\n\n"
          "Unity: drop the LOD0..LOD2 files in and add a LOD Group, or use the GLB with glTFast.\n"
          "Unreal: import the GLB (Interchange) or the OBJ; set the ORM texture to linear (not sRGB).\n"
          "Godot: drag the GLB into your project; it is ready to use.\n"
          "Blender: File > Import > glTF.\n")
    (folder / "README.txt").write_text(readme)
    pack.append("README.txt")
    zip_name = f"{n}-game-pack.zip"
    with zipfile.ZipFile(folder / zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for name in pack:
            z.write(folder / name, f"{n}/{name}")

    files.update(glb=files["lods"][0], obj=f"{n}_LOD0.obj", stl=f"{n}.stl", pack=zip_name)
    step("Finished", 1.0)
    return {"files": files, "stats": stats, "finish": asdict(opts)}
