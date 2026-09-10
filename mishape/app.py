from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .engine import analyze, deform, get_cage, regrid
from .parameters import schema
from .generation import generate, generation_schema
from .importers import import_obj

ROOT = Path(__file__).resolve().parent
STATE = Path(os.environ.get("MISHAPE_STATE", ROOT / ".state"))
STATE.mkdir(parents=True, exist_ok=True)
ASSETS = ROOT / "assets"
MODELS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
LOCK = threading.RLock()
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mishape")
app = FastAPI(title="MiShape", version=__version__)


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    parameters: dict[str, float] = Field(default_factory=dict)
    controls: list[dict] = Field(default_factory=list, max_length=1024)
    options: dict = Field(default_factory=lambda: {"symmetry": True, "preserve_wheels": True})


class Export(Recipe):
    format: str = "obj"
    part_id: int | None = None


class CageChange(Recipe):
    cage: dict


class Calibration(Recipe):
    length_m: float = Field(gt=1, lt=20)


class Batch(Recipe):
    model_id: str
    count: int = Field(12, ge=1, le=50)
    seed: int = Field(42, ge=0, le=2**32-1)
    strength: float = Field(.35, ge=.01, le=1)
    vary: list[str] = Field(default_factory=lambda: ["vehicle_length", "vehicle_width", "vehicle_height", "windscreen_angle", "greenhouse_tapering"])


def safe_id(value):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
        raise HTTPException(400, "无效标识符")
    return value


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def validate_model(model):
    if not isinstance(model, dict) or not {"vertices", "faces"}.issubset(model):
        raise ValueError("工程模型必须包含 vertices 与 faces 数组")
    v = np.asarray(model["vertices"], dtype=float).reshape(-1, 3)
    raw_faces = np.asarray(model["faces"])
    if raw_faces.dtype.kind not in "iu":
        raise ValueError("面索引必须是整数")
    f = raw_faces.reshape(-1, 3)
    if not len(v) or not len(f) or len(v) > 1500000 or len(f) > 2000000:
        raise ValueError("模型为空或超过 150 万顶点 / 200 万三角面的限制")
    if not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ValueError("模型包含无效坐标或面索引")
    if np.ptp(v, axis=0).min() <= 1e-8:
        raise ValueError("需要完整的三维车辆模型")
    model.setdefault("face_labels", [0] * len(f))
    if len(model["face_labels"]) != len(f):
        raise ValueError("组件标签数量与三角面不一致")
    model.setdefault("parts", [{"id": 0, "name": "导入车身", "color": "#788e75"}])
    if not isinstance(model["parts"], list) or any(not isinstance(p, dict) or not isinstance(p.get("id"), int) or not isinstance(p.get("name"), str) for p in model["parts"]):
        raise ValueError("组件必须包含整数 id 和 name")
    ids = {p["id"] for p in model["parts"]}
    if len(ids) != len(model["parts"]):
        raise ValueError("组件 id 不能重复")
    if not set(model["face_labels"]).issubset(ids):
        raise ValueError("组件标签引用未知组件")
    model.setdefault("metadata", {})
    if not isinstance(model["metadata"], dict):
        raise ValueError("模型 metadata 必须是对象")
    polygons = model.get("polygons")
    if polygons is not None:
        if not isinstance(polygons, list):
            raise ValueError("polygons 必须是多边形索引列表")
        for polygon in polygons:
            indices = polygon.get("vertices", polygon.get("indices", [])) if isinstance(polygon, dict) else polygon
            if not isinstance(indices, list) or len(indices) < 3 or any(not isinstance(i, int) or i < 0 or i >= len(v) for i in indices):
                raise ValueError("每个 polygon 必须包含至少三个有效顶点索引")
        for key in ("polygon_labels", "polygon_part_ids"):
            if key in model and (not isinstance(model[key], list) or len(model[key]) != len(polygons) or not set(model[key]).issubset(ids)):
                raise ValueError("多边形标签必须与 polygons 一一对应")
    model.setdefault("source_face_ids", list(range(len(f))))
    return model


def save_model(model, name=None):
    validate_model(model)
    mid = uuid.uuid4().hex
    model = copy.deepcopy(model)
    model["id"] = mid
    model["name"] = name or model.get("name", "MiShape design")
    path = STATE / f"{mid}.json"
    path.write_text(dump(model))
    with LOCK:
        MODELS[mid] = model
        while len(MODELS) > 8:
            MODELS.pop(next(iter(MODELS)))
    return mid


def get_model(mid):
    safe_id(mid)
    with LOCK:
        model = MODELS.get(mid)
    if model is None:
        path = STATE / f"{mid}.json"
        if not path.is_file():
            raise HTTPException(404, "模型不存在，请重新打开车辆")
        model = json.loads(path.read_text())
        with LOCK:
            MODELS[mid] = model
            while len(MODELS) > 8:
                MODELS.pop(next(iter(MODELS)))
    return model


def result_model(mid, recipe=None):
    model = get_model(mid)
    if recipe is None:
        recipe = {"parameters": {}, "controls": [], "options": {
            "symmetry": True, "preserve_wheels": True,
            "cage": {"type": "fitted", "dimensions": [9, 3, 4], "padding_mm": 30}}}
    cage = get_cage(model, recipe.get("parameters"), recipe.get("controls"), recipe.get("options"))
    return {"model_id": mid, "model": model, "analysis": analyze(model), "cage": cage, "recipe": recipe}


def catalog():
    path = ASSETS / "catalog.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data.get("assets", data.get("models", []))


@app.exception_handler(ValueError)
async def invalid_request(request, exc):
    return Response(dump({"detail": str(exc)}), status_code=422, media_type="application/json")


@app.get("/api/health")
def health():
    return {"status": "ok", "product": "MiShape", "version": __version__, "local": True}


@app.get("/api/bootstrap")
def bootstrap():
    return {"version": __version__, "assets": catalog(), "parameters": schema(), "generation": generation_schema()}


@app.post("/api/models/open")
def open_asset(data: dict):
    asset = next((a for a in catalog() if a["id"] == data.get("asset_id")), None)
    if asset is None:
        raise HTTPException(404, "车辆素材不存在")
    path = (ASSETS / asset.get("path", f'{asset["id"]}/viewport.json')).resolve()
    if not path.is_relative_to(ASSETS.resolve()) or not path.is_file():
        raise HTTPException(404, "车辆素材文件尚未准备完成")
    model = json.loads(path.read_text())
    model.setdefault("metadata", {})["asset_id"] = asset["id"]
    mid = save_model(model, asset["name"])
    return result_model(mid)


@app.get("/api/models/{mid}")
def model_info(mid: str):
    return result_model(mid)


@app.post("/api/models/{mid}/deform")
def preview(mid: str, req: Recipe):
    result = deform(get_model(mid), **req.model_dump())
    info = result.get("metadata", {}).get("mishape", {})
    return {"vertices": result["vertices"], "analysis": info["analysis"] if "analysis" in info else analyze(result), "quality": info.get("quality", {}), "cage": info["cage"] if "cage" in info else get_cage(get_model(mid), **req.model_dump()), "parameters": req.parameters}


@app.post("/api/models/{mid}/cage")
def change_cage(mid: str, req: CageChange):
    model = get_model(mid)
    changed = regrid(model, req.parameters, req.controls, req.options, req.cage)
    recipe = {key: changed[key] for key in ("parameters", "controls", "options")}
    result = deform(model, **recipe)
    info = result["metadata"]["mishape"]
    return {"vertices": result["vertices"], "analysis": info["analysis"], "quality": info["quality"],
            "cage": info["cage"], "parameters": recipe["parameters"], "recipe": recipe,
            "regrid": changed["resampling"]}


@app.post("/api/models/{mid}/calibrate")
def calibrate_model(mid: str, req: Calibration):
    candidate = deform(get_model(mid), **req.model_dump(exclude={"length_m"}))
    vertices = np.asarray(candidate["vertices"]).reshape(-1, 3)
    scale = req.length_m / np.ptp(vertices[:, 0])
    candidate["vertices"] = (vertices * scale).ravel().tolist()
    metadata = candidate["metadata"]
    metadata["landmarks"] = {key: (np.asarray(p) * scale).tolist() for key, p in metadata.get("landmarks", {}).items()}
    metadata["scale_status"] = "user_calibrated"
    metadata["reference_length_m"] = req.length_m
    metadata["scale_method"] = "user_known_length"
    metadata["calibration_parent"] = {"model_id": mid, "recipe": req.model_dump(), "uniform_scale": float(scale)}
    metadata.pop("mishape", None)
    return result_model(save_model(candidate, candidate["name"] + " · 已校准"))


@app.post("/api/generate")
def generate_vehicle(req: dict):
    model = generate(req)
    return result_model(save_model(model, model.get("name", model.get("metadata", {}).get("name", "MiShape · Parametric GT"))))


def normalize_import(model, up_axis, front_axis, length_m, units):
    axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1], "-x": [-1, 0, 0], "-y": [0, -1, 0], "-z": [0, 0, -1]}
    if up_axis not in axes or front_axis not in axes:
        raise ValueError("请选择有效坐标轴")
    up, front = np.array(axes[up_axis]), np.array(axes[front_axis])
    if np.dot(up, front):
        raise ValueError("向上与车头方向必须正交")
    rotation = np.stack([-front, np.cross(up, -front), up])
    v = np.asarray(model["vertices"], dtype=float).reshape(-1, 3) @ rotation.T
    if not len(v) or not np.isfinite(v).all() or np.ptp(v[:, 0]) <= 1e-8:
        raise ValueError("模型缺少有效车长，检查向上与车头坐标轴")
    if units == "fit":
        v *= length_m / np.ptp(v[:, 0])
    elif units == "mm":
        v *= .001
    elif units != "m":
        raise ValueError("未知单位设置")
    v[:, :2] -= (v[:, :2].min(0) + v[:, :2].max(0)) / 2
    v[:, 2] -= v[:, 2].min()
    model["vertices"] = v.ravel().tolist()
    model.setdefault("metadata", {}).update(units="m", scale_method="user_length_calibration" if units == "fit" else "user_declared_units",
        scale_status="user_calibrated" if units == "fit" else "declared", original_up=up_axis, original_front=front_axis,
        reference_length_m=length_m if units == "fit" else float(np.ptp(v[:, 0])),
        texture_note="保留材质基色；此导入入口暂不解析外部纹理依赖")
    return validate_model(model)


def import_mesh(data, ext, up_axis, front_axis, length_m, units):
    if ext == "obj":
        return normalize_import(import_obj(data), up_axis, front_axis, length_m, units)
    loaded = trimesh.load(io.BytesIO(data), file_type=ext, force="scene", process=False)
    objects = []
    if isinstance(loaded, trimesh.Scene):
        for node in loaded.graph.nodes_geometry:
            transform, geom = loaded.graph[node]
            mesh = loaded.geometry[geom].copy()
            if not isinstance(mesh, trimesh.Trimesh):
                continue
            mesh.apply_transform(transform)
            objects.append((str(node), mesh))
    else:
        objects = [("导入模型", loaded)]
    if not objects:
        raise ValueError("文件中没有可读多边形表面")
    vertices, faces, labels, parts, polygons, polygon_labels = [], [], [], [], [], []
    offset = 0
    colors = ["#7e9177", "#35464a", "#718074", "#bbc4c0"]
    for pid, (name, mesh) in enumerate(objects):
        vs = np.asarray(mesh.vertices)
        fs = np.asarray(mesh.faces, dtype=int) + offset
        vertices.extend(vs.tolist()); faces.extend(fs.tolist()); labels.extend([pid] * len(fs))
        polygons.extend(fs.tolist()); polygon_labels.extend([pid] * len(fs))
        color = colors[pid % len(colors)]
        try:
            color = "#" + "".join(f"{int(c):02x}" for c in mesh.visual.main_color[:3])
        except (AttributeError, ValueError):
            pass
        parts.append({"id": pid, "name": name, "color": color, "material": {"base_color": color, "roughness": .34, "metallic": .3}})
        offset += len(vs)
    model = {"vertices": np.asarray(vertices).ravel().tolist(), "faces": np.asarray(faces).ravel().tolist(), "face_labels": labels, "parts": parts,
        "metadata": {"source_format": ext, "import_note": "STL/PLY/GLB 导入以三角面为编辑拓扑；OBJ 和原生 MiShape 工程保留 polygon。"}, "polygons": polygons, "polygon_labels": polygon_labels}
    return normalize_import(model, up_axis, front_axis, length_m, units)


@app.post("/api/import")
async def upload(file: UploadFile = File(...), up_axis: str = Form("z"), front_axis: str = Form("-x"), length_m: float = Form(4.43), units: str = Form("fit")):
    data = await file.read(180 * 1024 * 1024 + 1)
    if len(data) > 180 * 1024 * 1024:
        raise HTTPException(413, "文件不能超过 180 MB")
    if not np.isfinite(length_m) or not 1 < length_m < 20:
        raise ValueError("校准车长应在 1–20 m 范围内")
    name = Path(file.filename or "vehicle").name
    ext = Path(name).suffix.lower().lstrip(".")
    if ext == "json":
        saved = json.loads(data)
        if not isinstance(saved, dict):
            raise ValueError("MiShape 工程需要 JSON 对象")
        model = saved.get("base_model", saved.get("model", saved))
        validate_model(model)
        return result_model(save_model(model, saved.get("name", name)), saved.get("recipe"))
    if ext not in {"obj", "stl", "ply", "glb"}:
        raise ValueError("支持 OBJ、STL、PLY、GLB 和 MiShape 工程 JSON；Blender 素材请使用内置车型或先导出 GLB")
    model = import_mesh(data, ext, up_axis, front_axis, length_m, units)
    return result_model(save_model(model, name))


def mesh_bytes(model, fmt, part_id=None):
    v = np.asarray(model["vertices"]).reshape(-1, 3)
    f = np.asarray(model["faces"]).reshape(-1, 3)
    labels = np.asarray(model["face_labels"])
    if part_id is not None and part_id not in set(labels):
        raise ValueError("选定组件没有多边形")
    if fmt == "obj":
        polygons = model.get("polygons")
        plabels = model.get("polygon_labels") or model.get("polygon_part_ids")
        if polygons is None or not len(polygons) or (part_id is not None and (plabels is None or len(plabels) != len(polygons))):
            polygons, plabels = f.tolist(), labels.tolist()
        parts = {p["id"]: p["name"] for p in model["parts"]}
        selected = []
        for i, poly in enumerate(polygons):
            if isinstance(poly, dict):
                pid = poly.get("part_id", 0); poly = poly.get("vertices", poly.get("indices", []))
            else:
                pid = plabels[i] if plabels is not None else 0
            if part_id is not None and pid != part_id:
                continue
            raw = np.asarray(poly)
            if raw.ndim != 1 or raw.size < 3 or raw.dtype.kind not in "iu" or raw.min() < 0 or raw.max() >= len(v):
                raise ValueError("多边形必须包含至少三个有效整数顶点索引")
            selected.append((pid, raw))
        if not selected:
            raise ValueError("选定组件没有可导出的多边形")
        # Component files contain only their own referenced vertices.  Full-model
        # exports keep original indices, polygon order, and unreferenced vertices.
        remap = None
        export_vertices = v
        if part_id is not None:
            used = np.unique(np.concatenate([poly for _, poly in selected]))
            export_vertices = v[used]
            remap = np.full(len(v), -1, dtype=np.int64)
            remap[used] = np.arange(len(used))
        lines = ["# MiShape polygon vehicle | units m | X front-to-rear, Y width, Z up"]
        lines += ["v %.8f %.8f %.8f" % tuple(p) for p in export_vertices]
        last = None
        for pid, poly in selected:
            if last != pid:
                lines.append("g " + re.sub(r"[\s/#]+", "_", parts.get(pid, f"part_{pid}")))
                last = pid
            if remap is not None:
                poly = remap[poly]
            lines.append("f " + " ".join(str(int(i) + 1) for i in poly))
        return ("\n".join(lines) + "\n").encode()
    if fmt in {"stl", "glb", "ply"}:
        if part_id is not None:
            f = f[labels == part_id]
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        if part_id is not None:
            mesh.remove_unreferenced_vertices()
        if fmt == "glb":
            scene = trimesh.Scene()
            for part in model["parts"]:
                if part_id is not None and part["id"] != part_id:
                    continue
                indices = labels == part["id"]
                if not np.any(indices):
                    continue
                m = trimesh.Trimesh(vertices=v, faces=np.asarray(model["faces"]).reshape(-1, 3)[indices], process=False)
                m.remove_unreferenced_vertices()
                color = part.get("material", {}).get("base_color", part.get("color", "#8a9883"))
                m.visual.face_colors = [int(color[i:i+2], 16) for i in (1, 3, 5)] + [255]
                # glTF is Y-up. Keep the internal design model Z-up and transform
                # only exported scene geometry; GLB import can invert this.
                m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float))
                scene.add_geometry(m, node_name=str(part["name"]), geom_name=str(part["id"]))
            return scene.export(file_type="glb")
        data = mesh.export(file_type=fmt)
        return data.encode() if isinstance(data, str) else data
    raise ValueError("支持 OBJ / STL / GLB / PLY / JSON")


@app.post("/api/models/{mid}/export")
def export_model(mid: str, req: Export):
    recipe = req.model_dump(exclude={"format", "part_id"})
    base = get_model(mid)
    if req.format == "json":
        body = dump({"schema": "mishape-project-v1", "name": base["name"], "base_model": base, "recipe": recipe}).encode()
    else:
        body = mesh_bytes(deform(base, **recipe), req.format, req.part_id)
    return Response(body, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="MiShape-{mid[:8]}.{req.format}"'})


def run_batch(jid, req):
    job = JOBS[jid]
    try:
        model = get_model(req.model_id)
        fields = {p["id"]: p for p in schema()}
        rng = np.random.default_rng(req.seed)
        directory = STATE / jid
        directory.mkdir(exist_ok=True)
        recipes = []
        # Stratified sampling gives coverage of each selected driver interval.
        samples = {key: (rng.permutation(req.count) + rng.random(req.count)) / req.count for key in req.vary}
        for i in range(req.count):
            if job.get("cancelled"):
                job["status"] = "cancelled"
                return
            params = dict(req.parameters)
            for key in req.vary:
                field = fields[key]
                delta = (samples[key][i] - .5) * (field["max"] - field["min"]) * req.strength
                params[key] = float(np.clip(params.get(key, 0) + delta, field["min"], field["max"]))
            recipe = {"parameters": params, "controls": req.controls, "options": req.options}
            result = deform(model, **recipe)
            info = result.get("metadata", {}).get("mishape", {})
            item = {"index": i, "name": f"VAR-{i+1:03d}", "recipe": recipe, "quality": info.get("quality", {}), "analysis": info.get("analysis", {})}
            (directory / f"{i}.json").write_text(dump({"vertices": result["vertices"], **item}))
            (directory / f"{i}.obj").write_bytes(mesh_bytes(result, "obj"))
            recipes.append(item)
            job["items"] = list(recipes)
            job["completed"] = i + 1
        manifest = {"schema": "mishape-batch-v1", "model_id": req.model_id, "seed": req.seed, "sampling": "stratified", "strength": req.strength, "count": req.count, "vary": req.vary, "items": recipes, "note": "形态变体；几何筛查不构成制造、碰撞或 CFD 认证。"}
        with zipfile.ZipFile(directory / "portfolio.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            archive.writestr("manifest.json", dump(manifest))
            archive.writestr("base-model.json", dump(model))
            for i, item in enumerate(recipes):
                archive.write(directory / f"{i}.obj", f'{item["name"]}/vehicle.obj')
                archive.writestr(f'{item["name"]}/recipe.json', dump(item))
        (directory / "manifest.json").write_text(dump(manifest))
        job["status"] = "complete"
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)


@app.post("/api/batches")
def batch(req: Batch):
    get_model(req.model_id)
    fields = {p["id"] for p in schema()}
    if not req.vary or len(req.vary) != len(set(req.vary)) or set(req.vary) - fields:
        raise ValueError("请选择有效且不重复的采样参数")
    deform(get_model(req.model_id), req.parameters, req.controls, req.options)
    if any(j["status"] == "running" for j in JOBS.values()):
        raise HTTPException(409, "已有批量任务正在运行")
    jid = uuid.uuid4().hex
    JOBS[jid] = {"id": jid, "model_id": req.model_id, "status": "running", "completed": 0, "count": req.count, "items": [], "seed": req.seed}
    EXECUTOR.submit(run_batch, jid, req)
    return JOBS[jid]


@app.get("/api/batches/{jid}")
def batch_status(jid: str):
    safe_id(jid)
    if jid not in JOBS:
        path = STATE / jid / "manifest.json"
        if not path.is_file():
            raise HTTPException(404, "批量任务不存在")
        data = json.loads(path.read_text())
        JOBS[jid] = {"id": jid, "status": "complete", "completed": data["count"], **data}
    return JOBS[jid]


@app.post("/api/batches/{jid}/cancel")
def cancel_batch(jid: str):
    job = batch_status(jid)
    job["cancelled"] = True
    return {"status": "cancelling"}


@app.get("/api/batches/{jid}/variants/{index}")
def batch_variant(jid: str, index: int):
    safe_id(jid)
    path = STATE / jid / f"{index}.json"
    if index < 0 or not path.is_file():
        raise HTTPException(404, "变体尚未生成")
    return FileResponse(path)


@app.get("/api/batches/{jid}/download")
def batch_download(jid: str):
    safe_id(jid)
    path = STATE / jid / "portfolio.zip"
    if not path.is_file():
        raise HTTPException(409, "变体组合尚未完成")
    return FileResponse(path, filename="MiShape-Portfolio.zip")


if ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
