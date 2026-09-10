"""MiShape's reproducible, component-aware polygon vehicle generator.

The surface is an original authored concept scaffold, not an OEM reconstruction
or a manufacturing-certified CAD model.  Geometry is in metres with the nose at
negative X, Y lateral, and Z up.  The public wire format uses flat numeric arrays.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from aeroshape.components import COMPONENTS
from aeroshape.engineering import EngineeringParams, generate as engineering_generate, preset
from aeroshape.parametric_quality import parametric_quality


REVISION = "mishape.polygon-generator.v1"
BODY_STYLES = {
    "fastback": ("Fastback · 溜背", "fastback"),
    "sedan": ("Sedan · 三厢", "notchback"),
    "estate": ("Estate · 旅行车", "estateback"),
    "hatchback": ("Hatchback · 两厢", "hatchback"),
    "suv": ("SUV · 运动多用途", "suv"),
}

# key, engine field, display label, unit, group, min, max, step, description
_FIELDS = [
    ("length", None, "Vehicle length · 整车长度", "m", "proportions", 3.90, 5.35, .01, "轴距与前后悬之和；前后悬按前悬占比分配。"),
    ("width", "body_width_m", "Body width · 车身宽度", "m", "proportions", 1.70, 2.10, .01, "车身外板的基础宽度，不含后视镜；实测包络在质量报告中。"),
    ("height", "roof_height_m", "Vehicle height · 车顶高度", "m", "proportions", 1.32, 1.88, .01, "轮胎接地点到车顶中心冠顶控制点的高度。"),
    ("wheelbase", "wheelbase_m", "Wheelbase · 轴距", "m", "proportions", 2.45, 3.20, .01, "前后轮轴心间距；保持整车长度时会改变两端悬长。"),
    ("front_overhang_ratio", None, "Front overhang share · 前悬占比", "ratio", "proportions", .38, .62, .005, "前悬占总悬长的比例；前悬与后悬的实际值自动计算。"),
    ("ride_height", "floor_height_m", "Floor height · 底板离地", "m", "proportions", .13, .30, .005, "中部底板中心名义离地高度；不是整车最小离地间隙。"),
    ("track", "track_m", "Track · 轮距", "m", "wheels", 1.50, 1.88, .01, "左右轮组中心距，前后轴相同。"),
    ("wheel_radius", "wheel_radius_m", "Wheel radius · 轮胎半径", "m", "wheels", .30, .43, .005, "轮胎和轮毂径向等比变化，轮拱随之适配。"),
    ("tyre_width", "tyre_width_m", "Tyre width · 胎宽", "m", "wheels", .19, .29, .005, "轮组轴向宽度；必须与轮距、车宽相容。"),
    ("cabin_length", "cabin_length_m", "Cabin length · 车顶控制段", "m", "cabin", 1.02, 2.55, .01, "前风挡上缘到后风挡上缘的纵向距离。"),
    ("windscreen_angle", "windshield_angle_deg", "Windscreen angle · 前风挡角", "°", "cabin", 25., 65., .5, "中心纵剖面上下缘弦线与水平面的夹角。"),
    ("backlight_angle", "rear_glass_angle_deg", "Backlight angle · 后风挡角", "°", "cabin", 22., 82., .5, "后风挡中心剖面弦线角；不等于每个面片的法向角。"),
    ("roof_crown", "roof_crown_m", "Roof crown · 车顶冠高", "m", "cabin", .01, .18, .005, "车顶中心相对前缘的冠高。"),
    ("roof_rear_drop", "rear_roof_drop_m", "Rear roof drop · 车顶后缘落差", "m", "cabin", -.12, .20, .005, "正值降低后缘，负值使后缘高于前缘。"),
    ("greenhouse_taper", "cabin_tumblehome_deg", "Greenhouse taper · 座舱内倾", "°", "cabin", 0., 18., .5, "腰线以上向内倾斜，左右对称。"),
    ("hood_height", "hood_height_m", "Hood height · 前盖后缘高度", "m", "front", .90, 1.18, .005, "前风挡下缘中心高度，与风挡角共同决定座舱前缘。"),
    ("hood_crown", "hood_crown_m", "Hood crown · 前盖冠高", "m", "front", 0., .06, .002, "前盖中部的平滑拱高，保持端部衔接。"),
    ("front_taper", "front_body_taper", "Front plan view · 车头收窄", "ratio", "front", 0., .20, .005, "从前轴向车头的连续平面收窄比例。"),
    ("decklid_height", "rear_deck_height_m", "Decklid height · 后甲板高度", "m", "rear", .84, 1.28, .005, "后风挡下缘中心高度，受后车顶和腰线约束。"),
    ("decklid_length", "rear_deck_length_m", "Decklid length · 尾箱平台长度", "m", "rear", 0., .85, .01, "0 为连续尾门；正值构成独立尾箱平台。"),
    ("rear_taper", "rear_body_taper", "Rear-end taper · 车尾收窄", "ratio", "rear", 0., .18, .005, "从后轴至车尾的平滑单侧收窄比例。"),
    ("rear_shoulder", "rear_shoulder_m", "Rear shoulder · 后肩外扩", "m", "rear", 0., .05, .002, "后翼子板单侧外扩量，会增加实测车身宽度。"),
    ("tailgate_angle", "tailgate_angle_deg", "Tailgate angle · 尾门角", "°", "rear", 55., 89., .5, "尾门中心上下缘弦线角，独立于后玻璃角度。"),
    ("tailgate_lower_height", "tailgate_lower_height_m", "Tailgate lower edge · 尾门下缘", "m", "rear", .62, 1.02, .005, "尾门与后围板的中心连接高度。"),
    ("rear_belt_height", "rear_belt_height_m", "Rear beltline · 后腰线高度", "m", "rear", .82, 1.16, .005, "后玻璃附近的侧面腰线；须低于玻璃下缘。"),
    ("diffuser_angle", "diffuser_angle_deg", "Diffuser angle · 扩散器角", "°", "aero", 2., 18., .5, "底部控制段入口到出口的中心弦线角。"),
    ("diffuser_length", "diffuser_length_m", "Diffuser length · 扩散器长度", "m", "aero", .30, .82, .01, "车尾底部渐升段的长度。"),
    ("diffuser_width", "diffuser_width_m", "Diffuser width · 扩散器宽度", "m", "aero", .85, 1.60, .01, "左右平滑过渡的底板控制带宽度。"),
]
_GROUPS = [
    ("proportions", "比例与姿态", "Proportions"),
    ("wheels", "轮组与轮拱", "Wheels & stance"),
    ("cabin", "座舱与车顶", "Greenhouse"),
    ("front", "前盖与车头", "Front architecture"),
    ("rear", "尾部构型", "Rear architecture"),
    ("aero", "底部气动造型", "Underbody"),
    ("components", "组件选型", "Component choices"),
]
_COMPONENT_KEYS = {f"{category}_style": category for category in COMPONENTS}
_FIELD_BY_KEY = {field[0]: field for field in _FIELDS}
_ENGINE_TO_PUBLIC = {field[1]: field[0] for field in _FIELDS if field[1]}


def _style(value: str) -> str:
    value = {"notchback": "sedan", "estateback": "estate"}.get(value, value)
    if value not in BODY_STYLES:
        raise ValueError(f"Unknown body_style {value!r}; choose {', '.join(BODY_STYLES)}.")
    return value


def _public_preset(body_style: str) -> dict[str, Any]:
    p = preset(BODY_STYLES[body_style][1], "v6").model_dump()
    values = {key: p[engine] for key, engine, *_ in _FIELDS if engine}
    values.update(
        length=p["wheelbase_m"] + p["front_overhang_m"] + p["rear_overhang_m"],
        front_overhang_ratio=p["front_overhang_m"] / (p["front_overhang_m"] + p["rear_overhang_m"]),
        **{key: p[key] for key in _COMPONENT_KEYS},
    )
    return values


def generation_schema() -> dict[str, Any]:
    """JSON-safe controls; numeric request values use the displayed unit (metres)."""
    defaults = _public_preset("fastback")
    fields = [dict(id=key, key=key, engine_key=engine, name=name, label=name.split(" · ")[-1], label_en=name.split(" · ")[0], unit=unit,
                   group=group, min=lo, max=hi, step=step, type="range",
                   default=defaults[key], description=description)
              for key, engine, name, unit, group, lo, hi, step, description in _FIELDS]
    for key, category in _COMPONENT_KEYS.items():
        label = {"mirror": "后视镜", "spoiler": "尾翼", "front": "前脸", "diffuser": "扩散器"}[category]
        fields.append(dict(id=key, key=key, name=label, label=label,
                           label_en=category.title(), group="components", type="select",
                           default=defaults[key], options=[dict(value=c["id"], label=c["name"], description=c["description"]) for c in COMPONENTS[category]]))
    return dict(
        schema=REVISION, units="m", front_axis="-X", up_axis="Z",
        body_styles=[dict(value=k, label=name) for k, (name, _) in BODY_STYLES.items()],
        fields=fields,
        groups=[dict(id=key, key=key, name=name, label=name, english=english,
                     fields=[f for f in fields if f["group"] == key]) for key, name, english in _GROUPS],
        presets={style: _public_preset(style) for style in BODY_STYLES},
        defaults=dict(body_style="fastback", **defaults),
        fine_shape=dict(count=70, min=-.22, max=.22, optional=True,
                        meaning="21 longitudinal metre offsets, 21 lateral scale coefficients, 21 vertical metre offsets, 7 crown metre offsets"),
        dimensional_constraints=[
            "front_overhang=(length-wheelbase)*front_overhang_ratio must be 0.68–1.12 m",
            "rear_overhang=(length-wheelbase)*(1-front_overhang_ratio) must be 0.55–1.18 m",
            "track+tyre_width <= width+0.10 m",
            "decklid_height-rear_belt_height >= 0.045 m",
            "decklid_height-tailgate_lower_height >= 0.10 m",
            "height-roof_crown-roof_rear_drop-decklid_height >= 0.16 m",
        ],
        validation_level="authored_parametric_polygon_surface",
        production_ready=False,
    )


def _parameters(request: dict[str, Any]) -> tuple[str, dict[str, Any], EngineeringParams, str | None]:
    if not isinstance(request, dict):
        raise ValueError("Generation request must be an object.")
    # Recipes and a flat slider state can both be replayed directly.
    nested = request.get("parameters", request.get("params", {}))
    if not isinstance(nested, dict):
        raise ValueError("parameters must be an object.")
    controls = {**nested, **{k: v for k, v in request.items() if k not in {"parameters", "params"}}}
    body = _style(controls.pop("body_style", controls.pop("body_type", "fastback")))
    revision = controls.pop("schema", REVISION)
    if revision != REVISION:
        raise ValueError(f"Unsupported generator recipe schema: {revision!r}.")
    paint = controls.pop("paint_color", None)
    if paint is not None and (not isinstance(paint, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", paint) is None):
        raise ValueError("paint_color must be a six-digit #RRGGBB colour.")
    seed = controls.pop("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
        raise ValueError("seed must be an integer in 0..2147483647.")
    fine_shape = controls.pop("fine_shape", None)
    # Accept physical engineering aliases, while keeping a single public recipe.
    for key in list(controls):
        if key in _ENGINE_TO_PUBLIC:
            public = _ENGINE_TO_PUBLIC[key]
            if public in controls and controls[public] != controls[key]:
                raise ValueError(f"Conflicting values for {public} and {key}.")
            controls[public] = controls.pop(key)
    unknown = set(controls) - set(_FIELD_BY_KEY) - set(_COMPONENT_KEYS)
    if unknown:
        raise ValueError(f"Unknown generation parameters: {', '.join(sorted(unknown))}.")
    values = {**_public_preset(body), **controls}
    for key, _, _, _, _, minimum, maximum, _, _ in _FIELDS:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
        if not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}.")
        values[key] = float(value)
    for key, category in _COMPONENT_KEYS.items():
        allowed = [item["id"] for item in COMPONENTS[category]]
        if values[key] not in allowed:
            raise ValueError(f"{key} must be one of {', '.join(allowed)}.")
    overhang = values["length"] - values["wheelbase"]
    front = overhang * values["front_overhang_ratio"]
    rear = overhang - front
    if not (.68 <= front <= 1.12 and .55 <= rear <= 1.18):
        raise ValueError(f"Length/wheelbase combination gives front/rear overhangs {front:.3f}/{rear:.3f} m; required 0.68–1.12/0.55–1.18 m. Adjust length, wheelbase or front_overhang_ratio.")
    physical = preset(BODY_STYLES[body][1], "v6").model_dump()
    physical.update({engine: values[key] for key, engine, *_ in _FIELDS if engine})
    physical.update(front_overhang_m=front, rear_overhang_m=rear, seed=seed,
                    **{key: values[key] for key in _COMPONENT_KEYS})
    if fine_shape is not None:
        physical["fine_shape"] = fine_shape
    parameters = EngineeringParams.model_validate(physical)
    recipe = dict(schema=REVISION, body_style=body, seed=seed, parameters=values)
    if fine_shape is not None:
        recipe["fine_shape"] = list(parameters.fine_shape)
    if paint is not None:
        recipe["paint_color"] = paint
    return body, recipe, parameters, paint


def generate(request: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate a detailed, split-labelled triangular vehicle and its exact recipe.

    Invalid coupled dimensions fail explicitly. A candidate that passes the
    generator's collapse guard but fails stricter surface screening remains
    inspectable, with ``quality.screen_pass=False`` and concrete reasons.
    """
    body, recipe, parameters, paint = _parameters({} if request is None else request)
    model = engineering_generate(parameters)
    quality = parametric_quality(model)
    quality.update(production_ready=False, scope="Concept polygon surface and authored component memberships",
                   manufacturing_tolerances="not_defined", class_a_surface="not_certified")
    vertices = model.vertices
    measurements = copy.deepcopy(model.metadata["generator_measurements"])
    body_ids = [p["id"] for p in model.parts if p.get("category") == "body"]
    body_vertices = vertices[np.unique(model.faces[np.isin(model.labels, body_ids)])]
    measurements.update(
        body_width_excluding_mirrors_m=float(np.ptp(body_vertices[:, 1])),
        wheel_radius_m=parameters.wheel_radius_m, track_m=parameters.track_m,
        floor_control_height_m=parameters.floor_height_m,
        dimensions_m=dict(length=float(np.ptp(vertices[:, 0])), width=float(np.ptp(vertices[:, 1])), height=float(np.ptp(vertices[:, 2]))),
        width_definition="dimensions_m.width includes mirrors; width parameter controls body scaffold",
    )
    materials = copy.deepcopy(model.metadata["material_palette"])
    if paint:
        original_paint = model.parts[0]["material"]
        for material in materials:
            if material == original_paint:
                material["base_color"] = paint
    parts = copy.deepcopy(model.parts)
    for part in parts:
        part["face_count"] = int(np.count_nonzero(model.labels == part["id"]))
        part["present"] = bool(part["face_count"])
        part["separable"] = True
        if paint and part["material"] == model.parts[0]["material"]:
            part["material"]["base_color"] = paint
    recipe_json = json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    recipe_hash = hashlib.sha256(recipe_json.encode()).hexdigest()
    material_ids = list(model.metadata["face_material_ids"])
    metadata = dict(
        name=f"MiShape {BODY_STYLES[body][0].split(' · ')[0]}", source="generated",
        generator=REVISION, generator_revision="aeroshape.engineering.v6", body_style=body,
        units="m", coordinates="X front-to-rear (nose -X); Y lateral; Z up", front_axis="-X", up_axis="Z",
        vertex_count=len(vertices), face_count=len(model.faces), component_count=len(np.unique(model.labels)),
        bounds=dict(min=vertices.min(axis=0).tolist(), max=vertices.max(axis=0).tolist()),
        dimensions=measurements["dimensions_m"], measurements=measurements,
        quality=quality, recipe=recipe, recipe_hash=recipe_hash,
        geometry_hash=model.metadata["base_geometry_hash"],
        material_palette=materials,
        component_choices=copy.deepcopy(model.metadata["component_choices"]),
        source_provenance=copy.deepcopy(model.metadata["provenance"]),
        license="CC0-1.0", production_ready=False, cfd_ready=False,
        notes=["Original authored vehicle concept, not a Porsche/OEM reproduction.",
               "Exterior surfaces, wheel assemblies, lights, glazing and trim carry persistent semantic part IDs.",
               "Closed body driver does not establish a boolean union, panel thickness, watertight whole assembly or Class-A CAD continuity.",
               "Self-intersection, component clearance and manufacturing feasibility require further validation.",
               "The seed labels reproducible recipes; it does not add hidden vertex noise."],
    )
    return dict(vertices=vertices.reshape(-1).tolist(), faces=model.faces.reshape(-1).tolist(),
                face_labels=model.labels.tolist(), parts=parts, face_material_ids=material_ids,
                face_materials=material_ids, source_face_ids=list(range(len(model.faces))),
                materials=materials, metadata=metadata)
