"""Polygon-preserving vehicle design kernel.

Canonical coordinates: metres, front=-X, left=+Y, up=+Z, ground=0.
All edits are evaluated from the input (rest) model, never accumulated.  A
7×3×4 tensor lattice interpolates displacement with local cubic support.
This is a styling mesh workflow, not a Class-A or manufacturing validator.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import json
import math
import re
import threading

import numpy as np

from .parameters import schema, validate_parameters

_DIMS = (7, 3, 4)
_CACHE: OrderedDict = OrderedDict()
_LOCK = threading.RLock()


def _arrays(model):
    try:
        v = np.asarray(model['vertices'], dtype=np.float64).reshape(-1, 3)
        raw_f = np.asarray(model['faces'])
        f = raw_f.astype(np.int64).reshape(-1, 3)
    except (KeyError, TypeError, ValueError):
        raise ValueError('Expected vertices and triangular faces as flat numeric arrays.') from None
    if len(v) < 3 or len(f) < 1 or not np.isfinite(v).all():
        raise ValueError('Mesh must contain finite vertices and at least one triangle.')
    if not np.array_equal(raw_f.reshape(-1), f.reshape(-1)):
        raise ValueError('Triangle vertex indices must be integers.')
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError('Triangle vertex index is outside the vertex array.')
    if np.any(np.ptp(v, axis=0) <= 1e-9):
        raise ValueError('Vehicle mesh must have a nonzero extent on every axis.')
    labels = np.asarray(model.get('face_labels', model.get('labels', np.zeros(len(f)))), dtype=np.int64)
    if labels.size != len(f):
        raise ValueError('face_labels must have one entry per triangle.')
    return v, f, labels.reshape(-1)


def _wheel_groups(v, f, labels, parts):
    ids = []
    for part in parts:
        name = (str(part.get('key', '')) + ' ' + str(part.get('name', ''))).lower()
        is_wheel = (any(t in name for t in ('wheel', 'tire', 'tyre', '车轮', '轮胎', '轮毂'))
                    or re.search(r'(^|[^a-z])rims?([^a-z]|$)', name) is not None)
        excluded = any(t in name for t in ('arch', 'fender', 'steering', '方向盘', '轮拱'))
        if is_wheel and not excluded:
            ids.append(part['id'])
    wheel_faces = np.isin(labels, ids)
    if not wheel_faces.any():
        return []
    selected = np.unique(f[wheel_faces])
    # A shared body/wheel vertex cannot be independently rigid without a crack.
    shared = np.intersect1d(selected, np.unique(f[~wheel_faces])) if (~wheel_faces).any() else np.array([], int)
    selected = np.setdiff1d(selected, shared)
    midpoint = (v[:, 0].min() + v[:, 0].max()) / 2
    groups = []
    for front in (True, False):
        for left in (True, False):
            ids = selected[((v[selected, 0] < midpoint) == front) & ((v[selected, 1] >= 0) == left)]
            if len(ids) < 6:
                continue
            lo, hi = v[ids].min(0), v[ids].max(0)
            groups.append(dict(name=('front' if front else 'rear') + ('_left' if left else '_right'),
                               indices=ids, center=(lo + hi) / 2, radius=(hi[2] - lo[2]) / 2,
                               shared_vertices=int(len(shared))))
    return groups


def _semantic_landmarks(v, f, labels, parts, tokens):
    ids = [p['id'] for p in parts if any(t in (str(p.get('key', '')) + ' ' + str(p.get('name', ''))).lower() for t in tokens)]
    ff = f[np.isin(labels, ids)]
    if not len(ff):
        return None
    q = v[np.unique(ff)]
    # Median sections suppress a few window corners/outlier vertices.
    lo, hi = np.quantile(q[:, 2], [.04, .96])
    if hi - lo < .08:
        return None
    bottom = np.median(q[q[:, 2] <= lo + .13 * (hi - lo)], axis=0)
    top = np.median(q[q[:, 2] >= hi - .13 * (hi - lo)], axis=0)
    bottom[1] = top[1] = 0
    return bottom, top


def _analysis(v, f, labels, model, wheels, overrides=None):
    lo, hi = v.min(0), v.max(0)
    length, width, height = hi - lo
    body = np.ones(len(v), bool)
    for w in wheels:
        body[w['indices']] = False
    body_v = v[body] if body.any() else v
    center = body_v[np.abs(body_v[:, 1]) <= max(.20 * width, .08)]
    if len(center) < 20:
        center = body_v
    def top_at(x):
        q = center[np.abs(center[:, 0] - x) < length * .04]
        if not len(q):
            q = center[np.argsort(np.abs(center[:, 0] - x))[:max(3, len(center) // 80)]]
        return np.array([x, 0., float(np.quantile(q[:, 2], .97))])
    xs = np.linspace(lo[0] + .04 * length, hi[0] - .04 * length, 70)
    zs = np.array([top_at(x)[2] for x in xs])
    roof = xs[zs >= np.max(zs) - .085 * height]
    roof_front = float(roof.min()) if len(roof) else lo[0] + .46 * length
    roof_rear = float(roof.max()) if len(roof) else lo[0] + .69 * length
    # Geometry-only defaults are priors and explicitly remain low confidence.
    windshield_base = top_at(max(lo[0] + .22 * length, roof_front - .14 * length))
    windshield_top = top_at(roof_front)
    rear_top = top_at(roof_rear)
    rear_base = top_at(min(hi[0] - .08 * length, roof_rear + .14 * length))
    wind = _semantic_landmarks(v, f, labels, model.get('parts', []), ('windshield', 'windscreen', '前风挡'))
    rear = _semantic_landmarks(v, f, labels, model.get('parts', []), ('rear_glass', 'rear wind', 'backlight', '后风挡'))
    if wind is not None:
        windshield_base, windshield_top = wind
    if rear is not None:
        rear_base, rear_top = rear
    front_groups = [w for w in wheels if w['name'].startswith('front')]
    rear_groups = [w for w in wheels if w['name'].startswith('rear')]
    front_axle = np.mean([w['center'] for w in front_groups], 0) if front_groups else np.array([lo[0] + .22 * length, 0, .24 * height])
    rear_axle = np.mean([w['center'] for w in rear_groups], 0) if rear_groups else np.array([hi[0] - .23 * length, 0, .24 * height])
    front_axle[1] = rear_axle[1] = 0
    nose = body_v[body_v[:, 0] <= lo[0] + .06 * length]
    tail = body_v[body_v[:, 0] >= hi[0] - .06 * length]
    def lower(q, x):
        if not len(q):
            return np.array([x, 0, .16 * height])
        return np.array([x, 0, float(np.quantile(q[:, 2], .07))])
    landmarks = dict(front_tip=np.array([lo[0], 0, .45 * height]), rear_tip=np.array([hi[0], 0, .45 * height]),
                     front_axle=front_axle, rear_axle=rear_axle,
                     windshield_base=windshield_base, windshield_top=windshield_top,
                     rear_glass_top=rear_top, rear_glass_base=rear_base,
                     hood_front=top_at(lo[0] + .10 * length), decklid=top_at(hi[0] - .09 * length),
                     front_bumper_lower=lower(nose, lo[0]), rear_bumper_lower=lower(tail, hi[0]),
                     diffusor_start=np.array([rear_axle[0], 0, max(lo[2] + .07, .09 * height)]),
                     roof_front=windshield_top.copy(), roof_rear=rear_top.copy())
    provided = dict(model.get('metadata', {}).get('landmarks', {}))
    provided.update(overrides or {})
    for key, point in provided.items():
        if key not in landmarks:
            raise ValueError(f'Unknown landmark: {key}')
        point = np.asarray(point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f'Landmark {key} requires three finite metre coordinates.')
        landmarks[key] = point
    if landmarks['front_axle'][0] >= landmarks['rear_axle'][0]:
        raise ValueError('front_axle must precede rear_axle along X (front = negative X).')
    measurements = _measure(v, landmarks, wheels)
    semantic_keys = set()
    if wind is not None:
        semantic_keys.add('windscreen_angle')
    if rear is not None:
        semantic_keys.add('backlight_angle')
    calibrated = bool(overrides)
    scale_status = model.get('metadata', {}).get('scale_status', 'declared')
    unverified_scale = scale_status in ('estimated', 'unknown', 'uncalibrated')
    landmark_dependencies = dict(
        wheelbase={'front_axle', 'rear_axle'}, front_overhang={'front_axle'}, rear_overhang={'rear_axle'},
        approach_angle={'front_axle', 'front_bumper_lower'}, windscreen_angle={'windshield_base', 'windshield_top'},
        backlight_angle={'rear_glass_top', 'rear_glass_base'}, decklid_height={'decklid'},
        hood_angle={'hood_front', 'windshield_base'}, vehicle_pitch={'front_bumper_lower', 'rear_bumper_lower'},
        diffusor_angle={'diffusor_start', 'rear_bumper_lower'})
    for key, item in measurements.items():
        exact = key in ('total_length', 'vehicle_length', 'vehicle_width', 'vehicle_height')
        axle_based = key in ('wheelbase', 'front_overhang', 'rear_overhang') and bool(front_groups and rear_groups)
        needed = landmark_dependencies.get(key, set())
        has_landmarks = bool(needed) and needed.issubset(provided)
        manual = has_landmarks and needed.issubset(overrides or {})
        item['method'] = ('mesh_bounds' if exact else ('user_landmarks' if manual else ('source_landmarks' if has_landmarks else
                          ('named_wheel_bounds' if axle_based else ('named_surface_sections' if key in semantic_keys else 'geometric_estimate')))))
        item['confidence'] = 'high' if exact else ('medium' if axle_based or key in semantic_keys or has_landmarks else 'low')
        item['estimated'] = not exact or (unverified_scale and item['unit'] == 'mm')
        item['scale_status'] = scale_status if item['unit'] == 'mm' else 'scale_independent'
        if unverified_scale and item['unit'] == 'mm':
            item['confidence'] = 'low'
    warnings = []
    if unverified_scale:
        warnings.append('素材采用估计尺度，尺寸读数尚未按实车已知尺寸校准；包围盒数值仅在当前模型尺度下成立。')
    if not wheels:
        warnings.append('未识别到带名称的轮胎组件；轴位来自比例估计，车轮刚性保护不可用。')
    if not wind or not rear:
        warnings.append('部分角度由轮廓启发式估计；请校准风挡、轴心与保险杠地标后使用。')
    warnings.append('总包围盒可能包含后视镜、扰流板等附件；未执行制造公差、碰撞或法规验证。')
    return dict(measurements=measurements, landmarks={k: q.tolist() for k, q in landmarks.items()},
                bounds=dict(min=lo.tolist(), max=hi.tolist(), size=(hi - lo).tolist()),
                wheels=[dict(name=w['name'], center=w['center'].tolist(), radius=float(w['radius']), vertices=len(w['indices'])) for w in wheels],
                coordinate_system=dict(length='X', front='-X', width='Y', up='+Z', unit='m', ground=0),
                scale_calibration=dict(status=scale_status, verified=scale_status in ('calibrated', 'user_calibrated', 'measured', 'authored'), reference_length_m=model.get('metadata', {}).get('reference_length_m')),
                warnings=warnings, calibration='user_landmarks' if calibrated else ('source_landmarks' if provided else 'automatic_estimate'))


def _angle(a, b):
    d = np.asarray(b) - np.asarray(a)
    return math.degrees(math.atan2(abs(d[2]), max(abs(d[0]), 1e-9)))


def _measure(v, landmarks, wheels):
    q = {k: np.asarray(p) for k, p in landmarks.items()}
    lo, hi = v.min(0), v.max(0)
    length, width, height = hi - lo
    front, rear = q['front_axle'], q['rear_axle']
    front_contact = front.copy(); front_contact[2] = 0
    body = np.ones(len(v), bool)
    for w in wheels:
        body[w['indices']] = False
    under = v[body & (v[:, 0] >= front[0]) & (v[:, 0] <= rear[0])]
    clearance = float(np.quantile(under[:, 2], .02)) if len(under) else float(lo[2])
    def section_width(x, zmin=None):
        mask = np.abs(v[:, 0] - x) < .05 * length
        if zmin is not None:
            mask &= v[:, 2] >= zmin
        return float(np.ptp(v[mask, 1])) if mask.sum() > 1 else float(width)
    roof_width = section_width((q['windshield_top'][0] + q['rear_glass_top'][0]) / 2, lo[2] + .78 * height)
    front_width, rear_width = section_width(lo[0] + .07 * length), section_width(hi[0] - .07 * length)
    shoulder = v[(v[:, 2] >= lo[2] + .25 * height) & (v[:, 2] <= lo[2] + .60 * height)]
    body_width = float(np.ptp(shoulder[:, 1])) if len(shoulder) > 1 else float(width)
    pitch = math.degrees(math.atan2(q['front_bumper_lower'][2] - q['rear_bumper_lower'][2], max(length, 1e-9)))
    vals = dict(total_length=(length * 1000, 'mm'), vehicle_length=(length * 1000, 'mm'),
                vehicle_width=(width * 1000, 'mm'), vehicle_height=(height * 1000, 'mm'),
                wheelbase=((rear[0] - front[0]) * 1000, 'mm'), front_overhang=((front[0] - lo[0]) * 1000, 'mm'),
                rear_overhang=((hi[0] - rear[0]) * 1000, 'mm'), approach_angle=(_angle(front_contact, q['front_bumper_lower']), 'deg'),
                windscreen_angle=(_angle(q['windshield_base'], q['windshield_top']), 'deg'),
                backlight_angle=(_angle(q['rear_glass_top'], q['rear_glass_base']), 'deg'),
                decklid_height=(q['decklid'][2] * 1000, 'mm'), rear_end_tapering=((body_width - rear_width) * 1000, 'mm'),
                greenhouse_tapering=((body_width - roof_width) * 1000, 'mm'), hood_angle=(_angle(q['hood_front'], q['windshield_base']), 'deg'),
                front_plan_view=((body_width - front_width) * 1000, 'mm'), ride_height=(clearance * 1000, 'mm'), vehicle_pitch=(pitch, 'deg'),
                diffusor_angle=(_angle(q['diffusor_start'], q['rear_bumper_lower']), 'deg'))
    defs = {p['id']: p['definition'] for p in schema()}
    result = {k: dict(value=round(float(value), 4), unit=unit, mode='absolute', definition=defs.get(k, '前后轴中心在X方向的距离。')) for k, (value, unit) in vals.items()}
    result['vehicle_pitch']['definition'] = '前后保险杠下缘高度差推算的静态侧视倾角；非底盘姿态传感器测量。'
    result['ride_height']['definition'] = '轴间非轮胎顶点Z坐标第2百分位距Z=0地面的高度估计。'
    result['vehicle_length']['definition'] = '测量值为端点总长；此驱动的编辑只拉伸轴间段，保持前后悬长度。'
    result['rear_end_tapering']['definition'] = '下车身肩宽减去尾端7%截面宽度；尽量排除后视镜。'
    result['greenhouse_tapering']['definition'] = '下车身肩宽减去车顶中央上部截面宽度；尽量排除后视镜。'
    result['front_plan_view']['definition'] = '下车身肩宽减去前端7%截面宽度；尽量排除后视镜。'
    return result


def _key(v, f, labels, model, landmarks):
    h = sha256()
    for arr in (v, f, labels):
        h.update(arr.tobytes())
    h.update(json.dumps([model.get('parts', []), model.get('metadata', {}).get('landmarks', {}), model.get('metadata', {}).get('scale_status'), model.get('metadata', {}).get('reference_length_m'), landmarks], sort_keys=True, default=str).encode())
    return h.hexdigest()


def _cubic_axis(t, size):
    """Catmull-Rom displacement weights with linear boundary extrapolation."""
    u = np.clip(t, 0, 1) * (size - 1)
    base = np.minimum(np.floor(u).astype(int), size - 2)
    s = u - base
    weights = np.stack((-.5*s + s*s - .5*s*s*s, 1 - 2.5*s*s + 1.5*s*s*s,
                        .5*s + 2*s*s - 1.5*s*s*s, -.5*s*s + .5*s*s*s), axis=1)
    indices = base[:, None] + np.arange(-1, 3)[None, :]
    # A ghost displacement = 2*d(boundary)-d(next), preserving affine fields.
    before = indices[:, 0] < 0
    weights[before, 1] += 2 * weights[before, 0]
    weights[before, 2] -= weights[before, 0]
    weights[before, 0] = 0
    after = indices[:, 3] >= size
    weights[after, 2] += 2 * weights[after, 3]
    weights[after, 1] -= weights[after, 3]
    weights[after, 3] = 0
    return np.clip(indices, 0, size - 1), weights


def _binding(points, lo, hi):
    t = (points - lo) / (hi - lo)
    return [_cubic_axis(t[:, axis], n) for axis, n in enumerate(_DIMS)]


def _interpolate(binding, displacement):
    n = len(binding[0][0])
    result = np.zeros((n, 3))
    grid = displacement.reshape(*_DIMS, 3)
    ix, wx = binding[0]; iy, wy = binding[1]; iz, wz = binding[2]
    for a in range(4):
        for b in range(4):
            xy = wx[:, a] * wy[:, b]
            for c in range(4):
                weight = xy * wz[:, c]
                if np.any(weight):
                    result += weight[:, None] * grid[ix[:, a], iy[:, b], iz[:, c]]
    return result


def _prepare(model, landmarks=None):
    v, f, labels = _arrays(model)
    key = _key(v, f, labels, model, landmarks)
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    wheels = _wheel_groups(v, f, labels, model.get('parts', []))
    analysis = _analysis(v, f, labels, model, wheels, landmarks)
    lo, hi = v.min(0), v.max(0)
    grid = np.stack(np.meshgrid(*[np.linspace(lo[a], hi[a], count) for a, count in enumerate(_DIMS)], indexing='ij'), axis=-1).reshape(-1, 3)
    lmkeys = list(analysis['landmarks'])
    lmarray = np.array([analysis['landmarks'][k] for k in lmkeys])
    prepared = dict(v=v.copy(), f=f, labels=labels, analysis=analysis, wheels=wheels, lo=lo, hi=hi, grid=grid,
                    binding=_binding(v, lo, hi), landmark_keys=lmkeys, landmark_array=lmarray,
                    landmark_binding=_binding(lmarray, lo, hi), key=key)
    with _LOCK:
        _CACHE[key] = prepared
        while len(_CACHE) > 2:
            _CACHE.popitem(last=False)
    return prepared


def analyze(model, landmarks=None):
    """Measure the rest polygon model; automatic landmarks are explicitly estimates."""
    return deepcopy(_prepare(model, landmarks)['analysis'])


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _driver_displacement(points, p, data, wheel_mode=False):
    lo, hi = data['lo'], data['hi']
    length, width, height = hi - lo
    lm = {k: np.array(v) for k, v in data['analysis']['landmarks'].items()}
    x, y, z = points.T
    front_axle, rear_axle = lm['front_axle'][0], lm['rear_axle'][0]
    core = np.clip((x - front_axle) / max(rear_axle - front_axle, .1), 0, 1)
    front = _smooth((front_axle - x) / max(front_axle - lo[0], .1))
    rear = _smooth((x - rear_axle) / max(hi[0] - rear_axle, .1))
    upper = _smooth((z - lo[2] - .45 * height) / max(.4 * height, .1))
    lower = 1 - _smooth((z - lo[2] - .2 * height) / max(.4 * height, .1))
    delta = np.zeros_like(points)
    def mm(k):
        return p.get(k, 0) / 1000
    delta[:, 0] += (x - (lo[0] + hi[0]) / 2) / length * mm('total_length')
    delta[:, 0] += (core - .5) * mm('vehicle_length')
    delta[:, 1] += y / width * mm('vehicle_width')
    if wheel_mode:
        return delta
    delta[:, 0] += -front * mm('front_overhang') + rear * mm('rear_overhang')
    delta[:, 2] += (z - lo[2]) / height * mm('vehicle_height') + mm('ride_height')
    delta[:, 1] -= y / max(width / 2, .1) * .5 * (rear * mm('rear_end_tapering') + front * mm('front_plan_view') + upper * mm('greenhouse_tapering'))
    rear_upper = _smooth((x - lm['rear_glass_top'][0]) / max(hi[0] - lm['rear_glass_top'][0], .2))
    delta[:, 2] += rear_upper * (1 - lower) * mm('decklid_height')
    measures = data['analysis']['measurements']
    for driver, base_key, top_key, region, sign in (
            ('windscreen_angle', 'windshield_base', 'windshield_top', 'front', 1),
            ('backlight_angle', 'rear_glass_base', 'rear_glass_top', 'rear', -1)):
        if not p.get(driver):
            continue
        base_pt, top_pt = lm[base_key], lm[top_key]
        rise = abs(top_pt[2] - base_pt[2]); run = abs(top_pt[0] - base_pt[0])
        angle = np.clip(measures[driver]['value'] + p[driver], 8, 85)
        shift = sign * (rise / np.tan(np.deg2rad(angle)) - run)
        longitudinal = np.exp(-((x - top_pt[0]) / max(.17 * length, .3)) ** 2)
        gate = _smooth((z - base_pt[2]) / max(rise, .1))
        delta[:, 0] += np.clip(shift, -.45, .45) * longitudinal * gate
    if p.get('hood_angle'):
        run = max(abs(lm['windshield_base'][0] - lm['hood_front'][0]), .15)
        old = np.deg2rad(measures['hood_angle']['value'])
        new = np.deg2rad(np.clip(measures['hood_angle']['value'] + p['hood_angle'], -25, 50))
        hood_x = _smooth((lm['windshield_base'][0] - x) / run)
        hood_z = np.exp(-((z - lm['hood_front'][2]) / max(.25 * height, .1)) ** 2)
        delta[:, 2] -= run * (np.tan(new) - np.tan(old)) * hood_x * hood_z
    for key, mask, run in (('approach_angle', front, max(front_axle - lo[0], .1)),
                           ('diffusor_angle', rear, max(hi[0] - lm['diffusor_start'][0], .1))):
        if p.get(key):
            angle = measures[key]['value']
            dz = run * (np.tan(np.deg2rad(np.clip(angle + p[key], -15, 55))) - np.tan(np.deg2rad(np.clip(angle, -15, 55))))
            delta[:, 2] += np.clip(dz, -.4, .4) * mask * lower
    if p.get('vehicle_pitch'):
        radians = np.deg2rad(p['vehicle_pitch'])
        center = np.array([(front_axle + rear_axle) / 2, 0, (lm['front_axle'][2] + lm['rear_axle'][2]) / 2])
        q = points + delta - center
        rotated = q.copy()
        rotated[:, 0] = np.cos(radians) * q[:, 0] + np.sin(radians) * q[:, 2]
        rotated[:, 2] = -np.sin(radians) * q[:, 0] + np.cos(radians) * q[:, 2]
        delta = rotated + center - points
    return delta


def _control_displacement(controls, symmetry):
    result = np.zeros((np.prod(_DIMS), 3))
    specified = {}
    for control in controls or []:
        try:
            raw_id = control['id']
            node = int(raw_id)
            if node != float(raw_id):
                raise ValueError
            d = np.asarray(control['delta'], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise ValueError('Each control requires an integer id and delta:[dx,dy,dz] in metres.') from None
        if not 0 <= node < len(result) or d.shape != (3,) or not np.isfinite(d).all() or np.linalg.norm(d) > .75:
            raise ValueError('Control id is invalid or its displacement exceeds 0.75 m.')
        if node in specified:
            raise ValueError(f'Duplicate control node {node}.')
        specified[node] = d
    for node, d in specified.items():
        result[node] = d
    if symmetry:
        visited = set()
        mirror = np.array([1, -1, 1])
        for node, d in specified.items():
            if node in visited:
                continue
            i, j, k = np.unravel_index(node, _DIMS)
            partner = int(np.ravel_multi_index((i, _DIMS[1] - 1 - j, k), _DIMS))
            if partner == node:
                result[node, 1] = 0
            else:
                value = (d + specified[partner] * mirror) / 2 if partner in specified else d
                result[node] = value
                result[partner] = value * mirror
            visited.update((node, partner))
    return result


def _cage_json(data, displacement):
    points, edges = [], []
    for node, (rest, moved) in enumerate(zip(data['grid'], data['grid'] + displacement)):
        ijk = np.unravel_index(node, _DIMS)
        mirror = int(np.ravel_multi_index((ijk[0], _DIMS[1] - 1 - ijk[1], ijk[2]), _DIMS))
        points.append(dict(id=node, index=list(map(int, ijk)), rest=rest.tolist(), position=moved.tolist(), mirror_id=mirror))
        for axis in range(3):
            if ijk[axis] + 1 < _DIMS[axis]:
                other = list(ijk); other[axis] += 1
                edges.append([node, int(np.ravel_multi_index(tuple(other), _DIMS))])
    return dict(dimensions=list(_DIMS), points=points, edges=edges, unit='m', method='local_cubic_displacement',
                coordinate_system='X length / Y width / Z up; front=-X', editable=True)


def get_cage(model, parameters=None, controls=None, options=None):
    options = options or {}
    p = validate_parameters(parameters)
    data = _prepare(model, options.get('landmarks'))
    d = _driver_displacement(data['grid'], p, data)
    d += _control_displacement(controls, options.get('symmetry', True))
    return _cage_json(data, d)


def _quality(data, moved, rigid_errors, preserve_wheels):
    v, f = data['v'], data['f']
    original_tri, moved_tri = v[f], moved[f]
    old_n = np.cross(original_tri[:, 1] - original_tri[:, 0], original_tri[:, 2] - original_tri[:, 0])
    new_n = np.cross(moved_tri[:, 1] - moved_tri[:, 0], moved_tri[:, 2] - moved_tri[:, 0])
    old_area, new_area = np.linalg.norm(old_n, axis=1) / 2, np.linalg.norm(new_n, axis=1) / 2
    tolerance = max(np.max(data['hi'] - data['lo']) ** 2 * 1e-12, 1e-14)
    good = old_area > tolerance
    flips = int(np.sum(((old_n * new_n).sum(1) < 0) & good & (new_area > tolerance)))
    added_degenerate = int(np.sum((new_area <= tolerance) & good))
    displacement = np.linalg.norm(moved - v, axis=1)
    finite = bool(np.isfinite(moved).all())
    warnings = []
    if flips:
        warnings.append(f'{flips} 个三角面相对基准法线翻转；请减小参数幅度并检查。')
    if added_degenerate:
        warnings.append(f'{added_degenerate} 个新增退化三角面；请减小参数幅度。')
    if not data['wheels']:
        warnings.append('没有命名轮胎组件，无法核实轮胎刚性。')
    if float(moved[:, 2].min()) < -.005:
        warnings.append('部分顶点位于Z=0地面以下；请检查离地高度和轮胎接地。')
    return dict(status='pass' if finite and not flips and not added_degenerate else 'review', finite=finite,
                nonfinite_values=int((~np.isfinite(moved)).sum()), triangles=int(len(f)), vertices=int(len(v)),
                source_degenerate_faces=int((~good).sum()), new_degenerate_faces=added_degenerate, normal_flips=flips,
                max_displacement_mm=round(float(displacement.max() * 1000), 5),
                rms_displacement_mm=round(float(np.sqrt(np.mean(displacement ** 2)) * 1000), 5),
                topology_preserved=True, wheel_rigidity=dict(enabled=bool(preserve_wheels), groups=len(data['wheels']),
                    verified=bool(data['wheels']) and preserve_wheels, max_shape_error_mm=round(max(rigid_errors, default=0.) * 1000, 8)),
                self_intersection='not_checked', manufacturing_validation='not_performed', warnings=warnings)


def deform(model, parameters=None, controls=None, options=None):
    """Return a polygon model with identical indices/labels and evaluated geometry.

    Parameters are deltas in schema units. Controls are ``{id, delta:[m,m,m]}``.
    ``options`` supports symmetry, preserve_wheels, and explicit landmarks.
    Repeated calls with the same input are deterministic. Original vertices,
    polygons, labels, UVs, and any unrelated metadata remain untouched.
    """
    options = options or {}
    p = validate_parameters(parameters)
    data = _prepare(model, options.get('landmarks'))
    preserve_wheels = bool(options.get('preserve_wheels', True))
    manual = _control_displacement(controls, options.get('symmetry', True))
    cage_displacement = _driver_displacement(data['grid'], p, data) + manual
    v = data['v']
    if not p and not np.any(manual):
        moved = v.copy()  # Exact identity; do not pass coordinates through arithmetic.
    else:
        moved = v + _interpolate(data['binding'], cage_displacement)
    rigid_errors = []
    wheel_moves = {}
    if preserve_wheels:
        for wheel in data['wheels']:
            center = wheel['center'][None, :]
            d = _driver_displacement(center, p, data, wheel_mode=True)[0]
            d += _interpolate(_binding(center, data['lo'], data['hi']), manual)[0]
            ids = wheel['indices']
            moved[ids] = v[ids] + d
            wheel_moves[wheel['name']] = d
            rigid_errors.append(float(np.max(np.linalg.norm((moved[ids] - moved[ids].mean(0)) - (v[ids] - v[ids].mean(0)), axis=1))))
    lmarray = data['landmark_array'] + _interpolate(data['landmark_binding'], cage_displacement)
    lm = {k: q.tolist() for k, q in zip(data['landmark_keys'], lmarray)}
    moved_wheels = []
    for wheel in data['wheels']:
        w = dict(wheel)
        w['center'] = (moved[w['indices']].min(0) + moved[w['indices']].max(0)) / 2
        moved_wheels.append(w)
    if preserve_wheels:
        for axle in ('front', 'rear'):
            wheels = [w for w in moved_wheels if w['name'].startswith(axle)]
            if wheels:
                # Preserve a calibrated axle landmark; only apply the wheel groups'
                # mean translation, rather than silently replacing its rest location.
                center = np.array(data['analysis']['landmarks'][axle + '_axle'])
                center = center + np.mean([wheel_moves[w['name']] for w in wheels], axis=0)
                center[1] = 0
                lm[axle + '_axle'] = center.tolist()
    analysis = deepcopy(data['analysis'])
    measured = _measure(moved, lm, moved_wheels)
    for key, item in measured.items():
        item.update({k: analysis['measurements'][key][k] for k in ('method', 'confidence', 'estimated', 'scale_status')})
        item['delta_from_source'] = round(item['value'] - analysis['measurements'][key]['value'], 4)
    analysis.update(measurements=measured, landmarks=lm,
                    bounds=dict(min=moved.min(0).tolist(), max=moved.max(0).tolist(), size=np.ptp(moved, axis=0).tolist()),
                    wheels=[dict(name=w['name'], center=w['center'].tolist(), radius=float(w['radius']), vertices=len(w['indices'])) for w in moved_wheels])
    quality = _quality(data, moved, rigid_errors, preserve_wheels)
    out = dict(model)
    # Preserve original polygon indices and per-face labels without re-triangulation.
    out['vertices'] = moved.reshape(-1).tolist() if p or np.any(manual) else deepcopy(model['vertices'])
    out['metadata'] = deepcopy(model.get('metadata', {}))
    out['metadata']['landmarks'] = deepcopy(lm)
    out['metadata']['landmark_provenance'] = 'transformed_source_landmarks_and_geometric_estimates'
    out['metadata']['mishape'] = dict(version='1.0', parameters=p, controls=deepcopy(controls or []),
                                     options=dict(symmetry=bool(options.get('symmetry', True)), preserve_wheels=preserve_wheels),
                                     source_hash=data['key'], analysis=analysis, quality=quality,
                                     cage=_cage_json(data, cage_displacement),
                                     evaluation='rest_model_delta', certification='styling_geometry_only')
    return out


def sample_parameters(count=12, seed=42, ranges=None):
    """Reproducible Latin-hypercube samples in user-selected delta ranges."""
    if isinstance(count, bool) or int(count) != count or not 1 <= int(count) <= 500:
        raise ValueError('Batch count must be an integer between 1 and 500.')
    count = int(count)
    ranges = ranges if ranges is not None else {'total_length': [-150, 150], 'vehicle_width': [-75, 75], 'vehicle_height': [-60, 60], 'vehicle_pitch': [-1, 1]}
    specs = {p['id']: p for p in schema()}
    rng = np.random.default_rng(int(seed))
    samples = [{} for _ in range(count)]
    for key in sorted(ranges):
        if key not in specs:
            raise ValueError(f'Unknown batch parameter: {key}')
        pair = np.asarray(ranges[key], float)
        if pair.shape != (2,) or not np.isfinite(pair).all() or pair[0] > pair[1] or pair[0] < specs[key]['min'] or pair[1] > specs[key]['max']:
            raise ValueError(f'Invalid batch range for {key}.')
        values = pair[0] + (pair[1] - pair[0]) * ((rng.permutation(count) + rng.random(count)) / count)
        for index, value in enumerate(values):
            samples[index][key] = round(float(value), 6)
    return samples
