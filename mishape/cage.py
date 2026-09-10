"""Rest-space, symmetric vehicle envelopes for curvilinear displacement FFD.

The fitted cage uses geometry-derived X sections, not a warped display of a box
binding. Its section profile is sampled independently of editable node density,
so changing density does not change the reference coordinate system.
"""
from __future__ import annotations

import numpy as np


def settings(value=None):
    """Canonical JSON settings. Absence intentionally replays the original box."""
    if value is None:
        return dict(type='box', dimensions=[7, 3, 4], padding_mm=0.0)
    if not isinstance(value, dict):
        raise ValueError('cage must be an object with type, dimensions and padding_mm.')
    unknown = set(value) - {'type', 'dimensions', 'padding_mm'}
    if unknown:
        raise ValueError('Unknown cage setting: ' + ', '.join(sorted(unknown)))
    kind = value.get('type', 'fitted')
    if kind not in ('fitted', 'box'):
        raise ValueError('cage.type must be fitted or box.')
    raw = value.get('dimensions', [9, 3, 4] if kind == 'fitted' else [7, 3, 4])
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError('cage.dimensions requires [longitudinal, width, height] counts.')
    try:
        dims = [int(n) for n in raw]
        if any(isinstance(n, (bool, str)) or n != actual for n, actual in zip(raw, dims)):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Cage node counts must be integers.') from None
    if not (5 <= dims[0] <= 17 and dims[1] in (3, 5, 7) and 3 <= dims[2] <= 7):
        raise ValueError('Cage density requires X=5–17, Y=3/5/7 and Z=3–7 (at most 833 nodes).')
    raw_pad = value.get('padding_mm', 30 if kind == 'fitted' else 0)
    try:
        pad = float(raw_pad)
    except (TypeError, ValueError):
        raise ValueError('cage.padding_mm must be a finite number from 0 to 150.') from None
    if isinstance(raw_pad, (str, bool)) or not np.isfinite(pad) or not 0 <= pad <= 150:
        raise ValueError('cage.padding_mm must be a finite number from 0 to 150.')
    return dict(type=kind, dimensions=dims, padding_mm=pad)


def _smooth(values, passes=1):
    out = np.asarray(values).copy()
    for _ in range(passes):
        padded = np.pad(out, (1, 1), mode='edge')
        out = .2 * padded[:-2] + .6 * padded[1:-1] + .2 * padded[2:]
    return out


def _sample_surface(v, f, labels, parts, wheels):
    # Explicit appendage labels are excluded. Broad body groups may contain
    # mirrors, so robust section statistics additionally suppress protrusions.
    excluded = []
    for part in parts:
        name = (str(part.get('key', '')) + ' ' + str(part.get('name', ''))).lower()
        if any(token in name for token in ('mirror', '后视镜', 'antenna', '天线', 'exhaust', '排气', 'interior', '内饰', 'steering')):
            excluded.append(part['id'])
    keep = np.ones(len(v), dtype=bool)
    for wheel in wheels:
        keep[wheel['indices']] = False
    if excluded:
        keep[np.unique(f[np.isin(labels, excluded)])] = False
    source = v[keep]
    if len(source) < 8:
        source = v
        keep[:] = True
    # Sparse imported polygon shells need surface samples, not only corner
    # vertices, to infer the cabin/hood transition between their end faces.
    if len(source) < 600:
        triangles = v[f[np.all(keep[f], axis=1)]]
        extra = []
        for a in np.linspace(0, 1, 9):
            for b in np.linspace(0, 1 - a, 9):
                extra.append(triangles[:, 0] * a + triangles[:, 1] * b + triangles[:, 2] * (1 - a - b))
        if extra:
            source = np.vstack([source] + extra)
    return source, int((~keep).sum())


def build(v, f, labels, parts, wheels, config):
    dims = tuple(config['dimensions'])
    padding = config['padding_mm'] / 1000
    lo, hi = v.min(0), v.max(0)
    if config['type'] == 'box':
        lower, upper = lo - padding, hi + padding
        grid = np.stack(np.meshgrid(*[np.linspace(lower[a], upper[a], n) for a, n in enumerate(dims)], indexing='ij'), axis=-1).reshape(-1, 3)
        return dict(grid=grid, lower=lower, upper=upper, settings=config, dimensions=dims,
                    method='local_cubic_displacement', profile=None,
                    diagnostics=dict(profile_source='axis_aligned_bounds', excluded_vertices=0))

    source, excluded = _sample_surface(v, f, labels, parts, wheels)
    # A fixed profile resolution avoids re-fitting the body on each density
    # adjustment. Horizontal sections use absolute Y, enforcing symmetry.
    xs = np.linspace(lo[0], hi[0], 65)
    z_levels = np.linspace(0, 1, 17)
    length = hi[0] - lo[0]
    height = hi[2] - lo[2]
    global_width = max(float(np.quantile(np.abs(source[:, 1]), .975)), .1)
    sections, bottoms, tops = [], [], []
    for x in xs:
        q = source[np.abs(source[:, 0] - x) <= length * .021]
        if len(q) < 16:
            q = source[np.argsort(np.abs(source[:, 0] - x))[:min(len(source), max(32, len(source) // 160))]]
        sections.append(q)
        bottoms.append(float(np.quantile(q[:, 2], .015)))
        # Trim/tire density and side mirror tips do not determine roof height.
        center = q[np.abs(q[:, 1]) <= .82 * global_width]
        if len(center) < 12:
            center = q
        tops.append(float(np.quantile(center[:, 2], .985)))
    bottom = _smooth(bottoms, 2) - padding
    top = np.maximum(_smooth(tops, 1) + padding, bottom + .12 * height)
    widths = np.empty((len(xs), len(z_levels)))
    for i, q in enumerate(sections):
        zmin, zmax = bottom[i] + padding, top[i] - padding
        span = max(zmax - zmin, .12 * height)
        local_width = min(float(np.quantile(np.abs(q[:, 1]), .975)), global_width * 1.04)
        row = np.full(len(z_levels), np.nan)
        for j, t in enumerate(z_levels):
            around = q[np.abs(q[:, 2] - (zmin + span * t)) <= max(.10 * span, .025)]
            if len(around) >= 3:
                row[j] = min(float(np.quantile(np.abs(around[:, 1]), .98)), local_width)
        valid = np.isfinite(row)
        if valid.any():
            row = np.interp(z_levels, z_levels[valid], row[valid])
        else:
            row[:] = local_width
        # Prevent collapsed parameter coordinates at pointed hood/roof ends.
        widths[i] = np.maximum(_smooth(row, 1), max(.12 * local_width, .055)) + padding
    for j in range(len(z_levels)):
        widths[:, j] = _smooth(widths[:, j], 2)
    # Extend first/last section to the padded X endpoints, preserving the
    # measured station spacing through the vehicle itself.
    if padding:
        xs = np.r_[lo[0] - padding, xs, hi[0] + padding]
        bottom = np.r_[bottom[0], bottom, bottom[-1]]
        top = np.r_[top[0], top, top[-1]]
        widths = np.vstack((widths[0], widths, widths[-1]))
    profile = dict(x=xs, bottom=bottom, top=top, half_width=widths, z_levels=z_levels)
    shape = dict(lower=np.array([xs[0], lo[1] - padding, lo[2] - padding]),
                 upper=np.array([xs[-1], hi[1] + padding, hi[2] + padding]),
                 settings=config, dimensions=dims, method='curvilinear_section_cubic_displacement', profile=profile)
    uvw = np.stack(np.meshgrid(*[np.linspace(0, 1, n) for n in dims], indexing='ij'), axis=-1).reshape(-1, 3)
    shape['grid'] = positions(uvw, shape)
    coords = coordinates(source, shape)
    outside = np.any((coords < -1e-6) | (coords > 1 + 1e-6), axis=1)
    shape['diagnostics'] = dict(profile_source='robust_symmetric_body_sections', section_samples=len(xs),
                                excluded_vertices=excluded, source_coverage_percent=round(float((~outside).mean() * 100), 2),
                                outside_binding='clamp_to_nearest_section_boundary',
                                approximate=True)
    return shape


def _width(x, t, profile):
    xs, zs = profile['x'], profile['z_levels']
    i = np.clip(np.searchsorted(xs, x, side='right') - 1, 0, len(xs) - 2)
    sx = np.clip((x - xs[i]) / (xs[i + 1] - xs[i]), 0, 1)
    z = np.clip(t, 0, 1) * (len(zs) - 1)
    j = np.minimum(np.floor(z).astype(int), len(zs) - 2)
    sz = z - j
    w = profile['half_width']
    return ((1 - sx) * ((1 - sz) * w[i, j] + sz * w[i, j + 1]) +
            sx * ((1 - sz) * w[i + 1, j] + sz * w[i + 1, j + 1]))


def positions(uvw, shape):
    """Map section coordinates to their physical rest positions."""
    if shape['profile'] is None:
        return shape['lower'] + uvw * (shape['upper'] - shape['lower'])
    p = shape['profile']
    x = p['x'][0] + uvw[:, 0] * (p['x'][-1] - p['x'][0])
    bottom, top = np.interp(x, p['x'], p['bottom']), np.interp(x, p['x'], p['top'])
    z = bottom + uvw[:, 2] * (top - bottom)
    y = (uvw[:, 1] * 2 - 1) * _width(x, uvw[:, 2], p)
    return np.column_stack((x, y, z))


def coordinates(points, shape):
    """Inverse rest-envelope map; extrapolation is clamped by cubic bindings."""
    if shape['profile'] is None:
        return (points - shape['lower']) / (shape['upper'] - shape['lower'])
    p = shape['profile']
    x, y, z = points.T
    bottom, top = np.interp(x, p['x'], p['bottom']), np.interp(x, p['x'], p['top'])
    tz = (z - bottom) / (top - bottom)
    ty = .5 + y / (2 * _width(x, tz, p))
    tx = (x - p['x'][0]) / (p['x'][-1] - p['x'][0])
    return np.column_stack((tx, ty, tz))
