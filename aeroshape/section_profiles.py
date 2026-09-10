"""Triangle-plane sections for density-independent vehicle silhouette fitting."""
from __future__ import annotations
import numpy as np


def triangle_plane_segments(vertices, faces, y):
    """Intersect in bounded chunks; each result is one exact surface segment."""
    segments = []
    for start in range(0, len(faces), 200000):
        t = vertices[faces[start:start+200000]]
        t = t[(t[:, :, 1].min(1) < y) & (t[:, :, 1].max(1) > y)]
        if not len(t):
            continue
        points = np.full((len(t), 3, 3), np.nan)
        for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            va, vb = t[:, a], t[:, b]
            valid = ((va[:, 1] <= y) & (vb[:, 1] > y)) | ((vb[:, 1] <= y) & (va[:, 1] > y))
            ratio = (y-va[valid, 1])/(vb[valid, 1]-va[valid, 1])
            points[valid, k] = va[valid]+ratio[:, None]*(vb[valid]-va[valid])
        valid = np.isfinite(points[:, :, 0])
        complete = valid.sum(1) == 2
        if complete.any():
            segments.append(points[complete][valid[complete]].reshape(-1, 2, 3))
    return np.concatenate(segments) if segments else np.empty((0, 2, 3))


def segment_envelope(segments, stations, axis=0, other_axis=2):
    upper = np.full(len(stations), np.nan)
    lower = np.full(len(stations), np.nan)
    delta = segments[:, 1, axis]-segments[:, 0, axis]
    keep = np.abs(delta) > 1e-10
    seg, delta = segments[keep], delta[keep]
    lo, hi = seg[:, :, axis].min(1), seg[:, :, axis].max(1)
    for i, station in enumerate(stations):
        valid = (lo <= station) & (hi >= station)
        if valid.any():
            hit = seg[valid]
            value = hit[:, 0, other_axis]+(station-hit[:, 0, axis])/delta[valid]*(hit[:, 1, other_axis]-hit[:, 0, other_axis])
            upper[i], lower[i] = value.max(), value.min()
    return upper, lower


def section_envelope(vertices, faces, y, stations):
    """Upper/lower z on actual segments, using off-centre cuts to avoid seams."""
    return segment_envelope(triangle_plane_segments(vertices, faces, y), stations)


def vehicle_section_profiles(vertices, faces, x_bounds, width, bins=48, rear_heights=None):
    """Three exact sagittal cuts measured both along x and along rear height."""
    x = x_bounds[0]+(np.arange(bins)+.5)/bins*(x_bounds[1]-x_bounds[0])
    top_cuts, rear_cuts = [], []
    def filled(values, minimum):
        valid = np.isfinite(values)
        if valid.sum() < minimum:
            raise ValueError('参考车纵剖面覆盖不足，请检查是否只导入了零件或坐标方向。')
        return np.interp(np.arange(len(values)), np.flatnonzero(valid), values[valid])
    for y in (.005, -.08*width, .08*width):
        segments = triangle_plane_segments(vertices, faces, y)
        upper, _ = segment_envelope(segments, x)
        top_cuts.append(filled(upper, bins*.8))
        if rear_heights is not None:
            rear, _ = segment_envelope(segments, rear_heights, axis=2, other_axis=0)
            rear_cuts.append(filled(rear, len(rear_heights)*.7)-x_bounds[1])
    return dict(stations=x, upper=np.median(top_cuts, axis=0),
                rear=np.median(rear_cuts, axis=0) if rear_cuts else None)


def vehicle_upper_profile(vertices, faces, x_bounds, width, bins=48):
    profile = vehicle_section_profiles(vertices, faces, x_bounds, width, bins)
    return profile['stations'], profile['upper']
