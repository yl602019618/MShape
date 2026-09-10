"""Reproducible surface-fit measurements and a local triangle deformation screen.

Distances use area-uniform surface samples and nearest *triangles*, not nearest
mesh vertices. The maximum is a sampled statistic, never a Hausdorff bound or a
manufacturing/CFD certification. All model coordinates are expected in metres.
"""
from __future__ import annotations

import numpy as np

from .model import Model, array_hash
from .spatial import TriangleIndex


_MAX_SAMPLES = 100_000
_MAX_PARTS = 64
_FACE_CHUNK = 100_000


def _sample_arguments(count, seed):
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise ValueError('Surface sample count must be an integer.')
    if not 1 <= int(count) <= _MAX_SAMPLES:
        raise ValueError(f'Surface sample count must be 1..{_MAX_SAMPLES}.')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Surface sampling seed must be a nonnegative integer.')
    return int(count), int(seed)


def _areas(vertices, faces):
    """Avoid materialising all 1.5M source triangles just to compute areas."""
    result = np.empty(len(faces), dtype=np.float64)
    for start in range(0, len(faces), _FACE_CHUNK):
        tri = vertices[faces[start:start + _FACE_CHUNK]]
        result[start:start + len(tri)] = .5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return result


def _surface_points(vertices, faces, count, seed):
    area = _areas(vertices, faces)
    total = float(area.sum())
    if not np.isfinite(area).all() or not np.isfinite(total) or total <= 0:
        raise ValueError('Surface sampling requires finite, positive triangle area.')
    rng = np.random.default_rng(seed)
    # Face probability is proportional to area; tessellation density has no
    # statistical weight. Degenerate faces have zero probability.
    cumulative = np.cumsum(area)
    face_ids = np.searchsorted(cumulative, rng.random(count) * cumulative[-1], side='right')
    face_ids = np.minimum(face_ids, len(faces) - 1)
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.c_[1. - root, root * (1. - uv[:, 1]), root * uv[:, 1]]
    return np.einsum('ni,nij->nj', bary, vertices[faces[face_ids]])


def sample_surface(model: Model, count: int, seed: int = 42) -> np.ndarray:
    """Return ``count`` deterministic, area-uniform triangle samples in metres.

    A fixed seed reproduces points for the same mesh. Retessellation need not
    reproduce individual points, but does not bias their area distribution.
    """
    count, seed = _sample_arguments(count, seed)
    return _surface_points(model.vertices, model.faces, count, seed)


def _distances(points, vertices, faces):
    # The local BVH is released before building the opposite direction's BVH.
    # Each model is indexed once for the overall bidirectional measurement.
    index = TriangleIndex(vertices, faces)
    nearest, _, _ = index.closest(points)
    return np.linalg.norm(points - nearest, axis=1)


def _statistics(distances):
    mm = np.asarray(distances, dtype=np.float64) * 1000.
    return dict(mean_mm=float(mm.mean()), rms_mm=float(np.sqrt(np.mean(mm * mm))),
                p95_mm=float(np.percentile(mm, 95)), max_sampled_mm=float(mm.max()),
                sample_count=len(mm))


def _comparison(a_vertices, a_faces, b_vertices, b_faces, count, seed):
    ap = _surface_points(a_vertices, a_faces, count, seed)
    bp = _surface_points(b_vertices, b_faces, count, seed)
    ab = _distances(ap, b_vertices, b_faces)
    ba = _distances(bp, a_vertices, a_faces)
    result = _statistics(np.concatenate((ab, ba)))
    result.update(source_to_target=_statistics(ab), target_to_source=_statistics(ba),
                  samples_per_direction=count)
    return result


def _semantic_part_ids(model):
    by_key = {}
    for part in model.parts:
        key = part.get('key')
        if not isinstance(key, str) or not key or key in ('unknown', 'unassigned') or key.startswith('island_'):
            continue
        by_key.setdefault(key, []).append(int(part['id']))
    present = set(np.unique(model.labels).tolist())
    return {key: ids for key, ids in by_key.items() if present.intersection(ids)}


def surface_metrics(a: Model, b: Model, samples: int = 4000, seed: int = 42,
                    include_parts: bool = False) -> dict:
    """Measure area-sampled unsigned distance in both directions, in mm.

    ``samples`` is the count *per direction*. Overall statistics give the two
    directions equal weight. Optional component distances compare only shared,
    present semantic keys, using at most 300 samples per direction per part.
    Matching keys do not by themselves verify a reference's segmentation.
    """
    samples, seed = _sample_arguments(samples, seed)
    result = _comparison(a.vertices, a.faces, b.vertices, b.faces, samples, seed)
    result.update(
        schema='aeroshape.surface-fit-metrics.v1',
        method='bidirectional_area_uniform_samples_to_nearest_triangle_surface',
        sampling='triangle_area_probability_with_uniform_barycentric_points',
        aggregation='equal_weight_per_direction', seed=seed, units='mm',
        maximum_scope='sampled_maximum_not_Hausdorff_distance_or_upper_bound',
        source_geometry_hash=array_hash(a.vertices, a.faces),
        target_geometry_hash=array_hash(b.vertices, b.faces),
        source_faces=len(a.faces), target_faces=len(b.faces))
    if include_parts:
        source = _semantic_part_ids(a)
        target = _semantic_part_ids(b)
        shared = sorted(source.keys() & target.keys())
        count = min(samples, 300)
        breakdown = []
        for i, key in enumerate(shared[:_MAX_PARTS]):
            af = a.faces[np.isin(a.labels, source[key])]
            bf = b.faces[np.isin(b.labels, target[key])]
            part = dict(key=key, source_faces=len(af), target_faces=len(bf))
            try:
                part.update(_comparison(a.vertices, af, b.vertices, bf, count, seed + i + 1))
                part['status'] = 'measured'
            except ValueError as exc:
                # A source label can cover degenerate faces only. Keep an
                # explicit missing measurement rather than emit NaN or zero.
                part.update(status='not_measurable', error=str(exc))
            breakdown.append(part)
        result.update(parts=breakdown, shared_semantic_parts=len(shared),
                      parts_omitted_by_limit=max(0, len(shared) - _MAX_PARTS),
                      part_comparison='shared_semantic_keys_not_verified_segmentation',
                      part_samples_per_direction=count)
    return result


def deformation_screen(base: Model, vertices: np.ndarray) -> dict:
    """Screen local triangle damage relative to a reference; no collision test.

    Reference slivers below a scale/area threshold do not dominate the minimum
    area ratio. Newly collapsed triangles are counted separately on every
    reference face above the absolute degeneracy threshold. A normal reversal
    means a >=90-degree rotation relative to the original triangle, a
    conservative local diagnostic rather than a test for self-intersection.
    """
    candidate = np.asarray(vertices, dtype=np.float64)
    result = dict(scope='local_triangle_screen_only', cfd_ready=False,
                  self_intersections='not_checked', component_clearance='not_checked')
    if candidate.shape != base.vertices.shape or not np.isfinite(candidate).all():
        result.update(screen_pass=False, finite=bool(np.isfinite(candidate).all()),
                      shape_matches=bool(candidate.shape == base.vertices.shape),
                      min_area_ratio=None, new_collapsed_faces=None,
                      normal_flipped_faces=None, normal_reversal_faces=None,
                      errors=['Candidate vertex shape differs from the reference or contains non-finite coordinates.'])
        return result
    area0 = _areas(base.vertices, base.faces)
    area1 = _areas(candidate, base.faces)
    if not np.isfinite(area0).all() or not np.isfinite(area1).all():
        result.update(screen_pass=False, finite=False, shape_matches=True,
                      min_area_ratio=None, new_collapsed_faces=None,
                      normal_flipped_faces=None, normal_reversal_faces=None,
                      errors=['Triangle calculations produced non-finite values.'])
        return result
    scale = float(np.ptp(base.vertices, axis=0).max())
    area_floor = max(scale * scale * 1e-14, np.finfo(float).tiny)
    valid = area0 > area_floor
    significant_floor = max(area_floor * 100., float(np.median(area0[valid])) * 1e-6) if valid.any() else area_floor
    significant = area0 > significant_floor
    collapsed = area1 <= area_floor
    new_collapsed = int(np.count_nonzero(valid & collapsed))
    flipped = 0
    for start in range(0, len(base.faces), _FACE_CHUNK):
        end = min(start + _FACE_CHUNK, len(base.faces))
        f = base.faces[start:end]
        old, new = base.vertices[f], candidate[f]
        n0 = np.cross(old[:, 1] - old[:, 0], old[:, 2] - old[:, 0])
        n1 = np.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0])
        reversed_normals = np.einsum('ij,ij->i', n0, n1) <= 0
        flipped += int(np.count_nonzero(reversed_normals & significant[start:end] & ~collapsed[start:end]))
    minimum = float(np.min(area1[significant] / area0[significant])) if significant.any() else None
    errors = []
    if minimum is None:
        errors.append('Reference has no significant nondegenerate triangles.')
    elif minimum < .025:
        errors.append('A significant triangle has less than 2.5% of its reference area.')
    if new_collapsed:
        errors.append(f'{new_collapsed} previously nondegenerate triangles collapsed.')
    if flipped:
        errors.append(f'{flipped} significant triangles rotated through at least 90 degrees.')
    result.update(screen_pass=not errors, finite=True, shape_matches=True,
                  min_area_ratio=minimum, new_collapsed_faces=new_collapsed,
                  normal_flipped_faces=flipped, normal_reversal_faces=flipped,
                  degenerate_faces=int(collapsed.sum()), reference_degenerate_faces=int((~valid).sum()),
                  significant_faces=int(significant.sum()), area_floor_m2=area_floor,
                  significant_area_floor_m2=significant_floor,
                  max_displacement_m=float(np.linalg.norm(candidate - base.vertices, axis=1).max()),
                  minimum_allowed_area_ratio=.025, errors=errors)
    return result
