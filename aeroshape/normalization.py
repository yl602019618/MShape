"""Coordinate and half-surface normalization with explicit face provenance."""
from __future__ import annotations
import numpy as np
from .model import Model, array_hash

MAX_FACES = 2_000_000
FACE_ARRAYS = ('raw_face_labels', 'source_face_indices', 'semantic_source_labels',
               'segmentation_face_scores', 'segmentation_face_methods', 'face_material_ids')


def resolve_units(vertices: np.ndarray, units: str) -> tuple[str, float]:
    scales = {'m': 1., 'cm': .01, 'mm': .001}
    if units == 'auto':
        # These mutually exclusive intervals describe a road-vehicle length.
        # Outside them do not silently shrink arbitrary geometry.
        length = float(np.ptp(vertices, axis=0).max())
        candidates = [name for name, scale in scales.items() if 2.5 <= length*scale <= 8.]
        units = candidates[0] if len(candidates) == 1 else 'm'
    if units not in scales:
        raise ValueError('Units must be auto, m, cm, or mm.')
    return units, scales[units]


def half_surface_report(vertices: np.ndarray) -> dict:
    lo = vertices.min(0); hi = vertices.max(0); span = hi-lo
    tolerance = max(float(span.max())*2e-6, 1e-8)
    negative = abs(hi[1]) <= tolerance and lo[1] < -10*tolerance
    positive = abs(lo[1]) <= tolerance and hi[1] > 10*tolerance
    plane_vertices = int((np.abs(vertices[:, 1]) <= tolerance).sum())
    car_proportions = (2.5 <= span[0] <= 8. and .12 <= span[1]/span[0] <= .30
                       and .12 <= span[2]/span[0] <= .55)
    detected = bool((negative or positive) and plane_vertices >= 3 and car_proportions)
    return dict(detected_half=detected, one_sided=bool(negative or positive),
                side='negative_y' if negative else ('positive_y' if positive else 'both_or_offset'),
                source_bounds_m=[lo.tolist(), hi.tolist()], plane_y_m=0.,
                plane_vertices=plane_vertices, tolerance_m=tolerance,
                detection='one_sided_Y0_boundary_and_passenger_vehicle_proportions')


def normalize_reference(model: Model, symmetry='auto', *, center=True, weld=True) -> Model:
    """Normalize a metre/Z-up reference, mirroring only a verified Y=0 half.

    Auto mode requires a one-sided Y=0 boundary and plausible half-car proportions.
    Explicit mirror still rejects full/offset geometry. Original face IDs survive
    winding reversal and cap removal in ``source_face_indices``.
    """
    if symmetry not in ('auto', 'keep', 'mirror'):
        raise ValueError('Symmetry must be auto, keep, or mirror.')
    v = model.vertices.copy(); f = model.faces.copy(); labels = model.labels.copy()
    meta = dict(model.metadata)
    report = half_surface_report(v)
    # Older importers centered a half-width bounding box, moving its Y=0 edge.
    previous = np.asarray(meta.get('source_translation_m', [0., 0., 0.]), float)
    recovered = False
    if symmetry != 'keep' and not report['one_sided'] and previous.shape == (3,) and previous[1] != 0:
        candidate = v.copy(); candidate[:, 1] -= previous[1]
        probe = half_surface_report(candidate)
        if probe['detected_half']:
            v = candidate; report = probe; recovered = True
    mirror = symmetry == 'mirror' or (symmetry == 'auto' and report['detected_half'])
    if mirror and not report['one_sided']:
        raise ValueError('Mirror requires a one-sided surface meeting the original Y=0 symmetry plane.')
    if mirror:
        tol = report['tolerance_m']
        on_plane = np.abs(v[:, 1]) <= tol
        v[on_plane, 1] = 0.
        # A half-model closure on the symmetry plane becomes an internal wall.
        keep = ~on_plane[f].all(1)
        count = int(keep.sum())
        if 2*count > MAX_FACES:
            raise ValueError('Mirrored reference exceeds the 2M-face limit.')
        origin = np.asarray(meta.get('source_face_indices', np.arange(len(f))), dtype=np.int32)
        if origin.shape != (len(f),):
            raise ValueError('Source face correspondence is stale.')
        original_f = f[keep]
        reflected = v.copy(); reflected[:, 1] *= -1
        f = np.vstack((original_f, original_f[:, [0, 2, 1]]+len(v)))
        v = np.vstack((v, reflected))
        part_pairs = {p['id']: p.get('pair_id', p['id']) for p in model.parts}
        base_labels = labels[keep]
        mirrored_labels = np.empty_like(base_labels)
        for old, new in part_pairs.items():
            mirrored_labels[base_labels == old] = new
        labels = np.r_[base_labels, mirrored_labels]
        for key in FACE_ARRAYS:
            if key in meta:
                a = np.asarray(meta[key])
                if a.shape != (len(keep),):
                    raise ValueError(f'{key} must remain face aligned.')
                if key == 'semantic_source_labels':
                    source_parts = meta.get('semantic_source_parts', model.parts)
                    source_keys = {p.get('key', ''): p['id'] for p in source_parts}
                    source_pairs = {}
                    for part in source_parts:
                        name = part.get('key', '')
                        other = name[:-1]+('l' if name.endswith('r') else 'r')
                        source_pairs[part['id']] = part.get('pair_id', source_keys.get(other, part['id']))
                    reflected_source = a[keep].copy()
                    for old, new in source_pairs.items():
                        reflected_source[a[keep] == old] = new
                    meta[key] = np.r_[a[keep], reflected_source].tolist()
                else:
                    meta[key] = np.tile(a[keep], 2).tolist()
        meta['source_face_indices'] = np.tile(origin[keep], 2).tolist()
        report.update(mirrored=True, removed_symmetry_cap_faces=int((~keep).sum()),
                      original_faces=len(keep), result_faces=len(f), winding_reversed=True)
    else:
        report.update(mirrored=False, original_faces=len(f), result_faces=len(f))
    # Exact duplicate welding: no physical gap repair or surface smoothing.
    before = len(v)
    if weld or mirror:
        tol = max(float(np.ptp(v, axis=0).max())*1e-9, 1e-12)
        _, ids, remap = np.unique(np.round(v/tol).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
        if len(ids) != len(v):
            # Preserve existing vertex IDs whenever there are no duplicates.
            # If welding is necessary, retain the first-occurrence order.
            stable = np.argsort(ids)
            inverse = np.empty(len(ids), dtype=np.int64); inverse[stable] = np.arange(len(ids))
            v = v[ids[stable]]; f = inverse[remap[f]]
    good = (f[:, 0] != f[:, 1]) & (f[:, 0] != f[:, 2]) & (f[:, 1] != f[:, 2])
    if not good.all():
        for key in FACE_ARRAYS:
            if key in meta:
                meta[key] = np.asarray(meta[key])[good].tolist()
        f = f[good]; labels = labels[good]
    if not len(f):
        raise ValueError('No nondegenerate surface faces remain.')
    used, inv = np.unique(f, return_inverse=True)
    if len(used) != len(v):
        v = v[used]; f = inv.reshape(-1, 3)
    reindexed = before != len(v)
    translation = np.zeros(3)
    if center:
        translation[:2] = -(v.max(0)[:2]+v.min(0)[:2])/2
        translation[2] = -v[:, 2].min()
        v += translation
    report.update(mode=symmetry, recovered_original_plane=recovered, result_faces=len(f))
    previous_report = meta.get('normalization', {})
    if previous_report.get('mirrored') and not mirror:
        # Reopening/fitting an already normalized model must not erase the
        # import provenance that explains its reconstructed second half.
        report = {**previous_report, 'last_pass_reused_full_reference': True}
    meta['normalization'] = report
    accumulated = previous.copy() if previous.shape == (3,) else np.zeros(3)
    if recovered:accumulated[1] = 0.
    meta['source_translation_m'] = (accumulated+translation).tolist()
    meta['welded_vertex_count'] = int(meta.get('welded_vertex_count', 0)+before-len(v))
    meta['dropped_index_degenerate_faces'] = int((~good).sum())
    meta['base_geometry_hash'] = array_hash(v, f)
    if mirror or reindexed:
        # Old attachment vertex IDs cannot survive a topology-changing import.
        for key in ('driver_vertices', 'rigid_followers', 'seam_ties', 'segmentation', 'semantic_schema'):
            meta.pop(key, None)
        meta['attachment_binding_status'] = 'requires_binding_after_topology_normalization'
    return Model(v, f, labels, model.parts, meta)
