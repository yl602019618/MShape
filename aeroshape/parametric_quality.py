"""Fast, fixed-reference surface guard for the authored 30-part generator.

The fixed presets for each body and generator revision define the accepted
smoothness reference over the original 30-component topology. This is an optimization constraint, not a self-intersection
test or manufacturing certification. Reference STL triangles are never involved.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .engineering import BODY_TYPES, generate, preset, reference
from .geometry import face_geometry
from .mesh_tools import vertex_manifold_check
from .model import Model


LIMITS = {
    'symmetry_tolerance_m': 1e-8,
    'min_triangle_area_ratio': .25,
    'principal_stretch_min': .30,
    'principal_stretch_max': 2.50,
    'originally_smooth_dihedral_deg': 15.,
    'max_new_crease_deg': 20.,
    'p999_new_crease_deg': 10.,
}


def _part_identity(parts):
    # Rendering names/colors and review state are not component memberships.
    return [{k: p.get(k) for k in ('id', 'key', 'source_members', 'pair_id', 'category')}
            for p in parts]


@lru_cache(maxsize=1)
def _topology():
    base = reference()
    driver = np.asarray(base.metadata['driver_vertices'], dtype=np.int32)
    mask = np.zeros(len(base.vertices), dtype=bool)
    mask[driver] = True
    face_mask = mask[base.faces].all(axis=1)
    f = base.faces[face_mask]
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    owners = np.tile(np.arange(len(f)), 3)
    keys = edges.min(axis=1).astype(np.int64)*len(base.vertices)+edges.max(axis=1)
    order = np.argsort(keys)
    _, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    paired = starts[counts == 2]
    adj = np.c_[owners[order[paired]], owners[order[paired+1]]]
    winding_ok = bool(np.all(edges[order[paired]] == edges[order[paired+1], ::-1]))
    compact = np.full(len(base.vertices), -1, dtype=np.int32)
    compact[driver] = np.arange(len(driver))
    driver_model = Model(base.vertices[driver], compact[f], base.labels[face_mask], base.parts)
    manifold = vertex_manifold_check(driver_model)
    ce = compact[edges]
    graph = coo_matrix((np.ones(len(ce), dtype=np.uint8), (ce[:, 0], ce[:, 1])),
                       shape=(len(driver), len(driver))).tocsr()
    component_count = int(connected_components(graph, directed=False, return_labels=False))
    d, reflection = cKDTree(base.vertices).query(base.vertices*[1, -1, 1])
    if float(d.max()) > LIMITS['symmetry_tolerance_m']:
        raise ValueError('Authored scaffold reflection correspondence is invalid.')
    report = dict(driver_faces=len(f), driver_vertices=len(driver),
                  boundary_edges=int((counts == 1).sum()),
                  nonmanifold_edges=int((counts > 2).sum()),
                  nonmanifold_vertices=manifold['nonmanifold_vertices'],
                  driver_connected_components=component_count, winding_consistent=winding_ok)
    report['closed_manifold_driver'] = bool(np.all(counts == 2) and winding_ok
                                            and manifold['vertex_manifold'] and component_count == 1)
    return dict(model=base, driver_faces=f, face_mask=face_mask, adjacency=adj,
                reflection=reflection, report=report, parts=_part_identity(base.parts))


@lru_cache(maxsize=3)
def _configured(spoiler):
    from .optional_components import configure_optional
    topo=_topology();base=topo['model']
    mask=topo['face_mask']
    if spoiler=='none':
        pid=next(p['id'] for p in base.parts if p['key']=='spoiler')
        mask=mask[base.labels!=pid]
        base=configure_optional(base,'none')
    return base,mask


@lru_cache(maxsize=48)
def _prepared(body_type, revision, spoiler='lip'):
    base = generate(preset(body_type, revision=revision).model_copy(update={'spoiler_style':spoiler}))
    topo = _topology()
    f = topo['driver_faces']
    _, normals, areas = face_geometry(base.vertices, base.faces)
    driver_normals = normals[_configured(spoiler)[1]]
    adj = topo['adjacency']
    angles = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i',
        driver_normals[adj[:, 0]], driver_normals[adj[:, 1]]), -1., 1.)))
    smooth = angles < LIMITS['originally_smooth_dihedral_deg']
    triangle = base.vertices[f]
    e0 = triangle[:, 1]-triangle[:, 0]
    e1 = triangle[:, 2]-triangle[:, 0]
    length = np.linalg.norm(e0, axis=1)
    along = np.einsum('ij,ij->i', e0, e1)/length
    height = np.linalg.norm(np.cross(e0, e1), axis=1)/length
    return dict(model=base, normals=driver_normals, areas=areas,
                smooth_adjacency=adj[smooth], angles=angles[smooth],
                length=length, along=along, height=height)


def parametric_quality(model: Model, baseline: Model | None = None) -> dict:
    """Return a JSON-safe hard screen and smoothness penalty against a fixed, revision-specific preset.

    ``baseline`` is optional compatibility input and must be the unchanged preset
    for this body type. A fitted or deformed baseline cannot relax the guard.
    All face/vertex identities and every smooth driver adjacency are checked;
    no random vertex sampling can hide an isolated spike. Caches hold only the
    authored topology and five preset references, never candidate results.
    """
    errors = []
    revision = model.metadata.get('generator_parameters', {}).get('generator_revision', 'v2')
    result = dict(schema='aeroshape.parametric-quality.v2', screen_pass=False,
                  errors=errors, baseline='fixed_body_and_generator_revision_preset',
                  generator_revision=revision,
                  limits=dict(LIMITS), smoothness_penalty=None,
                  self_intersections='not_checked', component_clearance='not_checked',
                  cfd_ready=False, validation_level='authored_parametric_surface_screen')
    body = model.metadata.get('body_type')
    if revision not in ('v2', 'v3', 'v4', 'v5', 'v6'):
        errors.append('Unsupported generator revision.')
        return result
    if body not in BODY_TYPES:
        errors.append('Missing or unsupported parametric body type.')
        return result
    topo = _topology()
    spoiler=model.metadata.get('generator_parameters',{}).get('spoiler_style','lip') if revision in ('v5','v6') else 'lip'
    if spoiler not in ('none','lip','sport'):
        errors.append('Unsupported optional component style.');return result
    canon,driver_mask = _configured(spoiler)
    expected=29 if spoiler=='none' else 30
    same_vertices = model.vertices.shape == canon.vertices.shape
    same_faces = np.array_equal(model.faces, canon.faces)
    same_labels = np.array_equal(model.labels, canon.labels)
    same_parts = _part_identity(model.parts) == topo['parts']
    same_bindings = all(model.metadata.get(key) == canon.metadata.get(key)
                        for key in ('driver_vertices', 'rigid_followers', 'seam_ties'))
    present = len(np.unique(model.labels))
    result['topology'] = dict(topo['report'], vertex_identity_preserved=same_vertices,
                            face_identity_preserved=same_faces,
                            component_face_membership_preserved=same_labels,
                            authored_parts_preserved=same_parts,
                            driver_follower_identity_preserved=same_bindings,
                            components_present=present, expected_components=expected, semantic_slots=30)
    if not all((same_vertices, same_faces, same_labels, same_parts, same_bindings, present == expected)):
        errors.append('Authored vertex/face identities, 30 component memberships or driver/follower bindings changed.')
        return result
    if not topo['report']['closed_manifold_driver']:
        errors.append('Authored driver is not one closed, consistently wound manifold.')
        return result
    v = np.asarray(model.vertices)
    if not np.isfinite(v).all():
        errors.append('Non-finite vertex coordinates.')
        return result
    prepared = _prepared(body, revision, spoiler)
    if revision in ('v5','v6'):
        from .windscreen import windshield_report
        result['windshield_profile']=windshield_report(model)
        if not result['windshield_profile']['pass']:
            errors.append('Windshield or A-pillar profile bends inward beyond 1 mm.')
    if baseline is not None and not (np.array_equal(baseline.vertices, prepared['model'].vertices)
            and np.array_equal(baseline.faces, canon.faces)
            and np.array_equal(baseline.labels, canon.labels)):
        errors.append('Quality baseline must be the unchanged same-body and same-revision engineering preset.')
        return result
    reflection_error = np.linalg.norm(v[topo['reflection']]-v*[1, -1, 1], axis=1)
    symmetry = float(reflection_error.max())
    result['symmetry'] = dict(max_m=symmetry, tolerance_m=LIMITS['symmetry_tolerance_m'],
                              method='authored_reflection_vertex_identity')
    if symmetry > LIMITS['symmetry_tolerance_m']:
        errors.append('Exact authored bilateral symmetry was lost.')
    _, normals, areas = face_geometry(v, model.faces)
    area_ratio = areas/prepared['areas']
    minimum_area = float(area_ratio.min())
    result['min_area_ratio'] = minimum_area
    if minimum_area < LIMITS['min_triangle_area_ratio']:
        errors.append('Triangle area collapsed below 25% of the same-body preset.')
    dn = normals[driver_mask]
    reversals = int((np.einsum('ij,ij->i', dn, prepared['normals']) <= 0).sum())
    result['driver_normal_reversal_faces'] = reversals
    if reversals:
        errors.append('Driver faces rotated through 90 degrees relative to the preset.')
    # Exact tangent-map singular values, with cached reference triangle basis.
    t = v[topo['driver_faces']]
    a = (t[:, 1]-t[:, 0])/prepared['length'][:, None]
    b = ((t[:, 2]-t[:, 0])-a*prepared['along'][:, None])/prepared['height'][:, None]
    g00 = np.einsum('ij,ij->i', a, a)
    g11 = np.einsum('ij,ij->i', b, b)
    g01 = np.einsum('ij,ij->i', a, b)
    disc = np.sqrt(np.maximum((g00-g11)**2+4*g01*g01, 0.))
    smin = np.sqrt(np.maximum((g00+g11-disc)*.5, 0.))
    smax = np.sqrt(np.maximum((g00+g11+disc)*.5, 0.))
    low, high = float(smin.min()), float(smax.max())
    result['surface_strain'] = dict(principal_min=low, principal_max=high,
                                    triangles_checked=len(t), scope='all_driver_triangles')
    if low < LIMITS['principal_stretch_min'] or high > LIMITS['principal_stretch_max']:
        errors.append('Driver principal stretch outside [0.30, 2.50] of the same-body preset.')
    adj = prepared['smooth_adjacency']
    angle = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i',
        dn[adj[:, 0]], dn[adj[:, 1]]), -1., 1.)))
    increase = np.maximum(angle-prepared['angles'], 0.)
    peak = float(increase.max())
    p999 = float(np.percentile(increase, 99.9))
    rms = float(np.sqrt(np.mean(increase**2)))
    result['smooth_driver_crease'] = dict(adjacencies_checked=len(adj),
        reference_dihedral_below_deg=LIMITS['originally_smooth_dihedral_deg'],
        max_increase_deg=peak, p999_increase_deg=p999, rms_increase_deg=rms,
        new_crease_edges=int((increase > LIMITS['max_new_crease_deg']).sum()),
        method='all_originally_smooth_driver_edge_dihedrals')
    if peak > LIMITS['max_new_crease_deg'] or p999 > LIMITS['p999_new_crease_deg']:
        errors.append('New sharp creases or spikes appeared on originally smooth driver surfaces.')
    # A dimensionless continuous term remains useful before a hard bound is hit.
    result['smoothness_penalty'] = float((rms/5.)**2+(peak/20.)**2+(p999/10.)**2)
    result['screen_pass'] = not errors
    return result


def cheap_screen(model: Model, baseline: Model | None = None) -> dict:
    """Optimizer entry point; same complete guard, cached topology/reference data."""
    return parametric_quality(model, baseline)
