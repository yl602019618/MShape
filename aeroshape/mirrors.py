"""Reference-proportioned mirror fixtures on the original authored topology.

Only dimensions and broad silhouette proportions were measured from the user's
labelled samples. No reference triangles, vertex offsets, or OEM surfaces are
embedded. Call this before the rigid parent-anchor transform in revision v4;
legacy generator revisions retain their original fixture exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .model import Model
from .spatial import TriangleIndex


STYLES = {
    'aero': dict(housing_m=[.136, .214, .139], glass_m=[.004, .183, .115]),
    'compact': dict(housing_m=[.127, .198, .128], glass_m=[.004, .169, .106]),
    'touring': dict(housing_m=[.145, .224, .148], glass_m=[.004, .194, .123]),
    'sport': dict(housing_m=[.146, .218, .126], glass_m=[.004, .188, .104]),
}


def _islands(model: Model, key: str):
    part = next(p for p in model.parts if p['key'] == key)
    faces = model.faces[model.labels == part['id']]
    ids, local = np.unique(faces, return_inverse=True)
    f = local.reshape(-1, 3)
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(ids), len(ids))).tocsr()
    n, labels = connected_components(graph, directed=False)
    groups = {int((labels == i).sum()): ids[labels == i] for i in range(n)}
    if n != 3 or set(groups) != {642, 162, 30}:
        raise ValueError('Mirror reshaping requires the authored housing, glass and mount topology.')
    return groups[642], groups[162], groups[30]


def _rounded_rectangle(q, exponent=3.6):
    """Smooth radial circle-to-rounded-rectangle map; no nearest-point fitting."""
    radius = np.linalg.norm(q, axis=1)
    lp = np.sum(np.abs(q)**exponent, axis=1)**(1/exponent)
    scale = np.divide(radius, lp, out=np.ones_like(radius), where=lp > 1e-12)
    return q*scale[:, None]


def reshape_mirrors(model: Model, style: str = 'aero') -> tuple[np.ndarray, dict]:
    """Return a new full vertex array and JSON-safe fixture provenance.

    Faces, semantic labels, rendering material slots and rigid follower vertex
    identities are unchanged. The left fixture is authored once and its exact
    original reflection correspondence constructs the right fixture.
    """
    if style not in STYLES:
        raise ValueError(f'Unknown mirror variant: {style}')
    v0 = model.vertices
    out = v0.copy()
    housing, glass, mount = _islands(model, 'mirror_l')
    spec = STYLES[style]
    size = np.asarray(spec['housing_m'])
    lens = np.asarray(spec['glass_m'])
    center = np.array([-.770, 1.109, 1.147])

    # Recover the original sphere's analytic coordinates, including its shear.
    u = v0[housing] - np.array([-.770, 1.056, 1.137])
    u[:, 1] -= .035*u[:, 0]/.165
    u /= [.165, .100, .062]
    if not np.allclose(np.linalg.norm(u, axis=1), 1., atol=1e-6):
        raise ValueError('Mirror source must be the unmodified canonical scaffold.')
    yz = _rounded_rectangle(u[:, 1:])
    shell = np.zeros_like(u)
    shell[:, 1:] = yz*size[1:]/2
    # A rounded leading cover and a broad, nearly planar rear face support the
    # glass perimeter. Both halves remain monotone in the original x latitude.
    ux = u[:, 0]
    shell[:, 0] = size[0]/2*np.where(ux > 0, np.tanh(6*ux)/np.tanh(6), ux)
    # Slight outward sweep, shared exactly by the lens rather than a handed
    # distortion of the two assemblies.
    shell[:, 0] += .035*shell[:, 1]
    out[housing] = shell + center

    q = (v0[glass] - [-.622, 1.064, 1.139])/[.009, .081, .047]
    if not np.allclose(np.linalg.norm(q, axis=1), 1., atol=1e-6):
        raise ValueError('Mirror glass source must be the canonical scaffold.')
    panel = q*lens/2
    panel[:, 1:] = _rounded_rectangle(q[:, 1:])*lens[1:]/2
    # The thin lens sits 1 mm aft of the housing's maximum rear plane, with a
    # small physical seam; it cannot cross the closed authored housing surface.
    panel[:, 0] += size[0]/2+lens[0]/2+.001+.035*panel[:, 1]
    out[glass] = panel+center

    # Three existing ten-vertex rings form a tapered foot instead of a thin
    # circular rod. The body flange stays exactly at its original vertices.
    old_path = np.array([[-.86, .932, 1.065], [-.845, .970, 1.110], [-.790, 1.015, 1.130]])
    tangent = np.gradient(old_path, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    a = np.cross(tangent, [0., 0., 1.]); a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(tangent, a)
    angles = np.arange(10)*2*np.pi/10
    old_rings = old_path[:, None, :]+.020*(np.cos(angles)[None, :, None]*a[:, None, :]
                                         +np.sin(angles)[None, :, None]*b[:, None, :])
    distance, match = cKDTree(v0[mount]).query(old_rings.reshape(-1, 3))
    if distance.max() > 1e-7 or len(np.unique(match)) != 30:
        raise ValueError('Mirror mounting-ring correspondence is invalid.')
    rings = old_rings.copy()
    inner_y = center[1]-size[1]/2
    rings[1] = np.array([-.814, .971, 1.104])+np.c_[.039*np.cos(angles),
                                                                  np.zeros(10), -.025*np.sin(angles)]
    rings[2] = np.array([center[0], inner_y-.001, center[2]])+np.c_[.030*np.cos(angles),
                                                                    np.zeros(10), -.022*np.sin(angles)]
    out[mount[match]] = rings.reshape(-1, 3)

    left = np.concatenate((housing, glass, mount))
    reflected = v0[left]*[1, -1, 1]
    distance, right = cKDTree(v0).query(reflected)
    if distance.max() > 1e-8 or len(np.unique(right)) != len(left):
        raise ValueError('Mirror reflection correspondence is invalid.')
    out[right] = out[left]*[1, -1, 1]
    report = dict(schema='aeroshape.reference-proportioned-mirrors.v1', style=style,
        shape_source='original_parametric_fixture_with_measured_reference_proportions',
        reference_sample='E_S_WW_WM_001', reference_labels=['Mirrors', 'Mirrors_Body', 'Mirrors_Glass'],
        reference_units='mm converted to m', source_triangles_copied=False,
        housing_nominal_dimensions_m=size.tolist(), glass_nominal_dimensions_m=lens.tolist(),
        reference_housing_with_neck_dimensions_m=[.1248, .2704, .1398],
        reference_glass_dimensions_m=[.0147, .1824, .1174],
        housing_and_glass_sweep_ratio=.035, mount_flange_preserved=True,
        topology_preserved=True, material_slots_preserved=True,
        construction='rounded transverse housing, rounded-rectangle glass, tapered three-ring foot',
        interfaces='separate housing/glass/mount islands with small seams, not boolean-unioned CAD',
        mirror_vertices_per_side=len(left), mirror_faces_per_side=1656)
    return out, report


def mirror_seating_translation(model: Model, vertices: np.ndarray, *, side='l',
                               clearance_m=.0005) -> tuple[np.ndarray, dict]:
    """Seat the complete rigid mirror on the actual body with a small seam.

    Return a translation to apply to ALL vertices of this mirror assembly after
    its parent-anchor rotation, before constructing its reflected peer. Moving
    by (nearest distance - clearance) along the exact closest point pair cannot
    cross the body: every surface distance can decrease by at most the magnitude
    of that translation. This corrects the old template's floating mounting foot.
    """
    if side not in ('l', 'r') or not 0 <= clearance_m <= .002:
        raise ValueError('Mirror seating requires side l/r and a seam between 0 and 2 mm.')
    _, _, mount = _islands(model, 'mirror_'+side)
    mount_faces = model.faces[np.isin(model.faces, mount).all(axis=1)]
    driver = np.asarray(model.metadata['driver_vertices'], dtype=np.int32)
    is_driver = np.zeros(len(vertices), bool); is_driver[driver] = True
    body_faces = model.faces[is_driver[model.faces].all(axis=1)]
    # A vertex-to-vertex distance supplies an upper bound. Any closer triangle
    # must overlap the foot's box expanded by that bound, so this broad phase
    # does not discard a possible nearer body triangle.
    bound = float(cKDTree(vertices[driver]).query(vertices[mount])[0].min())+1e-9
    foot = vertices[mount]; triangle = vertices[body_faces]
    candidates = ((triangle.max(axis=1) >= foot.min(0)-bound).all(axis=1)
                  & (triangle.min(axis=1) <= foot.max(0)+bound).all(axis=1))
    nearest = TriangleIndex(vertices, mount_faces).distance(TriangleIndex(vertices, body_faces[candidates]))
    if not nearest['complete']:
        raise ValueError('Mirror mounting distance check did not complete.')
    distance = nearest['distance_m']
    if distance < 1e-8:
        raise ValueError('Mirror mounting foot intersects the body before seating; parameters rejected.')
    if distance <= clearance_m:
        translation = np.zeros(3)
    else:
        translation = (np.asarray(nearest['point_b'])-nearest['point_a'])*(1-clearance_m/distance)
    return translation, dict(mode='rigid_translation_to_nearest_body_triangle',
        original_mount_gap_m=distance, target_mount_gap_m=min(clearance_m, distance),
        translation_m=translation.tolist(), seating_distance_m=float(np.linalg.norm(translation)),
        distance_check_complete=True, body_triangles_considered=int(candidates.sum()),
        geometry_preserved='entire_fixture_moves_rigidly; no vertices_projected_individually')
