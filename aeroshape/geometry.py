from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .model import Model, default_parts



def _invalidate_bindings_after_relabel(model):
    # A reference binding names semantic parent parts. After segmentation or
    # painting those names no longer guarantee the same parent triangles.
    if model.metadata.get('rigid_followers'):
        model.metadata['rigid_followers']=[]
        model.metadata['attachment_binding_status']='invalidated_by_relabel_requires_rebinding'
    model.metadata.pop('design_regions',None)
    # Keep canonical templates on manual edits; clear canonical provenance when
    # switching to an incompatible legacy 12-class/island schema.
    if model.metadata.get('semantic_schema')=='aeroshape.design30.v1':
        from .semantic import KEYS,UNKNOWN
        canonical=all((p['id'] in range(30) and p.get('key')==KEYS[p['id']]) or
                      (p['id']==UNKNOWN and p.get('key')=='unassigned') for p in model.parts)
        if not canonical:
            for key in ('semantic_schema','semantic_source_labels','semantic_source_parts',
                        'segmentation_face_scores','segmentation_face_methods','segmentation','design_sets'):
                model.metadata.pop(key,None)
    else:model.metadata.pop('design_sets',None)
    return model

def face_geometry(v, f):
    t = v[f]
    cross = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    return t.mean(axis=1), cross / np.maximum(norm[:, None], 1e-30), norm * .5


def edges_and_counts(f):
    edges = np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def normalized_coordinates(v):
    lo, hi = v.min(axis=0), v.max(axis=0)
    return (v-lo)/np.maximum(hi-lo, 1e-12)


def suggest_regions(model: Model) -> Model:
    """Position-based suggestion, NEVER a learned/OEM semantic segmentation."""
    c, n, _ = face_geometry(model.vertices, model.faces)
    lo, hi = model.vertices.min(axis=0), model.vertices.max(axis=0)
    p = (c-lo)/np.maximum(hi-lo, 1e-12)
    x, z = p[:, 0], p[:, 2]
    lab = np.full(len(c), 7, dtype=np.int32)
    upper = n[:, 2] > .3
    lab[(x < .18)] = 1
    lab[(x >= .18) & (x < .37) & upper & (z > .35)] = 2
    lab[(x >= .32) & (x < .46) & upper & (z > .58)] = 3
    lab[(x >= .44) & (x < .68) & upper & (z > .68)] = 4
    lab[(x >= .68) & (x < .85) & upper & (z > .53)] = 5
    lab[x >= .85] = 6
    lab[(z < .20) & (n[:, 2] < -.2)] = 8
    lab[(x > .75) & (z < .25) & (n[:, 2] < -.2)] = 11
    # Preserve only genuinely named source wheels/mirrors, not guesses from position.
    for part in model.parts:
        key = (part.get('key','') + ' ' + part.get('name','')).lower()
        if any(s in key for s in ('wheel', 'tire', 'tyre', '轮胎', '车轮')):
            lab[model.labels == part['id']] = 9
        if any(s in key for s in ('mirror', '后视镜')):
            lab[model.labels == part['id']] = 10
    out = model.clone(labels=lab, parts=default_parts())
    out.metadata['annotation_status'] = 'heuristic_requires_manual_review'
    return _invalidate_bindings_after_relabel(out)


def split_connected(model: Model) -> Model:
    """Find edge-connected components without cutting or duplicating master vertices."""
    f = model.faces
    e = np.sort(np.vstack((f[:, [0,1]], f[:,[1,2]], f[:,[2,0]])), axis=1)
    fi = np.tile(np.arange(len(f)), 3)
    order = np.lexsort((e[:,1], e[:,0]))
    same = np.all(e[order][1:] == e[order][:-1], axis=1)
    a, b = fi[order][:-1][same], fi[order][1:][same]
    graph = coo_matrix((np.ones(len(a)*2), (np.r_[a,b], np.r_[b,a])), shape=(len(f),len(f)))
    count, labels = connected_components(graph, directed=False)
    if count > 1000:
        raise ValueError(f'{count} mesh islands found. Weld/repair the input before component splitting (limit 1000).')
    colors = [p['color'] for p in default_parts()]
    parts = [dict(id=i, key=f'island_{i}', name=f'网格岛 {i+1}', color=colors[i%len(colors)], locked=False)
             for i in range(count)]
    out = model.clone(labels=labels, parts=parts)
    out.metadata['annotation_status'] = 'connected_islands_not_semantic_parts'
    return _invalidate_bindings_after_relabel(out)


def paint(model: Model, *, part_id: int, center=None, radius=.15, bounds=None,
          mirror=True, source_part=None, face_ids=None) -> Model:
    if part_id not in [p['id'] for p in model.parts]:
        raise ValueError('Unknown target part ID.')
    c = model.vertices[model.faces].mean(axis=1)
    chosen = np.zeros(len(c), bool)
    if center is not None:
        q = np.asarray(center, float)
        if q.shape != (3,) or not np.isfinite(q).all() or not 0 < radius < 100:
            raise ValueError('Invalid brush center/radius.')
        chosen |= np.linalg.norm(c-q, axis=1) <= radius
        if mirror:
            q[1] *= -1
            chosen |= np.linalg.norm(c-q, axis=1) <= radius
    if bounds is not None:
        b = np.asarray(bounds, float)
        if b.shape != (2,3) or not np.isfinite(b).all() or np.any(b[1] <= b[0]):
            raise ValueError('Invalid paint bounds.')
        chosen |= np.all((c >= b[0]) & (c <= b[1]), axis=1)
        if mirror:
            cm = c.copy(); cm[:,1] *= -1
            chosen |= np.all((cm >= b[0]) & (cm <= b[1]), axis=1)
    if face_ids is not None:
        ids = np.asarray(face_ids, int)
        if len(ids) and (ids.min() < 0 or ids.max() >= len(c)):
            raise ValueError('Face ID out of range.')
        chosen[ids] = True
        if mirror and len(ids):
            q = c[ids].copy(); q[:,1] *= -1
            dist, partners = cKDTree(c).query(q)
            chosen[partners[dist < 1e-4*max(np.ptp(model.vertices, axis=0))]] = True
    if source_part is not None:
        chosen &= model.labels == source_part
    out = model.clone(); out.labels[chosen] = part_id
    # Canonical left/right labels must reflect to their partner, not the same ID.
    if mirror and model.metadata.get('semantic_schema')=='aeroshape.design30.v1':
        part=next(p for p in model.parts if p['id']==part_id)
        if part.get('pair_id') is not None:
            side=part['key'][-1]
            opposite=(c[:,1]<0) if side=='l' else (c[:,1]>0)
            out.labels[chosen & opposite]=part['pair_id']
    out.metadata['annotation_status'] = 'user_edited_requires_review'
    if model.metadata.get('semantic_schema')=='aeroshape.design30.v1':
        from .semantic import refresh_summary
        scores=np.asarray(out.metadata['segmentation_face_scores']);methods=np.asarray(out.metadata['segmentation_face_methods'])
        scores[chosen]=1.;methods[chosen]=5
        out.metadata['segmentation_face_scores']=scores.tolist();out.metadata['segmentation_face_methods']=methods.tolist()
        refresh_summary(out)
    return _invalidate_bindings_after_relabel(out) if np.any(model.labels!=out.labels) else out


def region_grow(model: Model, seed_face: int, angle_degrees: float=25, limit: int=100000):
    """Adjacent-face grow with seed normal and per-edge angle barriers."""
    if not 0 <= seed_face < len(model.faces) or not 0 < angle_degrees < 90:
        raise ValueError('Invalid seed or angle.')
    m = model.mesh()
    adj = m.face_adjacency
    normals = m.face_normals
    cosine = np.cos(np.deg2rad(angle_degrees))
    keep = (normals[adj[:,0]]*normals[adj[:,1]]).sum(axis=1) > cosine
    adj = adj[keep]
    good = normals @ normals[seed_face] > cosine
    adj = adj[good[adj].all(axis=1)]
    n = len(model.faces)
    g = coo_matrix((np.ones(len(adj)*2), (np.r_[adj[:,0],adj[:,1]], np.r_[adj[:,1],adj[:,0]])), shape=(n,n)).tocsr()
    from scipy.sparse.csgraph import breadth_first_order
    ids = breadth_first_order(g, seed_face, directed=False, return_predecessors=False)
    if len(ids) > limit:
        raise ValueError('Region exceeds limit; lower the angle or use the brush.')
    return ids


def preview_indices(model: Model, target=180000):
    """Vertex-clustering DISPLAY LOD; master geometry and labels stay untouched."""
    v, f = model.vertices, model.faces
    if len(f) <= target:
        return np.arange(len(v)), f.copy(), np.arange(len(f))
    extent = np.ptp(v, axis=0).max()
    best = None
    step = extent / np.sqrt(target)
    for _ in range(16):
        keys = np.floor((v-v.min(axis=0))/step).astype(np.int64)
        _, representatives, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        ff = inverse[f]
        good = (ff[:,0]!=ff[:,1]) & (ff[:,0]!=ff[:,2]) & (ff[:,1]!=ff[:,2])
        ids = np.flatnonzero(good); ff = ff[good]
        # Keep coincident faces only once per label; preserve their source face ID.
        keys_f = np.column_stack((np.sort(ff, axis=1), model.labels[ids]))
        _, first = np.unique(keys_f, axis=0, return_index=True)
        best = (representatives, ff[first], ids[first])
        if len(first) <= target:
            return best
        step *= 1.35
    return best


def refine_uniform(model: Model, levels=1, max_faces=2000000) -> Model:
    """Conforming 1-to-4 triangle refinement. Not a quality-improving remesher."""
    if levels not in (1,2,3) or len(model.faces)*4**levels > max_faces:
        raise ValueError('Refinement exceeds 2M-face limit or invalid level count.')
    v, f, labels = model.vertices.copy(), model.faces.copy(), model.labels.copy()
    import json
    meta=json.loads(json.dumps(model.metadata))
    for _ in range(levels):
        e = np.sort(np.vstack((f[:,[0,1]], f[:,[1,2]], f[:,[2,0]])), axis=1)
        unique, inverse = np.unique(e, axis=0, return_inverse=True)
        mid = inverse.reshape(3,-1).T + len(v)
        if 'driver_vertices' in meta:
            old=np.asarray(meta['driver_vertices'],int);ok=np.isin(unique,old).all(1)
            meta['driver_vertices']=np.r_[old,np.flatnonzero(ok)+len(v)].tolist()
        for binding in meta.get('rigid_followers',[]):
            old=np.asarray(binding['vertices'],int);ok=np.isin(unique,old).all(1)
            binding['vertices']=np.r_[old,np.flatnonzero(ok)+len(v)].tolist()
        for field in ('face_material_ids','segmentation_face_scores','segmentation_face_methods','semantic_source_labels',
                      'raw_face_labels','source_face_indices'):
            if field in meta:meta[field]=np.tile(meta[field],4).tolist()
        v = np.vstack((v, v[unique].mean(axis=1)))
        a,b,c = f.T; ab,bc,ca = mid.T
        f = np.vstack((np.c_[a,ab,ca], np.c_[ab,b,bc], np.c_[ca,bc,c], np.c_[ab,bc,ca]))
        labels = np.tile(labels, 4)
    out = model.clone(vertices=v, faces=f, labels=labels,metadata=meta)
    out.metadata['refinement'] = dict(method='uniform_1_to_4', levels=levels,
                                      parent_geometry_hash=model.metadata['base_geometry_hash'])
    from .model import array_hash
    out.metadata['base_geometry_hash'] = array_hash(v,f)
    if out.metadata.get('semantic_schema')=='aeroshape.design30.v1':
        from .semantic import refresh_summary
        refresh_summary(out)
    return out
