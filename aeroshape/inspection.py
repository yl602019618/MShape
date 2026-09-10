"""Hash-bound static surface validation. No dynamics, containment or CFD claim."""
from __future__ import annotations
import hashlib,json,time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pydantic import BaseModel,Field,ConfigDict
from typing import Literal
from .model import Model,array_hash
from .spatial import TriangleIndex
from .geometry import edges_and_counts

class ClearancePair(BaseModel):
    model_config=ConfigDict(extra='forbid')
    part_a:int=Field(ge=0)
    part_b:int|None=Field(None,ge=0)
    threshold_m:float=Field(.008,ge=0,le=.2)

class InspectionOptions(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    scope:Literal['driver','assembly']='driver'
    tolerance_m:float=Field(1e-8,ge=1e-10,le=1e-5)
    max_pairs:int=Field(20000,ge=1,le=50000)
    max_triangle_tests:int=Field(20_000_000,ge=100,le=50_000_000)
    check_clearance:bool=True
    pairs:list[ClearancePair]=Field(default_factory=list,max_length=8)
    wheel_clearance_m:float=Field(.008,ge=0,le=.1)


def driver_faces(model):
    if model.metadata.get('driver_vertices'):
        mask=np.zeros(len(model.vertices),bool);mask[model.metadata['driver_vertices']]=True
        return np.flatnonzero(mask[model.faces].all(1))
    e,_=edges_and_counts(model.faces);n=len(model.vertices)
    graph=coo_matrix((np.ones(len(e)*2),(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(n,n)).tocsr()
    _,labels=connected_components(graph,directed=False);largest=np.bincount(labels).argmax()
    return np.flatnonzero((labels[model.faces]==largest).all(1))


def inspect(model:Model,options:InspectionOptions|dict|None=None):
    opt=options if isinstance(options,InspectionOptions) else InspectionOptions.model_validate(options or {})
    start=time.perf_counter();main=driver_faces(model)
    faceids=main if opt.scope=='driver' else np.arange(len(model.faces))
    index=TriangleIndex(model.vertices,model.faces[faceids])
    report=index.intersections(tolerance=opt.tolerance_m,max_pairs=opt.max_pairs,max_tests=opt.max_triangle_tests)
    pairs=np.asarray(report.pop('pairs'),np.int64).reshape(-1,2)
    global_pairs=faceids[pairs] if len(pairs) else np.empty((0,2),np.int64)
    report.update(scope=opt.scope,scope_face_count=len(faceids),whole_assembly_checked=opt.scope=='assembly',
        pairs=global_pairs.tolist(),expected_shared_edge_vertex_contacts_excluded=True,
        intentional_assembly_contacts_not_automatically_ignored=True)
    clearance=[];main_index=index if opt.scope=='driver' else TriangleIndex(model.vertices,model.faces[main])
    if opt.check_clearance:
        requests=opt.pairs or [ClearancePair(part_a=p['id'],threshold_m=opt.wheel_clearance_m) for p in model.parts if p['key'].startswith('wheel_')]
        keys={p['id']:p['key'] for p in model.parts}
        for pair in requests:
            fa=np.flatnonzero(model.labels==pair.part_a)
            fb=np.flatnonzero(model.labels==pair.part_b) if pair.part_b is not None else main
            if not len(fa) or not len(fb):raise ValueError('Clearance pair contains an empty/unknown region.')
            if np.intersect1d(fa,fb).size:raise ValueError('Clearance sets overlap. Choose independent components, not a region against itself.')
            idxa=TriangleIndex(model.vertices,model.faces[fa]);idxb=main_index if pair.part_b is None else TriangleIndex(model.vertices,model.faces[fb])
            r=idxa.distance(idxb,max_tests=opt.max_triangle_tests)
            r.update(part_a=pair.part_a,part_b=pair.part_b,key_a=keys[pair.part_a],key_b=keys.get(pair.part_b,'driver_skin'),threshold_m=pair.threshold_m,
                source_face_a=int(fa[r['face_a']]) if r['face_a']>=0 else None,source_face_b=int(fb[r['face_b']]) if r['face_b']>=0 else None)
            r['status']='incomplete' if not r['complete'] else 'contact_or_intersection' if r['distance_m']<=opt.tolerance_m else 'below_threshold' if r['distance_m']<pair.threshold_m else 'clear'
            clearance.append(r)
    completed=report['complete'] and all(r['complete'] for r in clearance) and (not opt.check_clearance or bool(clearance))
    passed=completed and not report['count'] and all(r['status']=='clear' for r in clearance)
    issues=sorted(set(global_pairs.ravel().tolist()))
    result=dict(schema='aeroshape.static-surface-inspection.v1',geometry_hash=array_hash(model.vertices,model.faces),
        annotation_hash=model.annotation_hash(),options=opt.model_dump(),self_intersection=report,component_clearance=clearance,
        clearance_scope_count=len(clearance),clearance_scope_status=('evaluated' if clearance else 'not_requested' if not opt.check_clearance else 'no_default_wheel_targets'),complete=completed,requested_checks_passed=passed,issue_face_ids=issues[:8000],issue_face_ids_truncated=len(issues)>8000,
        elapsed_s=time.perf_counter()-start,cfd_ready=False,
        unchecked=['solid containment / signed penetration depth','wheel steering / suspension swept envelope','manufacturability / CAD continuity','external flow volume mesh']+
            (['accessory self-intersections / all assembly pairs'] if opt.scope=='driver' else []),
        scope_note='Static unsigned surface distances. An attached mirror naturally has contact at its root; no blanket pair exclusion.',
        numerical_limit='float64 with explicit tolerance, not exact-arithmetic certification')
    return result
