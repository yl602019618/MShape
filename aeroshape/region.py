"""Topology-aware regional handles with bounded graph-biharmonic weights.

This is NOT volumetric BBW, CAD fairing, ARAP, or a global injectivity proof.
Selection is along mesh edges, never Euclidean proximity across separate sheets.
A sparse scalar weight solve is cached independently of the handle magnitude.
"""
from __future__ import annotations
from collections import OrderedDict
from threading import RLock
import hashlib
import json
import time
import warnings
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from scipy.optimize import minimize
from .model import Model, array_hash
from .geometry import edges_and_counts, face_geometry


class RegionSpec(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    seed_faces: list[int]=Field(min_length=1,max_length=64)
    pin_faces: list[int]=Field(default_factory=list,max_length=256)
    radius: float=Field(default=.18,ge=.005,le=3.)
    transition: float=Field(default=.38,ge=.01,le=3.)
    cross_parts: bool=True
    follow_attachments: bool=True
    axis: Literal['x','y','z','normal']='z'
    tension: float=Field(default=.02,ge=.001,le=1.)
    max_strain: float=Field(default=.35,ge=.01,le=1.)
    max_attachment_residual: float=Field(default=.003,gt=0,le=.05)


_CACHE: OrderedDict[str,dict]=OrderedDict()
_BINDINGS: OrderedDict[str,list]=OrderedDict()
_CACHE_LOCK=RLock()


def _bounded_qp(H,b):
    """Minimize .5 x^T H x + b^T x on [0,1], reporting projected KKT residual.

    A sparse unconstrained solve is an initial guess only. Clipping that guess
    is not used as the final answer: the bound-constrained optimizer must pass
    a projected-gradient test or the operation is rejected.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error',MatrixRankWarning)
        x=spsolve(H,-b)
    if not np.isfinite(x).all(): raise ValueError('Ill-conditioned regional solve; repair the source mesh.')
    scale=max(float(np.max(np.abs(H.diagonal()))),1.)
    H=H/scale;b=b/scale
    def gradient(x):return np.asarray(H@x+b)
    def residual(x):
        g=gradient(x);p=g.copy();p[(x<1e-9)&(g>0)]=0;p[(x>1-1e-9)&(g<0)]=0
        return float(np.max(np.abs(p))) if len(p) else 0.
    if x.min()>=-1e-10 and x.max()<=1+1e-10:
        x=np.clip(x,0,1)
        return x,dict(method='sparse_direct_feasible',iterations=1,projected_kkt=residual(x))
    def fun(x):
        g=gradient(x);return .5*float(x@(g+b)),g
    result=minimize(fun,np.clip(x,0,1),jac=True,method='L-BFGS-B',bounds=[(0.,1.)]*len(x),
                    options=dict(maxiter=1800,maxcor=20,ftol=1e-15,gtol=1e-8,maxls=40))
    residual_value=residual(result.x)
    if residual_value>2e-5:
        raise ValueError(f'Regional solver did not reach the numerical tolerance ({residual_value:.2g}). Reduce the region or improve the mesh.')
    return result.x,dict(method='bounded_L-BFGS-B',iterations=int(result.nit),projected_kkt=residual_value)


def _ids(values,n,name):
    a=np.asarray(values,dtype=np.int64)
    if len(a) and (a.min()<0 or a.max()>=n): raise ValueError(f'{name} indices do not match this reference mesh.')
    return np.unique(a)


def _mirror_map(v):
    q=v.copy();q[:,1]*=-1
    dist,partner=cKDTree(v).query(q)
    tol=max(float(np.ptp(v,axis=0).max())*1e-7,1e-9)
    good=(dist<=tol)&(partner[partner]==np.arange(len(v)))
    return partner,good


def prepare_region(model:Model,spec:RegionSpec,mirror=True):
    key=hashlib.sha256((model.metadata['base_geometry_hash']+model.annotation_hash()+
         spec.model_dump_json()+str(mirror)+':region-v03.2').encode()).hexdigest()
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key);return _CACHE[key]
    start=time.perf_counter();v=model.vertices;f=model.faces;n=len(v)
    seed_faces=_ids(spec.seed_faces,len(f),'Seed face');pin_faces=_ids(spec.pin_faces,len(f),'Pin face')
    locked_parts=[p['id'] for p in model.parts if p.get('locked')]
    locked=np.zeros(n,bool);locked[np.unique(f[np.isin(model.labels,locked_parts)])]=True
    driver=np.zeros(n,bool)
    if 'driver_vertices' in model.metadata:
        driver[_ids(model.metadata['driver_vertices'],n,'Driver vertex')]=True
    else:driver[:]=True
    if not np.all(driver[f[seed_faces]]):
        raise ValueError('Select the continuous body skin, not an independent accessory. Use rigid motion or explicitly bind that accessory to a parent surface.')
    selected_parts=set(int(x) for x in model.labels[seed_faces])
    if mirror:
        for p in model.parts:
            if p['id'] in list(selected_parts) and p.get('pair_id') is not None:selected_parts.add(p['pair_id'])
    permitted=driver.copy()
    hard=locked.copy()
    if not spec.cross_parts:
        sf=np.isin(model.labels,list(selected_parts))
        permitted[:]=False;permitted[np.unique(f[sf])]=True;permitted &=driver
        hard[np.unique(f[~sf])]=True
    hard[np.unique(f[pin_faces])]=True
    # Anchor a locked follower's parent footprint instead of letting the skin
    # separate from an accessory that the user deliberately froze.
    if model.metadata.get('rigid_followers') and locked.any():
        for binding in _binding_data(model):
            if not binding.get('error') and locked[binding['ids']].any():hard[np.unique(binding['faces'])]=True
    partner=good=None
    if mirror:
        partner,good=_mirror_map(v)
        if not good[permitted].all():
            raise ValueError('This surface lacks verified mirror vertex correspondence. Disable symmetry or remesh it symmetrically; nearest-neighbour symmetry is not silently assumed.')
        hard|=hard[partner]
        permitted&=permitted[partner]
    seed=np.unique(f[seed_faces])
    if mirror:seed=np.unique(np.r_[seed,partner[seed]])
    if np.any(hard[seed]) or not np.all(permitted[seed]):
        raise ValueError('The handle overlaps a locked/pinned/interface vertex. Move the handle or enable cross-panel blending.')
    edges,_=edges_and_counts(f)
    edges=edges[permitted[edges].all(axis=1)]
    if mirror:
        edges=np.unique(np.sort(np.vstack((edges,partner[edges])),axis=1),axis=0)
    lengths=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
    if np.any(lengths<1e-12):raise ValueError('Zero-length design edges must be cleaned before solving.')
    a,b=edges.T
    graph=coo_matrix((np.r_[lengths,lengths],(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
    dist=dijkstra(graph,directed=False,indices=seed,min_only=True,limit=spec.radius+spec.transition)
    support=np.isfinite(dist)&permitted
    if mirror:support|=support[partner]
    core=(dist<=spec.radius)&support
    if mirror:core|=core[partner]
    if np.any(core&hard):
        raise ValueError('The core region intersects fixed constraints. Reduce the core radius or move/remove the pin.')
    # The outer two vertex rings are zero-displacement collars. On a discrete
    # triangle surface this suppresses an abrupt slope change; it is not CAD G1.
    outside=~support|hard
    collar=hard.copy()
    for _ in range(2):
        crossing=outside[a]!=outside[b]
        ids=np.unique(edges[crossing]);collar[ids]=True;outside=outside|collar
    fixed=~support|collar|hard
    if mirror:fixed|=fixed[partner]
    core&=~fixed
    core[seed]=True
    if fixed[seed].any():raise ValueError('Transition band is narrower than the mesh spacing near the handle. Increase it.')
    free=support&~fixed&~core
    if not free.any():raise ValueError('No transition vertices. Increase the transition width.')
    if free.sum()>18000:raise ValueError('Interactive solve exceeds 18,000 free vertices. Reduce the region or use a coarse design surface.')
    active=support.copy()
    for _ in range(2):
        ids=np.unique(edges[active[a]|active[b]]);active[ids]=True
    local=np.flatnonzero(active);index=np.full(n,-1,dtype=int);index[local]=np.arange(len(local))
    le=edges[active[a]&active[b]];la,lb=index[le].T
    leng=np.linalg.norm(v[le[:,0]]-v[le[:,1]],axis=1)
    weights=np.median(leng)/leng
    A=coo_matrix((np.r_[weights,weights],(np.r_[la,lb],np.r_[lb,la])),shape=(len(local),len(local))).tocsr()
    degree=np.asarray(A.sum(axis=1)).ravel();L=diags(degree)-A
    # Symmetric positive semidefinite graph bending + membrane regularization.
    Q=(L.T@diags(1/np.maximum(degree,1e-12))@L+spec.tension*L).tocsr()
    fi=np.flatnonzero(free[local]);ci=np.flatnonzero(core[local])
    H=Q[fi][:,fi].tocsc();rhs=np.asarray(Q[fi][:,ci].sum(axis=1)).ravel()
    x,solver=_bounded_qp(H,rhs)
    w=np.zeros(n);w[local[fi]]=x;w[core]=1
    if mirror:w=.5*(w+w[partner])
    w[fixed]=0;w[core]=1
    _,normals,_=face_geometry(v,f[seed_faces[:1]])
    direction=np.eye(3)[['x','y','z'].index(spec.axis)] if spec.axis!='normal' else normals[0]
    basis=w[:,None]*direction
    if mirror:
        sign=np.sign(v[f[seed_faces[0]]].mean(axis=0)[1]) or 1.
        basis[:,1]=w*direction[1]*np.sign(v[:,1])*sign
        basis[:,1][np.abs(v[:,1])<1e-9]=0
    basis[hard]=0
    center=v[f[seed_faces[0]]].mean(axis=0)
    moving_faces=np.any(w[f]>1e-8,axis=1)
    touched=sorted(int(x) for x in np.unique(model.labels[moving_faces]))
    info=dict(method='bounded_graph_biharmonic_surface_handle',distance='shortest_mesh_edge_path',
              core_vertices=int(core.sum()),transition_vertices=int(free.sum()),support_vertices=int(support.sum()),
              affected_parts=touched,core_radius_m=spec.radius,transition_width_m=spec.transition,
              weight_min=float(w.min()),weight_max=float(w.max()),fixed_motion_exact=True,
              mirror_correspondence_verified=mirror,solver=solver,prepare_seconds=time.perf_counter()-start,
              scope='discrete_surface_continuity_not_CAD_G1_or_volume_injectivity')
    result=dict(weights=w,basis=basis,center=center,direction=direction,info=info,core=core,fixed=fixed,
                mirror_partner=partner,cache_key=key)
    with _CACHE_LOCK:
        _CACHE[key]=result
        while len(_CACHE)>12:_CACHE.popitem(last=False)
    return result


def _binding_data(model:Model):
    key=model.metadata['base_geometry_hash']+model.annotation_hash()
    with _CACHE_LOCK:
        if key in _BINDINGS:return _BINDINGS[key]
    from trimesh.triangles import closest_point, points_to_barycentric
    driver=np.zeros(len(model.vertices),bool);driver[_ids(model.metadata.get('driver_vertices',list(range(len(driver)))),len(driver),'Driver vertex')]=True
    bindings=[]
    for binding in model.metadata.get('rigid_followers',[]):
        ids=_ids(binding['vertices'],len(model.vertices),'Follower vertex')
        face_ids=np.flatnonzero(np.isin(model.labels,binding['parent_parts'])&driver[model.faces].all(axis=1))
        if not len(ids) or not len(face_ids):continue
        # Reference attachment anchors, chosen deterministically. Candidate triangle
        # search is local/approximate; max binding distance is an explicit guard.
        sample=ids[np.linspace(0,len(ids)-1,min(36,len(ids)),dtype=int)]
        tri=model.vertices[model.faces[face_ids]]
        k=min(16,len(tri));tree=cKDTree(tri.mean(axis=1))
        _,candidates=tree.query(model.vertices[sample],k=k)
        candidates=np.asarray(candidates).reshape(len(sample),k)
        tt=tri[candidates].reshape(-1,3,3);pp=np.repeat(model.vertices[sample],k,axis=0)
        cp=closest_point(tt,pp).reshape(len(sample),k,3)
        distance=np.linalg.norm(cp-model.vertices[sample,None,:],axis=2)
        best=distance.argmin(axis=1);rows=np.arange(len(sample))
        chosen=candidates[rows,best];points=cp[rows,best]
        valid=distance[rows,best]<=binding.get('max_binding_distance_m',.35)
        if valid.sum()<3:
            bindings.append(dict(ids=ids,error='Insufficient close parent anchors',name=binding['name']));continue
        points=points[valid];chosen=chosen[valid]
        bary=points_to_barycentric(tri[chosen],points)
        bindings.append(dict(ids=ids,faces=model.faces[face_ids[chosen]],bary=bary,points=points,name=binding['name'],
                             reference_distance_max=float(distance[rows,best][valid].max())))
    with _CACHE_LOCK:
        _BINDINGS[key]=bindings
        while len(_BINDINGS)>6:_BINDINGS.popitem(last=False)
    return bindings


def follow_attachments(model:Model,vertices,locked,mirror=False):
    out=np.array(vertices,copy=True);reports=[];bindings=_binding_data(model)
    partner=good=None
    if mirror:partner,good=_mirror_map(model.vertices)
    lookup={tuple(b['ids']):b for b in bindings};done=set()
    # Prefer +Y. A mirrored pair is fit jointly; independent least-squares fits
    # could break exact symmetry because the two triangle tessellations differ.
    for binding in sorted(bindings,key=lambda b:-float(model.vertices[b['ids'],1].mean())):
        ids=binding['ids'];name=binding['name']
        if name in done:continue
        done.add(name)
        if binding.get('error'):
            reports.append(dict(name=name,status='binding_failed',error=binding['error']));continue
        peer=None
        if mirror:
            if not good[ids].all():
                reports.append(dict(name=name,status='binding_failed',error='Follower mirror correspondence is not verified.'));continue
            peer=lookup.get(tuple(np.sort(partner[ids])))
            if peer is None or peer.get('error'):
                reports.append(dict(name=name,status='binding_failed',error='Mirrored follower binding is missing.'));continue
            done.add(peer['name'])
        joint_ids=np.unique(np.r_[ids,peer['ids']]) if peer else ids
        if locked[joint_ids].any():
            reports.append(dict(name=name,status='locked_not_followed'));continue
        a=binding['points'];target=np.einsum('ij,ijk->ik',binding['bary'],vertices[binding['faces']])
        S=np.array([1.,-1.,1.])
        if peer is not None:
            pa=peer['points'];pt=np.einsum('ij,ijk->ik',peer['bary'],vertices[peer['faces']])
            solve_a=np.vstack((a,pa*S));solve_t=np.vstack((target,pt*S))
        else:solve_a,solve_t=a,target
        if np.max(np.linalg.norm(solve_t-solve_a,axis=1))<1e-13:continue
        ca=solve_a.mean(axis=0);cb=solve_t.mean(axis=0)
        u,s,vt=np.linalg.svd((solve_a-ca).T@(solve_t-cb))
        signs=np.eye(3);signs[-1,-1]=np.linalg.det(u@vt)
        R=u@signs@vt;t=cb-ca@R
        out[ids]=model.vertices[ids]@R+t
        if peer is not None:
            # Includes the self-mirrored central accessory. Joint symmetric anchors
            # make R commute with reflection up to floating point roundoff.
            if peer is not binding:out[partner[ids]]=out[ids]*S
            residual=np.r_[np.linalg.norm(a@R+t-target,axis=1),np.linalg.norm((pa*S@R+t)*S-pt,axis=1)]
        else:residual=np.linalg.norm(a@R+t-target,axis=1)
        reports.append(dict(name=name,paired_name=peer['name'] if peer else None,status='rigid_fit',anchors=len(solve_a),
                            anchor_residual_max_m=float(residual.max()),anchor_residual_rms_m=float(np.sqrt(np.mean(residual**2))),
                            rotation_determinant=float(np.linalg.det(R)),
                            symmetry_constraint=mirror,reference_attachment_distance_max_m=binding['reference_distance_max']))
    return out,reports
