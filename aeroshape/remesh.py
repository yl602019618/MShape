"""Conservative LOCAL split/collapse/flip/relax/project surface remesher.

The design master is immutable. This creates a separate simulation-surface
candidate with an explicit barycentric correspondence. Boundary, semantic and
material interface vertices are pinned. It does not claim an exact Hausdorff
bound, CFD volume meshing, or identical left/right tessellation.
"""
from __future__ import annotations
import copy,time
import numpy as np
from pydantic import BaseModel,Field,ConfigDict
from .model import Model,array_hash
from .geometry import face_geometry,edges_and_counts
from .mesh_tools import triangle_min_angles
from .spatial import TriangleIndex

class RemeshOptions(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    target_edge_m:float=Field(.035,ge=.008,le=.12)
    iterations:int=Field(2,ge=1,le=4)
    max_deviation_m:float=Field(.002,ge=.0001,le=.01)
    max_region_faces:int=Field(30000,ge=10,le=60000)
    max_new_vertices:int=Field(15000,ge=1,le=50000)


def edge_table(f):
    edges=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);edges.sort(1)
    unique,inv,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    faces=np.tile(np.arange(len(f)),3);order=np.argsort(inv,kind='stable');offset=np.r_[0,np.cumsum(counts)]
    return unique,counts,faces[order],offset


def constraints(f,labels,mats,selected,nv):
    edges,counts,incident,offset=edge_table(f);protect=np.zeros(len(edges),bool)
    for k in range(len(edges)):
        ids=incident[offset[k]:offset[k+1]]
        protect[k]=(counts[k]!=2 or not selected[ids].all() or len(set(labels[ids]))!=1 or len(set(mats[ids]))!=1)
    frozen=np.zeros(nv,bool);frozen[np.unique(edges[protect])]=True
    return edges,counts,incident,offset,protect,frozen


def quality(v,f,selected):
    a=triangle_min_angles(v,f[selected]);edges,_=edges_and_counts(f[selected]);length=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
    return dict(faces=int(selected.sum()),min_angle_deg=float(a.min()),p10_min_angle_deg=float(np.percentile(a,10)),
                faces_below_10_deg=int((a<10).sum()),fraction_below_10_deg=float((a<10).mean()),
                median_edge_m=float(np.median(length)),p95_edge_m=float(np.percentile(length,95)))


def local_remesh(model:Model,face_mask,options:RemeshOptions|dict):
    opt=options if isinstance(options,RemeshOptions) else RemeshOptions.model_validate(options)
    start=time.perf_counter();selected=np.asarray(face_mask,bool).copy()
    if selected.shape!=(len(model.faces),) or not selected.any():raise ValueError('Select a nonempty local surface region for remeshing.')
    if selected.sum()>opt.max_region_faces:raise ValueError(f'Local region has {selected.sum()} faces; limit is {opt.max_region_faces}. Select a smaller region.')
    # Only a continuous driver skin is remeshed. Accessories and tyre assemblies stay unchanged.
    driver=np.zeros(len(model.vertices),bool);driver[model.metadata.get('driver_vertices',list(range(len(driver))))]=True
    selected &= driver[model.faces].all(1)
    if not selected.any():raise ValueError('The selection has no driver-surface triangles.')
    original_selected=selected.copy();orig_v=model.vertices.copy();orig_f=model.faces.copy()
    v=orig_v.copy();f=orig_f.copy();labels=model.labels.copy()
    mats=np.asarray(model.metadata.get('face_material_ids',np.zeros(len(f))),np.int32).copy()
    parent=np.arange(len(f),dtype=np.int64)
    before=quality(v,f,selected)
    # Projection index contains just the selected original patch, not unrelated close sheets.
    source_fids=np.flatnonzero(selected);surface=TriangleIndex(orig_v,orig_f[source_fids])
    pinned_positions={};total_split=total_collapse=total_flip=0;created=0
    for iteration in range(opt.iterations):
        edges,counts,incident,offset,protected,frozen=constraints(f,labels,mats,selected,len(v))
        if iteration==0:
            initial_fixed=np.flatnonzero(frozen);pinned_positions={int(i):v[i].copy() for i in initial_fixed}
        length=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
        ids=np.flatnonzero(~protected&(length>opt.target_edge_m*4/3))
        if created+len(ids)>opt.max_new_vertices:raise ValueError('Refinement would exceed the vertex budget; increase target edge length.')
        mid={};new=[]
        for k in ids:
            a,b=map(int,edges[k]);mid[(a,b)]=len(v)+len(new);new.append((v[a]+v[b])/2)
        if new:
            v=np.vstack((v,new));nf=[];nl=[];nm=[];ns=[];np_= []
            for i,(a,b,c) in enumerate(f):
                a,b,c=int(a),int(b),int(c)
                ab=mid.get(tuple(sorted((a,b))));bc=mid.get(tuple(sorted((b,c))));ca=mid.get(tuple(sorted((c,a))))
                split_count=sum(x is not None for x in (ab,bc,ca))
                if split_count==0:tri=[(a,b,c)]
                elif split_count==3:tri=[(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]
                elif split_count==1:
                    if ab is not None:tri=[(a,ab,c),(ab,b,c)]
                    elif bc is not None:tri=[(b,bc,a),(bc,c,a)]
                    else:tri=[(c,ca,b),(ca,a,b)]
                else:
                    if ab is None:a,b,c=b,c,a;ab,bc,ca=bc,ca,ab
                    elif bc is None:a,b,c=c,a,b;ab,bc,ca=ca,ab,bc
                    # ab and bc present; ca absent
                    tri=[(ab,b,bc),(a,ab,c),(ab,bc,c)]
                nf.extend(tri);nl.extend([labels[i]]*len(tri));nm.extend([mats[i]]*len(tri));ns.extend([selected[i]]*len(tri));np_.extend([parent[i]]*len(tri))
            f=np.asarray(nf,np.int32);labels=np.asarray(nl,np.int32);mats=np.asarray(nm,np.int32);selected=np.asarray(ns,bool);parent=np.asarray(np_,np.int64)
            total_split+=len(ids);created+=len(ids)
        # Collapse only interior edges with a valid link and noninverted surviving faces.
        edges,counts,incident,offset,protected,frozen=constraints(f,labels,mats,selected,len(v))
        length=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
        candidates=np.flatnonzero(~protected&~frozen[edges].any(1)&(length<.72*opt.target_edge_m))
        candidates=candidates[np.argsort(length[candidates])]
        vertex_faces=[set() for _ in range(len(v))]
        for face,t in enumerate(f):
            for x in t:vertex_faces[int(x)].add(face)
        alive=np.ones(len(f),bool);touched=set();collapse_count=0
        for k in candidates:
            a,b=map(int,edges[k])
            if a in touched or b in touched:continue
            af=vertex_faces[a];bf=vertex_faces[b];shared=af&bf
            if len(shared)!=2:continue
            na=set(f[list(af)].ravel())-{a};nb=set(f[list(bf)].ravel())-{b}
            opposite=set(f[list(shared)].ravel())-{a,b}
            if na&nb!=opposite:continue
            impacted=sorted((af|bf)-shared)
            if not impacted:continue
            old=f[impacted].copy();candidate=old.copy();candidate[candidate==b]=a
            newpos=(v[a]+v[b])/2
            projected,_,_=surface.closest(newpos[None]);newpos=projected[0]
            oldv=v[a].copy();v[a]=newpos
            _,nn,aa=face_geometry(v,candidate)
            v[a]=oldv;_,on,oa=face_geometry(v,old)
            if (aa<np.maximum(oa*.1,1e-14)).any() or ((nn*on).sum(1)<.7).any():continue
            # Keep local edge lengths bounded to prevent over-collapse.
            v[a]=newpos
            if triangle_min_angles(v,candidate).min()<.5:v[a]=oldv;continue
            for face in af|bf:
                for vertex in f[face]:vertex_faces[int(vertex)].discard(face)
            for face in shared:alive[face]=False
            f[impacted]=candidate
            for face in impacted:
                for vertex in f[face]:vertex_faces[int(vertex)].add(face)
            touched.update(na|nb|{a,b});collapse_count+=1
        f=f[alive];labels=labels[alive];mats=mats[alive];selected=selected[alive];parent=parent[alive]
        total_collapse+=collapse_count
        # Flip for quality only, preserving orientation and near-planar surface error.
        edges,counts,inc,off,protected,frozen=constraints(f,labels,mats,selected,len(v))
        angle=triangle_min_angles(v,f);_,normal,areas=face_geometry(v,f)
        touched_faces=set();existing=set(map(tuple,edges));flips=0
        for k in np.flatnonzero(~protected):
            ia,ib=map(int,inc[off[k]:off[k+1]])
            if ia in touched_faces or ib in touched_faces:continue
            a,b=map(int,edges[k]);c=next(int(x) for x in f[ia] if x not in (a,b));d=next(int(x) for x in f[ib] if x not in (a,b))
            if c==d or tuple(sorted((c,d))) in existing:continue
            if normal[ia]@normal[ib]<.995:continue
            if abs((v[d]-v[a])@normal[ia])>opt.max_deviation_m*.25:continue
            cand=np.array([[c,d,a],[d,c,b]],np.int32);_,nn,aa=face_geometry(v,cand)
            for j in range(2):
                if nn[j]@normal[ia]<0:cand[j]=cand[j,[0,2,1]]
            _,nn,aa=face_geometry(v,cand)
            if (nn@normal[ia]).min()<.995 or abs(aa.sum()/(areas[ia]+areas[ib])-1)>.005:continue
            q=triangle_min_angles(v,cand)
            if q.min()<=min(angle[ia],angle[ib])+.15:continue
            f[[ia,ib]]=cand;existing.discard(tuple(sorted((a,b))));existing.add(tuple(sorted((c,d))));touched_faces.update((ia,ib));flips+=1
        total_flip+=flips
        # Tangential relaxation projected to the original patch. Vertices on ANY
        # semantic/material/selection boundary remain pinned exactly.
        edges,_,_,_,_,frozen=constraints(f,labels,mats,selected,len(v))
        referenced=np.zeros(len(v),bool);referenced[np.unique(f)]=True
        movable=np.flatnonzero(~frozen&referenced)
        if len(movable):
            sums=np.zeros_like(v);degree=np.zeros(len(v));np.add.at(sums,edges[:,0],v[edges[:,1]]);np.add.at(sums,edges[:,1],v[edges[:,0]])
            np.add.at(degree,edges[:,0],1);np.add.at(degree,edges[:,1],1)
            normals=np.zeros_like(v);_,fn,fa=face_geometry(v,f)
            for k in range(3):np.add.at(normals,f[:,k],fn*fa[:,None])
            normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-30)
            delta=sums[movable]/degree[movable,None]-v[movable]
            delta-=normals[movable]*np.einsum('ij,ij->i',delta,normals[movable])[:,None]
            proposed=v[movable]+.30*delta;projected,_,_=surface.closest(proposed)
            # Backtracking checks all faces, not just the moved vertex positions.
            accepted=False;old=v[movable].copy();_,n0,a0=face_geometry(v,f)
            for factor in (1.,.5,.25,.125):
                v[movable]=old+factor*(projected-old);_,n1,a1=face_geometry(v,f)
                if (a1>=np.maximum(a0*.2,1e-16)).all() and ((n1*n0).sum(1)>.5).all():accepted=True;break
            if not accepted:v[movable]=old
    # Compact only now, so every pinned original ID can be checked first.
    for i,p in pinned_positions.items():
        if not np.array_equal(v[i],p):raise ValueError('A protected boundary moved; result rejected.')
    used,inverse=np.unique(f,return_inverse=True);f=inverse.reshape(-1,3).astype(np.int32);v=v[used]
    # Exact closest-point barycentric mapping for vertex field transfer. Not identity IDs.
    # Unchanged original vertices retain a known incident face; do not map an
    # accessory to a nearby body triangle just because it is spatially close.
    sfid=np.empty(len(v),np.int64);bary=np.zeros((len(v),3));sampled_dist=np.zeros(len(v))
    known=used<len(orig_v);unchanged=known.copy()
    unchanged[known]=np.linalg.norm(v[known]-orig_v[used[known]],axis=1)<1e-14
    incident=np.full(len(orig_v),len(orig_f),np.int64)
    for corner in range(3):np.minimum.at(incident,orig_f[:,corner],np.arange(len(orig_f)))
    sfid[unchanged]=incident[used[unchanged]]
    bary[unchanged]=(orig_f[sfid[unchanged]]==used[unchanged,None]).astype(float)
    changed=~unchanged
    if changed.any():
        q,localfid,bc=surface.closest(v[changed]);sfid[changed]=source_fids[localfid];bary[changed]=bc
        sampled_dist[changed]=np.linalg.norm(v[changed]-q,axis=1)
    local_new=TriangleIndex(v,f[selected])
    new_tri=v[f[selected]];old_tri=orig_v[orig_f[original_selected]]
    new_samples=np.vstack((new_tri.mean(1),(new_tri[:,0]+new_tri[:,1])*.5,(new_tri[:,1]+new_tri[:,2])*.5,(new_tri[:,2]+new_tri[:,0])*.5))
    old_samples=np.vstack((old_tri.reshape(-1,3),old_tri.mean(1)))
    q,_,_=surface.closest(new_samples);q2,_,_=local_new.closest(old_samples)
    forward=float(np.linalg.norm(q-new_samples,axis=1).max());reverse=float(np.linalg.norm(q2-old_samples,axis=1).max())
    deviation=max(float(sampled_dist.max()),forward,reverse)
    if deviation>opt.max_deviation_m:raise ValueError(f'Sampled bidirectional deviation {deviation*1000:.3f} mm exceeds limit; original design unchanged.')
    before_edges,before_count=edges_and_counts(orig_f);after_edges,after_count=edges_and_counts(f)
    if np.sum(before_count==1)!=np.sum(after_count==1) or np.sum(after_count>2)>np.sum(before_count>2):raise ValueError('Remeshing worsened edge topology; result rejected.')
    meta=copy.deepcopy(model.metadata)
    for key in ('driver_vertices','rigid_followers','seam_ties','segmentation_face_scores','segmentation_face_methods','semantic_source_labels','segmentation',
                'raw_face_labels','source_face_indices'):
        meta.pop(key,None)
    meta.update(base_geometry_hash=array_hash(v,f),parent_geometry_hash=array_hash(orig_v,orig_f),
                face_material_ids=mats.tolist(),geometry_role='remeshed_simulation_surface_candidate',
                canonical_topology_preserved=False,annotation_status='inherited_labels_new_tessellation_requires_review',
                correspondence='see correspondence.npz; source_triangle + barycentric_weights',
                recipe_seed_status='invalid_after_remeshing_keep_editing_original_design_master',cfd_ready=False)
    # Preserve authored labels, but do not pretend a stale review is still current.
    meta.pop('semantic_schema',None)
    result=Model(v,f,labels,copy.deepcopy(model.parts),meta)
    report=dict(method='protected_local_split_collapse_flip_tangential_relax_project',options=opt.model_dump(),
        before=before,after=quality(v,f,selected),split_edges=total_split,collapsed_edges=total_collapse,flipped_edges=total_flip,
        pinned_original_vertices=len(pinned_positions),pinned_motion_m=0.,outside_selection_unchanged=True,
        max_sampled_bidirectional_deviation_m=deviation,exact_Hausdorff_bound=False,
        canonical_topology_preserved=False,master_geometry_unchanged=True,elapsed_s=time.perf_counter()-start,
        symmetry='original_shape_within_sampled_tolerance_not_mirrored_tessellation',self_intersections='requires_post_check',cfd_ready=False)
    return result,dict(source_face_id=sfid,source_barycentric=bary,source_distance_m=sampled_dist,face_parent_id=parent),report
