"""Conservative surface diagnostics and explicit topology edits.

No fill-all-holes, voxel union, or automatic destruction of physical panel gaps.
All topology edits create a new reference revision, invalidating old face seeds.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .model import Model, array_hash
from .geometry import edges_and_counts, face_geometry
from .quality import triangle_min_angles, baseline_report



def vertex_manifold_check(model):
    """Verify connected incident-face fans and edge incidences at each vertex.

    A single fan with at most two incident faces per edge is a topological
    manifold neighbourhood (disk/half-disk). This does not detect intersections
    between geometrically overlapping but topologically unrelated triangles.
    """
    f=model.faces;n=len(model.vertices);nf=len(f)
    e=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
    owner=np.tile(np.arange(nf),3);order=np.lexsort((e[:,1],e[:,0]));se=e[order]
    match=np.all(se[:-1]==se[1:],axis=1)
    fa=owner[order[:-1][match]];fb=owner[order[1:][match]];edge=se[:-1][match]
    c0=[];c1=[]
    for endpoint in (0,1):
        c0.append(3*fa+np.argmax(f[fa]==edge[:,endpoint,None],axis=1))
        c1.append(3*fb+np.argmax(f[fb]==edge[:,endpoint,None],axis=1))
    a=np.concatenate(c0);b=np.concatenate(c1)
    g=coo_matrix((np.ones(2*len(a),dtype=np.uint8),(np.r_[a,b],np.r_[b,a])),shape=(3*nf,3*nf)).tocsr()
    _,cc=connected_components(g,directed=False)
    pairs=np.unique(np.c_[f.ravel(),cc],axis=0);fans=np.bincount(pairs[:,0],minlength=n)
    unique,counts=edges_and_counts(f)
    bad=np.zeros(n,bool);bad[fans>1]=True;bad[np.unique(unique[counts>2])]=True
    deg=(f[:,0]==f[:,1])|(f[:,0]==f[:,2])|(f[:,1]==f[:,2]);bad[np.unique(f[deg])]=True
    return dict(nonmanifold_vertices=int(bad.sum()),nonmanifold_vertex_ids=np.flatnonzero(bad)[:1000].tolist(),
                unused_vertices=int((fans==0).sum()),method='incident_face_fan_connectivity_and_edge_incidence',
                vertex_manifold=bool(not bad.any()))

def audit_mesh(model:Model,vertices=None,angle_threshold=10.):
    v=model.vertices if vertices is None else np.asarray(vertices,float);f=model.faces
    edges,counts=edges_and_counts(f);boundary=edges[counts==1]
    angle=triangle_min_angles(v,f);_,_,areas=face_geometry(v,f)
    bad=np.flatnonzero(angle<angle_threshold)
    loops=[]
    if len(boundary):
        used,inv=np.unique(boundary,return_inverse=True);ef=inv.reshape(-1,2)
        g=coo_matrix((np.ones(len(ef)*2),(np.r_[ef[:,0],ef[:,1]],np.r_[ef[:,1],ef[:,0]])),shape=(len(used),len(used))).tocsr()
        n,components=connected_components(g,directed=False)
        degree=np.asarray(g.sum(axis=1)).ravel()
        for i in range(n):
            ids=used[components==i];eb=boundary[np.isin(boundary[:,0],ids)]
            faces=np.flatnonzero(np.isin(f,ids).any(axis=1))
            loops.append(dict(id=i,vertex_count=len(ids),edge_count=len(eb),
                              is_simple_closed_chain=bool(np.all(degree[components==i]==2)),
                              perimeter_m=float(np.linalg.norm(v[eb[:,0]]-v[eb[:,1]],axis=1).sum()),
                              bounds_m=[v[ids].min(0).tolist(),v[ids].max(0).tolist()],
                              parts=np.unique(model.labels[faces]).tolist(),classification='unreviewed_do_not_autofill'))
    _,first,counts_faces=np.unique(np.sort(f,axis=1),axis=0,return_index=True,return_counts=True)
    report=baseline_report(model.clone(vertices=v))
    report.update(boundary_chains=loops,bad_angle_threshold_degrees=angle_threshold,
                  bad_angle_face_count=len(bad),angle_percentiles=np.percentile(angle,[0,1,5,50,95]).tolist(),
                  duplicate_index_faces=int(np.sum(counts_faces-1)),
                  bad_face_ids=bad[:5000].tolist(),bad_face_ids_truncated=len(bad)>5000,
                  boundary_segments_m=v[boundary[:5000]].reshape(-1,3).tolist(),
                  schema='mesh-audit-v03',vertex_manifold='not_certified',
                  global_intersection_status='not_checked',
                  readiness='blocked_pending_global_geometry_and_fluid_domain_validation')
    driver=model.metadata.get('driver_vertices')
    if driver is not None:
        skin=np.isin(f,np.asarray(driver,int)).all(axis=1)
        if skin.any():
            report['driver_surface']=dict(faces=int(skin.sum()),min_angle=float(angle[skin].min()),
               below_10_degrees=int((angle[skin]<10).sum()),angle_p05=float(np.percentile(angle[skin],5)))
    report.update(vertex_manifold_check(model))
    report['issue_groups']=[dict(part_id=p['id'],name=p['name'],bad_angle_faces=int(((model.labels==p['id'])&(angle<angle_threshold)).sum()))
                             for p in model.parts if np.any((model.labels==p['id'])&(angle<angle_threshold))]
    return report


def _reindexed(model,v,f,labels,source_faces,old_to_new,provenance):
    meta=json.loads(json.dumps(model.metadata))
    for field in ('face_material_ids','segmentation_face_scores','segmentation_face_methods','semantic_source_labels',
                  'raw_face_labels','source_face_indices'):
        if field in meta:meta[field]=np.asarray(meta[field])[source_faces].tolist()
    if meta.get('semantic_schema')=='aeroshape.design30.v1':
        meta['segmentation']['review']={'status':'pending','reviewer':None}
        meta['annotation_status']='topology_changed_requires_review'
    if 'driver_vertices' in meta:
        ids=old_to_new[np.asarray(meta['driver_vertices'],int)];meta['driver_vertices']=np.unique(ids[ids>=0]).tolist()
    for binding in meta.get('rigid_followers',[]):
        ids=old_to_new[np.asarray(binding['vertices'],int)];binding['vertices']=np.unique(ids[ids>=0]).tolist()
    meta['parent_geometry_hash']=model.metadata['base_geometry_hash']
    meta['base_geometry_hash']=array_hash(v,f)
    meta.setdefault('topology_history',[]).append(provenance)
    meta['recipe_seed_status']='old_face_seed_recipes_require_reselection'
    out=Model(v,f,labels,model.parts,meta)
    if meta.get('semantic_schema')=='aeroshape.design30.v1':
        from .semantic import refresh_summary
        refresh_summary(out)
    return out


def clean_mesh(model:Model):
    """Remove index/area degeneracies and exact duplicate index faces, no welding."""
    f=model.faces;v=model.vertices
    _,_,area=face_geometry(v,f);tol=float(np.ptp(v,axis=0).max())**2*1e-14
    keep=(area>tol)&(f[:,0]!=f[:,1])&(f[:,1]!=f[:,2])&(f[:,0]!=f[:,2])
    candidates=np.flatnonzero(keep);keys=np.sort(f[candidates],axis=1)
    _,first,inv=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    # Conflicting labels on identical faces must be resolved by the user.
    for i in np.flatnonzero(np.bincount(inv)>1):
        if len(np.unique(model.labels[candidates[inv==i]]))>1:
            raise ValueError('Duplicate faces carry different semantic labels. Resolve labels before cleaning.')
    source=np.sort(candidates[first]);nf=f[source]
    used,remap=np.unique(nf,return_inverse=True)
    old_to_new=np.full(len(v),-1,int);old_to_new[used]=np.arange(len(used))
    nf=remap.reshape(-1,3).astype(np.int32)
    result=_reindexed(model,v[used],nf,model.labels[source],source,old_to_new,
                      dict(action='remove_degenerate_and_duplicate_index_faces',removed_faces=len(f)-len(nf),welded_vertices=0))
    mesh=result.mesh();mesh.fix_normals(multibody=True)
    result=result.clone(faces=np.asarray(mesh.faces,dtype=np.int32))
    result.metadata['base_geometry_hash']=array_hash(result.vertices,result.faces)
    return result


def stitch_parts(model:Model,part_a:int,part_b:int,tolerance=1e-6):
    """Weld only mutually matched boundary vertices of TWO explicit parts.

    Unequal edge tessellation is rejected: this does not synthesize zipper faces.
    Physical gaps should be preserved by displacement ties instead of welding.
    """
    if part_a==part_b or not 0<tolerance<=.001:raise ValueError('Choose two different parts and a tolerance in (0, 1 mm].')
    def part_boundary(part):
        ff=model.faces[model.labels==part]
        if not len(ff):raise ValueError('Unknown or empty part.')
        e,c=edges_and_counts(ff);return np.unique(e[c==1])
    aa=part_boundary(part_a);bb=part_boundary(part_b)
    common=np.intersect1d(aa,bb);aa=np.setdiff1d(aa,common);bb=np.setdiff1d(bb,common)
    if not len(aa) or not len(bb):raise ValueError('No distinct open boundary vertices to stitch. Shared interfaces already have C0 continuity.')
    da,ia=cKDTree(model.vertices[bb]).query(model.vertices[aa]);db,ib=cKDTree(model.vertices[aa]).query(model.vertices[bb])
    choose=(da<=tolerance)&(ib[ia]==np.arange(len(aa)))
    pairs=np.c_[aa[choose],bb[ia[choose]]]
    if len(pairs)<2:raise ValueError('No mutually matching seam edge. Tolerance is not a hole-repair radius.')
    v=model.vertices.copy();old_to_new=np.arange(len(v));old_to_new[pairs[:,1]]=pairs[:,0]
    v[pairs[:,0]]=(v[pairs[:,0]]+v[pairs[:,1]])*.5
    f=old_to_new[model.faces]
    used,inv=np.unique(f,return_inverse=True)
    compact=np.full(len(v),-1,int);compact[used]=np.arange(len(used));old_to_new=compact[old_to_new]
    result=_reindexed(model,v[used],inv.reshape(-1,3).astype(np.int32),model.labels,np.arange(len(f)),old_to_new,
                      dict(action='explicit_pair_boundary_stitch',parts=[part_a,part_b],tolerance_m=tolerance,matched_vertices=len(pairs)))
    before=baseline_report(model);after=baseline_report(result)
    if after['nonmanifold_edges']>before['nonmanifold_edges'] or after['degenerate_faces']>before['degenerate_faces'] or after['boundary_edges']>=before['boundary_edges']:
        raise ValueError('Stitch failed topology checks; the original mesh is unchanged. Matching vertex samples are required.')
    return result


def improve_triangulation(model:Model,passes=2,plane_tolerance=.0001):
    """Quality-improving edge flips: unchanged vertices, labels/material barriers.

    This is a limited diagonal optimizer, NOT isotropic remeshing. Each candidate
    must be nearly planar, preserve orientation, and improve its minimum angle.
    """
    if passes not in (1,2,3,4):raise ValueError('Use 1..4 local passes.')
    f=model.faces.copy();v=model.vertices;total=0;largest_deviation=0.
    mat=np.asarray(model.metadata.get('face_material_ids',model.labels))
    source=np.asarray(model.metadata.get('semantic_source_labels',model.labels))
    raw_source=np.asarray(model.metadata.get('raw_face_labels',source))
    methods=np.asarray(model.metadata.get('segmentation_face_methods',np.zeros(len(f),int)))
    scores=np.asarray(model.metadata.get('segmentation_face_scores',np.zeros(len(f))),float).copy()
    for _ in range(passes):
        mesh=model.clone(faces=f).mesh();adj=mesh.face_adjacency;adj_e=mesh.face_adjacency_edges
        angle=triangle_min_angles(v,f);_,normals,_=face_geometry(v,f)
        same=(model.labels[adj[:,0]]==model.labels[adj[:,1]])&(mat[adj[:,0]]==mat[adj[:,1]])
        same&=(source[adj[:,0]]==source[adj[:,1]])&(methods[adj[:,0]]==methods[adj[:,1]])
        same&=raw_source[adj[:,0]]==raw_source[adj[:,1]]
        small=np.minimum(angle[adj[:,0]],angle[adj[:,1]])<18
        flat=(normals[adj[:,0]]*normals[adj[:,1]]).sum(1)>.996
        ids=np.flatnonzero(same&small&flat)
        ids=ids[np.argsort(np.minimum(angle[adj[ids,0]],angle[adj[ids,1]]))]
        touched=set();edges=set(map(tuple,edges_and_counts(f)[0]));count=0
        for i in ids:
            fa,fb=map(int,adj[i]);a,b=map(int,adj_e[i])
            if fa in touched or fb in touched:continue
            c=next(int(x) for x in f[fa] if x not in (a,b));d=next(int(x) for x in f[fb] if x not in (a,b))
            if c==d or tuple(sorted((c,d))) in edges:continue
            deviation=abs(float((v[d]-v[a])@normals[fa]))
            if deviation>plane_tolerance:continue
            candidate=np.array([[c,d,a],[d,c,b]],np.int32)
            _,nn,_=face_geometry(v,candidate)
            for j in range(2):
                if nn[j]@normals[fa]<0:candidate[j]=candidate[j,[0,2,1]]
            _,nn,_=face_geometry(v,candidate)
            if (nn@normals[fa]).min()<.99:continue
            aa=triangle_min_angles(v,candidate)
            if aa.min()<=min(angle[fa],angle[fb])+.25:continue
            # Do not accept a huge area change, inverted quad, or overlapping triangles.
            _,_,new_area=face_geometry(v,candidate);_,_,old_area=face_geometry(v,f[[fa,fb]])
            if abs(float(new_area.sum()/old_area.sum())-1)>.002:continue
            f[fa],f[fb]=candidate;edges.remove(tuple(sorted((a,b))));edges.add(tuple(sorted((c,d))))
            scores[fa]=scores[fb]=min(scores[fa],scores[fb])
            touched.update((fa,fb));count+=1;largest_deviation=max(largest_deviation,deviation)
        total+=count
        if count==0:break
    out=model.clone(faces=f)
    if total and 'source_face_indices' in out.metadata:
        out.metadata.pop('source_face_indices')
        out.metadata['source_face_correspondence_status']='invalidated_by_retriangulation'
    out.metadata['parent_geometry_hash']=model.metadata['base_geometry_hash'];out.metadata['base_geometry_hash']=array_hash(v,f)
    out.metadata.setdefault('topology_history',[]).append(dict(action='protected_near_planar_diagonal_flips',flips=total,
        max_opposite_plane_deviation_m=largest_deviation,vertex_motion_m=0.,passes=passes,
        global_Hausdorff_bound='not_measured',labels_preserved=True,materials_preserved=True))
    out.metadata['recipe_seed_status']='old_face_seed_recipes_require_reselection'
    if out.metadata.get('semantic_schema')=='aeroshape.design30.v1':
        from .semantic import refresh_summary
        out.metadata['segmentation_face_scores']=scores.tolist()
        out.metadata['annotation_status']='retriangulated_requires_review'
        refresh_summary(out)
    before=baseline_report(model);after=baseline_report(out)
    if after['boundary_edges']!=before['boundary_edges'] or after['nonmanifold_edges']!=before['nonmanifold_edges'] or not after['winding_consistent']:
        raise ValueError('Local retriangulation failed topology checks; original mesh unchanged.')
    return out
