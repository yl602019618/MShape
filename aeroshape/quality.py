from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .geometry import edges_and_counts, face_geometry
from .model import Model
from .deform import Deformation, DesignRequest


def reflection_residual(v):
    q=v.copy(); q[:,1]*=-1
    dist=cKDTree(v).query(q)[0]
    return dict(mean_m=float(dist.mean()), max_m=float(dist.max()), p95_m=float(np.percentile(dist,95)),
                method='nearest_reflected_vertex_not_continuous_surface_distance')


def triangle_min_angles(v,f):
    t=v[f]
    a=t[:,1]-t[:,0]; b=t[:,2]-t[:,0]; c=t[:,2]-t[:,1]
    lens=[np.linalg.norm(x,axis=1) for x in (a,b,c)]
    cos_a=np.einsum('ij,ij->i',a,b)/np.maximum(lens[0]*lens[1],1e-30)
    cos_b=np.einsum('ij,ij->i',-a,c)/np.maximum(lens[0]*lens[2],1e-30)
    aa=np.arccos(np.clip(cos_a,-1,1)); bb=np.arccos(np.clip(cos_b,-1,1))
    return np.rad2deg(np.minimum(np.minimum(aa,bb),np.maximum(np.pi-aa-bb,0)))



def surface_stretches(reference,vertices,faces):
    """Principal stretches of each triangle's 2D tangent map into 3D.

    Uses every supplied triangle, not off-surface samples. No claim about
    volume Jacobians or distant self-intersections follows from these values.
    """
    t0=np.asarray(reference)[faces];t1=np.asarray(vertices)[faces]
    e0=t0[:,1]-t0[:,0];e1=t0[:,2]-t0[:,0]
    length=np.linalg.norm(e0,axis=1)
    along=np.einsum('ij,ij->i',e0,e1)/np.maximum(length,1e-30)
    height=np.linalg.norm(np.cross(e0,e1),axis=1)/np.maximum(length,1e-30)
    a=(t1[:,1]-t1[:,0])/np.maximum(length[:,None],1e-30)
    b=((t1[:,2]-t1[:,0])-a*along[:,None])/np.maximum(height[:,None],1e-30)
    g00=np.einsum('ij,ij->i',a,a);g11=np.einsum('ij,ij->i',b,b)
    g01=np.einsum('ij,ij->i',a,b)
    disc=np.sqrt(np.maximum((g00-g11)**2+4*g01*g01,0))
    small=np.sqrt(np.maximum((g00+g11-disc)*.5,0))
    large=np.sqrt(np.maximum((g00+g11+disc)*.5,0))
    return small,large

def baseline_report(model: Model):
    _,n,a=face_geometry(model.vertices,model.faces)
    edges,counts=edges_and_counts(model.faces)
    length=max(np.ptp(model.vertices,axis=0))
    return dict(vertices=len(model.vertices), faces=len(model.faces),
                parts_with_faces=len(np.unique(model.labels)),
                bounds_m=[model.vertices.min(axis=0).tolist(),model.vertices.max(axis=0).tolist()],
                dimensions_m=np.ptp(model.vertices,axis=0).tolist(),
                boundary_edges=int((counts==1).sum()), nonmanifold_edges=int((counts>2).sum()),
                watertight_edge_incidence=bool(np.all(counts==2)),
                winding_consistent=bool(model.mesh().is_winding_consistent),
                degenerate_faces=int((a<length**2*1e-14).sum()),
                min_angle_degrees=float(triangle_min_angles(model.vertices,model.faces).min()),
                reflection=reflection_residual(model.vertices),
                self_intersections='not_checked', inter_component_collisions='not_checked',
                cfd_ready=False)


def check_design(model: Model, vertices, field: Deformation, request: DesignRequest):
    v=np.asarray(vertices,float); f=model.faces
    errors=[]; warnings=[]
    if not np.isfinite(v).all():
        return dict(screen_pass=False, errors=['Non-finite vertices'], warnings=[], cfd_ready=False)
    _,n0,a0=face_geometry(model.vertices,f)
    _,n1,a1=face_geometry(v,f)
    edges,counts=edges_and_counts(f)
    l0=np.linalg.norm(model.vertices[edges[:,0]]-model.vertices[edges[:,1]],axis=1)
    l1=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
    stretch=l1/np.maximum(l0,1e-20)
    area_ratio=a1/np.maximum(a0,1e-30)
    flipped=int(((n0*n1).sum(axis=1)<=0).sum())
    displacement=np.linalg.norm(v-model.vertices,axis=1)
    length=np.ptp(model.vertices,axis=0).max()
    degenerate=int((a1 < length**2*1e-14).sum())
    jac=field.jacobian_sample()
    locked_motion=float(displacement[field.locked].max()) if field.locked.any() else 0.
    wheel_ids=[p['id'] for p in model.parts if p.get('category')=='wheels' or any(k in (p.get('key','')+' '+p.get('name','')).lower()
                                                for k in ('wheel','tire','tyre','轮胎','车轮'))]
    body_vertices=np.unique(f[~np.isin(model.labels,wheel_ids)])
    clearance=float(v[body_vertices,2].min()) if len(body_vertices) else float(v[:,2].min())
    if degenerate: errors.append(f'{degenerate} degenerate triangles.')
    if flipped: errors.append(f'{flipped} faces rotated through >=90 degrees relative to the reference.')
    if float(area_ratio.min()) < .12: errors.append('Triangle area collapsed below 12% of reference.')
    if stretch.min()<.25 or stretch.max()>4: errors.append('Edge stretch outside [0.25, 4].')
    if displacement.max()>request.max_displacement+1e-10: errors.append('Maximum displacement exceeded.')
    if locked_motion>1e-8: errors.append('A locked part moved.')
    if v[:,2].min() < -1e-8: errors.append('Geometry crosses the ground plane Z=0.')
    if clearance < request.min_body_clearance-1e-8: errors.append('Minimum non-wheel ground clearance violated.')
    if jac['min_det'] is not None and jac['min_det'] < request.min_jacobian: errors.append('Sampled deformation Jacobian below the threshold.')
    if np.any(counts!=2): warnings.append('Master surface is not a closed edge-manifold; inspect holes and nonmanifold edges.')
    if jac['max_gradient_norm'] is not None and jac['max_gradient_norm']>=1: warnings.append('Sampled displacement gradient >=1; the small-gradient sufficient condition is not met.')
    if min(triangle_min_angles(v,f))<10: warnings.append('Triangles below 10 degrees remain; surface remeshing may be required.')
    warnings.append('Global self-intersection, assembly clearance, and manufacturability are not certified.')
    warnings.append('A passed screen is not approval for CFD: repair and validate the external fluid mesh separately.')
    regional=[region['info'] for _,region in field.regions]
    attachments=field.attachment_report
    tangent_strain=None
    if field.regions:
        use_faces=f
        if 'driver_vertices' in model.metadata:
            use_faces=f[np.isin(f,model.metadata['driver_vertices']).all(axis=1)]
        smin,smax=surface_stretches(model.vertices,v,use_faces)
        tangent_strain=dict(principal_min=float(smin.min()),principal_max=float(smax.max()),
                           triangles_checked=len(use_faces),scope='all_driver_triangles_tangent_maps_not_volume')
        max_strain=min(op.region.max_strain for op,_ in field.regions)
        # All triangle edge directions, not a sampled off-surface NN Jacobian.
        if stretch.min()<1-max_strain or stretch.max()>1+max_strain:
            errors.append(f'Regional edge strain exceeds the configured {max_strain:.0%} limit.')
        if smin.min()<1-max_strain or smax.max()>1+max_strain:
            errors.append(f'Regional principal surface strain exceeds the configured {max_strain:.0%} limit.')
        tolerance=min(op.region.max_attachment_residual for op,_ in field.regions)
        if any(p.get('anchor_residual_max_m',0)>tolerance or p.get('status')=='binding_failed' for p in attachments):
            errors.append('A rigid attachment cannot follow within the configured anchor residual tolerance.')
        # Surface normal reversal is conservative for large rotations, not an intersection test.
    return dict(screen_pass=not errors,surface_strain=tangent_strain,regional_handles=regional,attachments=attachments, errors=errors, warnings=warnings,
                max_displacement_m=float(displacement.max()), min_body_clearance_m=clearance,
                locked_motion_max_m=locked_motion, degenerate_faces=degenerate,
                normal_reversal_faces=flipped, min_area_ratio=float(area_ratio.min()),
                edge_stretch_min=float(stretch.min()), edge_stretch_max=float(stretch.max()),
                min_angle_degrees=float(triangle_min_angles(v,f).min()),
                sampled_jacobian=jac, reflection=reflection_residual(v),
                boundary_edges=int((counts==1).sum()), nonmanifold_edges=int((counts>2).sum()),
                self_intersections='not_checked', inter_component_collisions='not_checked',
                cfd_ready=False, validation_level='local_surface_screen_only')


def evaluate(model: Model, request: DesignRequest):
    field=Deformation(model,request.operations)
    vertices=field.apply()
    return vertices,check_design(model,vertices,field,request)
