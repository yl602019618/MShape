"""Component-aware, finite-dimensional reference fitting for labeled sedans."""
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=1)
def authored_exterior_mask():
    # The design30 IDs group wheel-house liners with fenders. Use the original
    # authored membership, not colors or a view-dependent visibility heuristic.
    from .detailed import _build
    source=_build(design_skin=True)
    remove={'seals','hood_trim','roof_trim','rear_trim'}
    keep=~np.isin(source.labels,[p['id'] for p in source.parts if p['key'] in remove])
    internal=[p['id'] for p in source.parts if p['key'] in ('underbody','diffuser','splitter') or
        p['key'].startswith(('liner_', 'mirror', 'tire_', 'rim_', 'brake_'))]
    return ~np.isin(source.labels[keep],internal)

def exterior_mask(model, *, reference=False):
    if reference:
        names=model.metadata.get('raw_label_names',[])
        raw=np.asarray(model.metadata.get('raw_face_labels',[]))
        if raw.shape!=(len(model.faces),) or not names:
            raise ValueError('精细组件拟合需要与三角面精确对应的原始分割标签。')
        internal=[i for i,n in enumerate(names) if n.lower().startswith(('underbody','mirror','wheel','tire','tyre'))]
        return ~np.isin(raw,internal)
    from .engineering import reference as authored
    base=authored();mask=authored_exterior_mask()
    if model.metadata.get('generator_parameters',{}).get('spoiler_style')=='none':
        mask=mask[base.labels!=9]
    if len(mask)!=len(model.faces):raise ValueError('Exterior evaluation requires the authored topology.')
    return mask

def area_constraints(model, basis, prepared, threshold=.55, minimum=.275):
    """Linearized oriented triangle areas, all near-bound faces (not samples)."""
    f=model.faces
    tri=model.vertices[f];a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0]
    cross=np.cross(a,b);area=np.linalg.norm(cross,axis=1)*.5
    ids=np.flatnonzero(area/prepared['areas']<threshold)
    a=a[ids];b=b[ids];f=f[ids];normal=cross[ids]/np.maximum(2*area[ids,None],1e-15)
    da=basis[f[:,1]]-basis[f[:,0]];db=basis[f[:,2]]-basis[f[:,0]]
    derivative=.5*np.einsum('ni,nij->nj',normal,
        np.cross(da,b[:,:,None],axisa=1,axisb=1,axisc=1)+np.cross(a[:,:,None],db,axisa=1,axisb=1,axisc=1))
    return derivative/prepared['areas'][ids,None],minimum-area[ids]/prepared['areas'][ids]

def shape_constraints(model, basis, prepared):
    """Conservative local QP constraints; exact guards still accept each step."""
    from .parametric_quality import _topology
    from .windscreen import profile_vertices
    D,lower=area_constraints(model,basis,prepared);upper=np.full(len(D),np.inf)
    f=_topology()['driver_faces'];tri=model.vertices[f]
    a=(tri[:,1]-tri[:,0])/prepared['length'][:,None]
    b=((tri[:,2]-tri[:,0])-a*prepared['along'][:,None])/prepared['height'][:,None]
    tangent=np.stack((a,b),axis=2)
    u,s,vt=np.linalg.svd(tangent,full_matrices=False)
    selected=np.flatnonzero((s[:,0]>2.15)|(s[:,1]<.42))
    ff=f[selected]
    da=(basis[ff[:,1]]-basis[ff[:,0]])/prepared['length'][selected,None,None]
    db=((basis[ff[:,2]]-basis[ff[:,0]])-da*prepared['along'][selected,None,None])/prepared['height'][selected,None,None]
    change=np.stack((da,db),axis=2)
    rows=[D];lo=[lower];hi=[upper]
    for k in (0,1):
        derivative=np.einsum('ni,nijk,nj->nk',u[selected,:,k],change,vt[selected,k,:])
        rows.append(derivative);lo.append(.32-s[selected,k]);hi.append(2.45-s[selected,k])
    for ids in profile_vertices().values():
        points=model.vertices[ids][:,[0,2]];a=points[0];delta=points[-1]-a
        norm=np.linalg.norm(delta);normal=np.array([-delta[1],delta[0]])/norm
        t=(points-a)@delta/(norm*norm)
        b=basis[ids][:,[0,2]];gradient=np.einsum('i,nij->nj',normal,b-(1-t)[:,None,None]*b[0]-t[:,None,None]*b[-1])
        rows.append(gradient);lo.append(-.0005-(points-a)@normal);hi.append(np.full(len(ids),np.inf))
    return np.concatenate(rows),np.concatenate(lo),np.concatenate(hi)

def infer_arches(model):
    """Infer axle circles from the two upper exterior wheel-opening boundaries.

    These are body openings, not measured tyres. Require a well-supported arc
    in both axles; unknown geometry is an error, never an invented measurement.
    """
    from scipy.optimize import least_squares
    f=model.faces[exterior_mask(model,reference=True)]
    v=model.vertices[np.unique(f)];length=np.ptp(v[:,0]);mid=(v[:,0].min()+v[:,0].max())/2
    lateral=np.quantile(np.abs(v[:,1]),.98)*.86
    rows=[]
    for name,lo,hi in [('front',-.43,-.14),('rear',.10,.41)]:
        boundary=[]
        for x in np.linspace(mid+lo*length,mid+hi*length,180):
            points=v[(np.abs(v[:,0]-x)<.008)&(np.abs(v[:,1])>lateral)]
            if len(points):boundary.append(points[np.argmin(points[:,2])][[0,2]])
        boundary=np.array(boundary)
        if len(boundary)<40:raise ValueError('无法从外板边界可靠提取前后轮拱。')
        peak=boundary[np.argmax(boundary[:,1])];arc=boundary[np.abs(boundary[:,0]-peak[0])<.24]
        def fun(t):return np.linalg.norm(arc-t[:2],axis=1)-t[2]
        fit=least_squares(fun,[peak[0],peak[1]-.38,.38],bounds=([peak[0]-.10,peak[1]-.55,.30],[peak[0]+.10,peak[1]-.25,.52]),loss='soft_l1',f_scale=.003)
        rms=float(np.sqrt(np.mean(fun(fit.x)**2)))
        if len(arc)<30 or rms>.008 or not fit.success:raise ValueError('轮拱圆弧拟合不可靠；请使用普通工程参数拟合。')
        rows.append(dict(axle=name,center_x_m=float(fit.x[0]),center_z_m=float(fit.x[1]),radius_m=float(fit.x[2]),boundary_rms_mm=rms*1000,samples=len(arc)))
    return rows

def _raw_parts(model):
    exterior_mask(model,reference=True)
    names=model.metadata['raw_label_names'];raw=np.asarray(model.metadata['raw_face_labels'])
    return {str(name).lower():model.faces[raw==i] for i,name in enumerate(names)}

def seed_from_components(reference, parameters):
    """Recover dimensional landmarks, then backtrack the coupled roof proposal."""
    from .engineering import EngineeringParams,generate
    from .parametric_quality import parametric_quality
    from .fitting import _central_floor
    if parameters.body_type!='notchback':
        raise ValueError('组件精细拟合目前验证的是三厢轿车；其他构型请使用工程参数拟合。')
    ref=reference.clone();parts=_raw_parts(ref)
    required={'body','body_fender','body_hood','body_roof','body_tail','body_rear_fascia','body_front_fascia','windows','underbody_smooth','front_intakes','lights_front'}
    if not required.issubset(parts):raise ValueError('当前组件精细拟合需要原始 Body / Roof / Hood / Windows 等分割标签。')
    arches=infer_arches(ref);a,b=arches
    radius=float(np.clip(np.mean([a['radius_m'],b['radius_m']])-.05,.30,.43))
    delta=np.array([-(a['center_x_m']+b['center_x_m'])/2,0.,radius-np.mean([a['center_z_m'],b['center_z_m']])])
    ref.vertices+=delta
    def vertices(key):return ref.vertices[np.unique(parts[key])]
    body=ref.vertices[np.unique(ref.faces[exterior_mask(ref,reference=True)])]
    width=float(np.ptp(body[:,1]));wb=b['center_x_m']-a['center_x_m']
    p=parameters.model_dump();p.update(generator_revision='v6',fine_shape=None,
        wheelbase_m=wb,front_overhang_m=float(-wb/2-body[:,0].min()),rear_overhang_m=float(body[:,0].max()-wb/2),
        wheel_radius_m=radius,body_width_m=width,track_m=width-parameters.tyre_width_m-.05,
        floor_height_m=float(_central_floor(ref)),roof_height_m=float(body[:,2].max()))
    roof=vertices('body_roof');roof=roof[np.abs(roof[:,1])<.04]
    glass=vertices('windows');glass=glass[np.abs(glass[:,1])<.04]
    hood=vertices('body_hood');hood=hood[np.abs(hood[:,1])<.04]
    if min(len(roof),len(glass),len(hood))<20:raise ValueError('中心玻璃与车顶标签缺少足够的纵向边界点。')
    p['hood_height_m']=float(hood[:,2].max())
    initial=EngineeringParams.model_validate(p)
    if not parametric_quality(generate(initial))['screen_pass']:
        raise ValueError('实测轮拱和车身尺寸超出当前连续母版的适配范围。')
    def ends(v):
        ends=[]
        for x in (v[:,0].min(),v[:,0].max()):
            q=v[np.abs(v[:,0]-x)<.003];ends.append(np.median(q,axis=0))
        return ends
    r0,r1=ends(roof);g0,g1=ends(glass[glass[:,0]<r0[0]+.05]);g2,g3=ends(glass[glass[:,0]>r1[0]-.05])
    def angle(a,b):return float(np.degrees(np.arctan2(abs(a[2]-b[2]),abs(a[0]-b[0]))))
    proposal=dict(roof_crown_m=p['roof_height_m']-r0[2],rear_roof_drop_m=r0[2]-r1[2],
        cabin_length_m=r1[0]-r0[0],windshield_angle_deg=angle(g0,r0),rear_glass_angle_deg=angle(r1,g3),
        hood_height_m=g0[2],rear_deck_height_m=g3[2],
        rear_deck_length_m=body[:,0].max()-(g3[2]-p['tailgate_lower_height_m'])/np.tan(np.deg2rad(p['tailgate_angle_deg']))-g3[0])
    proposal={k:float(v) for k,v in proposal.items()}
    # The dimensional proposal is not itself a surface. Limit it against the
    # unchanged preset before allowing the shared smooth field to refine it.
    selected=initial;fraction=0.
    for t in (.5,.375,.25,.125):
        try:
            q=EngineeringParams.model_validate({**p,**{k:p[k]+t*(v-p[k]) for k,v in proposal.items()}})
            quality=parametric_quality(generate(q))
            if quality['screen_pass'] and quality['surface_strain']['principal_max']<2.4:
                selected=q;fraction=t;break
        except ValueError:continue
    ref.metadata['component_fitting_alignment_delta_m']=delta.tolist()
    return ref,selected,dict(arches=arches,alignment_delta_m=delta.tolist(),
        wheelbase_from_body_arches_m=wb,tyres_observed=False,
        tyre_assumptions=dict(radial_arch_gap_m=.05,radius_m=radius,width_m=selected.tyre_width_m,track_m=selected.track_m,
            rule='mean arch radius minus 50 mm; track places tyre envelope 25 mm inside maximum body width per side'),
        measured_component_proposal=proposal,accepted_component_proposal_fraction=fraction,
        dimensional_seed_parameters=initial.model_dump(),starting_parameters=selected.model_dump())

def refine(reference, parameters, progress=lambda _:None, max_iterations=28):
    """Joint robust component ICP over 70 smooth controls; zero vertex unknowns."""
    import time
    from scipy.optimize import minimize,LinearConstraint
    from .engineering import generate
    from .shape_controls import basis,regularizer,COUNT,LIMIT
    from .fit_metrics import _areas,_surface_points,surface_metrics,_statistics
    from .spatial import TriangleIndex
    from .parametric_quality import parametric_quality,_prepared
    from .model import Model
    from .inspection import inspect
    started=time.perf_counter()
    ref,p,measurements=seed_from_components(reference,parameters)
    base=generate(p);B=basis(base.vertices,p.wheelbase_m,p.wheel_radius_m)
    ext=exterior_mask(base);rext=exterior_mask(ref,reference=True);parts=_raw_parts(ref)
    dm=np.zeros(len(base.vertices),bool);dm[base.metadata['driver_vertices']]=True;driver=dm[base.faces].all(1)
    fm=np.asarray(base.metadata['face_material_ids'])
    groups=[('exterior',list(range(30)),None,2.,2600,True),
        ('hood',[0],['body_hood'],.3,900,True),('roof',[1,7],['body_roof'],.3,900,True),
        ('windows',[5,6,16,22],['windows'],.3,1200,True),
        ('side',[12,13,14,15,18,19,20,21],['body','body_fender'],.05,900,False),
        ('front',[2,28],['body_front_fascia','front_intakes','lights_front'],.1,900,True),
        ('rear',[3,4,29],['body','body_rear_fascia','body_tail'],.05,900,False),
        ('underbody',[8,10,11],['underbody_smooth'],.10,900,True)]
    data=[]
    progress('建立原始标签对应关系；共用 70 个粗尺度样条控制')
    for k,(name,labels,source,weight,count,bidirectional) in enumerate(groups):
        mask=np.isin(base.labels,labels)&driver
        if name=='windows':mask&=fm==1
        if name=='side':mask|=(np.isin(base.labels,[16,22])&(fm!=1)&driver)
        if name!='underbody':mask&=ext
        faces=base.faces[mask];rf=ref.faces[rext] if source is None else np.concatenate([parts[n] for n in source])
        area=_areas(base.vertices,faces);rng=np.random.default_rng(734+k)
        tri=faces[rng.choice(len(faces),count,p=area/area.sum())]
        uv=rng.random((count,2));u=np.sqrt(uv[:,0]);bary=np.c_[1-u,u*(1-uv[:,1]),u*uv[:,1]]
        sample0=np.einsum('ni,nij->nj',bary,base.vertices[tri]);sb=np.einsum('ni,nijk->njk',bary,B[tri])
        points=_surface_points(ref.vertices,rf,count,1834+k)
        t=ref.vertices[rf];n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
        data.append(dict(name=name,faces=faces,index=TriangleIndex(ref.vertices,rf),normal=n,points=points,sample0=sample0,b=sb,weight=weight,bidirectional=bidirectional))
    c=np.zeros(COUNT);R=regularizer();history=[];prepared=_prepared(p.body_type,'v6',p.spoiler_style)
    for iteration in range(max_iterations):
        progress(f'组件联合拟合 {iteration+1}/{max_iterations} · 检查面片拉伸、前风挡和新增折痕')
        m=generate(p.model_copy(update={'fine_shape':tuple(c)}));As=[];ys=[];errors={}
        for g in data:
            points=g['sample0']+np.einsum('nij,j->ni',g['b'],c)
            closest,fi,_=g['index'].closest(points);n=g['normal'][fi]
            pairs=[(g['b'],g['sample0'],closest,n)];distances=[np.linalg.norm(points-closest,axis=1)]
            if g['bidirectional']:
                back,bfi,bb=TriangleIndex(m.vertices,g['faces']).closest(g['points']);bi=g['faces'][bfi]
                backB=np.einsum('ni,nijk->njk',bb,B[bi]);back0=np.einsum('ni,nij->nj',bb,base.vertices[bi])
                bn=np.cross(m.vertices[bi[:,1]]-m.vertices[bi[:,0]],m.vertices[bi[:,2]]-m.vertices[bi[:,0]])
                bn/=np.maximum(np.linalg.norm(bn,axis=1)[:,None],1e-12)
                pairs.append((backB,back0,g['points'],bn));distances.append(np.linalg.norm(back-g['points'],axis=1))
            errors[g['name']]=float(np.sqrt(np.mean(np.concatenate(distances)**2))*1000)
            for b,zero,target,normal in pairs:
                dist=np.linalg.norm(zero+np.einsum('nij,j->ni',b,c)-target,axis=1)
                weight=np.sqrt(g['weight']/len(dist)*np.minimum(1.,.06/np.maximum(dist,1e-9)))
                As.extend([np.einsum('ni,nij->nj',normal,b)*weight[:,None],(b*weight[:,None,None]*.3).reshape(-1,COUNT)])
                ys.extend([np.einsum('ni,ni->n',normal,target-zero)*weight,((target-zero)*weight[:,None]*.3).ravel()])
        A=np.concatenate(As+[.12*R,.05*np.eye(COUNT)]);y=np.concatenate(ys+[np.zeros(len(R)+COUNT)])
        H=A.T@A;g=A.T@y;D,lo,hi=shape_constraints(m,B,prepared)
        sol=minimize(lambda t:.5*t@H@t-g@t,c,jac=lambda t:H@t-g,method='SLSQP',bounds=[(-LIMIT,LIMIT)]*COUNT,
            constraints=[LinearConstraint(D,D@c+lo,D@c+hi)],options={'maxiter':120,'ftol':1e-9})
        proposed=sol.x;accepted=False;attempts=[];change=0.
        for fraction in (1.,.5,.25,.125,.0625,.03125):
            trial=c+(proposed-c)*fraction
            try:quality=parametric_quality(generate(p.model_copy(update={'fine_shape':tuple(trial)})))
            except ValueError as exc:quality={'screen_pass':False,'errors':[str(exc)]}
            attempts.append(dict(fraction=fraction,errors=quality['errors']))
            if quality['screen_pass']:
                change=float(np.max(np.abs(trial-c)));c=trial;accepted=True;break
        history.append(dict(iteration=iteration+1,training_rms_mm=errors,qp_success=bool(sol.success),accepted=accepted,line_search=attempts,max_control_change_m=change))
        if not accepted or (iteration>8 and change<.00015):break
    progress('完整车身自交与四轮静态间隙检查')
    final=None
    for fraction in (1.,.875,.75,.5,.25,0.):
        pp=p.model_copy(update={'fine_shape':tuple(c*fraction)});model=generate(pp)
        q=parametric_quality(model)
        if not q['screen_pass']:continue
        inspection=inspect(model)
        if inspection['complete'] and inspection['requested_checks_passed']:
            final=(pp,model,inspection,fraction);break
    if final is None:raise ValueError('拟合参数未通过车身自交和轮组间隙检查。')
    pp,model,inspection,fraction=final
    progress('使用独立采样评估外观表面与原始组件的剩余误差')
    def surface(m,mask):return Model(m.vertices,m.faces[mask],m.labels[mask],m.parts)
    aligned_old=generate(parameters);aligned_old.vertices+=np.asarray(measurements['alignment_delta_m'])
    target=surface(ref,rext)
    before=surface_metrics(surface(aligned_old,exterior_mask(aligned_old)),target,samples=10000,seed=98173)
    after=surface_metrics(surface(model,exterior_mask(model)),target,samples=10000,seed=98173)
    if after['rms_mm']>=before['rms_mm']:raise ValueError('精细拟合未改善独立外观表面误差；保留现有工程拟合结果。')
    indexes=[TriangleIndex(m.vertices,m.faces[exterior_mask(m)]) for m in (aligned_old,model)]
    breakdown=[]
    for key,faces in parts.items():
        if key.startswith(('underbody','mirror','wheel','tire','tyre')):continue
        points=_surface_points(ref.vertices,faces,1500,73191+len(breakdown))
        values=[]
        for index in indexes:
            nearest,_,_=index.closest(points);values.append(_statistics(np.linalg.norm(points-nearest,axis=1)))
        breakdown.append(dict(source_component=key,before=values[0],after=values[1]))
    report=dict(schema='aeroshape.component-spline-fit.v1',measurements=measurements,
        method='joint_component_robust_symmetric_C2_spline_ICP_with_constrained_quadratic_steps',
        free_shape_controls=COUNT,per_vertex_degrees_of_freedom=0,source_vertices_transferred=0,
        coefficient_bound_m=LIMIT,longitudinal_stations=7,vertical_layers=3,
        regularization=dict(second_difference=.12,coefficient_prior=.05,huber_distance_m=.06),
        exterior_fit=dict(before=before,after=after,scope='outer body and glass; excludes wheels, mirrors, underbody, wheel-house liners, splitter and diffuser on both meshes'),
        source_component_distances=breakdown,component_metric_scope='area-uniform original labeled reference surfaces to nearest generated exterior triangles; one direction, independent of inferred design30 boundaries',
        iterations=history,static_inspection=inspection,clearance_backtracking_fraction=fraction,
        evaluation_seed=98173,training_seeds=[734,1834],elapsed_seconds=round(time.perf_counter()-started,3))
    return ref,pp,model,report,aligned_old
