"""Versioned 30-region schema and conservative, inspectable label proposals.

No learned model or OEM label enumeration is implied. The mesh is never cut,
reordered, welded or moved here. Optional components are proposed only from
source names or explicit mappings, not invented to fill all thirty classes.
"""
from __future__ import annotations
import hashlib
import re
from typing import Literal
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .model import Model, array_hash
from .geometry import face_geometry
from .design30 import _groups

SCHEMA = 'aeroshape.design30.v1'
UNKNOWN = 30  # Sentinel, not a thirty-first design class.
ROWS = _groups()
KEYS = [row[0] for row in ROWS]
KEY_TO_ID = {k:i for i,k in enumerate(KEYS)}
BODY_TYPES = ('fastback','notchback','estateback','suv','hatchback','unknown')


def schema_parts() -> list[dict]:
    parts=[]
    for i,(key,name,_) in enumerate(ROWS):
        category='body';color='#82a7b6';catname='车身表面'
        if key.startswith('wheel_'): category='wheels';color='#354252';catname='轮组'
        elif key in ('windshield','rear_glass','panorama') or key.startswith('side_glass'):
            category='glazing';color='#446474';catname='玻璃区域'
        elif key in ('splitter','spoiler','diffuser','underbody') or key.startswith('mirror_'):
            category='aero';color='#beaa86';catname='气动附件与底部'
        elif key.endswith('lights'):category='lighting';color='#c9d8dc';catname='灯具'
        # Distinct surface-region colors; imported material shading remains separate.
        if category=='body':color=['#8eafc5','#b4a38c','#78a59f','#8b9eb6','#b19794'][i%5]
        p=dict(id=i,key=key,name=name,color=color,category=category,category_name=catname,
               locked=key.startswith('wheel_'),material=dict(base_color=color,roughness=.4,metallic=.1))
        if key.endswith(('r','l')):
            counterpart=key[:-1]+('l' if key.endswith('r') else 'r')
            if counterpart in KEY_TO_ID:p['pair_id']=KEY_TO_ID[counterpart]
        parts.append(p)
    parts.append(dict(id=UNKNOWN,key='unassigned',name='待判定 / 非目标部件',color='#b5769a',
                      category='detail',category_name='待校正',locked=True))
    return parts


class SegmentationOptions(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    body_type: Literal['fastback','notchback','estateback','suv','hatchback','unknown']='unknown'
    use_source_names: bool=True
    preserve_manual: bool=True
    source_map: dict[str,str]=Field(default_factory=dict)
    smooth_iterations: int=Field(default=2,ge=0,le=6)
    smoothing: float=Field(default=.25,ge=0,le=1)
    unknown_threshold: float=Field(default=.05,ge=0,le=.8)
    # Overrides are normalized longitudinal / vertical coordinates, not metres.
    landmarks: dict[str,float]=Field(default_factory=dict)
    @model_validator(mode='after')
    def mapping_valid(self):
        for target in self.source_map.values():
            if target not in KEY_TO_ID and target not in ('unassigned','wheels','mirrors','doors','side_glass','windows','front_fenders','body_skin'):
                raise ValueError(f'Unknown canonical key: {target}')
        allowed={'hood_end','roof_start','roof_end','rear_glass_end','belt_height','front_axle','rear_axle'}
        if set(self.landmarks)-allowed:raise ValueError('Unsupported landmark key.')
        if any(not np.isfinite(x) or not 0<x<1 for x in self.landmarks.values()):
            raise ValueError('Landmark fractions must be finite and inside (0,1).')
        return self


def face_topology(f:np.ndarray):
    """Manifold edge adjacency. Never bridge nearby disconnected surfaces."""
    e=np.sort(np.vstack((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
    owner=np.tile(np.arange(len(f)),3)
    order=np.lexsort((e[:,1],e[:,0])); ee=e[order]; oo=owner[order]
    same=np.all(ee[1:]==ee[:-1],axis=1)
    valid=same.copy()
    # Edges with 3+ incident faces are not label-propagation connections.
    crowded=np.r_[False,same[:-1]] | np.r_[same[1:],False]
    valid &= ~crowded
    a,b=oo[:-1][valid],oo[1:][valid]
    graph=coo_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(len(f),len(f))).tocsr()
    _,cc=connected_components(graph,directed=False)
    return a,b,cc


def _normalized(model:Model):
    c,n,area=face_geometry(model.vertices,model.faces)
    lo=model.vertices.min(0); span=np.maximum(np.ptp(model.vertices,axis=0),1e-12)
    return c,n,area,(c-lo)/span,lo,span


def _outward_analysis_normals(model,c,n,area,a,b,cc):
    """Interpret closed inward CFD shells without changing source winding."""
    degree=np.bincount(np.r_[a,b],minlength=len(model.faces))
    closed=np.bincount(cc,weights=(degree!=3).astype(np.int32))==0
    origin=(model.vertices.min(0)+model.vertices.max(0))/2
    volumes=np.bincount(cc,weights=np.einsum('ij,ij->i',c-origin,n)*area/3)
    tolerance=max(float(np.prod(np.ptp(model.vertices,axis=0)))*1e-10,1e-12)
    reversed_components=np.flatnonzero(closed&(volumes < -tolerance))
    if len(reversed_components):
        n=n.copy();n[np.isin(cc,reversed_components)]*=-1
    return n,dict(method='closed_component_signed_volume_analysis_only',
                  reversed_components=reversed_components.tolist(),
                  reversed_faces=int(np.isin(cc,reversed_components).sum()),
                  source_faces_changed=False)


def estimate_landmarks(model:Model, options:SegmentationOptions, *, _geometry=None) -> dict:
    # Roof plateau from an area-weighted upper central-skin station profile.
    if _geometry is None:
        c,n,area,p,lo,span=_normalized(model)
        a,b,cc=face_topology(model.faces)
        n,_=_outward_analysis_normals(model,c,n,area,a,b,cc)
    else:c,n,area,p,lo,span=_geometry
    defaults=dict(hood_end=.34,roof_start=.46,roof_end=.69,rear_glass_end=.84,
                  belt_height=.58,front_axle=.22,rear_axle=.77)
    if options.body_type in ('estateback','suv','hatchback'):
        defaults.update(roof_end=.83,rear_glass_end=.93)
    stations=np.linspace(0,1,81); heights=np.full(80,np.nan)
    center=(np.abs(c[:,1])<.28*span[1]) & (n[:,2]>.15)
    for i in range(80):
        mask=center & (p[:,0]>=stations[i]) & (p[:,0]<stations[i+1])
        if mask.any():heights[i]=np.quantile(p[mask,2],.85)
    roof=np.flatnonzero((heights>.90)&(stations[:-1]>.28)&(stations[:-1]<.90))
    origin={k:'body_type_prior' for k in defaults}
    if len(roof)>3:
        rs=float(stations[roof.min()]);re=float(stations[roof.max()+1])
        if .35<rs<.60 and .58<re<.94 and re-rs>.10:
            defaults['roof_start']=rs;defaults['roof_end']=re
            defaults['hood_end']=max(.20,rs-.12)
            defaults['rear_glass_end']=min(.98,re+(.09 if options.body_type in ('estateback','suv','hatchback') else .15))
            for key in ('roof_start','roof_end','hood_end','rear_glass_end'):origin[key]='upper_profile_estimate'
    defaults.update(options.landmarks)
    origin.update({k:'user_override' for k in options.landmarks})
    if not defaults['hood_end']<defaults['roof_start']<defaults['roof_end']<defaults['rear_glass_end']:
        raise ValueError('Longitudinal landmarks must satisfy hood_end < roof_start < roof_end < rear_glass_end.')
    if not defaults['front_axle']<defaults['rear_axle']:raise ValueError('front_axle must precede rear_axle.')
    return dict(values=defaults,origin=origin,units='normalized_bbox_fraction',
                note='Profile and axle priors are proposals, not detected physical seams or measured wheel centers.')


def _name_key(s:str) -> str|None:
    s=re.sub(r'^\d+[_ -]+','',s.strip().lower())
    s=re.sub(r'_mat\d+$','',s)
    s=re.sub(r'[\s.\-/]+','_',s).strip('_')
    # Trimesh gives repeated ASCII STL solid blocks suffixes such as Body_2.
    s=re.sub(r'_\d+$','',s)
    if s in KEY_TO_ID:return s
    synonyms={'bonnet':'hood','engine_hood':'hood','front_windshield':'windshield','windscreen':'windshield',
              'rear_window':'rear_glass','back_window':'rear_glass','trunk':'tailgate','boot':'tailgate',
              'body_roof':'roof','floor':'underbody','under_body':'underbody','panoramic_roof':'panorama',
              'front_grille':'front_bumper','grille':'front_bumper','front_lip':'splitter',
              'headlights':'front_lights','headlamp':'front_lights','taillights':'rear_lights',
              'taillamp':'rear_lights','wheels':'wheels','wheel':'wheels','tires':'wheels','tyres':'wheels',
              'mirrors':'mirrors','mirror':'mirrors','side_mirror':'mirrors','side_mirrors':'mirrors',
              'doors':'doors','door':'doors','side_windows':'side_glass','side_glass':'side_glass',
              'body':'body_skin','body_hood':'hood','body_fender':'front_fenders',
              'body_rear_fascia':'rear_bumper','body_tail':'tailgate',
              'body_front_fascia':'front_bumper','mirrors_body':'mirrors','mirrors_glass':'mirrors',
              'underbody_smooth':'underbody','front_intakes':'front_bumper',
              'lights_front':'front_lights','windows':'windows'}
    if s in synonyms:return synonyms[s]
    # Explicit corner aliases; all are proposals until human review.
    for token,corner in [('front_left','fl'),('front_right','fr'),('rear_left','bl'),('rear_right','br')]:
        if s in (f'wheel_{token}',f'tire_{token}',f'tyre_{token}',f'{token}_wheel'):
            return 'wheel_'+corner
    # Detailed authored assets have finer names which map by their definitions.
    for key,_,members in ROWS:
        if s in members:return key
    return None


def _resolved_targets(key:str, c:np.ndarray, p:np.ndarray, lm:dict, normals=None) -> np.ndarray:
    result=np.full(len(c),UNKNOWN,np.int32)
    if key in KEY_TO_ID:result[:]=KEY_TO_ID[key]
    elif key=='windows':
        # The original DrivAer windows class combines front, rear and side
        # glass. Keep that source annotation while proposing its design split.
        side=(np.abs(normals[:,1])>.5) if normals is not None else (np.abs(c[:,1])>.65*np.max(np.abs(c[:,1])))
        result[:]=np.where(p[:,0]<(lm['roof_start']+lm['roof_end'])/2,
                           KEY_TO_ID['windshield'],KEY_TO_ID['rear_glass'])
        result[side]=np.where(c[side,1]>=0,KEY_TO_ID['side_glass_l'],KEY_TO_ID['side_glass_r'])
    elif key in ('wheels','mirrors','doors','side_glass','front_fenders'):
        for side in ('l','r'):
            m=(c[:,1]>=0) if side=='l' else (c[:,1]<0)
            if key=='mirrors':result[m]=KEY_TO_ID['mirror_'+side]
            elif key=='side_glass':result[m]=KEY_TO_ID['side_glass_'+side]
            elif key=='front_fenders':result[m]=KEY_TO_ID['front_fender_'+side]
            else:
                split=(lm['front_axle']+lm['rear_axle'])/2 if key=='wheels' else (lm['roof_start']+lm['roof_end'])/2
                for isfront in (True,False):
                    mask=m & ((p[:,0]<split) if isfront else (p[:,0]>=split))
                    k=('wheel_'+('f' if isfront else 'b')+side) if key=='wheels' else (('front_door_' if isfront else 'rear_door_')+side)
                    result[mask]=KEY_TO_ID[k]
    return result


def _symmetry_capability(model:Model, driver_ids:np.ndarray) -> dict:
    v=model.vertices
    if not len(driver_ids):return dict(vertex_correspondence=False,max_residual_m=None,note='No verified driver skin.')
    query=v[driver_ids].copy();query[:,1]*=-1
    dist,_=cKDTree(v).query(query)
    tol=max(np.ptp(v,axis=0))*1e-6
    return dict(vertex_correspondence=bool(np.all(dist<=tol)),max_residual_m=float(dist.max()),tolerance_m=float(tol),
                note='Vertex-nearest mirror check only, not a continuous-surface symmetry proof.')


def segment30(model:Model, options:SegmentationOptions|None=None) -> Model:
    options=options or SegmentationOptions()
    c,n,area,p,lo,span=_normalized(model);x=p[:,0];z=p[:,2]
    a,b,cc=face_topology(model.faces)
    n,orientation_report=_outward_analysis_normals(model,c,n,area,a,b,cc)
    landmark_report=estimate_landmarks(model,options,_geometry=(c,n,area,p,lo,span));lm=landmark_report['values']
    warnings=['Automatic labels need human boundary review; scores are heuristic reliability scores, not calibrated probabilities.',
              'No claim of OEM part boundaries, CFD readiness, attachment binding, or identical topology across vehicles.']
    if options.body_type=='unknown':warnings.append('Body type unknown; generic passenger-car priors used.')
    if not (2.5<span[0]<7.5 and .8<span[1]<3.8 and .6<span[2]<3.8):
        warnings.append('Dimensions outside passenger-car prior: check metres, axes, full vehicle versus partial asset.')
    side=np.clip(np.abs(n[:,1]),0,1);upper=np.clip(n[:,2],0,1)
    lateral=np.clip(np.abs(c[:,1])/(.5*span[1]),0,1)
    # A compact cost matrix; 30 labels at <=2M faces still bounded by import guard.
    costs=np.full((len(c),30),8,dtype=np.float32)
    def interval(a,b,values=x):return np.maximum(a-values,0)+np.maximum(values-b,0)
    def assign(k,v):costs[:,KEY_TO_ID[k]]=v
    assign('hood',5*interval(.04,lm['hood_end'])+1.3*(1-upper)+2*np.maximum(lm['belt_height']*.7-z,0))
    assign('windshield',7*interval(lm['hood_end'],lm['roof_start'])+1.1*(1-upper)+2*np.maximum(.56-z,0))
    assign('roof',7*interval(lm['roof_start'],lm['roof_end'])+1.2*(1-upper)+4*np.maximum(.83-z,0))
    assign('rear_glass',7*interval(lm['roof_end'],lm['rear_glass_end'])+1.0*(1-upper)+3*np.maximum(.63-z,0))
    assign('tailgate',6*interval(lm['rear_glass_end'],1)+.7*(1-upper)+2*np.maximum(.40-z,0))
    assign('front_bumper',5*np.maximum(x-.13,0)+.55*upper+2*np.maximum(z-.62,0))
    assign('rear_bumper',5*np.maximum(.88-x,0)+.55*upper+2*np.maximum(z-.62,0))
    assign('underbody',3*np.maximum(z-.18,0)+1.1*np.maximum(n[:,2]+.2,0))
    door_mid=(lm['roof_start']+lm['roof_end'])/2
    for s in ('r','l'):
        opposite=(c[:,1]>1e-10) if s=='r' else (c[:,1]<-1e-10)
        base=(1-side)*.65+(1-lateral)*.45+opposite*6
        assign('front_fender_'+s,base+5*interval(.05,lm['hood_end'])+3*np.maximum(z-lm['belt_height'],0))
        assign('front_door_'+s,base+5*interval(lm['hood_end'],door_mid)+4*np.maximum(z-lm['belt_height'],0))
        assign('rear_door_'+s,base+5*interval(door_mid,lm['roof_end'])+4*np.maximum(z-lm['belt_height'],0))
        assign('rear_quarter_'+s,base+5*interval(lm['roof_end'],.98)+3*np.maximum(z-lm['belt_height'],0))
        assign('side_glass_'+s,base+5*interval(lm['hood_end'],lm['roof_end'])+4*np.maximum(lm['belt_height']-z,0))
    sizes=np.bincount(cc,weights=area); main=int(np.argmax(sizes))
    # Disconnected objects default to unknown, except conservative cylindrical wheel candidates.
    disconnected=cc!=main
    geometry_labels=np.argmin(costs,axis=1).astype(np.int32)
    method=np.zeros(len(c),np.uint8)  # 0 geometry,1 source name,2 explicit map,3 component,4 unknown,5 manual,6 transfer
    fixed=np.zeros(len(c),bool);targets=geometry_labels.copy()
    targets[disconnected]=UNKNOWN;method[disconnected]=4
    inferred_components=[]
    # Limit component scanning; pathological STL soups are not an assembly detector.
    if len(sizes)<=1000:
        for comp in range(len(sizes)):
            if comp==main:continue
            ids=np.flatnonzero(cc==comp); verts=model.vertices[np.unique(model.faces[ids])]
            mn=verts.min(0);mx=verts.max(0);extent=mx-mn;center=(mn+mx)/2
            q=(center-lo)/span
            wheel=(len(ids)>=40 and .05<extent[0]/span[0]<.20 and .06<extent[2]/span[0]<.21
                   and .70<extent[0]/max(extent[2],1e-12)<1.40 and extent[1]<.8*extent[0]
                   and q[2]<.34 and abs(center[1])>.24*span[1]
                   and min(abs(q[0]-lm['front_axle']),abs(q[0]-lm['rear_axle']))<.15)
            if wheel:
                targets[ids]=_resolved_targets('wheels',c[ids],p[ids],lm);method[ids]=3;fixed[ids]=True
                inferred_components.append(dict(component=int(comp),target='wheel',faces=len(ids)))
    # Attached mirror housings can share the body's topology in production STL.
    # Propose only a pair of small, localized outboard lobes that extends beyond
    # the measured door-panel width; a generic box/sedan side is insufficient.
    bulk=(x>.45)&(x<.65)&(z>.20)&(z<.55)&(cc==main)
    if bulk.sum()>=20:
        bulk_half=float(np.quantile(np.abs(c[bulk,1]),.99))
        half_width=float(np.max(np.abs(c[:,1])))
        if bulk_half>0 and half_width>bulk_half*1.055:
            seed=(np.abs(c[:,1])>max(bulk_half*1.035,half_width*.92))&(x>.25)&(x<.52)&(z>.38)&(z<.84)&(cc==main)
            lobes=[]
            for sign,side_key in ((-1,'r'),(1,'l')):
                ids=np.flatnonzero(seed&(c[:,1]*sign>0))
                if len(ids)<30:break
                mn=c[ids].min(0);mx=c[ids].max(0);extent=mx-mn
                if not (.005*span[0]<extent[0]<.10*span[0] and .015*span[0]<extent[2]<.20*span[0]):break
                lobes.append((sign,side_key,mn,mx))
            if len(lobes)==2 and abs((lobes[0][2][0]+lobes[0][3][0])-(lobes[1][2][0]+lobes[1][3][0]))<.04*span[0]:
                for sign,side_key,mn,mx in lobes:
                    mask=(cc==main)&(~fixed)&(c[:,1]*sign>bulk_half*.965)
                    mask&=(c[:,0]>mn[0]-.01*span[0])&(c[:,0]<mx[0]+.01*span[0])
                    mask&=(c[:,2]>mn[2]-.02*span[2])&(c[:,2]<mx[2]+.02*span[2])
                    targets[mask]=KEY_TO_ID['mirror_'+side_key];method[mask]=3;fixed[mask]=True
                    inferred_components.append(dict(component=main,target='mirror_'+side_key,faces=int(mask.sum()),
                        detection='attached_bilateral_outboard_lobe_proposal',panel_half_width_m=bulk_half))
    mapping_log=[]
    # Re-segmentation uses retained original source assignments, not its own predictions.
    source_labels=np.asarray(model.metadata.get('semantic_source_labels',model.labels),np.int32)
    source_parts=model.metadata.get('semantic_source_parts',model.parts)
    if source_labels.shape!=model.labels.shape:raise ValueError('Source label provenance is stale after topology changes.')
    for part in source_parts:
        explicit=None
        for key in (str(part['id']),part.get('key',''),part.get('name','')):
            if key in options.source_map:explicit=options.source_map[key];break
        inferred=None
        if explicit is None and options.use_source_names:
            inferred=_name_key(part.get('key','')) or _name_key(part.get('name',''))
        key=explicit if explicit is not None else inferred
        ids=np.flatnonzero(source_labels==part['id'])
        if key is None or not len(ids):continue
        if key=='body_skin':
            body_ids=[KEY_TO_ID[k] for k in ('hood','roof','front_bumper','rear_bumper','tailgate',
                       'front_fender_l','front_fender_r','front_door_l','front_door_r',
                       'rear_door_l','rear_door_r','rear_quarter_l','rear_quarter_r')]
            targets[ids]=np.asarray(body_ids)[np.argmin(costs[np.ix_(ids,body_ids)],axis=1)]
        else:
            targets[ids]=_resolved_targets(key,c[ids],p[ids],lm,n[ids])
        method[ids]=2 if explicit is not None else 1;fixed[ids]=True
        mapping_log.append(dict(source_id=part['id'],source_name=part.get('name',''),target=key,
                                mode='explicit' if explicit is not None else 'name_alias_proposal',faces=len(ids)))
    if options.preserve_manual and model.metadata.get('semantic_schema')==SCHEMA:
        prior_methods=np.asarray(model.metadata.get('segmentation_face_methods',[]))
        if prior_methods.shape==targets.shape:
            manual=prior_methods==5;targets[manual]=model.labels[manual];method[manual]=5;fixed[manual]=True
    # Graph ICM: hard semantic anchors, normal-sensitive manifold-only neighbors.
    targets=np.ascontiguousarray(targets)
    if len(a) and options.smooth_iterations:
        weight=np.exp(-((1-np.clip(np.einsum('ij,ij->i',n[a],n[b]),-1,1))/.25))*options.smoothing
        g=coo_matrix((np.r_[weight,weight],(np.r_[a,b],np.r_[b,a])),shape=(len(c),len(c))).tocsr()
        degree=np.asarray(g.sum(1)).ravel()
        for _ in range(options.smooth_iterations):
            # Avoid dense N x N matrices; 30 sparse graph-vector products.
            best=np.full(len(c),np.inf);new=targets.copy()
            for k in range(30):
                cost=costs[:,k]+degree-np.asarray(g@(targets==k)).ravel()
                improve=cost<best;new[improve]=k;best[improve]=cost[improve]
            editable=(~fixed)&(~disconnected);targets[editable]=new[editable]
    ordered=np.partition(costs,1,axis=1)[:,:2];margin=np.abs(ordered[:,1]-ordered[:,0])
    score=np.clip(.22+margin/(.6+margin),0,.78)
    score[method==1]=.85;score[method==2]=1.;score[method==3]=.58;score[method==5]=1.
    # Poor absolute geometric fit and near-boundary ambiguity reduce reliability.
    bestcost=np.min(costs,axis=1);score[(method==0)&(bestcost>1.5)]*=.3
    unknown=(~fixed)&((score<options.unknown_threshold)|disconnected)
    targets[unknown]=UNKNOWN;method[targets==UNKNOWN]=4;score[targets==UNKNOWN]=0
    # Mirror face propagation only for near-exact, involutive centroid+normal matches.
    reflected=c.copy();reflected[:,1]*=-1;dist,partners=cKDTree(c).query(reflected)
    tol=span.max()*1e-6
    involutive=partners[partners]==np.arange(len(c))
    pairmap=np.arange(31)
    for part in schema_parts():pairmap[part['id']]=part.get('pair_id',part['id'])
    rn=n.copy();rn[:,1]*=-1
    compatible=(dist<tol)&involutive&(np.einsum('ij,ij->i',rn,n[partners])>.98)
    ids=np.flatnonzero(compatible&(np.arange(len(c))<partners))
    label_conflicts=0
    for sideids in [ids]:
        other=partners[sideids]
        conflicts=fixed[sideids]&fixed[other]&(pairmap[targets[sideids]]!=targets[other])
        label_conflicts=int(conflicts.sum())
        allowed=~conflicts
        i=sideids[allowed];j=other[allowed]
        choose_j=(fixed[j]&~fixed[i])|((fixed[j]==fixed[i])&(score[j]>score[i]))
        src=np.where(choose_j,j,i);dst=np.where(choose_j,i,j)
        targets[dst]=pairmap[targets[src]];score[dst]=score[src];method[dst]=method[src]
    if label_conflicts:warnings.append(f'{label_conflicts} mirrored source-label pairs conflict; not silently overwritten.')
    # An automatically detected main component is a driver proposal, not an attachment solver.
    driver=np.unique(model.faces[cc==main])
    if float(area[cc==main].sum()/max(area.sum(),1e-30))<.55:
        warnings.append('Largest connected skin has less than 55% of total area. Separate panels will not automatically deform together; inspect and explicitly stitch/bind intended interfaces.')
    parts=schema_parts()
    meta=dict(model.metadata)
    # Dropping bindings is safer than applying old parent IDs to new semantics.
    same_semantics=np.array_equal(targets,model.labels) and all(
        (p['id'] in range(30) and p.get('key')==KEYS[p['id']]) or
        (p['id']==UNKNOWN and p.get('key')=='unassigned') for p in model.parts)
    if not same_semantics:
        if meta.get('rigid_followers'):
            warnings.append('Existing rigid follower bindings invalidated by semantic relabel; rebind before attachment editing.')
        meta['rigid_followers']=[];meta['driver_vertices']=driver.tolist()
        meta['attachment_binding_status']='not_created_by_importer'
    else:
        meta.setdefault('driver_vertices',driver.tolist())
        old_parts={p['id']:p for p in model.parts}
        for part in parts:
            part['locked']=bool(part['locked'] or old_parts.get(part['id'],{}).get('locked',False))
        meta['attachment_binding_status']='retained_existing_bindings_under_identical_semantics'

    meta['semantic_source_labels']=source_labels.tolist();meta['semantic_source_parts']=source_parts
    meta['segmentation_face_scores']=np.round(score,3).tolist();meta['segmentation_face_methods']=method.tolist()
    meta['semantic_schema']=SCHEMA;meta['body_type']=options.body_type
    meta['annotation_status']='auto_proposed_requires_review'
    meta['design_sets']={
        'roof':['roof','panorama','side_glass_l','side_glass_r'],
        'rear':['tailgate','rear_glass','spoiler','rear_quarter_l','rear_quarter_r','rear_lights'],
        'bonnet':['hood','front_fender_l','front_fender_r'],
        'mirrors':['mirror_l','mirror_r'],'underbody':['underbody','diffuser']}
    region_stats=[]
    for part in parts:
        mask=targets==part['id']
        region_stats.append(dict(id=part['id'],key=part['key'],faces=int(mask.sum()),
            area_m2=float(area[mask].sum()),mean_score=float(score[mask].mean()) if mask.any() else None,
            status='present_proposed' if mask.any() else 'absent_or_not_observed'))
    meta['segmentation']=dict(schema=SCHEMA,algorithm='source_anchors_geometry_priors_normal_weighted_graph_icm_v1',
        normal_orientation=orientation_report,
        options=options.model_dump(),landmarks=landmark_report,regions=region_stats,source_mappings=mapping_log,
        inferred_components=inferred_components,mesh_components=len(sizes),main_skin_component=main,
        low_score_faces=int((score<.5).sum()),unknown_faces=int((targets==UNKNOWN).sum()),
        score_kind='heuristic_not_calibrated_probability',warnings=warnings,
        mirrored_face_coverage=float(compatible.mean()),source_symmetry_conflicts=label_conflicts,
        review=dict(status='pending',reviewer=None),geometry_changed=False,
        symmetry=_symmetry_capability(model,driver),cfd_ready=False)
    out=model.clone(labels=targets,parts=parts,metadata=meta)
    assert array_hash(out.vertices,out.faces)==array_hash(model.vertices,model.faces)
    return out


def topology_digest(faces:np.ndarray) -> str:
    return hashlib.sha256(np.asarray(faces,dtype='<i4').tobytes()).hexdigest()


def transfer_labels(reference:Model,target:Model,*,mode='mapped',vertex_map=None,
                    face_map=None,max_distance_m=.02) -> Model:
    """No equality-of-face-arrays shortcut without explicit correspondence.

    mapped: target vertex -> reference vertex permutation + target face -> ref
    face permutation, checked against connectivity. nearest: conservative
    centroid/normal proposal (NOT registration or closest triangle projection).
    """
    if reference.metadata.get('semantic_schema')!=SCHEMA:raise ValueError('Reference needs canonical design30 labels.')
    out=segment30(target,SegmentationOptions(body_type=target.metadata.get('body_type','unknown'),use_source_names=False))
    count=len(target.faces);scores=np.asarray(out.metadata['segmentation_face_scores']);labels=out.labels.copy()
    if mode=='mapped':
        if vertex_map is None or face_map is None:raise ValueError('Explicit vertex_map and face_map are required; equal face arrays alone do not prove correspondence.')
        vm=np.asarray(vertex_map);fm=np.asarray(face_map)
        if vm.dtype.kind not in 'iu' or fm.dtype.kind not in 'iu':raise ValueError('Correspondence arrays must be integer.')
        if vm.shape!=(len(target.vertices),) or fm.shape!=(count,):raise ValueError('Correspondence shape mismatch.')
        if len(vm)!=len(reference.vertices) or len(fm)!=len(reference.faces):raise ValueError('Mapped mode requires bijective topology.')
        if not np.array_equal(np.sort(vm),np.arange(len(vm))) or not np.array_equal(np.sort(fm),np.arange(len(fm))):
            raise ValueError('Correspondence must be a permutation.')
        if not np.array_equal(np.sort(vm[target.faces],axis=1),np.sort(reference.faces[fm],axis=1)):
            raise ValueError('Correspondence does not preserve face connectivity.')
        labels=reference.labels[fm].copy();scores[:]=1.;accepted=np.ones(count,bool)
    elif mode=='nearest':
        if not np.isfinite(max_distance_m) or max_distance_m<=0 or max_distance_m>.5:raise ValueError('Invalid transfer distance.')
        rc,rn,_=face_geometry(reference.vertices,reference.faces);tc,tn,_=face_geometry(target.vertices,target.faces)
        dist,fi=cKDTree(rc).query(tc)
        dot=np.einsum('ij,ij->i',tn,rn[fi]);accepted=(dist<=max_distance_m)&(dot>.8)
        labels[accepted]=reference.labels[fi[accepted]]
        scores[accepted]=np.minimum(.75,1-dist[accepted]/max_distance_m)
    else:raise ValueError('Transfer mode must be mapped or nearest.')
    if not np.array_equal(out.labels,labels):
        out.metadata['rigid_followers']=[]
        out.metadata['attachment_binding_status']='invalidated_by_label_transfer_requires_rebinding'
    out.labels=labels
    out.metadata['segmentation_face_scores']=np.round(scores,3).tolist()
    methods=np.asarray(out.metadata['segmentation_face_methods']);methods[accepted]=6
    out.metadata['segmentation_face_methods']=methods.tolist()
    out.metadata['annotation_status']='transferred_requires_review'
    out.metadata['label_transfer']=dict(mode=mode,accepted_faces=int(accepted.sum()),total_faces=count,
        reference_geometry_hash=array_hash(reference.vertices,reference.faces),reference_annotation_hash=reference.annotation_hash(),
        verified='connectivity_under_user_supplied_correspondence' if mode=='mapped' else 'distance_normal_proposal_not_registration',
        max_distance_m=max_distance_m if mode=='nearest' else None)
    refresh_summary(out)
    return out


def refresh_summary(model:Model) -> None:
    """Update derived label counts after paint/transfer; preserve original source."""
    if model.metadata.get('semantic_schema')!=SCHEMA:return
    score=np.asarray(model.metadata.get('segmentation_face_scores',np.zeros(len(model.faces))))
    _,_,area=face_geometry(model.vertices,model.faces)
    seg=model.metadata['segmentation']
    seg['regions']=[dict(id=p['id'],key=p['key'],faces=int((model.labels==p['id']).sum()),
        area_m2=float(area[model.labels==p['id']].sum()),mean_score=float(score[model.labels==p['id']].mean()) if (model.labels==p['id']).any() else None,
        status='present_proposed' if (model.labels==p['id']).any() else 'absent_or_not_observed') for p in model.parts]
    seg['unknown_faces']=int((model.labels==UNKNOWN).sum());seg['low_score_faces']=int((score<.5).sum())
    seg['review']=dict(status='pending',reviewer=None)
