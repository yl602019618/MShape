"""AeroGT v2: authored semantic scaffold + explicit dimensional landmarks.

Angles are sagittal CENTRELINE chord angles to horizontal, not every glass normal.
Tyres use uniform radial scaling and translation, never body-length stretching.
The assembly is a research concept and is not a boolean-unioned CFD solid.
"""
from __future__ import annotations
import copy
from functools import lru_cache
from typing import Literal, Annotated
import numpy as np
from scipy.interpolate import PchipInterpolator
from pydantic import BaseModel,ConfigDict,Field,model_validator
from scipy.spatial import cKDTree
from .model import Model,array_hash
from .design30 import design_model
from .components import variant
from .geometry import face_geometry

BODY_TYPES=('fastback','notchback','estateback','hatchback','suv')

class EngineeringParams(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    body_type:Literal['fastback','notchback','estateback','hatchback','suv']='fastback'
    # Absent on saved 0.6/0.8 records: replay their original map exactly.
    generator_revision:Literal['v2','v3','v4','v5','v6']='v2'
    wheelbase_m:float=Field(2.92,ge=2.45,le=3.20)
    front_overhang_m:float=Field(.992,ge=.68,le=1.12)
    rear_overhang_m:float=Field(1.0315,ge=.55,le=1.18)
    body_width_m:float=Field(1.97,ge=1.70,le=2.10)
    track_m:float=Field(1.774,ge=1.50,le=1.88)
    wheel_radius_m:float=Field(.355,ge=.30,le=.43)
    tyre_width_m:float=Field(.238,ge=.19,le=.29)
    floor_height_m:float=Field(.225,ge=.13,le=.30)
    roof_height_m:float=Field(1.51,ge=1.32,le=1.88)
    cabin_length_m:float=Field(1.34,ge=1.02,le=2.55)
    windshield_angle_deg:float=Field(32.,ge=25.,le=65.)
    rear_glass_angle_deg:float=Field(29.,ge=22.,le=82.)
    roof_crown_m:float=Field(.040,ge=.010,le=.18)
    rear_deck_height_m:float=Field(1.045,ge=.84,le=1.28)
    rear_shoulder_m:float=Field(.0,ge=-.035,le=.05)
    diffuser_length_m:float=Field(.57,ge=.30,le=.82)
    diffuser_angle_deg:float=Field(9.,ge=2.,le=18.)
    diffuser_width_m:float=Field(1.30,ge=.85,le=1.60)
    rear_deck_length_m:float=Field(.38,ge=0.,le=.85)
    tailgate_angle_deg:float=Field(78.,ge=55.,le=89.)
    rear_roof_drop_m:float=Field(.025,ge=-.12,le=.20)
    tailgate_lower_height_m:float=Field(.84,ge=.62,le=1.02)
    rear_belt_height_m:float=Field(.97,ge=.82,le=1.16)
    rear_body_taper:float=Field(.025,ge=0.,le=.18)
    cabin_tumblehome_deg:float=Field(4.,ge=0.,le=18.)
    hood_height_m:float=Field(1.055,ge=.90,le=1.18)
    hood_crown_m:float=Field(.015,ge=0.,le=.06)
    front_body_taper:float=Field(.025,ge=0.,le=.20)
    spoiler_style:Literal['none','lip','sport']='lip'
    mirror_style:str='aero'
    front_style:str='balanced'
    diffuser_style:str='smooth'
    fine_shape:tuple[Annotated[float,Field(ge=-.22,le=.22)], ...]|None=Field(None,min_length=70,max_length=70,exclude_if=lambda v:v is None)

    @model_validator(mode='after')
    def validate_fine_shape(self):
        if self.generator_revision!='v6' and (self.roof_crown_m>.085 or self.rear_roof_drop_m<0):
            raise ValueError('扩展车顶前后冠高需要 v6 生成器。')
        if self.fine_shape is not None and self.generator_revision!='v6':
            raise ValueError('精细造型控制需要 v6 生成器；旧配方保持原几何。')
        return self

    seed:int=Field(0,ge=0,le=2_147_483_647)

PRESETS={
 'fastback':{},
 'notchback':dict(cabin_length_m=1.15,rear_glass_angle_deg=43,roof_crown_m=.035,rear_deck_height_m=1.01,front_style='balanced'),
 'estateback':dict(wheelbase_m=2.96,rear_overhang_m=1.07,cabin_length_m=2.30,roof_height_m=1.57,rear_glass_angle_deg=65,rear_deck_height_m=1.07,mirror_style='touring'),
 'hatchback':dict(wheelbase_m=2.63,front_overhang_m=.82,rear_overhang_m=.68,body_width_m=1.81,track_m=1.60,wheel_radius_m=.325,tyre_width_m=.21,roof_height_m=1.48,floor_height_m=.19,cabin_length_m=1.99,windshield_angle_deg=40,rear_glass_angle_deg=66,rear_deck_height_m=.99,mirror_style='compact',diffuser_length_m=.39,diffuser_width_m=1.12,front_style='closed'),
 'suv':dict(wheelbase_m=2.91,front_overhang_m=.91,rear_overhang_m=.94,body_width_m=2.01,track_m=1.77,wheel_radius_m=.399,tyre_width_m=.25,roof_height_m=1.74,floor_height_m=.26,cabin_length_m=2.38,windshield_angle_deg=51,rear_glass_angle_deg=68,rear_deck_height_m=1.16,roof_crown_m=.045,mirror_style='touring',front_style='upright',diffuser_length_m=.55,diffuser_width_m=1.34)
}
EXPANDED_PRESETS={
 'fastback':dict(rear_deck_length_m=.30,rear_glass_angle_deg=32,cabin_length_m=1.34,rear_roof_drop_m=.025),
 'notchback':dict(rear_deck_length_m=.66,tailgate_angle_deg=81,rear_roof_drop_m=.015,rear_belt_height_m=.94),
 'estateback':dict(rear_deck_length_m=0.,tailgate_angle_deg=78,rear_roof_drop_m=.025,tailgate_lower_height_m=.84,rear_belt_height_m=.97,cabin_length_m=2.05,rear_glass_angle_deg=43),
 'hatchback':dict(rear_deck_length_m=0.,tailgate_angle_deg=77,rear_roof_drop_m=.035,tailgate_lower_height_m=.79,rear_belt_height_m=.90,cabin_length_m=1.80,rear_glass_angle_deg=55,hood_height_m=1.015),
 'suv':dict(rear_deck_length_m=0.,tailgate_angle_deg=84,rear_roof_drop_m=.025,tailgate_lower_height_m=.91,rear_belt_height_m=1.065,cabin_length_m=2.21,rear_glass_angle_deg=70,hood_height_m=1.12,cabin_tumblehome_deg=2.)
}
PARAMETER_INFO={
 'wheelbase_m':('轴距','mm','比例','前后轮轴心间距'),
 'front_overhang_m':('前悬','mm','比例','前轴至整车最前端的纵向距离'),
 'rear_overhang_m':('后悬','mm','比例','后轴至整车最后端的纵向距离'),
 'body_width_m':('车身基础宽度','mm','比例','基础外板宽度；后肩参数可能继续增加实测宽度'),
 'track_m':('轮距','mm','轮组','左右车轮中心距离，前后相同'),
 'wheel_radius_m':('轮胎外半径','mm','轮组','径向等比缩放；轮胎不随轴距拉伸'),
 'tyre_width_m':('轮胎宽度','mm','轮组','轴向胎宽，轮毂随同适配'),
 'floor_height_m':('中部底板离地','mm','比例','底板中心的名义高度，不是全车最小离地间隙'),
 'roof_height_m':('车顶中心控制点高度','mm','座舱','中心纵剖面的冠顶控制点'),
 'cabin_length_m':('车顶控制段长度','mm','座舱','前/后风挡上端间的纵向距离'),
 'windshield_angle_deg':('前风挡弦线角','°','座舱','中心剖面上下端连线与水平面夹角'),
 'rear_glass_angle_deg':('后风挡弦线角','°','座舱','中心剖面上下端连线与水平面夹角'),
 'roof_crown_m':('车顶纵向冠高','mm','座舱','车顶中间控制点相对前缘的高度；后缘落差由独立参数控制'),
 'rear_deck_height_m':('后风挡下端高度','mm','尾部','后玻璃下缘的中心控制点高度'),
 'rear_shoulder_m':('后肩单侧外扩','mm','尾部','后翼子板带内平滑外扩量'),
 'diffuser_length_m':('扩散器控制段长度','mm','底部','车尾之前的渐升底部段'),
 'diffuser_angle_deg':('扩散器中心弦线角','°','底部','控制段入口到出口的中心弦线角'),
 'diffuser_width_m':('扩散器控制带宽','mm','底部','两侧平滑过渡到未修改底板；非切开替换零件'),
 'rear_deck_length_m':('独立尾箱平台长度','mm','尾部架构','0 将原尾箱面连续变成尾门；保持车长/座舱长度时，平台变长会使座舱向前移'),
 'tailgate_angle_deg':('尾门弦线角','°','尾部架构','独立于后玻璃角度，尾门上下缘弦线与水平面的夹角'),
 'rear_roof_drop_m':('车顶后缘落差','mm','尾部架构','车顶后缘相对前缘的下降量；改变旅行车平顶或溜背趋势'),
 'tailgate_lower_height_m':('尾门下缘高度','mm','尾部架构','尾门到后围板的中心连接高度'),
 'rear_belt_height_m':('后侧腰线高度','mm','尾部架构','后玻璃下缘附近的外板腰线；须低于玻璃下缘'),
 'rear_body_taper':('后车身收窄比例','%','尾部架构','从后轴至尾端的单侧连续收窄比例'),
 'cabin_tumblehome_deg':('座舱额外内倾角','°','座舱','车窗带以上的额外向内倾角，沿腰线平滑衔接'),
 'hood_height_m':('发动机盖后缘高度','mm','前部架构','前玻璃下缘的中心高度；与玻璃角度共同决定座舱前缘'),
 'hood_crown_m':('发动机盖冠高','mm','前部架构','发动机盖中部的平滑冠高，端部保持衔接'),
 'front_body_taper':('车头收窄比例','%','前部架构','从前轴至车头的连续收窄比例'),
}

def preset(body_type,revision='v5'):
    if body_type not in PRESETS:raise ValueError('Unsupported body type')
    if revision not in ('v2','v3','v4','v5','v6'):raise ValueError('Unsupported generator revision')
    values={**PRESETS[body_type],**(EXPANDED_PRESETS[body_type] if revision in ('v3','v4','v5','v6') else {})}
    if revision in ('v5','v6'):values['spoiler_style']='lip' if body_type=='fastback' else 'none'
    return EngineeringParams(body_type=body_type,generator_revision=revision,**values)

def schema():
    from .shape_controls import schema as shape_schema
    props=EngineeringParams.model_json_schema()['properties'];fields=[]
    for key,(name,unit,group,meaning) in PARAMETER_INFO.items():
        p=props[key];scale=1000 if unit=='mm' else (100 if unit=='%' else 1)
        fields.append(dict(key=key,name=name,unit=unit,group=group,meaning=meaning,
            min=p['minimum']*scale,max=p['maximum']*scale,step=1 if unit=='mm' else .5,ui_scale=scale))
        if key=='rear_shoulder_m':
            fields[-1].update(min=0,revision_min={'v2':-35,'v3':-35,'v4':0,'v5':0})
        if key=='roof_crown_m':fields[-1].update(max=85,revision_max={'v6':180})
        if key=='rear_roof_drop_m':fields[-1].update(min=0,revision_min={'v6':-120})
    return dict(schema='aeroshape.engineering.v6',shape_controls=shape_schema(),fields=fields,body_types=list(BODY_TYPES),
                presets={k:preset(k).model_dump() for k in BODY_TYPES},legacy_presets={k:preset(k,'v2').model_dump() for k in BODY_TYPES},angles='centreline_chords_to_horizontal',
                revision_presets={revision:{k:preset(k,revision).model_dump() for k in BODY_TYPES} for revision in ('v2','v3','v4','v5','v6')},
                discrete_choices=['mirror_style','front_style','diffuser_style','spoiler_style'],random_seed_mode='parameter_sampling_only_no_hidden_vertex_noise')

KNOTS=np.array([-2.45,-2.22,-1.7,-1.05,-.85,-.40,0,.68,.96,1.33,1.70,2.12,2.43])
TOP=PchipInterpolator(KNOTS,[.765,.855,.995,1.055,1.17,1.463,1.505,1.493,1.442,1.265,1.05,1.015,.965],extrapolate=True)
BELT=PchipInterpolator(KNOTS,[.715,.825,.943,1.005,1.010,1.020,1.030,1.030,1.025,1.002,.975,.928,.889],extrapolate=True)

def smooth(x):
    q=np.clip(x,0,1);return q*q*(3-2*q)

def _refined_sidewall_offset(points):
    """Replace the authored door scallop with one shallow outward crown.

    Canonical coordinates keep the correction attached to both doors and the
    rear quarter through dimensional changes. The old -20 mm Gaussian trough
    otherwise becomes especially sharp above an arch, where the same profile
    is compressed into a short vertical span. Lower-rim and shared-belt
    vertices stay fixed; smooth spatial gates retain the nose and end joins.
    """
    x,y,z=np.asarray(points).T
    lower=np.full_like(x,.225)
    for axle in (-1.49,1.43):
        dx=x-axle
        arch=.355+np.sqrt(np.maximum(.405**2-dx*dx,0.))
        lower=np.maximum(lower,np.where(np.abs(dx)<=.405,arch,.225))
    t=np.clip((z-lower)/np.maximum(BELT(x)-lower,.05),0.,1.)
    wave=np.sin(np.pi*t)
    authored=(.016*np.exp(-((t-.82)/.10)**2)-.020*np.exp(-((t-.37)/.18)**2))*wave
    # A broad crown with zero endpoint slope retains the rim tangents and
    # avoids replacing the trough with an over-inflated arch shoulder.
    crown=(.003+.004*smooth((lower-.45)/.25))*wave*wave
    longitudinal=smooth((x+1.30)/.25)*(1-smooth((x-1.90)/.25))
    lateral=smooth((np.abs(y)-.65)/.20)
    return np.sign(y)*(crown-authored)*longitudinal*lateral

@lru_cache(maxsize=1)
def reference():return design_model()


def generate(params:EngineeringParams|dict) -> Model:
    p=params if isinstance(params,EngineeringParams) else EngineeringParams.model_validate(params)
    if p.generator_revision not in ('v5','v6') and p.spoiler_style!='lip':raise ValueError('旧版配方不支持选装尾翼；升级造型后可选择。')
    expanded=p.generator_revision in ('v3','v4','v5','v6')
    mirror=variant('mirror',p.mirror_style);nose=variant('front',p.front_style);diff=variant('diffuser',p.diffuser_style)
    base=reference();v0=base.vertices;v=v0.copy();by={x['key']:x['id'] for x in base.parts}
    # New fixtures use measured proportions on the same authored topology.
    # The old style map remains exact for saved v2/v3 parameter records.
    mirror_report=None
    if p.generator_revision in ('v4','v5','v6'):
        from .mirrors import reshape_mirrors
        v,mirror_report=reshape_mirrors(base,p.mirror_style)
    else:
        for side in ('l','r'):
            ids=np.unique(base.faces[base.labels==by['mirror_'+side]])
            sign=1 if side=='l' else -1;root=np.array([-.86,sign*.932,1.065]);delta=v[ids]-root
            w=smooth((np.abs(v[ids,1])-.938)/.066)
            v[ids]+=delta*(np.array(mirror['scale'])-1)*w[:,None]
    styled=base.clone(vertices=v);styled.metadata['base_geometry_hash']=array_hash(v,base.faces)
    # Basic coupled feasibility constraints; violations reject, never silently clamp.
    if p.generator_revision in ('v4','v5','v6') and p.rear_shoulder_m<0:
        raise ValueError('后肩外扩量不能为负；当前构型保持后门与后翼子板的饱满侧面。')
    if p.track_m+p.tyre_width_m>p.body_width_m+.10:raise ValueError('轮距 + 胎宽超出当前轮拱适配域；减小轮距/胎宽或增大车宽。')
    if p.roof_height_m-p.rear_deck_height_m<.22:raise ValueError('后风挡高度不足；降低后甲板或提高车顶。')
    xmin,xmax=float(v0[:,0].min()),float(v0[:,0].max());front=-p.wheelbase_m/2;rear=p.wheelbase_m/2
    target_min=front-p.front_overhang_m;target_max=rear+p.rear_overhang_m
    radial=(p.wheel_radius_m+.05)/.405
    xs=np.array([xmin,-1.895,-1.49,-1.085,1.025,1.43,1.835,xmax])
    xt=np.array([target_min,front-.405*radial,front,front+.405*radial,rear-.405*radial,rear,rear+.405*radial,target_max])
    if np.any(np.diff(xt)<.07):raise ValueError('轴距/前后悬不足以容纳轮拱和端面。')
    xmap=PchipInterpolator(xs,xt)
    def longitudinal(x):
        r=xmap(x)
        for wx,tx in [(-1.49,front),(1.43,rear)]:
            r=np.where(np.abs(x-wx)<=.405,tx+(x-wx)*radial,r)
        return r
    def lower(x,y,z):
        out=z+(p.floor_height_m-.225)
        for wx in (-1.49,1.43):
            weight=(1-smooth((np.abs(x-wx)-.405)/.20))*(1-smooth((z-.79)/.26))*smooth((np.abs(y)-.50)/.18)
            arch=p.wheel_radius_m+(z-.355)*radial
            out=out*(1-weight)+arch*weight
        if expanded:
            hood_shift=p.hood_height_m-(float(TOP(-1.06))+(p.floor_height_m-.225))
            hood_weight=smooth((x-xmin)/(-1.06-xmin))*(1-smooth((x+.90)/.30))*smooth((z-.45)/.50)
            out+=hood_shift*hood_weight
            # Smooth rear cross-section reshaping, with the floor fixed and a
            # positive gap between belt and upper profile even at the tail.
            belt_shift=(p.rear_belt_height_m-float(BELT(1.62))-(p.floor_height_m-.225))*smooth((x-.05)/1.35)
            tail_weight=smooth((x-1.62)/(xmax-1.62))
            tail_shift=p.tailgate_lower_height_m-.075-float(BELT(xmax))-(p.floor_height_m-.225)
            belt_shift=belt_shift*(1-tail_weight)+tail_shift*tail_weight
            out+=belt_shift*smooth((z-.40)/.50)
        return out
    lower_a=float(lower(np.array([-1.06]),np.array([0.]),np.array([TOP(-1.06)]))[0])
    roof_endpoint=p.roof_height_m-p.roof_crown_m
    a_x=float(longitudinal(np.array([-1.06]))[0]);a_upper=a_x+(roof_endpoint-lower_a)/np.tan(np.deg2rad(p.windshield_angle_deg))
    rear_upper=a_upper+p.cabin_length_m
    rear_lower=rear_upper+(roof_endpoint-p.rear_deck_height_m)/np.tan(np.deg2rad(p.rear_glass_angle_deg))
    if expanded:
        if p.rear_deck_height_m-p.rear_belt_height_m<.045:raise ValueError('后玻璃下缘须高于后腰线至少45 mm。')
        if p.rear_deck_height_m-p.tailgate_lower_height_m<.10:raise ValueError('尾门上下缘高度差须至少100 mm。')
        rear_roof_z=roof_endpoint-p.rear_roof_drop_m
        if rear_roof_z-p.rear_deck_height_m<.16:raise ValueError('后车顶下降过大，后玻璃可用高度不足160 mm。')
        # Solve from the tail forward. This gives the glass and the lower
        # tailgate independent chords and a deck that can reach exactly zero.
        gate_run=(p.rear_deck_height_m-p.tailgate_lower_height_m)/np.tan(np.deg2rad(p.tailgate_angle_deg))
        gate_start=target_max-gate_run
        rear_lower=gate_start-p.rear_deck_length_m
        rear_upper=rear_lower-(rear_roof_z-p.rear_deck_height_m)/np.tan(np.deg2rad(p.rear_glass_angle_deg))
        a_upper=rear_upper-p.cabin_length_m
        lower_a=p.hood_height_m
        a_x=a_upper-(roof_endpoint-lower_a)/np.tan(np.deg2rad(p.windshield_angle_deg))
        if a_x<target_min+.58 or a_x>front+1.05:raise ValueError('座舱与尾部参数使发动机盖或前风挡超出前舱布置域；缩短座舱/平台或调整玻璃角度。')
    if a_upper-a_x<.12 or (not expanded and rear_lower>=target_max-.12):
        raise ValueError('玻璃角度与座舱长度使后风挡越过尾部，或前风挡倒置；缩短车顶段、增大角度或增加后悬。')
    # Spatial warp across a continuous skin; semantic boundaries follow the same surface.
    sx=np.array([xmin,-1.06,-.40,.27,.94,1.62,xmax])
    tx=np.array([target_min,a_x,a_upper,(a_upper+rear_upper)/2,rear_upper,rear_lower,target_max])
    tz=np.array([float(lower(np.array([xmin]),np.array([0.]),np.array([TOP(xmin)]))[0]),lower_a,
                 roof_endpoint,p.roof_height_m,roof_endpoint,p.rear_deck_height_m,
                 .965+(p.floor_height_m-.225)])
    if expanded:
        # At zero deck the inherited trunk triangles become the upper and lower
        # tailgate. No station coincides, no face is deleted or flattened.
        deck_blend=float(smooth(p.rear_deck_length_m/.18))
        tail_middle_x=(rear_lower+target_max)/2*(1-deck_blend)+gate_start*deck_blend
        tail_middle_z=(p.rear_deck_height_m+p.tailgate_lower_height_m)/2*(1-deck_blend)+p.rear_deck_height_m*deck_blend
        sx=np.array([xmin,-1.06,-.40,.27,.94,1.62,1.98,xmax])
        tx=np.array([target_min,a_x,a_upper,(a_upper+rear_upper)/2,rear_upper,rear_lower,tail_middle_x,target_max])
        tz=np.array([float(lower(np.array([xmin]),np.array([0.]),np.array([TOP(xmin)]))[0]),lower_a,roof_endpoint,
                     p.roof_height_m,rear_roof_z,p.rear_deck_height_m,tail_middle_z,p.tailgate_lower_height_m])
    if np.any(np.diff(tx)<=(.001 if expanded else .035)):raise ValueError('座舱参数产生非单调纵向控制站点。')
    upper_map=PchipInterpolator(sx,tx);top_map=PchipInterpolator(sx,tz)
    width_scale=p.body_width_m/1.964
    def warp(points):
        x,y,z=points.T
        # Where upper and belt profiles meet at the nose, do not divide by zero.
        q=np.clip((z-BELT(x))/np.maximum(TOP(x)-BELT(x),.012),0,1)*smooth((x+1.25)/.19)
        low=lower(x,y,z);lowtop=lower(x,np.zeros_like(y),TOP(x))
        out=np.c_[longitudinal(x)+(upper_map(x)-longitudinal(x))*q,y*width_scale,low+(top_map(x)-lowtop)*q]
        if p.generator_revision in ('v5','v6'):
            # Use one chord for glass and A-pillars, with a small outward crown.
            # Blend down through the pillar into the sidewall. The .825
            # authored height ratio includes the complete glass/A-pillar
            # boundary while avoiding a hard cowl crease on lower side faces.
            t=np.clip((x+1.06)/.66,0.,1.)
            weight=smooth(t/.06)*(1-smooth((t-.94)/.06))*smooth(q/.825)
            endpoints=[]
            for at in (-1.06,-.40):
                xx=np.full_like(x,at)
                zz=BELT(xx)+(TOP(xx)-BELT(xx))*q
                low_at=lower(xx,y,zz)
                top_at=lower(xx,np.zeros_like(y),TOP(xx))
                endpoints.append(np.c_[longitudinal(xx)+(upper_map(xx)-longitudinal(xx))*q,
                    low_at+(top_map(xx)-top_at)*q])
            a,b=endpoints;delta=b-a
            norm=np.linalg.norm(delta,axis=1)
            outward=np.c_[-delta[:,1],delta[:,0]]/np.maximum(norm[:,None],1e-12)
            offset=.006*np.sin(np.pi*t)*q-np.einsum('ij,ij->i',out[:,[0,2]]-a,outward)
            # Correct inward sag without pulling the naturally convex roof
            # shoulder inward. A smooth positive part avoids a new crease.
            lift=.5*(offset+np.sqrt(offset*offset+.0005**2))
            out[:,[0,2]]+=outward*(lift*weight)[:,None]
        if p.generator_revision in ('v4','v5','v6'):
            out[:,1]+=_refined_sidewall_offset(points)*width_scale
        shoulder=smooth((x-.1)/.70)*(1-smooth((x-1.75)/.6))*smooth((np.abs(y)-.60)/.26)*np.exp(-((z-1.00)/.25)**4)
        out[:,1]+=np.sign(y)*p.rear_shoulder_m*shoulder
        if expanded:
            rear_taper=smooth((x-1.15)/(xmax-1.15))
            front_taper=1-smooth((x-xmin)/(-1.38-xmin))
            out[:,1]*=1-p.rear_body_taper*rear_taper-p.front_body_taper*front_taper
            cabin_gate=smooth((x+1.25)/.25)*(1-smooth((x-1.62)/.35))
            inward=np.maximum(out[:,2]-lower(x,y,BELT(x)),0)*np.tan(np.deg2rad(p.cabin_tumblehome_deg))
            out[:,1]-=np.sign(y)*inward*cabin_gate*smooth(np.abs(y)/.55)
            hood_gate=smooth((x-xmin)/.4)*(1-smooth((x+1.35)/.30))
            out[:,2]+=p.hood_crown_m*hood_gate*(1-np.clip(np.abs(y)/.98,0,1)**2)*smooth((z-.55)/.40)
        # Front options keep the windscreen interface and longitudinal endpoints fixed.
        front_weight=smooth((x-xmin)/.23)*(1-smooth((x+1.4)/.34))
        out[:,0]+=nose['sculpt'][0]*front_weight*(1-np.clip(np.abs(y)/1.,0,1)**2)*(1-smooth((z-.9)/.2))
        out[:,2]+=nose['sculpt'][1]*front_weight*smooth((z-.25)/.40)*(1-q)
        # Diffuser below the rear axle: blend inlet and side rails, preserving the shared skin.
        inlet=target_max-p.diffuser_length_m;t=np.clip((out[:,0]-inlet)/p.diffuser_length_m,0,1)
        lateral=1-smooth((np.abs(out[:,1])-.32*p.diffuser_width_m)/(.18*p.diffuser_width_m))
        # Add a displacement field around the floor, do not flatten all nearby faces.
        floor_ref=.225+.10*np.clip((x-1.8)/.63,0,1)**1.3
        gate=(1-smooth((z-floor_ref-.08)/.35))*lateral*smooth(t/.18)
        rise=p.diffuser_length_m*np.tan(np.deg2rad(p.diffuser_angle_deg))
        desired=p.floor_height_m+rise*t
        channel=diff['shape']*np.sin(2*np.pi*out[:,1]/max(p.diffuser_width_m,.1))**2*np.sin(np.pi*t)**2
        floor_warp=lower(x,y,floor_ref)
        out[:,2]+=(desired+channel-floor_warp)*gate
        if p.generator_revision=='v6' and p.fine_shape is not None:
            from .shape_controls import displacement
            out+=displacement(out,p.fine_shape,p.wheelbase_m,p.wheel_radius_m)
        return out
    out=warp(v)
    # Every tyre/hoop/spoke uses one circular radial scale, plus a separate axial width.
    wheel_ids=[]
    for ax,tx_,oldx in [('f',front,-1.49),('b',rear,1.43)]:
        for side,sign in [('l',1),('r',-1)]:
            ids=np.unique(base.faces[base.labels==by[f'wheel_{ax}{side}']]);wheel_ids.extend(ids)
            q=v[ids]-[oldx,sign*.887,.355]
            q[:,[0,2]]*=p.wheel_radius_m/.355;q[:,1]*=p.tyre_width_m/.238
            out[ids]=q+[tx_,sign*p.track_m/2,p.wheel_radius_m]
    # Mirrors/handles are rigid around exact on-parent anchor locations. This avoids
    # stretching mirrors when raising a cabin. Lamp strips/spoiler follow skin warp
    # because they are conformal features, not separate stiff bolt-on modules.
    follower_reports=[]
    for binding in styled.metadata.get('rigid_followers',[]):
        if not binding['name'].startswith(('mirror_assembly','handles_','drl_','taillamp_')):continue
        ids=np.asarray(binding['vertices']);side=1 if v[ids,1].mean()>0 else -1
        if side<0:continue  # construct the negative-Y peer by verified reference reflection
        parentids=np.flatnonzero(np.isin(base.labels,binding['parent_parts']))
        parentv=np.unique(base.faces[parentids]);center=v[ids].mean(0)
        if p.generator_revision in ('v4','v5','v6') and binding['name'].startswith('mirror_assembly'):
            # A new housing's centroid must not move the original mounting site.
            center=v0[ids].mean(0)
        distances=np.linalg.norm(v0[parentv]-center,axis=1);close=parentv[np.argsort(distances)[:24]]
        a=v0[close];b=warp(a);ca=a.mean(0);cb=b.mean(0)
        u,_,vt=np.linalg.svd((a-ca).T@(b-cb));signs=np.eye(3);signs[-1,-1]=np.linalg.det(u@vt);rot=u@signs@vt
        out[ids]=(v[ids]-ca)@rot+cb
        # A rigid fixture may protrude after parent-fit. Fit the ENTIRE fixture
        # to the prescribed front/rear envelope, not individual vertex clipping.
        envelope_dx=max(0.,target_min-float(out[ids,0].min()))-max(0.,float(out[ids,0].max())-target_max)
        out[ids,0]+=envelope_dx
        if p.generator_revision in ('v4','v5','v6') and binding['name'].startswith('mirror_assembly'):
            from .mirrors import mirror_seating_translation
            shift,seating=mirror_seating_translation(base,out)
            out[ids]+=shift
            mirror_report['seating']=seating
        refq=v0[ids].copy();refq[:,1]*=-1;d,partners=cKDTree(v0).query(refq)
        if d.max()>1e-7:raise ValueError('Authored component reflection correspondence lost.')
        out[partners]=out[ids]*[1,-1,1]
        follower_reports.append(dict(name=binding['name'],mode='rigid_anchor_fit',envelope_translation_m=envelope_dx,max_anchor_residual_m=float(np.linalg.norm((a-ca)@rot+cb+np.array([envelope_dx,0,0])-b,axis=1).max())))
    _,n0,a0=face_geometry(v0,base.faces);_,n1,a1=face_geometry(out,base.faces)
    area=float(np.min(a1/np.maximum(a0,1e-30)))
    if not np.isfinite(out).all() or area<.025:raise ValueError('Generated candidate collapses a surface triangle; parameters rejected.')
    if p.generator_revision in ('v5','v6') and p.spoiler_style=='sport':
        from .optional_components import shape_spoiler
        out=shape_spoiler(base,out,'sport')
    # Measurements on the requested scaffold AND nearest actual mesh vertices; never
    # claim an input angle alone proves the realised mesh has that angle everywhere.
    landmarks_ref=np.array([[-1.06,0,TOP(-1.06)],[-.4,0,TOP(-.4)],[.27,0,TOP(.27)],[.94,0,TOP(.94)],[1.62,0,TOP(1.62)]])
    # TOP(0.94) matches authored skin; q=1 at all centreline glass boundaries.
    landmarks=warp(landmarks_ref)
    nearest_dist,nearest=cKDTree(v0).query(landmarks_ref)
    actual=out[nearest]
    def chord(a,b):return float(np.degrees(np.arctan2(abs(b[2]-a[2]),abs(b[0]-a[0]))))
    centres=[]
    for key in ('wheel_fl','wheel_bl'):
        wv=out[np.unique(base.faces[base.labels==by[key]])]
        centres.append((wv.min(0)+wv.max(0))/2)
    measured=dict(wheelbase_m=float(centres[1][0]-centres[0][0]),front_overhang_m=float(front-out[:,0].min()),rear_overhang_m=float(out[:,0].max()-rear),
        actual_length_m=float(np.ptp(out[:,0])),actual_width_including_mirrors_m=float(np.ptp(out[:,1])),actual_height_m=float(np.ptp(out[:,2])),
        windshield_centreline_angle_deg=chord(actual[0],actual[1]),rear_glass_centreline_angle_deg=chord(actual[3],actual[4]),
        roof_crown_control_m=float(landmarks[2,2]-(landmarks[1,2]+landmarks[3,2])/2),
        angle_measurement='two actual mesh vertices nearest authored centreline boundary landmarks',landmark_reference_residual_max_m=float(nearest_dist.max()),
        control_landmarks_m=landmarks.tolist(),actual_landmarks_m=actual.tolist(),landmark_vertex_ids=nearest.tolist())
    if expanded:
        gate_source_x=1.62 if p.rear_deck_length_m<.001 else 1.98
        gate_reference=np.array([[gate_source_x,0.,TOP(gate_source_x)],[xmax,0.,TOP(xmax)]])
        gate_distance,gate_nearest=cKDTree(v0).query(gate_reference)
        gate_actual=out[gate_nearest]
        measured.update(rear_deck_control_length_m=p.rear_deck_length_m,tailgate_control_angle_deg=p.tailgate_angle_deg,
            tailgate_centreline_angle_deg=chord(gate_actual[0],gate_actual[1]),tailgate_actual_landmarks_m=gate_actual.tolist(),
            tailgate_landmark_reference_residual_max_m=float(gate_distance.max()),
            roof_rear_drop_control_m=p.rear_roof_drop_m,rear_architecture='closed_tailgate' if p.rear_deck_length_m<.001 else 'separate_rear_deck',
            tailgate_control_landmarks_m=[[gate_start,0.,p.rear_deck_height_m],[target_max,0.,p.tailgate_lower_height_m]],
            rear_profile_control_landmarks_m=np.c_[tx,np.zeros_like(tx),tz].tolist(),
            rear_deck_measurement='analytic profile control length; zero reuses the entire authored deck as descending tailgate')
    meta=copy.deepcopy(base.metadata)
    if mirror_report is not None:meta['mirror_geometry']=mirror_report
    meta.update(name=f'AeroGT {p.body_type.title()} · {p.seed:04d}',body_type=p.body_type,
        family_id='aerogt-engineering-'+p.generator_revision+'-'+p.body_type,asset_license='CC0-1.0',
        generator='aeroshape.engineering.v2',generator_parameters=p.model_dump(),
        generator_revision=p.generator_revision,
        generator_measurements=measured,generator_screen=dict(min_triangle_area_ratio=area,rigid_attachment_fits=follower_reports,
            status='dimensional_screen_only',self_intersection_status='not_checked',component_clearance_status='not_checked'),
        provenance=dict(source_type='generated',dataset_version={'v2':'aeroshape-0.6','v3':'aeroshape-0.9','v4':'aeroshape-0.10','v5':'aeroshape-0.11','v6':'aeroshape-0.12'}[p.generator_revision],dataset_origin_verification='original_authored_template',attribution='AeroShape original concept; not OEM/DrivAer'),
        component_choices=dict(mirror=p.mirror_style,front=p.front_style,diffuser=p.diffuser_style),
        component_interface='aerogt30-v2',base_geometry_hash=array_hash(out,base.faces),
        canonical_geometry_hash=array_hash(base.vertices,base.faces),canonical_topology_preserved=True,
        annotation_status='authored_semantics_preserved_by_vertex_identity',cfd_ready=False)
    # Preserve canonical 30 labels without invoking the low-accuracy inference path.
    meta['semantic_schema']='aeroshape.design30.v1'
    meta['segmentation_face_scores']=[1.]*len(base.faces);meta['segmentation_face_methods']=[2]*len(base.faces)
    meta['segmentation']={'method':'authored_scaffold','symmetry':{'vertex_correspondence':True},'landmarks':{'values':{'hood_end':.34,'roof_start':.46,'roof_end':.69,'rear_glass_end':.84,'belt_height':.58}},'review':{'status':'pending'},'scores_are_probabilities':False}
    from .semantic import refresh_summary
    model=base.clone(vertices=out,metadata=meta)
    if p.generator_revision in ('v5','v6'):
        from .optional_components import configure_optional
        model=configure_optional(model,p.spoiler_style)
        model.metadata['component_choices']['spoiler']=p.spoiler_style
        model.metadata['base_geometry_hash']=array_hash(model.vertices,model.faces)
    refresh_summary(model)
    return model


def sample_parameters(base:EngineeringParams,count:int,seed:int,variation:float=.35):
    if not 1<=count<=256 or not 0<=variation<=1:raise ValueError('Batch count 1..256; variation 0..1.')
    rng=np.random.default_rng(seed)
    ranges=dict(wheelbase_m=.16,front_overhang_m=.10,rear_overhang_m=.12,body_width_m=.08,roof_height_m=.08,
                cabin_length_m=.10,windshield_angle_deg=5.,rear_glass_angle_deg=5.,rear_shoulder_m=.025,
                diffuser_angle_deg=3.,diffuser_width_m=.10,roof_crown_m=.015)
    if base.generator_revision in ('v3','v4','v5','v6'):
        ranges.update(rear_deck_length_m=.10,tailgate_angle_deg=4.,rear_roof_drop_m=.025,tailgate_lower_height_m=.035,
            rear_belt_height_m=.025,rear_body_taper=.025,cabin_tumblehome_deg=2.,hood_height_m=.03,hood_crown_m=.01,front_body_taper=.025)
    props=EngineeringParams.model_json_schema()['properties']
    for i in range(count):
        d=base.model_dump()
        # Randomness is explicit in the emitted physical parameter vector, not hidden noise.
        for k,r in ranges.items():
            value=float(d[k]+rng.uniform(-1,1)*r*variation)
            # Native zero-deck and zero-taper presets sit on meaningful physical
            # boundaries; sampling stays in their explicit scalar domains.
            d[k]=float(np.clip(value,props[k]['minimum'],props[k]['maximum'])) if base.generator_revision in ('v3','v4','v5','v6') else value
            if base.generator_revision!='v6' and k=='roof_crown_m':d[k]=min(.085,d[k])
            if base.generator_revision!='v6' and k=='rear_roof_drop_m':d[k]=max(0.,d[k])
            if base.generator_revision in ('v4','v5','v6') and k=='rear_shoulder_m':d[k]=max(0.,d[k])
        d['seed']=int(rng.integers(1,2_147_483_647))
        yield EngineeringParams.model_validate(d)
