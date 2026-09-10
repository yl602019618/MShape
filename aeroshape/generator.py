from __future__ import annotations
from dataclasses import dataclass, asdict
from functools import lru_cache
import numpy as np
from scipy.interpolate import PchipInterpolator
from .design30 import design_model
from .model import Model, array_hash

ARCHETYPES={
 'fastback': dict(rear_drop=-.12, rear_lift=0.00, deck=0.00, height=1.00, width=1.00, length=1.00),
 'notchback': dict(rear_drop=-.20, rear_lift=-.02, deck=-.055, height=.99, width=.995, length=1.00),
 'estateback': dict(rear_drop=-.015, rear_lift=.16, deck=.055, height=1.035, width=1.01, length=1.015),
}

@dataclass(frozen=True)
class VehicleParams:
    archetype:str='fastback'
    length_scale:float=1.0
    width_scale:float=1.0
    cabin_height:float=0.0
    rear_roof:float=0.0
    rear_deck:float=0.0
    shoulder_width:float=0.0
    nose_length:float=0.0
    stance:float=0.0
    seed:int=0


def _smoothstep(t):
    t=np.clip(t,0.,1.); return t*t*(3.-2.*t)


def _part_masks(m:Model):
    by={p['key']:p['id'] for p in m.parts}
    wheel_ids=[p['id'] for p in m.parts if p['key'].startswith('wheel_')]
    wheel=np.isin(m.labels,wheel_ids)
    vwheel=np.zeros(len(m.vertices),bool); vwheel[np.unique(m.faces[wheel])]=True
    return by,vwheel


def generate_vehicle(params:VehicleParams|dict) -> Model:
    if isinstance(params,dict): params=VehicleParams(**params)
    if params.archetype not in ARCHETYPES: raise ValueError('archetype must be fastback, notchback, or estateback')
    if not (.92<=params.length_scale<=1.08 and .92<=params.width_scale<=1.08): raise ValueError('length/width scale outside design range')
    for value in [params.cabin_height,params.rear_roof,params.rear_deck,params.shoulder_width,params.nose_length,params.stance]:
        if not -.18<=value<=.18: raise ValueError('shape offsets must be within ±0.18 m')
    base=design_model(); v=base.vertices.copy(); by,vwheel=_part_masks(base)
    a=ARCHETYPES[params.archetype]
    # global proportion: preserve symmetry and topology. Wheel assemblies scale with the body longitudinally/laterally.
    v[:,0]*=params.length_scale*a['length']
    v[:,1]*=params.width_scale*a['width']
    # normalized longitudinal coordinate based on the authored reference span
    xmin,xmax=v[:,0].min(),v[:,0].max(); xn=(v[:,0]-xmin)/max(xmax-xmin,1e-9)
    # upper-body gate keeps underbody and wheel geometry stable while morphing greenhouse.
    z0=.82+params.stance; gate=_smoothstep((v[:,2]-z0)/.40)
    rear=_smoothstep((xn-.56)/.34)
    tail=_smoothstep((xn-.72)/.20)
    front=1-_smoothstep((xn-.18)/.22)
    center=1-np.minimum(1,np.abs(xn-.50)/.42)
    dz=(params.cabin_height + .035*(a['height']-1))*center*gate
    dz+=(a['rear_drop']+params.rear_roof)*rear*gate
    dz+=(a['rear_lift']+params.rear_deck)*tail*gate
    # notchback deck shoulder: recover a short horizontal rear deck while lowering rear glass.
    if params.archetype=='notchback':
        dz+=.105*_smoothstep((xn-.75)/.10)*(1-_smoothstep((xn-.93)/.05))*gate
    elif params.archetype=='estateback':
        dz+=.055*_smoothstep((xn-.62)/.15)*gate
    # nose proportion acts mostly ahead of front axle, but keeps topology unchanged.
    v[:,0]+=params.nose_length*front*(1-.35*gate)
    v[:,2]+=dz
    # shoulder width only influences upper side body, symmetrically.
    lateral=np.sign(v[:,1]); side_gate=_smoothstep((np.abs(v[:,1])-.55)/.28)*_smoothstep((v[:,2]-.65)/.35)
    v[:,1]+=lateral*params.shoulder_width*side_gate*(.45+.55*center)
    # stance is a body-only vertical offset. Locked wheel vertices remain fixed in Z.
    v[~vwheel,2]+=params.stance
    # deterministic small family variation: broad smooth modes only, not vertex noise.
    if params.seed:
        rng=np.random.default_rng(params.seed)
        amp=rng.uniform(-.012,.012,3)
        v[:,2]+=gate*(amp[0]*np.sin(np.pi*xn)+amp[1]*np.sin(2*np.pi*xn))*center
        v[:,1]+=np.sign(v[:,1])*amp[2]*side_gate*np.sin(np.pi*xn)
    meta=dict(base.metadata)
    meta.update(name=f"AeroGT {params.archetype.title()} · Parametric {params.seed:04d}",
                family_id=f"aerogt-generator-{params.archetype}",
                generator='aeroshape.generator.v1',generator_parameters=asdict(params),
                reference_type='original_parametric_concept_not_real_production_vehicle',
                source='AeroGT authored topology + smooth archetype morph',
                annotation_status='authored_30_region_template_propagated_by_shared_topology')
    meta['base_geometry_hash']=array_hash(v,base.faces)
    return base.clone(vertices=v,metadata=meta)


def sample_family(archetype:str,count:int=8,seed:int=42):
    if count<1 or count>512: raise ValueError('count must be 1..512')
    rng=np.random.default_rng(seed); out=[]
    for i in range(count):
        p=VehicleParams(archetype=archetype,
            length_scale=float(rng.uniform(.975,1.03)),width_scale=float(rng.uniform(.97,1.025)),
            cabin_height=float(rng.uniform(-.025,.045)),rear_roof=float(rng.uniform(-.045,.055)),
            rear_deck=float(rng.uniform(-.025,.045)),shoulder_width=float(rng.uniform(-.018,.025)),
            nose_length=float(rng.uniform(-.035,.055)),stance=float(rng.uniform(-.012,.012)),seed=seed*1000+i+1)
        out.append((p,generate_vehicle(p)))
    return out

@lru_cache(maxsize=3)
def archetype_reference(archetype:str):
    return generate_vehicle(VehicleParams(archetype=archetype)).clone()
