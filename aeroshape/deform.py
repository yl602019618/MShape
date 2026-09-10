from __future__ import annotations
from math import comb
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator
from scipy.spatial import cKDTree
from .model import Model
from .region import RegionSpec, prepare_region, follow_attachments


class Control(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    index: tuple[int,int,int]
    displacement: tuple[float,float,float]


class Operation(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    kind: Literal['bulge','width','taper','slope','ffd','rigid','region'] = 'bulge'
    name: str = Field(default='Design edit', max_length=100)
    bounds: tuple[tuple[float,float,float], tuple[float,float,float]]
    parts: list[int] = Field(default_factory=list, max_length=1000)
    mirror: bool = True
    amount: float = Field(default=0., ge=-2., le=2.)
    direction: tuple[float,float,float] = (0.,0.,1.)
    blend: float = Field(default=.08, ge=.001, le=2.)
    feather: float = Field(default=.15, ge=.02, le=.49)
    grid: tuple[int,int,int] = (4,3,3)
    controls: list[Control] = Field(default_factory=list, max_length=216)
    enabled: bool = True
    region: RegionSpec | None = None

    @model_validator(mode='after')
    def validate_geometry(self):
        if self.kind == "region" and self.region is None:
            raise ValueError("Regional handles require a region specification.")
        if any(h-l < 1e-6 for l,h in zip(*self.bounds)):
            raise ValueError('Each FFD box extent must be at least 1 micrometre.')
        if any(n < 2 or n > 6 for n in self.grid):
            raise ValueError('FFD grid dimensions must be 2..6.')
        for c in self.controls:
            if any(i < 0 or i >= n for i,n in zip(c.index,self.grid)):
                raise ValueError('Control index outside FFD grid.')
            if any(abs(d)>2 for d in c.displacement):
                raise ValueError('Control displacements limited to 2 m.')
        if any(abs(d)>10 for d in self.direction):
            raise ValueError('Direction component exceeds 10.')
        return self


class DesignRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    operations: list[Operation] = Field(default_factory=list, max_length=32)
    min_body_clearance: float = Field(default=0., ge=0., le=2.)
    max_displacement: float = Field(default=.5, gt=0., le=5.)
    min_jacobian: float = Field(default=.15, gt=0., le=.9)


def smoothstep(x):
    x = np.clip(x,0,1)
    return x*x*x*(10+x*(-15+6*x))


def raw_field(points: np.ndarray, op: Operation) -> np.ndarray:
    """Reference-coordinate field. Coefficients/displacements are in metres."""
    q = np.asarray(points, float)
    lo,hi = np.array(op.bounds)
    t = (q-lo)/(hi-lo)
    inside = np.all((t>=0)&(t<=1),axis=1)
    tc = np.clip(t,0,1)
    env = (smoothstep(tc/op.feather)*smoothstep((1-tc)/op.feather)).prod(axis=1)
    env *= inside
    out = np.zeros_like(q)
    if op.kind == 'ffd':
        for c in op.controls:
            w = np.ones(len(q))
            for axis,(idx,n) in enumerate(zip(c.index,op.grid)):
                z=tc[:,axis]; degree=n-1
                w *= comb(degree,idx)*z**idx*(1-z)**(degree-idx)
            out += w[:,None]*np.asarray(c.displacement)
    elif op.kind == 'bulge':
        out[:] = op.amount*np.asarray(op.direction)
    elif op.kind == 'width':
        out[:,1] = op.amount*(2*tc[:,1]-1)
    elif op.kind == 'taper':
        out[:,1] = -op.amount*(2*tc[:,1]-1)*smoothstep(tc[:,0])
    elif op.kind == 'slope':
        out[:,2] = op.amount*(2*tc[:,0]-1)
    elif op.kind == 'rigid':
        out[:] = op.amount*np.asarray(op.direction)
        if op.mirror:
            out[:,1] *= np.sign(q[:,1])
        return out  # Rigid selection uses complete part membership, not a fading box.
    return out*env[:,None]


class Deformation:
    """Additive rest-pose edits with symmetry and seam/locked-part constraints.

    For S=diag(1,-1,1), the symmetric field is (g(p)+S g(Sp))/2.
    All zero-displacement pins are mirrored as well. Therefore u(Sp)=S u(p)
    even for different left/right tessellations. This preserves a symmetric
    baseline surface; it does NOT make an asymmetric source mesh symmetric.
    """
    def __init__(self, model: Model, operations: list[Operation]):
        self.model=model
        self.tree=cKDTree(model.vertices)
        self.ops=[]
        self.regions=[]
        self.attachment_report=[]
        all_ids={p['id'] for p in model.parts}
        locked_ids={p['id'] for p in model.parts if p.get('locked',False)}
        locked=np.zeros(len(model.vertices),bool)
        if locked_ids:
            locked[np.unique(model.faces[np.isin(model.labels,list(locked_ids))])] = True
        self.locked=locked
        for op in operations:
            if not op.enabled: continue
            if op.kind == "region":
                self.regions.append((op,prepare_region(model,op.region,op.mirror)))
                continue
            if not set(op.parts).issubset(all_ids):
                raise ValueError('Operation references unknown part IDs.')
            selected=np.isin(model.labels,op.parts) if op.parts else np.ones(len(model.faces),bool)
            active=np.zeros(len(model.vertices),bool)
            active[np.unique(model.faces[selected])]=True
            nonselected=np.zeros(len(model.vertices),bool)
            nonselected[np.unique(model.faces[~selected])]=True
            boundary=active & nonselected
            active[locked]=False
            pins=boundary | locked
            pin_points=model.vertices[pins]
            # Mirrored pins also protect the opposite side after field projection.
            if op.mirror and len(pin_points):
                p=pin_points.copy(); p[:,1]*=-1
                pin_points=np.vstack((pin_points,p))
            pin_tree=cKDTree(pin_points) if len(pin_points) else None
            self.ops.append((op,active,pin_tree))

    def displacement(self, points: np.ndarray, chunk_size=100000) -> np.ndarray:
        q=np.asarray(points,float)
        if len(q)>chunk_size:
            return np.vstack([self.displacement(q[i:i+chunk_size],chunk_size) for i in range(0,len(q),chunk_size)])
        out=np.zeros_like(q)
        nearest=self.tree.query(q)[1]
        qm=q.copy(); qm[:,1]*=-1
        nearest_m=None
        for op,active,pins in self.ops:
            g=raw_field(q,op)*active[nearest,None]
            if op.mirror:
                if nearest_m is None: nearest_m=self.tree.query(qm)[1]
                gm=raw_field(qm,op)*active[nearest_m,None]
                gm[:,1]*=-1
                g=(g+gm)*.5
            if pins is not None:
                dist=pins.query(q)[0]
                # For rigid parts with shared seams this taper would cease to be rigid.
                # Such selections are rejected instead of silently distorting the part.
                if op.kind == 'rigid' and np.any(active & (pins.query(self.model.vertices)[0] < 1e-9)):
                    raise ValueError('Rigid selection shares vertices with a fixed part; separate the assembly first.')
                if op.kind != 'rigid':
                    g*=smoothstep(dist/op.blend)[:,None]
            out+=g
        return out

    def apply(self) -> np.ndarray:
        vertices=self.model.vertices+self.displacement(self.model.vertices)
        for op,region in self.regions:vertices+=op.amount*region['basis']
        if any(op.region.follow_attachments for op,_ in self.regions):
            vertices,self.attachment_report=follow_attachments(self.model,vertices,self.locked,mirror=all(op.mirror for op,_ in self.regions) and all(op.mirror for op,_,_ in self.ops))
        return vertices

    def jacobian_sample(self, max_points=1600) -> dict:
        if self.regions:
            return dict(min_det=None,max_gradient_norm=None,sample_count=0,
                        scope='surface_handle_no_volumetric_Jacobian',status='not_applicable')
        if not self.ops:
            return dict(min_det=1., max_gradient_norm=0., sample_count=0,
                        scope='sampled_not_global_injectivity_proof')
        points=[]
        for op,_,_ in self.ops:
            if op.kind=='rigid': continue
            lo,hi=np.array(op.bounds)
            grid=np.meshgrid(*[np.linspace(a+.03*(b-a),b-.03*(b-a),5) for a,b in zip(lo,hi)], indexing='ij')
            points.append(np.stack(grid,axis=-1).reshape(-1,3))
        # Surface locations catch seam/pin transition problems missed by box grids.
        ids=np.linspace(0,len(self.model.vertices)-1,min(500,len(self.model.vertices)),dtype=int)
        points.append(self.model.vertices[ids])
        p=np.vstack(points)
        if len(p)>max_points: p=p[np.linspace(0,len(p)-1,max_points,dtype=int)]
        eps=max(np.ptp(self.model.vertices,axis=0).max()*1e-5,1e-7)
        grad=np.zeros((len(p),3,3))
        for axis in range(3):
            d=np.zeros(3); d[axis]=eps
            grad[:,:,axis]=(self.displacement(p+d)-self.displacement(p-d))/(2*eps)
        det=np.linalg.det(grad+np.eye(3))
        norm=np.linalg.svd(grad,compute_uv=False)[:,0]
        return dict(min_det=float(det.min()), max_gradient_norm=float(norm.max()),
                    sample_count=len(p), scope='sampled_not_global_injectivity_proof')
