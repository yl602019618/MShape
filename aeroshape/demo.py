from __future__ import annotations
import numpy as np
from scipy.interpolate import PchipInterpolator
import trimesh
from .model import Model, default_parts
from .geometry import suggest_regions


def demo_model(family='fastback') -> Model:
    """Original illustrative loft, NOT a Tesla/DrivAer asset or CFD benchmark."""
    if family not in ('fastback','notchback','estateback'):
        raise ValueError('Choose fastback, notchback, or estateback.')
    knots = np.array([-2.35,-2.05,-1.4,-.65,-.15,.7,1.25,1.85,2.35])
    tops = {
      'fastback': [0.69,.79,.9,1.32,1.53,1.53,1.38,1.08,.82],
      'notchback':[0.69,.79,.9,1.32,1.53,1.51,1.25,.91,.83],
      'estateback':[0.69,.79,.9,1.32,1.53,1.55,1.54,1.48,1.14],
    }
    x = np.linspace(-2.35,2.35,97)
    theta = np.arange(72)*2*np.pi/72
    xx, tt = np.meshgrid(x,theta,indexing='ij')
    width = PchipInterpolator(knots,[.72,.84,.9,.91,.91,.90,.89,.86,.77])(xx)
    top = PchipInterpolator(knots,tops[family])(xx)
    c, s = np.cos(tt), np.sin(tt)
    # Rounded rectangular sections, narrower glazing above the belt line.
    yy = width * np.sign(c) * np.abs(c)**.52
    zz01 = .5+.5*np.sign(s)*np.abs(s)**.58
    yy *= 1 - .18*np.clip((zz01-.56)/.44,0,1)
    bottom = .25 + .035*np.exp(-((xx-2.2)/.45)**2)
    zz = bottom + (top-bottom)*zz01
    # Wheel-arch relief on the lower side walls; no independent panel tearing.
    for wx in [-1.43,1.45]:
        dx = xx-wx
        arch = .34+np.sqrt(np.maximum(.405**2-dx**2,0))
        arch_w = np.clip((np.abs(yy)-.56)/.24,0,1) * (np.abs(dx)<.405)
        low = np.clip((.56-zz01)/.56,0,1)
        zz += np.maximum(arch-zz,0)*arch_w*low
    v = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    f=[]
    n=len(theta)
    for i in range(len(x)-1):
        for j in range(n):
            a=i*n+j; b=i*n+(j+1)%n; c0=(i+1)*n+(j+1)%n; d=(i+1)*n+j
            f += [[a,b,c0],[a,c0,d]]
    for idx,base in [(0,0),(-1,(len(x)-1)*n)]:
        center=v[base:base+n].mean(axis=0)
        ci=len(v); v=np.vstack((v,center))
        for j in range(n): f.append([ci,base+j,base+(j+1)%n])
    shell=trimesh.Trimesh(v,np.array(f),process=False)
    shell.fix_normals()
    base=Model(shell.vertices,shell.faces,np.zeros(len(f),np.int32))
    base=suggest_regions(base)
    vertices=[base.vertices]; faces=[base.faces]; labels=[base.labels]; off=len(base.vertices)
    for wx in [-1.43,1.45]:
        for side in [-1,1]:
            wheel=trimesh.creation.cylinder(radius=.33,height=.21,sections=48)
            transform=trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0])
            wheel.apply_transform(transform); wheel.apply_translation([wx,side*.88,.33])
            vertices.append(wheel.vertices); faces.append(wheel.faces+off)
            labels.append(np.full(len(wheel.faces),9,np.int32)); off+=len(wheel.vertices)
    for side in [-1,1]:
        mirror=trimesh.creation.icosphere(subdivisions=2,radius=1.)
        mirror.vertices *= [.18,.11,.075]
        mirror.apply_translation([-.67,side*1.025,1.035])
        vertices.append(mirror.vertices); faces.append(mirror.faces+off)
        labels.append(np.full(len(mirror.faces),10,np.int32)); off+=len(mirror.vertices)
    meta=dict(name=f'AeroShape {family.title()}', family_id=f'original-demo-{family}',
              source='Original procedural demonstration geometry', asset_license='CC0-1.0',
              annotation_status='procedural_demonstration',
              limitations=['Not a production vehicle or an actual DrivAer model.',
                           'Assembly clearances and self-intersections have not been certified.',
                           'Not a CFD benchmark; validate or replace the geometry before simulation.'])
    return Model(np.vstack(vertices),np.vstack(faces),np.concatenate(labels),default_parts(),meta)
