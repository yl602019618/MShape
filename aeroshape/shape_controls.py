"""Coarse, symmetric C2 design fields over the dimensional car, never source meshes.

Seven longitudinal stations and three vertical layers are deliberately much
coarser than the mesh. All touching panels evaluate the same spatial function.
The X/Z field vanishes at the circular wheel arches; tyres remain independent.
"""
from functools import lru_cache
import numpy as np
from scipy.interpolate import BSpline

COUNT = 70
LIMIT = .22
STATIONS = ('车头', '前舱', '前柱', '座舱', '后柱', '尾箱', '车尾')

def smoother(t):
    t = np.clip(t, 0., 1.)
    return t*t*t*(10.+t*(-15.+6.*t))

def basis(points, wheelbase=2.92, wheel_radius=.355):
    x, y, z = np.asarray(points, dtype=float).T
    # Open cubic knot vector. Cubic extrapolation avoids a clipped derivative
    # at long rear overhangs; the hard geometry screen still bounds candidates.
    knots = np.r_[np.repeat(-2.55, 4), [-1.275, 0., 1.275], np.repeat(2.55, 4)]
    bx = BSpline(knots, np.eye(7), 3)(x)
    h = z/1.6
    bz = np.c_[(1-h)**2, 2*h*(1-h), h*h]
    tensor = (bx[:, :, None]*bz[:, None, :]).reshape(-1, 21)
    protection = np.ones_like(x)
    for axle in (-wheelbase/2, wheelbase/2):
        near = (1-smoother((np.abs(x-axle)-wheel_radius-.05)/.25))
        near *= (1-smoother((z-(2*wheel_radius+.075))/.22))*smoother((np.abs(y)-.45)/.25)
        protection *= 1-near
    result = np.zeros((len(x), 3, COUNT))
    result[:, 0, :21] = tensor*protection[:, None]
    result[:, 1, 21:42] = tensor*y[:, None]
    result[:, 2, 42:63] = tensor*protection[:, None]
    result[:, 2, 63:] = bx*(1-(y/1.05)**2)[:, None]*(smoother((z-.55)/.55)*protection)[:, None]
    return result

def displacement(points, controls, wheelbase=2.92, wheel_radius=.355):
    return np.einsum('nij,j->ni', basis(points, wheelbase, wheel_radius), np.asarray(controls))

@lru_cache(maxsize=1)
def regularizer():
    rows = []
    # Penalise bending of the control polygon and vertical oscillation.
    for start in (0, 21, 42):
        for j in range(3):
            for i in range(5):
                row = np.zeros(COUNT);row[start+i*3+j:start+(i+3)*3+j:3] = [1, -2, 1]
                rows.append(row)
        for i in range(7):
            row = np.zeros(COUNT);row[start+i*3:start+i*3+3] = [1, -2, 1]
            rows.append(row)
    for i in range(5):
        row = np.zeros(COUNT);row[63+i:63+i+3] = [1, -2, 1];rows.append(row)
    return np.array(rows)

def schema():
    rows=[]
    for axis, offset in [('纵向',0),('宽度',21),('高度',42)]:
        for i, station in enumerate(STATIONS):
            for j, layer in enumerate(('下部','腰部','上部')):
                rows.append(dict(index=offset+i*3+j, name=f'{station} · {layer}{axis}',
                    group=f'精细造型 · {axis}', min=-LIMIT*1000, max=LIMIT*1000, step=1, unit='mm', ui_scale=1000))
    for i, station in enumerate(STATIONS):
        rows.append(dict(index=63+i,name=f'{station} · 横向冠高',group='精细造型 · 冠高',
            min=-LIMIT*1000,max=LIMIT*1000,step=1,unit='mm',ui_scale=1000))
    return rows
