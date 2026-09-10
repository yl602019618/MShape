"""Measure real glass and A-pillar longitudinal chords on authored driver vertices."""
from functools import lru_cache
import numpy as np
from scipy.interpolate import PchipInterpolator

@lru_cache(maxsize=1)
def profile_vertices():
    from .engineering import reference,KNOTS,BELT
    base=reference();v=base.vertices;driver=np.asarray(base.metadata['driver_vertices'])
    width=PchipInterpolator(KNOTS,[.80,.912,.978,.954,.942,.940,.935,.947,.971,.982,.963,.932,.862])
    result={}
    for r in (0.,.64,.77,.80):
        mask=(v[driver,0]>=-1.06-1e-8)&(v[driver,0]<=-.40+1e-8)
        mask&=(abs(v[driver,1]-width(v[driver,0])*r)<1e-8)&(v[driver,2]>BELT(v[driver,0])+.01)
        ids=driver[mask];ids=ids[np.argsort(v[ids,0])]
        if len(ids)<15:raise ValueError('Authored windshield profile identity lost.')
        result[r]=ids
    return result

def windshield_report(model):
    rows=[]
    for r,ids in profile_vertices().items():
        points=model.vertices[ids][:,[0,2]];a=points[0];delta=points[-1]-a
        length=np.linalg.norm(delta)
        if length<.1:return {'pass':False,'max_inward_sag_mm':None,'reason':'degenerate windshield chord','profiles':rows}
        normal=np.array([-delta[1],delta[0]])/length
        distances=(points-a)@normal
        rows.append({'lateral_fraction':r,'vertices':len(ids),'inward_sag_mm':max(0.,float(-distances.min()*1000))})
    maximum=max(x['inward_sag_mm'] for x in rows)
    return {'pass':maximum<=1.,'max_inward_sag_mm':maximum,'limit_mm':1.,'profiles':rows,
            'scope':'glass centre, shoulder, shared A-pillar boundary and adjacent pillar longitudinal mesh profiles'}
