"""Explicit optional solids on stable semantic slots; omitted faces never export."""
import numpy as np


def configure_optional(model, style):
    if style not in ('none','lip','sport'):raise ValueError('Unknown spoiler style.')
    if style!='none':return model
    part=next(p['id'] for p in model.parts if p['key']=='spoiler')
    keep=model.labels!=part
    if keep.all():return model
    metadata=dict(model.metadata)
    for key in ('face_material_ids','segmentation_face_scores','segmentation_face_methods'):
        if len(metadata.get(key,[]))==len(keep):metadata[key]=np.asarray(metadata[key])[keep].tolist()
    metadata['rigid_followers']=[b for b in metadata.get('rigid_followers',[]) if not b['name'].startswith('spoiler')]
    metadata['optional_components']={'spoiler':False}
    metadata['component_slots']=30
    metadata['active_components']=29
    vertices=model.vertices.copy()
    unused=np.setdiff1d(np.unique(model.faces[~keep]),np.unique(model.faces[keep]))
    # Retain stable vertex indices for the closed driver and all other fixtures.
    # Inactive slots have no faces and sit inside the body, affecting no bounds.
    vertices[unused]=[0.,0.,.5]
    metadata['inactive_vertex_slots']=unused.tolist()
    return model.clone(vertices=vertices,faces=model.faces[keep],labels=model.labels[keep],metadata=metadata)


def shape_spoiler(model,vertices,style):
    out=vertices.copy()
    if style=='sport':
        pid=next(p['id'] for p in model.parts if p['key']=='spoiler')
        ids=np.unique(model.faces[model.labels==pid]);q=out[ids]
        center=(q.max(0)+q.min(0))/2
        out[ids]=center+(q-center)*[1.12,1.03,1.25]+[0,0,.008]
    return out
