from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
from threading import RLock
from typing import Literal
import io
import zipfile
import os
import uuid
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from .model import Model, array_hash
from .io import import_mesh, npz_bytes, load_npz, export_bundle, MAX_BYTES
from .demo import demo_model
from .detailed import detailed_model
from .design30 import design_model
from .generator import generate_vehicle, VehicleParams, archetype_reference, sample_family
from .region import prepare_region
from .mesh_tools import audit_mesh,clean_mesh,stitch_parts,improve_triangulation
from .geometry import preview_indices, suggest_regions, split_connected, paint, region_grow, refine_uniform
from .deform import DesignRequest,Operation
from .quality import baseline_report, evaluate
from .dataset import BatchRequest, sample_dataset

ROOT=Path(__file__).resolve().parents[1]
STATE=Path(os.environ.get('AEROSHAPE_STATE',str(ROOT/'.state')))
STATE.mkdir(parents=True,exist_ok=True)
app=FastAPI(title='AeroShape Studio',version='0.14.0',description='Vehicle design, annotation and offline aerodynamic surrogate predictions.')
store: OrderedDict[str,dict]=OrderedDict()
lock=RLock()


@app.exception_handler(ValueError)
async def invalid_request(_,exc):
    return JSONResponse(status_code=400,content={'detail':str(exc)})


def get_model(mid):
    if not isinstance(mid,str) or len(mid)!=32 or any(c not in '0123456789abcdef' for c in mid):
        raise HTTPException(404,'Unknown model ID.')
    with lock:
        if mid not in store:
            file=STATE/(mid+'.npz')
            if not file.is_file(): raise HTTPException(404,'Model not found.')
            m=load_npz(file.read_bytes())
            store[mid]=make_entry(m)
        store.move_to_end(mid)
        value=store[mid]
        while len(store)>6: store.popitem(last=False)
        return value


def make_entry(model):
    return dict(model=model,preview=preview_indices(model),baseline=baseline_report(model))


def save_model(model):
    mid=uuid.uuid4().hex
    entry=make_entry(model)
    # Atomic persistence; edits create a new immutable model revision.
    temporary=STATE/(mid+'.tmp')
    temporary.write_bytes(npz_bytes(model)); temporary.replace(STATE/(mid+'.npz'))
    with lock:
        store[mid]=entry
        while len(store)>6: store.popitem(last=False)
    return payload(mid,entry)


def payload(mid,entry):
    m=entry['model']; vi,ff,fi=entry['preview']
    parts=[]
    for p in m.parts:
        part=dict(p); part['face_count']=int((m.labels==p['id']).sum()); parts.append(part)
    meta={k:val for k,val in m.metadata.items() if k not in ('face_material_ids','driver_vertices','rigid_followers','segmentation_face_scores','segmentation_face_methods','semantic_source_labels','raw_face_labels','source_face_indices')}
    meta['rigid_follower_count']=len(m.metadata.get('rigid_followers',[]))
    return dict(id=mid,metadata=meta,parts=parts,baseline=entry['baseline'],
                face_materials=np.asarray(m.metadata.get('face_material_ids',np.zeros(len(m.faces),int)))[fi].tolist(),
                vertices=m.vertices[vi].ravel().tolist(), faces=ff.ravel().tolist(),
                face_labels=m.labels[fi].tolist(), source_face_ids=fi.tolist(),
                preview_is_lod=len(ff)<len(m.faces), preview_face_count=len(ff),
                annotation_hash=m.annotation_hash())


@app.get('/api/health')
def health():
    return dict(status='ok', version='0.14.0', cfd=False, surrogate=app.state.prediction_jobs.capability()['available'], renderer='offline WebGL2')


@app.get('/api/demo/{family}')
def demo(family: Literal['design30','aerogt','fastback','notchback','estateback','gt-fastback','gt-notchback','gt-estateback']):
    if family.startswith('gt-'): return save_model(archetype_reference(family[3:]))
    return save_model(design_model() if family=='design30' else detailed_model() if family=='aerogt' else demo_model(family))

class GenerateRequest(BaseModel):
    archetype: Literal['fastback','notchback','estateback']='fastback'
    length_scale: float=Field(default=1.0,ge=.92,le=1.08)
    width_scale: float=Field(default=1.0,ge=.92,le=1.08)
    cabin_height: float=Field(default=0.,ge=-.18,le=.18)
    rear_roof: float=Field(default=0.,ge=-.18,le=.18)
    rear_deck: float=Field(default=0.,ge=-.18,le=.18)
    shoulder_width: float=Field(default=0.,ge=-.18,le=.18)
    nose_length: float=Field(default=0.,ge=-.18,le=.18)
    stance: float=Field(default=0.,ge=-.18,le=.18)
    seed: int=Field(default=0,ge=0,le=2_147_483_647)

@app.post('/api/generate')
def generate(req:GenerateRequest):
    return save_model(generate_vehicle(VehicleParams(**req.model_dump())))

class GenerateBatchRequest(BaseModel):
    archetype: Literal['fastback','notchback','estateback']='fastback'
    count: int=Field(default=16,ge=1,le=256)
    seed: int=Field(default=42,ge=0,le=2_147_483_647)

@app.post('/api/generate-batch')
def generate_batch(req:GenerateBatchRequest):
    samples=sample_family(req.archetype,req.count,req.seed)
    manifest=[];buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for i,(params,m) in enumerate(samples):
            name=f'{req.archetype}-{i:04d}'
            z.writestr(name+'.npz',npz_bytes(m))
            manifest.append({'name':name,'parameters':params.__dict__,'geometry_hash':m.metadata['base_geometry_hash'],'faces':len(m.faces),'regions':len(set(m.labels.tolist()))})
        z.writestr('manifest.json',json.dumps({'schema':'aeroshape-generator-batch-v1','archetype':req.archetype,'seed':req.seed,'count':req.count,'samples':manifest},ensure_ascii=False,indent=2))
    return Response(buf.getvalue(),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="aerogt-generated-family.zip"'})


@app.post('/api/import')
async def upload(file: UploadFile=File(...), units:str=Form('auto'), up:str=Form('+z'),
                 longitudinal:str=Form('+x'), center:bool=Form(True), weld:bool=Form(True),
                 symmetry:str=Form('auto')):
    data=await file.read(MAX_BYTES+1)
    if len(data)>MAX_BYTES: raise HTTPException(413,'Upload exceeds 256 MiB.')
    try:
        m=import_mesh(data,file.filename or 'mesh',units=units,up=up,
                      longitudinal=longitudinal,center=center,weld=weld,symmetry=symmetry)
    except (ValueError,IndexError,KeyError,TypeError) as exc:
        raise HTTPException(400,f'Import failed: {exc}') from exc
    return save_model(m)


@app.get('/api/models/{mid}')
def read_model(mid:str):
    return payload(mid,get_model(mid))


@app.post('/api/models/{mid}/preview')
def preview(mid:str,req:DesignRequest):
    entry=get_model(mid); m=entry['model']
    v,q=evaluate(m,req); vi,_,_=entry['preview']
    return dict(vertices=v[vi].ravel().tolist(),quality=q,geometry_hash=array_hash(v,m.faces),
                recipe=req.model_dump(mode='json'))


class SegmentRequest(BaseModel):
    mode: Literal['suggest','components']


@app.post('/api/models/{mid}/segment')
def segment(mid:str,req:SegmentRequest):
    m=get_model(mid)['model']
    return save_model(suggest_regions(m) if req.mode=='suggest' else split_connected(m))


class PartRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id: int | None = None
    name: str = Field(min_length=1,max_length=100)
    locked: bool = False
    color: str = Field(default='#d7af81',pattern=r'^#[0-9a-fA-F]{6}$')


@app.post('/api/models/{mid}/part')
def edit_part(mid:str,req:PartRequest):
    m=get_model(mid)['model'].clone()
    if req.id is None:
        if m.metadata.get('semantic_schema')=='aeroshape.design30.v1':
            raise ValueError('The canonical vocabulary has exactly 30 regions. Use an existing label or unassigned; custom classes require a separate schema.')
        if len(m.parts)>=1000: raise ValueError('Part limit reached.')
        pid=max(p['id'] for p in m.parts)+1
        m.parts.append(dict(id=pid,key=f'custom_{pid}',name=req.name,locked=req.locked,color=req.color))
    else:
        part=next((p for p in m.parts if p['id']==req.id),None)
        if part is None: raise ValueError('Unknown part.')
        part.update(name=req.name,locked=req.locked,color=req.color)
    if m.metadata.get('semantic_schema')=='aeroshape.design30.v1':
        from .semantic import refresh_summary
        refresh_summary(m)
    return save_model(m)


class PaintRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    part_id: int
    center: tuple[float,float,float] | None = None
    radius: float = Field(default=.2,gt=0,le=10)
    bounds: tuple[tuple[float,float,float],tuple[float,float,float]] | None = None
    mirror: bool = True
    source_part: int | None = None
    face_ids: list[int] | None = Field(default=None,max_length=100000)


@app.post('/api/models/{mid}/paint')
def paint_faces(mid:str,req:PaintRequest):
    return save_model(paint(get_model(mid)['model'],**req.model_dump()))


class GrowRequest(BaseModel):
    seed_face: int = Field(ge=0)
    part_id: int
    angle_degrees: float = Field(default=25,gt=0,lt=90)
    mirror: bool = True


@app.post('/api/models/{mid}/grow')
def grow(mid:str,req:GrowRequest):
    m=get_model(mid)['model']; ids=region_grow(m,req.seed_face,req.angle_degrees)
    return save_model(paint(m,part_id=req.part_id,face_ids=ids,mirror=req.mirror))


class RefineRequest(BaseModel):
    design: DesignRequest
    levels: int = Field(default=1,ge=1,le=3)


@app.post('/api/models/{mid}/refine')
def refine(mid:str,req:RefineRequest):
    m=get_model(mid)['model']; v,q=evaluate(m,req.design)
    if not q['screen_pass']: raise HTTPException(422,dict(message='Design screen failed.',quality=q))
    committed=m.clone(vertices=v)
    committed.metadata['committed_recipe']=req.design.model_dump(mode='json')
    committed.metadata['parent_geometry_hash']=m.metadata['base_geometry_hash']
    return save_model(refine_uniform(committed,req.levels))


@app.post('/api/models/{mid}/export')
def export(mid:str,req:DesignRequest):
    m=get_model(mid)['model']; v,q=evaluate(m,req)
    if not q['screen_pass']: raise HTTPException(422,dict(message='Design screen failed; export rejected.',quality=q))
    data=export_bundle(m,v,req.model_dump(mode='json'),q)
    return Response(data,media_type='application/zip',headers={'Content-Disposition':'attachment; filename="aeroshape-design.zip"'})


@app.get('/api/models/{mid}/reference')
def reference(mid:str):
    return Response(npz_bytes(get_model(mid)['model']),media_type='application/octet-stream',
                    headers={'Content-Disposition':'attachment; filename="reference.npz"'})


@app.post('/api/models/{mid}/batch')
def batch(mid:str,req:BatchRequest):
    data,manifest=sample_dataset(get_model(mid)['model'],req)
    return Response(data,media_type='application/zip',headers={
        'Content-Disposition':'attachment; filename="aeroshape-dataset.zip"',
        'X-Accepted-Samples':str(manifest['accepted']),
        'X-Requested-Samples':str(manifest['requested']),
        'X-Attempted-Samples':str(manifest['attempted'])})


class RegionPreviewRequest(BaseModel):
    operation: Operation


@app.post('/api/models/{mid}/region')
def region_handle(mid:str,req:RegionPreviewRequest):
    entry=get_model(mid);op=req.operation
    if op.kind!='region':raise ValueError('Expected a regional surface handle.')
    prepared=prepare_region(entry['model'],op.region,op.mirror);vi,_,_=entry['preview']
    return dict(weights=prepared['weights'][vi].tolist(),basis=prepared['basis'][vi].ravel().tolist(),
                center=prepared['center'].tolist(),direction=prepared['direction'].tolist(),info=prepared['info'])


class AuditRequest(BaseModel):
    design: DesignRequest=Field(default_factory=DesignRequest)
    angle_threshold: float=Field(default=10,gt=0,lt=60)


@app.post('/api/models/{mid}/audit')
def audit(mid:str,req:AuditRequest):
    m=get_model(mid)['model'];v,q=evaluate(m,req.design)
    result=audit_mesh(m,v,req.angle_threshold);result['design_quality']=q
    result['geometry_hash']=array_hash(v,m.faces)
    return result


class MeshEditRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    action: Literal['clean','triangulate','stitch']
    part_a: int | None=None
    part_b: int | None=None
    tolerance: float=Field(default=.0001,gt=0,le=.001)


@app.post('/api/models/{mid}/mesh-edit')
def mesh_edit(mid:str,req:MeshEditRequest):
    m=get_model(mid)['model']
    if req.action=='clean':out=clean_mesh(m)
    elif req.action=='triangulate':out=improve_triangulation(m,plane_tolerance=req.tolerance)
    else:out=stitch_parts(m,req.part_a,req.part_b,req.tolerance)
    return save_model(out)


from .library_api import register_library
register_library(app,get_model,save_model,STATE,ROOT)

from .studio_api import register_studio
register_studio(app,get_model,save_model,STATE,ROOT)

from .fitting_api import register_fitting
app.state.model_payload=payload
register_fitting(app,get_model,save_model,STATE,ROOT)

from .annotation_api import register_annotation
register_annotation(app,STATE)

from .prediction_api import register_prediction
register_prediction(app,get_model,STATE,ROOT)

app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
