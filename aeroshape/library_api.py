"""HTTP integration for the vehicle catalog and semantic review workbench."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Literal
import numpy as np
from fastapi import UploadFile, File, Form, HTTPException, Query
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ConfigDict
from .io import MAX_BYTES,load_npz,npz_bytes
from .library import VehicleLibrary,review_labels
from .semantic import SegmentationOptions,segment30,schema_parts,SCHEMA,transfer_labels
from .drivaer import ImportOptions,import_collection,read_archive


class SaveRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model_id:str
    name:str|None=Field(default=None,max_length=160)
    parent_id:str|None=None


class ReviewRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    geometry_hash:str
    annotation_hash:str
    reviewer:str=Field(min_length=1,max_length=100)
    notes:str=Field(min_length=1,max_length=2000)
    allow_unassigned:bool=False


class TransferRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    reference_library_id:str
    mode:Literal['mapped','nearest']='nearest'
    vertex_map:list[int]|None=Field(default=None,max_length=2000000)
    face_map:list[int]|None=Field(default=None,max_length=2000000)
    max_distance_m:float=Field(default=.02,gt=0,le=.5)


def register_library(app,get_model,save_model,state:Path,root:Path):
    library=VehicleLibrary(state/'vehicle-library');app.state.vehicle_library=library

    @app.get('/api/library/schema')
    def get_schema():
        return dict(schema=SCHEMA,target_regions=30,unknown_sentinel=30,parts=schema_parts(),
            methods={0:'geometry_prior',1:'source_name_proposal',2:'explicit_mapping',3:'component_proposal',
                     4:'unassigned',5:'manual',6:'transferred'},scores_are_probabilities=False)

    @app.get('/api/library/vehicles')
    def list_vehicles(query:str=Query('',max_length=200),body_type:str='',source:str='',review_status:str='',
                      offset:int=Query(0,ge=0),limit:int=Query(24,ge=1,le=128)):
        return library.list(query=query,body_type=body_type,source=source,review_status=review_status,offset=offset,limit=limit)

    @app.get('/api/library/vehicles/{vid}')
    def detail(vid:str):return library.get(vid)

    @app.get('/api/library/vehicles/{vid}/thumbnail')
    def thumbnail(vid:str):
        return Response(library.thumbnail(vid),media_type='image/png',headers={'Cache-Control':'private,max-age=86400'})

    @app.post('/api/library/vehicles/{vid}/open')
    def open_vehicle(vid:str):return save_model(library.load(vid))

    @app.get('/api/library/vehicles/{vid}/reference')
    def reference(vid:str):
        return Response(npz_bytes(library.load(vid)),media_type='application/octet-stream',
                        headers={'Content-Disposition':'attachment; filename="vehicle-reference.npz"'})

    @app.get('/api/library/manifest')
    def manifest():return library.manifest()

    @app.post('/api/library/save')
    def save(req:SaveRequest):
        return library.add(get_model(req.model_id)['model'],name=req.name,parent_id=req.parent_id)

    @app.post('/api/library/import')
    async def import_files(files:list[UploadFile]=File(...),options:str=Form('{}')):
        if len(options)>65536:raise ValueError('Import options exceed 64 KiB.')
        opts=ImportOptions.model_validate_json(options)
        if len(files)>128:raise ValueError('At most 128 uploaded files per request.')
        inputs={};total=0
        for f in files:
            name=f.filename or 'mesh';data=await f.read(MAX_BYTES+1);total+=len(data)
            if total>MAX_BYTES:raise HTTPException(413,'Upload batch exceeds 256 MiB; split the batch.')
            if name in inputs:raise ValueError('Duplicate upload filenames; use an archive retaining distinct relative paths.')
            inputs[name]=data
        if len(inputs)==1:
            name,data=next(iter(inputs.items()))
            if name.lower().endswith(('.zip','.tar','.tar.gz','.tgz')):
                inputs=await run_in_threadpool(read_archive,data,name)
        return await run_in_threadpool(import_collection,inputs,opts,library.add)

    @app.post('/api/library/examples')
    def examples():
        results=[]
        for file,family,name in [('aerogt-design30.npz','fastback','AeroGT · 30 区参考 / 原创'),
                                 ('fastback.npz','fastback','Fastback · 工作流示例 / 原创'),
                                 ('notchback.npz','notchback','Notchback · 工作流示例 / 原创'),
                                 ('estateback.npz','estateback','Estateback · 工作流示例 / 原创')]:
            m=load_npz((root/'examples'/file).read_bytes())
            m.metadata.update(name=name,body_type=family,asset_license='CC0',
                provenance=dict(source_type='original_demo',dataset_origin_verification='bundled_authored_demo',
                                dataset_version='aeroshape-0.4',attribution='AeroShape authored concept; not DrivAerNet/OEM'))
            m=segment30(m,SegmentationOptions(body_type=family));results.append(library.add(m))
        return dict(vehicles=results,original_demos_only=True,drivaernet_geometries_bundled=0)

    @app.post('/api/models/{mid}/segment30')
    def semantic(mid:str,req:SegmentationOptions):return save_model(segment30(get_model(mid)['model'],req))

    @app.get('/api/models/{mid}/segmentation')
    def segmentation(mid:str):
        entry=get_model(mid);m=entry['model'];_,_,fi=entry['preview']
        if m.metadata.get('semantic_schema')!=SCHEMA:raise ValueError('This reference has not been segmented into design30.')
        return dict(report=m.metadata.get('segmentation'),
                    scores=np.asarray(m.metadata['segmentation_face_scores'])[fi].tolist(),
                    methods=np.asarray(m.metadata['segmentation_face_methods'])[fi].tolist(),
                    source_face_ids=fi.tolist(),source_parts=m.metadata.get('semantic_source_parts',[]))

    @app.post('/api/models/{mid}/review-labels')
    def review(mid:str,req:ReviewRequest):return save_model(review_labels(get_model(mid)['model'],**req.model_dump()))

    @app.post('/api/models/{mid}/transfer-labels')
    def transfer(mid:str,req:TransferRequest):
        ref=library.load(req.reference_library_id)
        return save_model(transfer_labels(ref,get_model(mid)['model'],mode=req.mode,vertex_map=req.vertex_map,
                                         face_map=req.face_map,max_distance_m=req.max_distance_m))
