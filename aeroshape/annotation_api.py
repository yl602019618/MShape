"""Dedicated annotation API; optimistic revisions protect simultaneous editors."""
from pathlib import Path
from typing import Annotated, Literal
import json
import struct
import numpy as np
from fastapi import File, Form, UploadFile, HTTPException
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from .annotation import AnnotationStore, parts, workflow, SCHEMA
from .io import import_mesh, MAX_BYTES

FaceID = Annotated[int, Field(strict=True, ge=0, le=2_000_000)]


class Options(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    units: Literal['auto', 'm', 'cm', 'mm'] = 'auto'
    up: Literal['+x', '-x', '+y', '-y', '+z', '-z'] = '+z'
    longitudinal: Literal['+x', '-x', '+y', '-y', '+z', '-z'] = '+x'
    body_type: Literal['fastback', 'notchback', 'estateback', 'hatchback', 'suv', 'unknown'] = 'unknown'
    propose: bool = True


class Start(BaseModel):
    model_config = ConfigDict(extra='forbid')
    library_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    propose: bool = True


class Selection(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    revision: int = Field(ge=1)
    tool: Literal['wand', 'cut', 'smooth', 'expand', 'shrink']
    seeds: list[FaceID] = Field(default_factory=list, max_length=100000)
    background: list[FaceID] = Field(default_factory=list, max_length=100000)
    selected: list[FaceID] = Field(default_factory=list, max_length=2_000_000)
    radius: float = Field(default=.45, ge=.005, le=3.)
    angle: float = Field(default=28., ge=1, le=90)
    strength: float = Field(default=2., ge=.1, le=6)
    rings: int = Field(default=2, ge=1, le=8)


class Edit(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(ge=1)
    action: Literal['assign', 'review', 'undo', 'redo', 'propose']
    faces: list[FaceID] = Field(default_factory=list, max_length=2_000_000)
    part: int = Field(default=30, ge=0, le=35)
    mirror: bool = False
    only_unassigned: bool = False
    replace_part: bool = False
    source_part: int | None = Field(default=None, ge=0, le=35)
    status: Literal['pending', 'confirmed', 'absent', 'skipped'] = 'pending'


def register_annotation(app, state: Path):
    store = AnnotationStore(state/'annotations'); app.state.annotation_store = store

    @app.get('/api/annotation/schema')
    def schema():
        return dict(schema=SCHEMA, parts=parts(), workflow=workflow(), unknown=30,
                    source_labels=['Body', 'Body_Fender', 'Body_Hood', 'Body_Rear_Fascia', 'Body_Roof',
                                   'Body_Tail', 'Body_front_fascia', 'Mirrors', 'Mirrors_Body',
                                   'Mirrors_Glass', 'Underbody_Smooth', 'front_intakes', 'lights_front', 'windows'])

    @app.get('/api/annotation/sessions')
    def sessions():
        rows = []
        with store.lock:
            for p in store.root.glob('*/head.json'):
                head = json.loads(p.read_text())
                rows.append({k: head[k] for k in ('id', 'name', 'revision', 'updated', 'faces')})
        return dict(sessions=sorted(rows, key=lambda r: r['updated'], reverse=True)[:100])

    @app.post('/api/annotation/sessions')
    def start(req: Start):
        library = app.state.vehicle_library
        model = library.load(req.library_id); record = library.get(req.library_id)
        body = model.metadata.get('body_type', 'unknown')
        if body not in ('fastback', 'notchback', 'estateback', 'hatchback', 'suv', 'unknown'): body = 'unknown'
        return store.create(model, record['name'], body, req.propose)

    @app.post('/api/annotation/import')
    async def upload(file: UploadFile = File(...), options: str = Form('{}')):
        if len(options) > 4096: raise ValueError('导入选项过长。')
        opts = Options.model_validate_json(options)
        data = await file.read(MAX_BYTES+1)
        if len(data) > MAX_BYTES: raise HTTPException(413, '单个文件不能超过 256 MiB。')
        def process():
            model = import_mesh(data, file.filename or 'vehicle.stl', units=opts.units, up=opts.up,
                                longitudinal=opts.longitudinal, center=True, weld=True, symmetry='keep')
            model.metadata['body_type'] = opts.body_type
            return store.create(model, Path(file.filename or 'vehicle').name, opts.body_type, opts.propose)
        return await run_in_threadpool(process)

    @app.get('/api/annotation/sessions/{sid}')
    def info(sid: str, inspect: bool = False): return store.info(sid, inspect)

    @app.get('/api/annotation/sessions/{sid}/mesh')
    def mesh(sid: str):
        with store.lock:
            m = store.surface(sid).model
            data = struct.pack('<II', len(m.vertices), len(m.faces)) + m.vertices.astype('<f4').tobytes() + m.faces.astype('<u4').tobytes()
        return Response(data, media_type='application/octet-stream')

    @app.get('/api/annotation/sessions/{sid}/labels')
    def labels(sid: str, revision: int):
        with store.lock:
            head, snap = store.load(sid); store.check_revision(head, revision)
            return Response(snap['labels'].astype('u1').tobytes(), media_type='application/octet-stream')

    @app.post('/api/annotation/sessions/{sid}/select')
    def select(sid: str, req: Selection): return store.selection(sid, req)

    @app.post('/api/annotation/sessions/{sid}/edit')
    def edit(sid: str, req: Edit): return store.edit(sid, req)

    @app.get('/api/annotation/sessions/{sid}/export')
    def export(sid: str):
        return Response(store.export(sid), media_type='application/zip',
                        headers={'Content-Disposition': 'attachment; filename="vehicle-annotation.zip"'})

    @app.post('/api/annotation/sessions/{sid}/publish')
    def publish(sid: str):
        return app.state.vehicle_library.add(store.export_model(sid), name=store.info(sid)['name']+' · 标注参考')
