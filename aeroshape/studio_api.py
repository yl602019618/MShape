from __future__ import annotations
import io,json,zipfile,uuid,hashlib,tempfile,os
from pathlib import Path
import numpy as np
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field
from fastapi.responses import Response,FileResponse
from starlette.background import BackgroundTask
from .engineering import EngineeringParams,generate,preset,schema,sample_parameters,BODY_TYPES
from .components import catalog
from .deform import DesignRequest
from .quality import evaluate
from .model import array_hash
from .io import npz_bytes
from .inspection import InspectionOptions,inspect,driver_faces
from .remesh import RemeshOptions,local_remesh
from .history_store import HistoryStore

class PublishRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model_id:str
    name:str|None=Field(None,max_length=160)
class InspectRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    design:DesignRequest=Field(default_factory=DesignRequest)
    options:InspectionOptions=Field(default_factory=InspectionOptions)
class RemeshRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    design:DesignRequest=Field(default_factory=DesignRequest)
    options:RemeshOptions=Field(default_factory=RemeshOptions)
    scope:Literal['changed','parts']='changed'
    parts:list[int]=Field(default_factory=list,max_length=30)
class HistoryRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    design:DesignRequest=Field(default_factory=DesignRequest)
    name:str=Field('设计版本',max_length=100)
    parent:str|None=None
class EngineeringBatch(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    parameters:EngineeringParams=Field(default_factory=EngineeringParams)
    count:int=Field(8,ge=1,le=256)
    seed:int=Field(42,ge=0,le=2_147_483_647)
    variation:float=Field(.35,ge=0,le=1)
    mixed_body_types:bool=False
    check_driver_intersections:bool=True


def checked_generator(parameters):
    model = generate(parameters)
    if model.metadata.get('generator_parameters', {}).get('generator_revision') in ('v3','v4','v5','v6'):
        from .fitting import _final_quality
        quality = _final_quality(model)
        if not quality['screen_pass']:
            raise ValueError('造型参数组合未通过连续车身质量检查：'+' '.join(quality['errors']))
        model.metadata['generator_screen'] = quality
    return model


def write_family(path:Path,req:EngineeringBatch):
    requested=req.count;accepted=0;attempted=0;failures=[];samples=[];hashes=set()
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        # Each model is released before the next one, not accumulated in a list.
        for attempt in range(requested*4):
            if accepted>=requested:break
            attempted+=1;typ=BODY_TYPES[accepted%len(BODY_TYPES)] if req.mixed_body_types else req.parameters.body_type
            base=preset(typ,revision=req.parameters.generator_revision) if req.mixed_body_types else req.parameters
            raw=None
            try:
                raw=next(sample_parameters(base,1,req.seed+attempt,req.variation));m=checked_generator(raw)
                h=array_hash(m.vertices,m.faces)
                if h in hashes:raise ValueError('Duplicate geometry rejected.')
                check=m.metadata.get('generator_screen',{}).get('driver_inspection')
                if req.check_driver_intersections or check is not None:
                    if check is None:check=inspect(m,InspectionOptions(check_clearance=False))
                    if not check['requested_checks_passed']:raise ValueError('Driver self-intersection check failed or incomplete.')
                name=f'{typ}-{accepted:04d}';archive.writestr(name+'.npz',npz_bytes(m))
                if check:archive.writestr(name+'.inspection.json',json.dumps(check,ensure_ascii=False,indent=2))
                samples.append(dict(name=name,parameters=raw.model_dump(),geometry_hash=h,measurements=m.metadata['generator_measurements'],
                                    component_choices=m.metadata['component_choices'],driver_intersections_checked=check is not None,cfd_ready=False))
                hashes.add(h);accepted+=1
            except (ValueError,RuntimeError) as exc:
                failures.append(dict(attempt=attempt,body_type=typ,parameters=raw.model_dump() if raw else None,error=str(exc)))
        manifest=dict(schema='aeroshape.engineering-batch.v2',requested=requested,accepted=accepted,attempted=attempted,
            seed=req.seed,variation=req.variation,mixed_body_types=req.mixed_body_types,samples=samples,rejections=failures,
            status='complete' if accepted==requested else 'partial',cfd_ready=False,component_clearance='not_checked_in_this_batch')
        archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


def register_studio(app,get_model,save_model,state,root):
    artifacts=Path(state)/'mesh-artifacts';artifacts.mkdir(exist_ok=True)
    history=HistoryStore(Path(state)/'design-history');app.state.design_history=history
    @app.get('/api/generator/schema')
    def generator_schema():return schema()
    @app.get('/api/components/catalog')
    def component_catalog():return catalog()
    @app.get('/api/components/{category}/{component_id}/thumbnail')
    def component_thumbnail(category:str,component_id:str):
        from .components import variant
        variant(category,component_id)
        path=Path(root)/'assets/components'/f'{category}-{component_id}.png'
        if not path.is_file():raise ValueError('Component thumbnail not built; run scripts/build_release_examples.py.')
        return Response(path.read_bytes(),media_type='image/png')
    @app.post('/api/generator/preview')
    def preview(req:EngineeringParams):return save_model(checked_generator(req))
    @app.post('/api/generator/publish')
    def publish(req:PublishRequest):
        m=get_model(req.model_id)['model']
        if m.metadata.get('generator')!='aeroshape.engineering.v2':raise ValueError('Publish expects an engineering-generator candidate.')
        if m.metadata.get('generator_screen',{}).get('status')=='displacement_controls_only_not_a_publishable_surface':
            raise ValueError('Displacement-control scaffolds are not publishable vehicle surfaces.')
        if m.metadata.get('calibration_schema') or m.metadata.get('generator_parameters',{}).get('generator_revision') in ('v3','v4','v5','v6'):
            from .fitting import SUPPORTED_VERSIONS, _final_quality
            if m.metadata.get('calibration_schema') and m.metadata['calibration_schema'] not in SUPPORTED_VERSIONS:
                raise ValueError('旧版参考网格形变已停用，请重新进行参数化拟合。')
            if not _final_quality(m)['screen_pass']:
                raise ValueError('参数化拟合车型未通过车身质量检查。')
        record=app.state.vehicle_library.add(m,name=req.name)
        if not history.list(m):
            _,q=evaluate(m,DesignRequest());history.commit(m,m.vertices,DesignRequest().model_dump(),q,'生成基准',kind='generated_baseline')
        return record
    @app.post('/api/generator/batch')
    def batch(req:EngineeringBatch):
        fd,name=tempfile.mkstemp(suffix='.zip',dir=artifacts);os.close(fd);path=Path(name)
        try:r=write_family(path,req)
        except Exception:path.unlink(missing_ok=True);raise
        return FileResponse(path,media_type='application/zip',filename='aerogt-engineering-family.zip',
            headers={'X-Accepted-Samples':str(r['accepted']),'X-Requested-Samples':str(r['requested'])},
            background=BackgroundTask(path.unlink,missing_ok=True))
    @app.post('/api/models/{mid}/inspect-surface')
    def inspect_surface(mid:str,req:InspectRequest):
        m=get_model(mid)['model'];v,q=evaluate(m,req.design)
        candidate=m.clone(vertices=v);candidate.metadata['base_geometry_hash']=array_hash(v,m.faces)
        r=inspect(candidate,req.options);r['local_deformation_screen']=q['screen_pass'];r['reference_geometry_hash']=array_hash(m.vertices,m.faces)
        r['design_recipe_sha256']=hashlib.sha256(req.design.model_dump_json().encode()).hexdigest()
        return r
    @app.post('/api/models/{mid}/prepare-mesh')
    def prepare_mesh(mid:str,req:RemeshRequest):
        m=get_model(mid)['model'];v,q=evaluate(m,req.design)
        if not q['screen_pass']:raise ValueError('Local design screening failed; cannot remesh this candidate.')
        if req.scope=='changed':
            displacement=np.linalg.norm(v-m.vertices,axis=1)
            mask=(displacement[m.faces]>1e-6).any(1)
        else:
            if not req.parts:raise ValueError('Choose at least one design region.')
            mask=np.isin(m.labels,req.parts)
        current=m.clone(vertices=v);current.metadata['base_geometry_hash']=array_hash(v,m.faces)
        new,mapping,report=local_remesh(current,mask,req.options)
        post=inspect(new,InspectionOptions(check_clearance=True))
        report['post_remesh_inspection']=post
        # Always a diagnostic candidate, never silently replace the design master.
        aid=uuid.uuid4().hex;path=artifacts/(aid+'.zip')
        buffer=io.BytesIO();np.savez_compressed(buffer,**mapping)
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('reference.npz',npz_bytes(m));z.writestr('deformed-design.npz',npz_bytes(current))
            z.writestr('remeshed-surface.npz',npz_bytes(new));z.writestr('surface.stl',new.mesh().export(file_type='stl'))
            z.writestr('correspondence.npz',buffer.getvalue());z.writestr('design.recipe.json',req.design.model_dump_json(indent=2))
            z.writestr('report.json',json.dumps(report,ensure_ascii=False,indent=2))
        report['artifact_id']=aid;report['download_url']=f'/api/mesh-artifacts/{aid}'
        (artifacts/(aid+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
        return report
    @app.get('/api/mesh-artifacts/{aid}')
    def download_artifact(aid:str):
        if len(aid)!=32 or any(c not in '0123456789abcdef' for c in aid):raise ValueError('Invalid artifact ID.')
        p=artifacts/(aid+'.zip')
        if not p.is_file():raise ValueError('Mesh artifact not found.')
        return FileResponse(p,filename='aeroshape-remeshed-candidate.zip',media_type='application/zip')
    @app.get('/api/models/{mid}/history')
    def list_history(mid:str):return dict(versions=history.list(get_model(mid)['model']),storage='server_immutable_snapshots')
    @app.post('/api/models/{mid}/history')
    def commit_history(mid:str,req:HistoryRequest):
        m=get_model(mid)['model'];v,q=evaluate(m,req.design)
        if not q['screen_pass']:raise ValueError('Cannot commit a design failing local screening.')
        return history.commit(m,v,req.design.model_dump(),q,req.name,parent=req.parent)
    @app.get('/api/history/{vid}/backup')
    def history_backup(vid:str):
        r=history.get(vid);buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('version.json',json.dumps(r,ensure_ascii=False,indent=2))
            z.writestr('reference.npz',history.bytes(r['reference_object']));z.writestr('result.npz',history.bytes(r['result_object']))
        return Response(buffer.getvalue(),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="design-version-backup.zip"'})
