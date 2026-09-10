"""Persistent, cancellable, single-flight inference jobs with immutable inputs."""
from __future__ import annotations
import hashlib
import json
import time
import uuid
import zipfile
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Literal
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from .deform import DesignRequest
from .quality import evaluate
from .prediction import capability, predict_model, write_json, PredictionCancelled


class PredictionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    design: DesignRequest=Field(default_factory=DesignRequest)
    with_wss: bool=True
    calibrate: bool=True
    chunk_size: int=Field(32768,ge=1024,le=262144)
    device: Literal['auto','cpu','cuda']='auto'


class PredictionJobs:
    def __init__(self,get_model,state,root):
        self.get_model=get_model;self.root=Path(root);self.path=Path(state)/'predictions'
        self.path.mkdir(parents=True,exist_ok=True);self.lock=RLock();self.active={};self.cap_cache=None;self.cap_time=0

    def capability(self):
        with self.lock:
            if self.cap_cache is None or time.monotonic()-self.cap_time>60:
                self.cap_cache=capability(self.root);self.cap_time=time.monotonic()
            return self.cap_cache

    def directory(self,pid):
        if len(pid)!=32 or any(c not in '0123456789abcdef' for c in pid):raise HTTPException(404,'预测任务不存在。')
        path=self.path/pid
        if not (path/'job.json').is_file():raise HTTPException(404,'预测任务不存在。')
        return path

    def read(self,pid):
        with self.lock:
            path=self.directory(pid);data=json.loads((path/'job.json').read_text())
            if data['status'] in ('queued','running') and pid not in self.active:
                data.update(status='failed',stage='interrupted',error='服务已重启，上次预测被中断，请重新预测。')
                write_json(path/'job.json',data)
            if data['status']=='succeeded':data['result']=json.loads((path/'result.json').read_text())
            return data

    def update(self,pid,**changes):
        with self.lock:
            path=self.directory(pid);data=json.loads((path/'job.json').read_text());data.update(changes,updated_at=time.time())
            write_json(path/'job.json',data)

    def submit(self,mid,req):
        model=self.get_model(mid)['model'].clone()
        cap=self.capability()
        if not cap['available']:raise HTTPException(503,'代理环境尚未就绪。'+cap['setup'])
        key=hashlib.sha256((mid+model.annotation_hash()+req.model_dump_json()).encode()).hexdigest()
        with self.lock:
            for pid,item in self.active.items():
                if item['key']==key:return self.read(pid)
            if self.active:raise HTTPException(409,'已有车辆正在预测。请等待完成，或先停止当前任务。')
            pid=uuid.uuid4().hex;directory=self.path/pid;directory.mkdir()
            record=dict(prediction_id=pid,model_id=mid,model_name=model.metadata.get('name','当前车辆'),
                        request=req.model_dump(),status='queued',stage='checking',created_at=time.time(),updated_at=time.time())
            write_json(directory/'job.json',record)
            stop=Event();self.active[pid]=dict(key=key,cancel=stop)
            Thread(target=self.work,args=(pid,mid,model,req,stop,cap),daemon=True).start()
            return record

    def work(self,pid,mid,model,req,stop,cap):
        directory=self.path/pid
        try:
            self.update(pid,status='running',stage='checking')
            vertices,quality=evaluate(model,req.design)
            if not quality['screen_pass']:raise ValueError('当前设计未通过几何筛查：'+'；'.join(quality.get('errors',[])))
            if stop.is_set():raise PredictionCancelled()
            result=predict_model(model.clone(vertices=vertices),self.root,directory,
                with_wss=req.with_wss,calibrate=req.calibrate,chunk_size=req.chunk_size,device=req.device,
                cancel=stop,runtime=cap,progress=lambda stage:self.update(pid,stage=stage))
            if stop.is_set():raise PredictionCancelled()
            result.update(prediction_id=pid,model_id=mid,design=req.design.model_dump(),
                design_recipe_sha256=hashlib.sha256(req.design.model_dump_json().encode()).hexdigest())
            write_json(directory/'result.json',result)
            with zipfile.ZipFile(directory/'prediction.zip','w',zipfile.ZIP_DEFLATED) as z:
                for name in ('job.json','result.json','input.stl','preview.json'):
                    if name!='job.json':z.write(directory/name,name)
                for f in (directory/'fields').iterdir():
                    if f.is_file():z.write(f,'fields/'+f.name)
            with self.lock:
                if stop.is_set():raise PredictionCancelled()
                self.update(pid,status='succeeded',stage='complete')
        except PredictionCancelled:self.update(pid,status='cancelled',stage='cancelled')
        except Exception as exc:self.update(pid,status='failed',stage='failed',error=str(exc)[:2500])
        finally:
            with self.lock:self.active.pop(pid,None)

    def cancel(self,pid):
        with self.lock:
            self.directory(pid)
            if pid in self.active:self.active[pid]['cancel'].set()
            return self.read(pid)


def register_prediction(app,get_model,state,root):
    jobs=PredictionJobs(get_model,state,root);app.state.prediction_jobs=jobs

    @app.get('/api/prediction/capability')
    def prediction_capability():return jobs.capability()

    @app.post('/api/models/{mid}/predict',status_code=202)
    def predict(mid:str,req:PredictionRequest):return jobs.submit(mid,req)

    @app.get('/api/predictions/{pid}')
    def result(pid:str):return jobs.read(pid)

    @app.post('/api/predictions/{pid}/cancel')
    def cancel(pid:str):return jobs.cancel(pid)

    def completed(pid):
        if jobs.read(pid)['status']!='succeeded':raise HTTPException(409,'预测尚未完成，暂无结果文件。')
        return jobs.directory(pid)

    @app.get('/api/predictions/{pid}/preview')
    def preview(pid:str):return FileResponse(completed(pid)/'preview.json',media_type='application/json')

    @app.get('/api/predictions/{pid}/download')
    def download(pid:str):return FileResponse(completed(pid)/'prediction.zip',media_type='application/zip',filename=f'aeroshape-prediction-{pid[:8]}.zip')
