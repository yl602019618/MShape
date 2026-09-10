"""Adapter for the user-supplied offline Transolver pressure/WSS checkpoints.

Only an immutable copy enters inference. Display LOD is built after full-mesh
inference and never participates in force integration.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Event
import numpy as np
from .model import Model, array_hash
from .geometry import preview_indices

SCHEMA = 'aeroshape-surrogate-result-v2'
FILES = ('infer_stl.py','model_t3.py','backbone/Transolver_chunk_opt_matrix_mul.py',
         'weights/pressure_best.pt','weights/wss_best.pt',
         'stats/abupt_norm_stats.json','stats/ours_fusion_stats.json')


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path, value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(path)


def predictor_root(root):return Path(root)/'predictor_package/no_operator_offline_package'


def predictor_python(root):
    configured=os.environ.get('AEROSHAPE_PREDICTOR_PYTHON')
    if configured:return configured
    for rel in ('.venv-predictor/bin/python','.venv-predictor/Scripts/python.exe'):
        p=Path(root)/rel
        if p.is_file():return str(p)
    return sys.executable


def capability(root):
    package=predictor_root(root);missing=[f for f in FILES if not (package/f).is_file()]
    python=predictor_python(root)
    script="""import importlib,json
r={'dependencies':{},'cuda':False}
for n in ('torch','numpy','scipy','pyvista','einops'):
 try:
  m=importlib.import_module(n);r['dependencies'][n]=getattr(m,'__version__','installed')
  if n=='torch':r['cuda']=m.cuda.is_available()
 except Exception:r['dependencies'][n]=None
print(json.dumps(r))
"""
    try:
        run=subprocess.run([python,'-c',script],capture_output=True,text=True,timeout=60,
                           env={**os.environ,'MPLCONFIGDIR':str(Path(root)/'.state/predictor-mpl')})
        runtime=json.loads(run.stdout.strip().splitlines()[-1]) if run.returncode==0 else {}
    except (OSError,subprocess.TimeoutExpired,ValueError,IndexError):runtime={}
    dependencies=runtime.get('dependencies',{})
    available=not missing and bool(dependencies) and all(dependencies.values())
    return dict(available=available,package_present=not missing,missing_files=missing,
                dependencies=dependencies,python=python,devices=['cpu','cuda'] if runtime.get('cuda') else ['cpu'],
                default_device='cuda' if runtime.get('cuda') else 'cpu',mode='offline-subprocess',
                note='本地代理估计 · 固定来流 30 m/s · 无需上传车辆',
                setup='运行 scripts/setup_predictor.py 安装独立推理环境。')


def align_input(model, package):
    v=model.vertices;lo=v.min(0);hi=v.max(0);span=hi-lo
    # Do not let the legacy loader silently mirror a half-car or rescale units.
    if span[0]<2 or span[0]>8 or span[1]<1.2 or span[1]>3.5 or span[2]<.5:
        raise ValueError('预测需要以米为单位的完整车辆。请核对尺寸，半车应先补齐后再预测。')
    if lo[1]>=-.001 or hi[1]<=.001:
        raise ValueError('车辆似乎只在对称面的一侧，请先补齐整车并核对朝向。')
    stats=json.loads((package/'stats/abupt_norm_stats.json').read_text())
    lower=np.asarray(stats['pos_min']);upper=np.asarray(stats['pos_max'])
    offset=np.array([(lower[0]+upper[0]-lo[0]-hi[0])/2,-(lo[1]+hi[1])/2,-lo[2]])
    transformed=v+offset;matrix=np.eye(4);matrix[:3,3]=offset
    low=transformed.min(0);high=transformed.max(0)
    extent=np.maximum(np.maximum(lower-low,high-upper),0)
    warnings=[]
    if extent.max()>1e-6:warnings.append('部分坐标超出训练包围盒；预测器会裁剪归一化坐标，结果可能偏离。')
    wheel_ids=[p['id'] for p in model.parts if p.get('key','').startswith(('wheel_','tire_','tyre_')) or p.get('key')=='wheels']
    if not wheel_ids or not np.isin(model.labels,wheel_ids).any():
        warnings.append('未识别到轮组标签；请核对模型是否含完整车轮。')
    return transformed,matrix,dict(inside_training_bbox=bool(extent.max()<=1e-6),
        out_of_domain_extent_m=extent.tolist(),warnings=warnings,
        bounds_m=dict(min=low.tolist(),max=high.tolist()),
        training_bounds_m=dict(min=lower.tolist(),max=upper.tolist()))


class PredictionCancelled(Exception):pass


def field_preview(path, matrix):
    with np.load(path,allow_pickle=False) as z:
        points=z['points'].astype(float)-matrix[:3,3];faces=z['faces'].astype(np.int32)
        pressure=z['pressure'].astype(float);wss=z['wss'].astype(float)
    if pressure.shape!=(len(points),) or not np.isfinite(pressure).all():raise ValueError('压力场无效。')
    if wss.size and (wss.shape!=(len(points),3) or not np.isfinite(wss).all()):raise ValueError('切应力场无效。')
    model=Model(points,faces,np.zeros(len(faces),np.int32),[dict(id=0,key='surface',name='预测表面',color='#80b7c3')])
    vi,ff,_=preview_indices(model,target=65000)
    fields={'pressure':pressure}
    if wss.size:fields['wss']=np.linalg.norm(wss,axis=1)
    result=dict(vertices=points[vi].ravel().tolist(),faces=ff.ravel().tolist(),
                source_faces=len(faces),preview_faces=len(ff),preview_is_lod=len(ff)<len(faces),fields={})
    for name,values in fields.items():
        # Robust color scale retains full-range values for export and reports clipping.
        low,high=np.quantile(values,[.02,.98]);full=[float(values.min()),float(values.max())]
        result['fields'][name]=dict(values=values[vi].tolist(),range=[float(low),float(high)],
                                   full_range=full,unit='m²/s²',normalization='density-normalized')
    return result


def predict_model(model, root, directory, *, with_wss=True, calibrate=True,
                  chunk_size=32768, device='auto', cancel=None, progress=None, runtime=None):
    directory=Path(directory);package=predictor_root(root);cancel=cancel or Event()
    progress=progress or (lambda stage:None);cap=runtime or capability(root)
    if not cap['available']:raise ValueError('代理环境尚未就绪。'+cap['setup'])
    effective=cap['default_device'] if device=='auto' else device
    if effective not in cap['devices']:raise ValueError('当前环境没有可用的 CUDA，请选择自动或 CPU。')
    transformed,matrix,domain=align_input(model,package)
    input_file=directory/'input.stl';model.clone(vertices=transformed).mesh().export(input_file,file_type='stl')
    output=directory/'fields';output.mkdir(exist_ok=True)
    command=[cap['python'],str(package/'infer_stl.py'),str(input_file),'--out-dir',str(output),
             '--chunk-size',str(chunk_size),'--device',effective,'--strict-input']
    if not with_wss:command.append('--no-wss')
    if not calibrate:command.append('--no-calib')
    progress('pressure');started=time.monotonic();log_path=directory/'inference.log';stage='pressure'
    env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
         'MPLCONFIGDIR':str(directory/'mpl')}
    with log_path.open('w') as log:
        process=subprocess.Popen(command,cwd=package,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            while process.poll() is None:
                if cancel.wait(.5):raise PredictionCancelled()
                if time.monotonic()-started>1800:raise ValueError('预测超过 30 分钟，已停止。可换用 CUDA 后重试。')
                tail=log_path.read_text(errors='replace')[-8000:]
                next_stage='integrating' if '[stage] integrating' in tail else 'wss' if '[stage] wss' in tail else 'pressure'
                if next_stage!=stage:stage=next_stage;progress(stage)
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
    if cancel.is_set():raise PredictionCancelled()
    if process.returncode:raise ValueError('代理推理失败：'+log_path.read_text(errors='replace')[-1800:])
    raw=json.loads((output/'input_cdcl.json').read_text())
    for key in ('cd_raw','cl_raw','a_ref'):
        if not isinstance(raw.get(key),(int,float)) or not np.isfinite(raw[key]):raise ValueError('代理输出系数无效。')
    if raw['a_ref']<=0:raise ValueError('代理输出迎风面积无效。')
    progress('visualizing')
    preview=field_preview(output/'input_fields.npz',matrix)
    write_json(directory/'preview.json',preview)
    result=dict(schema=SCHEMA,status='succeeded',geometry_hash=array_hash(model.vertices,model.faces),
        prediction=raw,effective_device=effective,requested_device=device,elapsed_seconds=round(time.monotonic()-started,2),
        domain=domain,input_to_training_transform=dict(matrix4x4=matrix.tolist(),reason='X center alignment; Y symmetry alignment; ground Z at zero; no scaling or mirroring'),
        input_sha256=sha256(input_file),predictor_files={f:sha256(package/f) for f in FILES},
        operating_conditions=dict(velocity_m_s=30.,density_kg_m3=1.,flow_direction='+X',lift_direction='+Z',fields='density-normalized'),
        cfd_status='not_run',surrogate_status='predicted',source_faces=len(model.faces),
        calibration_note='沿用提供模型的线性校准；在 559 辆标注车上拟合，不是独立泛化精度保证。',
        artifacts={f.name:dict(sha256=sha256(f),bytes=f.stat().st_size) for f in output.iterdir() if f.is_file()})
    return result
