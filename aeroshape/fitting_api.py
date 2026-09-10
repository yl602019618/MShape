"""Bounded background calibration jobs and reference-based generation HTTP API."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from threading import Event, RLock
from typing import Literal
import uuid
import zipfile

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from .engineering import EngineeringParams, sample_parameters
from .fitting import CalibrationStore
from .inspection import InspectionOptions, inspect
from .io import npz_bytes
from .model import array_hash


_ID_PATTERN = r'^[a-f0-9]{32}$'
_ACTIVE = {'queued', 'running'}
_LOG = logging.getLogger(__name__)


class FittingJobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    reference_library_id: str = Field(pattern=_ID_PATTERN)
    body_type: Literal['auto', 'fastback', 'notchback', 'estateback', 'hatchback', 'suv'] = 'auto'
    symmetry: Literal['auto', 'keep', 'mirror'] = 'auto'
    method: Literal['engineering','component'] = 'engineering'
    fit_strength: float = Field(.55, ge=.15, le=1.)
    detail_strength: float = Field(1., ge=1., le=1.)
    component_choices: dict[Literal['mirror','front','diffuser','spoiler'],str] = Field(default_factory=dict)


class FittingGenerateRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    calibration_id: str = Field(pattern=_ID_PATTERN)
    parameters: EngineeringParams
    detail_strength: float = Field(1., ge=1., le=1.)


class FittingBatchRequest(FittingGenerateRequest):
    count: int = Field(8, ge=1, le=256)
    seed: int = Field(42, ge=0, le=2_147_483_647)
    variation: float = Field(.35, ge=0., le=1.)
    check_driver_intersections: bool = True


def _time():
    return datetime.now(timezone.utc).isoformat()


def _strict_id(value, kind):
    if not isinstance(value, str) or re.fullmatch(_ID_PATTERN, value) is None:
        raise HTTPException(404, f'Unknown {kind} ID.')
    return value


class FittingJobs:
    """One worker, at most one further waiting job, and atomic small job files."""

    def __init__(self, path, calibrations, library, save_model):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.calibrations = calibrations
        self.library = library
        self.save_model = save_model
        self.lock = RLock()
        self.stopping = Event()
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='aeroshape-fit')
        for file in self.path.glob('*.json'):
            if re.fullmatch(_ID_PATTERN, file.stem) is None:
                continue
            try:
                job = json.loads(file.read_text(encoding='utf-8'))
                if not isinstance(job, dict) or job.get('id') != file.stem:
                    raise ValueError('Job ID does not match filename.')
                if job.get('status') in _ACTIVE:
                    job.update(status='interrupted', progress='服务重启，未完成的拟合已中断，请重新提交。',
                               error='Server restarted before calibration completed.', updated_at=_time())
                    self._write(job)
                self.jobs[job['id']] = job
            except (OSError, ValueError, TypeError):
                _LOG.warning('Ignoring unreadable fitting job %s', file.name)

    def _write(self, job):
        destination = self.path / (job['id'] + '.json')
        temporary = self.path / (job['id'] + '.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(job, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _update(self, jid, **changes):
        with self.lock:
            job = dict(self.jobs[jid])
            job.update(changes, updated_at=_time())
            self._write(job)
            self.jobs[jid] = job
            return copy.deepcopy(job)

    def get(self, jid):
        _strict_id(jid, 'fitting job')
        with self.lock:
            if jid not in self.jobs:
                raise HTTPException(404, 'Fitting job not found.')
            return copy.deepcopy(self.jobs[jid])

    def submit(self, request):
        # Read-only identity validation is performed before reserving queue space.
        try:
            self.library.get(request.reference_library_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        with self.lock:
            if self.stopping.is_set():
                raise HTTPException(503, 'Fitting service is shutting down.')
            if sum(job.get('status') in _ACTIVE for job in self.jobs.values()) >= 2:
                raise HTTPException(429, '已有拟合正在运行和排队，请等待完成后再提交。', headers={'Retry-After': '5'})
            jid = uuid.uuid4().hex
            now = _time()
            job = dict(id=jid, status='queued', progress='等待拟合工作线程',
                       created_at=now, updated_at=now, request=request.model_dump(),
                       polling_url=f'/api/fitting/jobs/{jid}', cfd_ready=False)
            self._write(job)
            self.jobs[jid] = job
            try:
                self.executor.submit(self._run, jid)
            except RuntimeError as exc:
                self._update(jid, status='failed', progress='无法启动拟合任务', error=str(exc))
                raise HTTPException(503, 'Unable to start fitting job.') from exc
            return copy.deepcopy(job)

    def _run(self, jid):
        def progress(message):
            if self.stopping.is_set():
                raise RuntimeError('Fitting interrupted because the service is shutting down.')
            self._update(jid, progress=str(message)[:500])

        try:
            if self.stopping.is_set():
                return
            job = self._update(jid, status='running', progress='读取参考车辆')
            req = FittingJobRequest.model_validate(job['request'])
            reference = self.library.load(req.reference_library_id)
            record, ref, fitted, model = self.calibrations.create(
                reference, body_type=req.body_type, symmetry=req.symmetry, fit_strength=req.fit_strength, progress=progress,
                **({'component_choices':req.component_choices} if req.component_choices else {}),
                **({'method':req.method} if req.method!='engineering' else {}))
            progress('保存拟合预览和可编辑模型')
            ids = {}
            for name, value in [('reference_model_id', ref), ('template_model_id', fitted), ('model_id', model)]:
                ids[name] = self.save_model(value)['id']
            if 'coarse-fit' in record.get('objects',{}):
                from .io import load_npz
                ids['coarse_model_id']=self.save_model(load_npz((self.calibrations.path(record['id'])/'coarse-fit.npz').read_bytes()))['id']
            # Geometry/preview arrays and the full report already have persistent
            # stores. Only their identifiers belong in the polling job metadata.
            self._update(jid, status='completed', progress='拟合完成，可继续调整参数和拆分组件',
                         result=dict(calibration_id=record['id'], **ids))
        except Exception as exc:
            self._update(jid, status='interrupted' if self.stopping.is_set() else 'failed',
                         progress='拟合已中断' if self.stopping.is_set() else '拟合失败',
                         error=str(exc)[:2000] or type(exc).__name__)

    def shutdown(self):
        self.stopping.set()
        with self.lock:
            for jid, job in list(self.jobs.items()):
                if job.get('status') == 'queued':
                    self._update(jid, status='interrupted', progress='服务关闭，排队的拟合已中断。',
                                 error='Server stopped before the job started.')
        self.executor.shutdown(wait=False, cancel_futures=True)


def write_fitting_family(path: Path, calibrations, request: FittingBatchRequest):
    """Write accepted candidates sequentially and account for every rejection."""
    # Unknown/broken calibration IDs are request errors, not 4*N failed samples.
    calibrations.info(request.calibration_id)
    accepted = []
    rejected = []
    hashes = set()
    attempted = 0
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for attempt in range(request.count * 4):
            if len(accepted) >= request.count:
                break
            attempted += 1
            parameters = None
            try:
                parameters = next(sample_parameters(request.parameters, 1, request.seed + attempt, request.variation))
                model = calibrations.generate(request.calibration_id, parameters, detail_strength=request.detail_strength)
                geometry_hash = array_hash(model.vertices, model.faces)
                if geometry_hash in hashes:
                    raise ValueError('Duplicate geometry rejected.')
                # Parametric generation already requires a complete driver
                # check. Reuse its hash-bound report instead of testing twice.
                inspection = model.metadata.get('generator_screen', {}).get('driver_inspection')
                if request.check_driver_intersections or inspection is not None:
                    if inspection is None:
                        inspection = inspect(model, InspectionOptions(check_clearance=False))
                    if not inspection.get('complete') or not inspection.get('requested_checks_passed'):
                        raise ValueError('Driver self-intersection check failed or incomplete.')
                name = f'calibrated-{parameters.body_type}-{len(accepted):04d}'
                archive.writestr(name + '.npz', npz_bytes(model))
                if inspection:
                    archive.writestr(name + '.inspection.json', json.dumps(inspection, ensure_ascii=False, allow_nan=False))
                accepted.append(dict(name=name, parameters=parameters.model_dump(), geometry_hash=geometry_hash,
                                     calibration_id=request.calibration_id,
                                     measurements=model.metadata.get('generator_measurements', {}),
                                     component_choices=model.metadata.get('component_choices', {}),
                                     driver_intersections_checked=inspection is not None, cfd_ready=False))
                hashes.add(geometry_hash)
            except (ValueError, RuntimeError) as exc:
                rejected.append(dict(attempt=attempt, parameters=parameters.model_dump() if parameters else None,
                                     error=str(exc)[:2000]))
        manifest = dict(schema='aeroshape.calibrated-batch.v1', calibration_id=request.calibration_id,
                        requested=request.count, accepted=len(accepted), attempted=attempted,
                        seed=request.seed, variation=request.variation, detail_strength=request.detail_strength,
                        samples=accepted, rejections=rejected,
                        status='complete' if len(accepted) == request.count else 'partial', cfd_ready=False,
                        component_clearance='not_checked_in_this_batch',
                        self_intersections='checked_per_accepted_driver' if accepted and all(s['driver_intersections_checked'] for s in accepted) else 'not_checked')
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
    return manifest


def register_fitting(app, get_model, save_model, state, root):
    state = Path(state)
    calibrations = CalibrationStore(state / 'calibrations')
    jobs = FittingJobs(state / 'fitting-jobs', calibrations, app.state.vehicle_library, save_model)
    artifacts = state / 'fitting-artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    app.state.calibration_store = calibrations
    app.state.fitting_jobs = jobs
    app.add_event_handler('shutdown', jobs.shutdown)

    def model_payload(mid):
        entry = get_model(mid)
        builder = getattr(app.state, 'model_payload', None)
        if callable(builder):
            return builder(mid, entry)
        # Reuse the app's established serializer without importing app.py back
        # into this registration module or creating another model revision.
        for route in app.routes:
            if getattr(route, 'path', None) == '/api/models/{mid}' and 'GET' in getattr(route, 'methods', set()):
                return route.endpoint(mid)
        raise RuntimeError('The model payload serializer is not registered.')

    def calibration_info(cid):
        _strict_id(cid, 'calibration')
        try:
            return calibrations.info(cid)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post('/api/fitting/jobs', status_code=202)
    def submit_job(req: FittingJobRequest):
        return jobs.submit(req)

    @app.get('/api/fitting/jobs/{jid}')
    def job_status(jid: str):
        job = jobs.get(jid)
        if job['status'] == 'completed':
            saved = job['result']
            record = calibration_info(saved['calibration_id'])
            job['result'] = dict(calibration_id=record['id'], parameters=record['parameters'], report=record['report'],
                                 template_model=model_payload(saved['template_model_id']),
                                 reference_model=model_payload(saved['reference_model_id']),
                                 model=model_payload(saved['model_id']))
            if saved.get('coarse_model_id'):job['result']['coarse_model']=model_payload(saved['coarse_model_id'])
        return job

    @app.get('/api/fitting/calibrations/{cid}')
    def calibration(cid: str):
        return calibration_info(cid)

    @app.post('/api/fitting/calibrations/{cid}/resume')
    def resume_calibration(cid: str):
        from .io import load_npz
        record=calibration_info(cid)
        folder=calibrations.path(cid)
        models={key:save_model(load_npz((folder/name).read_bytes())) for key,name in
                [('model','parametric-fit.npz'),('reference_model','reference.npz'),('template_model','preset.npz')]}
        if 'coarse-fit' in record.get('objects',{}):models['coarse_model']=save_model(load_npz((folder/'coarse-fit.npz').read_bytes()))
        return dict(calibration_id=cid,parameters=record['parameters'],report=record['report'],
                    reference_library_id=record.get('reference_library_id'),**models)

    @app.post('/api/fitting/generate')
    def generate_candidate(req: FittingGenerateRequest):
        calibration_info(req.calibration_id)
        return save_model(calibrations.generate(req.calibration_id, req.parameters, detail_strength=req.detail_strength))

    @app.post('/api/fitting/batch')
    def generate_batch(req: FittingBatchRequest):
        calibration_info(req.calibration_id)
        path = artifacts / (uuid.uuid4().hex + '.zip')
        try:
            manifest = write_fitting_family(path, calibrations, req)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return FileResponse(path, media_type='application/zip', filename='aeroshape-calibrated-family.zip',
                            headers={'X-Accepted-Samples': str(manifest['accepted']),
                                     'X-Requested-Samples': str(manifest['requested'])},
                            background=BackgroundTask(path.unlink, missing_ok=True))
