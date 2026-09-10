from __future__ import annotations
import copy
import io
import json
import math
import zipfile
import hashlib
from typing import Literal
import numpy as np
from scipy.stats import qmc
from pydantic import BaseModel, Field, ConfigDict, model_validator
from .model import Model, array_hash
from .deform import DesignRequest
from .quality import evaluate
from .io import npz_bytes


class Parameter(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    operation: int = Field(ge=0, le=31)
    field: Literal['amount','control'] = 'amount'
    control: int = Field(default=0, ge=0, le=215)
    axis: int = Field(default=2, ge=0, le=2)
    low: float = Field(ge=-2, le=2)
    high: float = Field(ge=-2, le=2)

    @model_validator(mode='after')
    def ordered(self):
        if self.high <= self.low: raise ValueError('Parameter high must exceed low.')
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    design: DesignRequest
    parameters: list[Parameter] = Field(min_length=1, max_length=64)
    count: int = Field(default=16, ge=1, le=256)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    sampler: Literal['sobol','lhs'] = 'sobol'
    attempts_factor: int = Field(default=4, ge=1, le=10)


def sample_dataset(model: Model, req: BatchRequest):
    raw=req.design.model_dump(mode='json')
    targets=[]
    for p in req.parameters:
        if p.operation>=len(raw['operations']): raise ValueError('Parameter operation does not exist.')
        op=raw['operations'][p.operation]
        if not op['enabled']: raise ValueError('Cannot sample a disabled operation.')
        if p.field=='control' and (op['kind']!='ffd' or p.control>=len(op['controls'])):
            raise ValueError('Parameter control does not exist.')
        if p.field=='amount' and op['kind']=='ffd':
            raise ValueError('FFD is sampled through control displacements, not amount.')
        target=(p.operation,p.field,p.control if p.field=='control' else 0,p.axis if p.field=='control' else 0)
        if target in targets: raise ValueError('The same parameter cannot be sampled twice.')
        targets.append(target)
    attempts=req.count*req.attempts_factor
    if attempts*len(model.faces)>120_000_000:
        raise ValueError('Batch exceeds the local interactive workload limit. Reduce count or use the CLI in smaller batches.')
    d=len(req.parameters)
    if req.sampler=='sobol':
        # Generate a full base-2 design, then consume in order. Never clip invalid geometries.
        samples=qmc.Sobol(d, scramble=True, seed=req.seed).random_base2(math.ceil(math.log2(attempts)))[:attempts]
    else:
        samples=qmc.LatinHypercube(d, seed=req.seed).random(attempts)
    low=np.array([p.low for p in req.parameters]); high=np.array([p.high for p in req.parameters])
    values=qmc.scale(samples,low,high)
    result=io.BytesIO(); log=[]; accepted=[]; seen=set()
    with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('base/reference.npz', npz_bytes(model))
        for sample_index,vector in enumerate(values):
            recipe=copy.deepcopy(raw)
            for p,val in zip(req.parameters,vector):
                op=recipe['operations'][p.operation]
                if p.field=='amount': op['amount']=float(val)
                else: op['controls'][p.control]['displacement'][p.axis]=float(val)
            request=DesignRequest.model_validate(recipe)
            v,quality=evaluate(model,request)
            # Micrometre-rounded correspondence-based duplicate screen within this family.
            duplicate_key=hashlib.sha256(np.round(v,6).astype('<f8').tobytes()).hexdigest()
            no_change=float(np.max(np.abs(v-model.vertices)))<1e-8
            duplicate=duplicate_key in seen or no_change
            reasons=quality['errors'] + (['duplicate_or_no_geometry_change'] if duplicate else [])
            record=dict(candidate_index=sample_index,parameters=vector.tolist(),
                        accepted=quality['screen_pass'] and not duplicate, reasons=reasons)
            log.append(record)
            if not record['accepted']: continue
            seen.add(duplicate_key)
            sample_id=f"{model.metadata['family_id']}_{req.seed}_{sample_index:05d}"
            # Never turn source/user-controlled family strings into archive paths.
            slug=hashlib.sha256(sample_id.encode()).hexdigest()[:16]
            folder=f'samples/{slug}'
            meta=dict(sample_id=sample_id, geometry_hash=array_hash(v,model.faces),
                      family_id=model.metadata['family_id'],
                      base_geometry_hash=model.metadata['base_geometry_hash'],
                      annotation_hash=model.annotation_hash(), candidate_index=sample_index,
                      parameter_vector=vector.tolist(), recipe=recipe, quality=quality,
                      split_group=model.metadata['family_id'], units='m',
                      geometry_status='screened_not_CFD_verified', cfd_status='not_run',
                      aerodynamic_labels=None, operating_conditions=None,
                      frontal_projected_area_m2=None, normalisation_transform=None,
                      source_license=model.metadata.get('asset_license','unknown'))
            z.writestr(folder+'/surface.npz',npz_bytes(model,v))
            z.writestr(folder+'/surface.stl',model.mesh(v).export(file_type='stl'))
            z.writestr(folder+'/design.json',json.dumps(meta,ensure_ascii=False,indent=2))
            accepted.append(dict(sample_id=sample_id,path=folder,geometry_hash=meta['geometry_hash']))
            if len(accepted)>=req.count: break
        manifest=dict(schema_version='aeroshape-dataset-0.1', engine_version='0.3.0',
                      numpy_version=np.__version__, scipy_version=__import__('scipy').__version__,
                      requested=req.count, accepted=len(accepted), attempted=len(log),
                      complete=len(accepted)==req.count, seed=req.seed, sampler=req.sampler,
                      parameter_definitions=[p.model_dump() for p in req.parameters],
                      base_geometry_hash=model.metadata['base_geometry_hash'],
                      annotation_hash=model.annotation_hash(), family_id=model.metadata['family_id'],
                      candidates=log,samples=accepted,
                      policy='Rejection sampling, no clipping or backtracking. Record failures. Family-grouped evaluation recommended.',
                      cfd_run=False)
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    return result.getvalue(),manifest
