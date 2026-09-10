"""Fit original engineering parameters, never reference mesh vertices.

The versioned engineering generator is the sole geometry factory. Calibration stores parameters
and an audit reference. The v6 generator also exposes a fixed, coarse C2 design
field; no source-dependent vertex residual is used during generation. Coordinates are metres.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .engineering import EngineeringParams, BODY_TYPES, generate, preset
from .fit_metrics import surface_metrics, sample_surface
from .io import npz_bytes
from .model import Model, array_hash
from .parametric_quality import parametric_quality
from .section_profiles import vehicle_section_profiles
from .semantic import SCHEMA, SegmentationOptions, segment30

VERSION = 'aeroshape.parametric-reference-fit.v5'
COMPONENT_VERSION = 'aeroshape.parametric-reference-fit.v6'
SUPPORTED_VERSIONS = (COMPONENT_VERSION, VERSION, 'aeroshape.parametric-reference-fit.v4', 'aeroshape.parametric-reference-fit.v3', 'aeroshape.parametric-reference-fit.v2')
WHEELS = ('wheel_fl', 'wheel_fr', 'wheel_bl', 'wheel_br')
# These are the ONLY automatically adjustable coordinates, all authored engineering
# controls. Values are prior scales, not permissions to displace individual vertices.
SCALES = dict(wheelbase_m=.24, front_overhang_m=.17, rear_overhang_m=.17,
    body_width_m=.15, track_m=.17, wheel_radius_m=.045, tyre_width_m=.025,
    roof_height_m=.12, cabin_length_m=.25, windshield_angle_deg=6.,
    rear_glass_angle_deg=10., rear_deck_height_m=.10,
    rear_deck_length_m=.20, tailgate_angle_deg=10., rear_roof_drop_m=.06,
    tailgate_lower_height_m=.10, rear_belt_height_m=.08,
    rear_body_taper=.05, cabin_tumblehome_deg=4., hood_height_m=.08,
    hood_crown_m=.02, front_body_taper=.05)


def part_vertices(model: Model, key: str) -> np.ndarray:
    ids = [p['id'] for p in model.parts if p.get('key') == key]
    return np.unique(model.faces[np.isin(model.labels, ids)])


def wheel_landmarks(model: Model) -> dict:
    found = {}
    for key in WHEELS:
        ids = part_vertices(model, key)
        if len(ids) < 16:
            continue
        points = model.vertices[ids]
        lo, hi = points.min(0), points.max(0)
        extent = hi-lo
        if not (.4 < extent[0] < 1.15 and .4 < extent[2] < 1.15 and
                .6 < extent[0]/extent[2] < 1.5 and .05 < extent[1] < .6):
            continue
        found[key] = dict(center=((lo+hi)/2).tolist(),
                          radius_m=float((extent[0]+extent[2])/4),
                          width_m=float(extent[1]), vertices=int(len(ids)))
    return found


def body_surface(model: Model) -> Model:
    excluded = [p['id'] for p in model.parts if p.get('key', '').startswith(('wheel_', 'mirror_'))
                or p.get('key') in ('unassigned', 'unknown')]
    mask = ~np.isin(model.labels, excluded)
    if not mask.any():
        raise ValueError('参考车没有可拟合的车身表面。')
    # A lightweight measurement view; source annotation metadata is not changed.
    return Model(model.vertices, model.faces[mask], model.labels[mask], model.parts)


def _features(model: Model, bins=48) -> dict:
    body = body_surface(model)
    v = body.vertices[np.unique(body.faces)]
    lo, hi = v.min(0), v.max(0)
    # Area sampling gives dense tessellations and sparse tessellations equal
    # weight. Quantiles and profile smoothing suppress mesh noise in the target,
    # and never modify generated geometry. A fixed seed stabilizes optimization.
    points = sample_surface(body, 48000, seed=734)
    station = np.clip(((points[:, 0]-lo[0])/(hi[0]-lo[0])*bins).astype(int), 0, bins-1)
    floor = _central_floor(model)
    rear_heights = floor+(hi[2]-floor)*np.linspace(.32, .93, 12)
    sections = vehicle_section_profiles(body.vertices, body.faces, (lo[0], hi[0]), hi[1]-lo[1], bins, rear_heights)
    upper = sections['upper']
    width = np.full(bins, np.nan)
    for i in range(bins):
        strip = points[station == i]
        if len(strip) >= 5: width[i] = np.quantile(np.abs(strip[:, 1]), .985)
    def fill(values):
        good = np.isfinite(values)
        if good.sum() < bins//2:
            raise ValueError('参考车纵剖面覆盖不足，请检查是否只导入了零件。')
        return gaussian_filter1d(np.interp(np.arange(bins), np.flatnonzero(good), values[good]), .65)
    return dict(upper=fill(upper), rear_envelope=sections['rear'], half_width=fill(width),
                length=float(hi[0]-lo[0]), width=float(2*np.quantile(np.abs(points[:, 1]), .999)),
                roof=float(hi[2]), front=float(lo[0]), rear=float(hi[0]))


def _central_floor(model: Model) -> float:
    ids = part_vertices(model, 'underbody')
    q = model.vertices[ids] if len(ids) else model.vertices
    centre = q[(np.abs(q[:, 0]) < .6) & (np.abs(q[:, 1]) < .4)]
    if len(centre) < 10:
        centre = q[np.abs(q[:, 1]) < .4]
    if len(centre) < 5:
        raise ValueError('缺少车轮且中央底板不足，无法确定可靠的垂直对齐。')
    # Source underbody includes small raised ribs; the robust central median is
    # a registration landmark, not an inferred engineering floor parameter.
    return float(np.median(centre[:, 2])) if len(ids) else float(np.quantile(centre[:, 2], .02))


def fit_parameters(reference: Model, body_type: str, progress: Callable,
                   fit_strength: float = .55, component_choices: dict | None = None) -> tuple[Model, EngineeringParams, dict]:
    if not np.isfinite(fit_strength) or not .15 <= fit_strength <= 1.:
        raise ValueError('fit_strength must be between 0.15 and 1.')
    ref = reference.clone()
    dim = np.ptp(ref.vertices, axis=0)
    if not (3. <= dim[0] <= 6.5 and 1.35 <= dim[1] <= 3.2 and .9 <= dim[2] <= 2.5):
        raise ValueError('参考尺寸不在乘用整车范围：请核对米/毫米、车头方向和半车补全。')
    prior = preset(body_type)
    if component_choices:
        from .components import variant
        for category,style in component_choices.items():variant(category,style)
        prior=EngineeringParams.model_validate({**prior.model_dump(),**{k+'_style':v for k,v in component_choices.items()}})
    original = generate(prior)
    wheels = wheel_landmarks(ref)
    front = [wheels[k]['center'][0] for k in ('wheel_fl', 'wheel_fr') if k in wheels]
    rear = [wheels[k]['center'][0] for k in ('wheel_bl', 'wheel_br') if k in wheels]
    wheel_measured = bool(front and rear)
    shift = np.zeros(3)
    shift[0] = -(np.mean(front)+np.mean(rear))/2 if wheel_measured else (
        (original.vertices[:, 0].min()+original.vertices[:, 0].max())/2-
        (ref.vertices[:, 0].min()+ref.vertices[:, 0].max())/2)
    ref.vertices[:, 0] += shift[0]
    if wheel_measured:
        shift[2] = -ref.vertices[:, 2].min()
        ground_source = 'observed_wheels_ground'
    else:
        shift[2] = _central_floor(original)-_central_floor(ref)
        ground_source = 'central_underbody_aligned_to_preset_floor_missing_wheels'
    ref.vertices[:, 2] += shift[2]
    ref.metadata.update(fitting_alignment_translation_m=shift.tolist(),
                        base_geometry_hash=array_hash(ref.vertices, ref.faces))
    target = _features(ref)
    wheels = wheel_landmarks(ref)
    measured = dict(body_width_m=target['width'])
    if wheel_measured:
        wb = float(np.mean(rear)-np.mean(front))
        measured.update(wheelbase_m=wb, front_overhang_m=-wb/2-target['front'],
            rear_overhang_m=target['rear']-wb/2,
            track_m=float(2*np.median([abs(w['center'][1]) for w in wheels.values()])),
            wheel_radius_m=float(np.median([w['radius_m'] for w in wheels.values()])),
            tyre_width_m=float(np.median([w['width_m'] for w in wheels.values()])))
    keys = [k for k in SCALES if wheel_measured or k not in ('track_m', 'wheel_radius_m', 'tyre_width_m')]
    scales = np.array([SCALES[k] for k in keys])
    center = np.array([getattr(prior, k) for k in keys])
    props = EngineeringParams.model_json_schema()['properties']
    lower = np.array([props[k]['minimum'] for k in keys])
    upper = np.array([props[k]['maximum'] for k in keys])
    prior_strength = .30*(.55/fit_strength)**2
    radius = 1.6*fit_strength/.55
    lower = np.maximum(lower, center-radius*scales)
    upper = np.minimum(upper, center+radius*scales)
    uncertainty = dict(wheelbase_m=.09, front_overhang_m=.09, rear_overhang_m=.09,
        body_width_m=.07, track_m=.07, wheel_radius_m=.025, tyre_width_m=.025)
    evaluations = 0
    rejected = 0
    cache = {}

    def huber(a):
        a = np.abs(a)
        return float(np.mean(np.where(a <= 1., .5*a*a, a-.5)))

    def objective(values):
        nonlocal evaluations, rejected
        values = np.clip(values, lower, upper)
        identity = tuple(np.round(values, 9))
        if identity in cache: return cache[identity]
        evaluations += 1
        if evaluations % 12 == 0:
            progress(f'仅拟合 {len(keys)} 个工程参数 · 已评估 {evaluations} 组 · 质量约束拒绝 {rejected} 组')
        d = prior.model_dump(); d.update(zip(keys, map(float, values)))
        try:
            p = EngineeringParams.model_validate(d)
            model = generate(p)
            quality = parametric_quality(model)
            if not quality['screen_pass']:
                raise ValueError('Parametric quality guard rejected this combination.')
            predicted = _features(model)
            profile_loss = huber((predicted['upper']-target['upper'])/.06)
            # The final quarter must carry enough weight to identify roof-end,
            # exposed deck and hatch/tailgate controls on two-box vehicles.
            rear_loss = .75*huber((predicted['upper'][-14:]-target['upper'][-14:])/.05)
            rear_height_loss = .5*huber((predicted['rear_envelope']-target['rear_envelope'])/.06)
            width_loss = .6*huber((predicted['half_width']-target['half_width'])/.05)
            dimension_loss = .6*huber(np.array([(getattr(p, k)-v)/uncertainty[k] for k, v in measured.items()]))
            # Length is still observed when no wheel landmarks are available.
            length_loss = .5*((predicted['length']-target['length'])/.12)**2
            prior_loss = prior_strength*float(np.sum(((values-center)/scales)**2))
            smooth_loss = .15*quality['smoothness_penalty']
            losses = dict(profile=profile_loss, rear_profile=rear_loss, rear_height_envelope=rear_height_loss,
                          width=width_loss, dimensions=dimension_loss,
                          length=length_loss, prior=prior_loss, smoothness=smooth_loss)
            answer = (float(sum(losses.values())), p, losses)
        except ValueError:
            rejected += 1
            answer = (float('inf'), None, None)
        cache[identity] = answer
        return answer

    best_values = center.copy()
    best = objective(best_values)
    initial_loss = dict(best[2])
    if not np.isfinite(best[0]):
        raise ValueError('原始车型母版未通过质量约束，无法开始参数拟合。')
    # Deterministic derivative-free coordinate search. Discrete profile sampling
    # must not cause finite-difference gradients or jagged source labels to drive
    # unconstrained glass angles. The FULL regularized loss ranks every candidate.
    for step in (.70, .35, .15):
        for axis in range(len(keys)):
            start = best_values.copy()
            for sign in (-1., 1.):
                values = np.clip(start + np.eye(1, len(keys), axis)[0]*sign*step*scales, lower, upper)
                candidate = objective(values)
                if candidate[0] < best[0]-1e-10:
                    best_values, best = values, candidate
    return ref, best[1], dict(method='bounded_original_parameter_coordinate_search_with_fixed_preset_prior',
        free_parameters=keys, degree_of_freedom_count=len(keys), per_vertex_degrees_of_freedom=0,
        held_parameters=[k for k in EngineeringParams.model_fields if k not in keys],
        prior_parameters=prior.model_dump(), prior_strength=prior_strength, fit_strength=fit_strength,
        prior_scales={k: SCALES[k] for k in keys}, parameter_bounds={k: [float(a), float(b)] for k, a, b in zip(keys, lower, upper)},
        initial_total_loss=float(sum(initial_loss.values())), final_total_loss=best[0],
        initial_loss_terms=initial_loss, final_loss_terms=best[2], evaluations=evaluations,
        quality_rejected_candidates=rejected, detected_wheels=wheels, wheel_landmarks_measured=wheel_measured,
        measured_dimensions=measured, alignment_translation_m=shift.tolist(), ground_alignment=ground_source,
        original_dimensions_m=dim.tolist(), silhouette_sampling='three_exact_triangle_plane_sections_48_stations_plus_area_uniform_width',
        angle_scope='original_generator_controls_only_no_inferred_label_regression')


def _annotate(model: Model, info: dict, p: EngineeringParams, quality: dict) -> Model:
    # Metadata only. Original components, materials, bindings and measurements
    # survive exactly as produced by engineering.generate(p).
    model.metadata.update(name=info.get('reference_name', 'Reference')+' · 参数化拟合',
        calibration_id=info['id'], calibration_schema=info['schema'],
        family_id='parametric-reference-'+info['id'],
        calibration_parameters=info['parameters'], geometry_role='parametric_reference_fit',
        topology_basis='authored_30_component_parametric_scaffold',
        canonical_topology_preserved=True, generator_screen=quality,
        reference_geometry_hash=info['source_hash'],
        reference_usage=('parameter_objective_only_no_source_vertices_or_per_vertex_residual' if p.generator_revision=='v6' else 'parameter_objective_only_no_source_vertices_or_displacement'),
        fitting_reference_provenance=info.get('reference_provenance', {}), cfd_ready=False)
    return model


def _final_quality(model: Model) -> dict:
    from .inspection import InspectionOptions, inspect
    quality = parametric_quality(model)
    if not quality['screen_pass']:
        return quality
    precision=model.metadata.get('generator_revision')=='v6'
    report = inspect(model, InspectionOptions(check_clearance=precision))
    quality['driver_inspection'] = report
    quality['self_intersections'] = 'checked_complete_driver' if report.get('complete') else 'incomplete'
    if precision:quality['component_clearance']='checked_four_wheels_against_driver'
    if not report.get('complete') or not report.get('requested_checks_passed'):
        quality['screen_pass'] = False
        quality['errors'].append('Driver self-intersection check failed or incomplete.')
    return quality


class CalibrationStore:
    def __init__(self, root: Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def path(self, cid: str):
        if not isinstance(cid, str) or not re.fullmatch(r'[a-f0-9]{32}', cid):
            raise ValueError('Invalid calibration ID.')
        return self.root/cid

    def info(self, cid):
        path = self.path(cid)/'calibration.json'
        if not path.is_file(): raise ValueError('Calibration not found or unfinished.')
        info = json.loads(path.read_text())
        if info.get('schema') not in SUPPORTED_VERSIONS:
            raise ValueError('旧版拟合使用参考网格形变，已停用；请重新拟合以获得原始 30 组件参数化车型。')
        if info.get('id') != cid: raise ValueError('Calibration ID mismatch.')
        return info

    def create(self, reference: Model, *, body_type='auto', symmetry='auto',
               fit_strength=.55, progress=lambda _: None, component_choices=None, method='engineering'):
        from .normalization import normalize_reference
        if method not in ('engineering','component'):raise ValueError('Unknown fitting method.')
        if method=='component':
            from .component_fit import _raw_parts
            _raw_parts(reference)
        started = time.perf_counter(); cid = uuid.uuid4().hex
        progress('统一参考坐标，检查半车和尺寸')
        ref = normalize_reference(reference, symmetry=symmetry)
        body = ref.metadata.get('body_type', 'fastback') if body_type == 'auto' else body_type
        if body not in BODY_TYPES: body = 'fastback'
        if ref.metadata.get('semantic_schema') != SCHEMA or 'segmentation' not in ref.metadata:
            ref = segment30(ref, SegmentationOptions(body_type=body))
        progress('检测尺寸；以原始车型母版约束有限参数')
        ref, p, measurements = fit_parameters(ref, body, progress, fit_strength, **({'component_choices':component_choices} if component_choices else {}))
        prior_parameters=preset(body).model_copy(update={k+'_style':v for k,v in (component_choices or {}).items()})
        base = generate(prior_parameters)
        model = generate(p)
        quality = _final_quality(model)
        progress('独立测量车身表面偏差，验证选装组件与连续车身')
        # Body-only comparisons give references without wheels the same scope.
        # Wheels remain present in every output and have separate landmark data.
        before = surface_metrics(body_surface(base), body_surface(ref), samples=4000)
        after = surface_metrics(body_surface(model), body_surface(ref), samples=4000)
        surface_backtracking = 1.
        if not quality['screen_pass'] or after['rms_mm'] > before['rms_mm']:
            # A useful contour loss is not a guarantee on an independent metric.
            # Backtrack ONLY the parameter vector; keep a safe preset if necessary.
            prior = prior_parameters.model_dump(); proposed = p.model_dump()
            for fraction in (.5, .25, 0.):
                trial = dict(prior)
                for key in SCALES: trial[key] = prior[key]+fraction*(proposed[key]-prior[key])
                try:
                    candidate_p = EngineeringParams.model_validate(trial)
                    candidate = generate(candidate_p)
                    candidate_quality = _final_quality(candidate)
                except ValueError:
                    # Nonlinear station and triangle constraints need not hold
                    # along the segment between two individually valid vectors.
                    continue
                if not candidate_quality['screen_pass']: continue
                metric = before if fraction == 0. else surface_metrics(body_surface(candidate), body_surface(ref), samples=4000)
                if metric['rms_mm'] <= before['rms_mm']:
                    p, model, quality, after, surface_backtracking = candidate_p, candidate, candidate_quality, metric, fraction
                    break
        if not quality['screen_pass']:
            raise ValueError('拟合结果未通过连续车身与自交检查。')
        component_report=None;coarse_model=None;record_version=VERSION
        if method=='component':
            from .component_fit import refine
            coarse_body_metric=after
            ref,p,model,component_report,coarse_model=refine(ref,p,progress)
            # Keep the legacy comparison in the identical rigid coordinate frame.
            base.vertices+=np.asarray(component_report['measurements']['alignment_delta_m'])
            quality=_final_quality(model)
            if not quality['screen_pass']:raise ValueError('精细拟合未通过最终几何检查。')
            after=surface_metrics(body_surface(model),body_surface(ref),samples=4000)
            component_report['whole_body_fit']=dict(before=coarse_body_metric,after=after,
                scope='body surface including underbody and wheel-house liners; excludes wheels and mirrors')
            record_version=COMPONENT_VERSION
        measurements['optimizer_proposal_total_loss'] = measurements['final_total_loss']
        measurements['optimizer_proposal_loss_terms'] = measurements['final_loss_terms']
        measurements['final_parameters'] = p.model_dump()
        measurements['final_loss_scope'] = 'accepted_optimizer_proposal'
        if surface_backtracking != 1.:
            measurements['final_total_loss'] = None
            measurements['final_loss_terms'] = None
            measurements['final_loss_scope'] = 'not_recomputed_after_parameter_backtracking_see_independent_surface_metrics'
        ref_present = [part['key'] for part in ref.parts if np.any(ref.labels == part['id'])]
        generated_present = [part['key'] for part in model.parts if np.any(model.labels == part['id'])]
        report = dict(schema=record_version, template_fit=dict(before=before, after=after,
            scope='body_surface_excluding_wheels_mirrors_and_unknown', parameter_backtracking_fraction=surface_backtracking),
            parameter_fit=measurements, quality=quality,
            components=dict(reference_present=ref_present, generated_present=generated_present,
                            generated_count=len(generated_present), expected_count=29 if p.spoiler_style=='none' else 30, semantic_slots=30,
                            missing_reference_components_use_authored_generator=True),
            generation=dict(factory='aeroshape.engineering.generate', exact_original_generator=True,
                generator_revision=p.generator_revision,
                reference_vertices_transferred=0, vertex_residual_field=False,
                vertices=len(model.vertices), faces=len(model.faces)),
            reference_hash=array_hash(ref.vertices, ref.faces),
            original_reference_hash=array_hash(reference.vertices, reference.faces),
            symmetry=ref.metadata.get('symmetry_completion', ref.metadata.get('normalization', {})),
            explanation='参考车只参与尺寸、剖面和尾部轮廓评分；最终几何完全由扩展参数化生成器重建，保留 30 个语义组件槽位，未选装尾翼时有 29 个实体组件。后甲板、尾门及座舱采用有限造型参数，独立报告双向车身表面误差。',
            limitations=['现有车型母版无法表达所有真车局部造型；优先保留连续性和原有组件结构。',
                '参考组件自动分区可能不准确，未将推断玻璃边界作为实测角度。',
                '通过拓扑、对称、拉伸、新增折痕和车身自交检查；整车装配间隙、制造与 CFD 有效性需独立检查。'],
            cfd_ready=False, elapsed_seconds=round(time.perf_counter()-started, 3))
        if component_report is not None:
            report['component_fit']=component_report
            report['parameter_fit'].update(final_loss_scope='coarse_loss_precedes_component_refinement',
                final_total_loss=None,final_loss_terms=None,final_parameters=p.model_dump())
            report['explanation']='先拟合工程比例，再从原始组件标签和轮拱边界提取造型约束，联合优化 70 个共享的粗尺度样条控制。相邻外板使用同一连续函数，轮拱、圆形轮胎与 30 个语义组件槽位保留。拟合参数可直接继续生成；没有复制参考车顶点。'
            report['limitations']=['精细模式目前验证一辆带原始分割标签的三厢轿车；其他构型使用工程参数拟合。',
                '参考没有轮胎：轮组按轮拱位置和明确间隙假设补充。',
                '灯具、格栅、后视镜和底部结构仍采用原有组件造型；近似外观不等于制造级 CAD 复刻。',
                '外观表面与含底盘车身分别报告；前者不包括轮组、后视镜、轮拱内衬、底板和底部空气动力附件。']
            report['generation'].update(finite_shape_controls=70,shared_spatial_spline=True)
        record = dict(id=cid, schema=record_version, parameters=p.model_dump(), report=report,
            reference_library_id=reference.metadata.get('library_id'),
            reference_name=ref.metadata.get('name', 'Reference'),
            reference_provenance=ref.metadata.get('provenance', {}),
            source_hash=array_hash(ref.vertices, ref.faces), template_hash=array_hash(base.vertices, base.faces),
            generated_hash=array_hash(model.vertices, model.faces))
        _annotate(model, record, p, quality)
        progress('保存参数化拟合及参考对照')
        folder = self.path(cid); folder.mkdir()
        objects = {}
        stored=[('reference', ref), ('preset', base), ('parametric-fit', model)]
        if coarse_model is not None:stored.append(('coarse-fit',coarse_model))
        for name, value in stored:
            data = npz_bytes(value); (folder/(name+'.npz')).write_bytes(data)
            objects[name] = hashlib.sha256(data).hexdigest()
        record['objects'] = objects
        temporary = folder/'calibration.json.tmp'
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
        temporary.replace(folder/'calibration.json')
        return record, ref, base, model

    def generate(self, cid: str, parameters: EngineeringParams | dict, detail_strength=1.) -> Model:
        if detail_strength != 1.:
            raise ValueError('detail_strength is obsolete; parametric generation does not transfer mesh details.')
        info = self.info(cid)
        p = EngineeringParams.model_validate(parameters.model_dump() if isinstance(parameters, EngineeringParams) else parameters)
        if p.body_type != info['parameters']['body_type']:
            raise ValueError('同一拟合基准仅支持原车身类型；请先对新类型重新拟合。')
        if p.generator_revision != info['parameters'].get('generator_revision', 'v2'):
            raise ValueError('生成器版本与拟合基准不同；请重新拟合以使用扩展造型参数。')
        model = generate(p)
        quality = _final_quality(model)
        if not quality['screen_pass']:
            raise ValueError('参数组合超出原始车身质量适配范围：'+' '.join(quality['errors']))
        return _annotate(model, info, p, quality)
