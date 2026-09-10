"""MiShape's public design-driver schema. Distances are editing deltas in mm."""
from __future__ import annotations
from copy import deepcopy

_PARAMETERS = [
    ('total_length', '整体长度', 'Total length', 'dimensions', 'mm', -600, 600, 10, '绕车身中心整体伸缩；轴距与前后悬按比例变化。'),
    ('vehicle_length', '轴间车身长度', 'Vehicle length / wheelbase stretch', 'dimensions', 'mm', -500, 500, 10, '拉伸前后轴之间的车身；前后悬保持长度，轴距随之变化。'),
    ('vehicle_width', '车宽', 'Vehicle width', 'dimensions', 'mm', -300, 300, 5, '左右对称改变整体车宽，车轮随轮距平移并保持形状。'),
    ('vehicle_height', '车高', 'Vehicle height', 'dimensions', 'mm', -250, 300, 5, '以地面为基准改变车身高度；保留轮胎直径。'),
    ('front_overhang', '前悬', 'Front overhang', 'proportions', 'mm', -300, 300, 5, '前轴前方的车身向车头延伸；前轴保持位置。'),
    ('rear_overhang', '后悬', 'Rear overhang', 'proportions', 'mm', -300, 300, 5, '后轴后方的车身向车尾延伸；后轴保持位置。'),
    ('approach_angle', '接近角', 'Approach angle', 'front', 'deg', -15, 15, .5, '前轮地面接触点至前保险杠下缘连线与地面的角度变化。'),
    ('windscreen_angle', '前风挡角', 'Windscreen angle', 'greenhouse', 'deg', -15, 15, .5, '前风挡下缘至上缘在XZ侧视平面内与地面的锐角变化。'),
    ('backlight_angle', '后风挡角', 'Backlight angle', 'greenhouse', 'deg', -15, 15, .5, '后风挡上缘至下缘在XZ侧视平面内与地面的锐角变化。'),
    ('decklid_height', '尾厢盖高度', 'Decklid height', 'rear', 'mm', -150, 150, 5, '车尾盖板区域的垂直位移。'),
    ('rear_end_tapering', '车尾收窄', 'Rear-end tapering', 'rear', 'mm', -200, 200, 5, '正值减小车尾总宽度，负值加宽。'),
    ('greenhouse_tapering', '座舱收窄', 'Greenhouse tapering', 'greenhouse', 'mm', -200, 200, 5, '正值减小车顶宽度，向腰线平滑过渡。'),
    ('hood_angle', '引擎盖角', 'Hood angle', 'front', 'deg', -12, 12, .5, '引擎盖前缘至前风挡下缘连线与地面的角度变化。'),
    ('front_plan_view', '车头俯视收窄', 'Front plan-view taper', 'front', 'mm', -200, 200, 5, '正值减小车头总宽度，向前轴过渡。'),
    ('ride_height', '车身离地高度', 'Vehicle ride height', 'stance', 'mm', -100, 150, 5, '车身相对车轮上下移动；轮胎接地点保持不动。'),
    ('vehicle_pitch', '车身俯仰', 'Vehicle pitch', 'stance', 'deg', -3, 3, .1, '绕横向轴旋转车身；正值为车头抬升，轮胎仍接地。'),
    ('diffusor_angle', '后扩散器角', 'Rear-end diffusor angle', 'rear', 'deg', -15, 15, .5, '尾部底板与地面夹角的变化；以底板前端为支点。'),
]


def schema() -> list[dict]:
    """Return independent, JSON-serializable slider definitions."""
    return [dict(id=p[0], key=p[0], label=p[1], label_en=p[2], group=p[3], unit=p[4],
                 min=p[5], max=p[6], step=p[7], default=0, mode='delta', definition=p[8])
            for p in _PARAMETERS]


def validate_parameters(parameters: dict | None) -> dict[str, float]:
    import math
    values = parameters or {}
    if not isinstance(values, dict):
        raise ValueError('parameters must be a dictionary of design-driver deltas.')
    specs = {p['id']: p for p in schema()}
    unknown = set(values) - set(specs)
    if unknown:
        raise ValueError('Unknown design parameters: ' + ', '.join(sorted(unknown)))
    result = {}
    for key, raw in values.items():
        if isinstance(raw, bool):
            raise ValueError(f'{key} must be a finite number.')
        try:
            value = float(raw)
        except (ValueError, TypeError):
            raise ValueError(f'{key} must be a finite number.') from None
        spec = specs[key]
        if not math.isfinite(value) or not spec['min'] <= value <= spec['max']:
            raise ValueError(f"{key} must lie between {spec['min']} and {spec['max']} {spec['unit']}.")
        if value:
            result[key] = value
    return result
