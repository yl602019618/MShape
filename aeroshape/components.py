"""Compatible ORIGINAL shape variants; not downloaded OEM interchangeable parts."""
from __future__ import annotations
import copy

COMPONENTS = {
 'mirror': [
  dict(id='aero',name='Aero · 宽扁圆角',scale=[1.,1.,1.],description='参考分割 STL 的比例重塑宽扁镜壳、圆角镜片与收分支座'),
  dict(id='compact',name='Compact · 紧凑',scale=[.84,.87,.85],description='更短、更薄的完整镜组；不是摄像头替代品'),
  dict(id='touring',name='Touring · 宽幅',scale=[1.07,1.12,1.16],description='适合旅行车和 SUV 的较大镜壳与宽幅镜片'),
  dict(id='sport',name='Sport · 低矮',scale=[1.18,.94,.77],description='较低的宽扁镜壳，保留完整镜片与支座'),
 ],
 'spoiler': [
  dict(id='none',name='无尾翼',description='连续车尾，不安装独立扰流唇；导出不含该零件'),
  dict(id='lip',name='贴合扰流唇',description='沿车尾轮廓的薄型扰流唇'),
  dict(id='sport',name='运动扰流唇',description='更宽、更高的车尾扰流唇'),
 ],
 'front': [
  dict(id='balanced',name='Balanced · 平衡前脸',sculpt=[0.,0.],description='基准圆润前保险杠'),
  dict(id='closed',name='Closed · 平滑封闭',sculpt=[.045,-.02],description='平滑中央表面；进气饰面不是通流风道'),
  dict(id='sport',name='Sport · 低鼻',sculpt=[-.05,-.035],description='低鼻、侧肩更明确的形态变体'),
  dict(id='upright',name='Upright · 直立',sculpt=[.02,.055],description='较直立的前端曲面，保留发动机盖接口'),
 ],
 'diffuser': [
  dict(id='smooth',name='Smooth · 平滑',shape=0.,description='连续平滑的底部上扬段'),
  dict(id='channel',name='Channel · 双通道',shape=.010,description='成对浅通道，不增加开放薄鳍'),
  dict(id='sport',name='Sport · 四通道',shape=.018,description='四条渐隐通道；不代表更低阻力'),
 ]
}

def catalog():
    out=copy.deepcopy(COMPONENTS)
    for category,items in out.items():
        for item in items:
            item.update(category=category,source='aeroshape_original',asset_license='CC0-1.0',
                interface='aerogt30-v2',compatible_body_types=['fastback','notchback','estateback','hatchback','suv'],
                placement='author-defined_mount_or_boundary_preserving_field',
                validation='geometric_candidate_not_manufacturing_or_regulatory_approval')
    return dict(schema='aeroshape.components.v1',categories=out,total=sum(map(len,out.values())),
        external_assets_bundled=0,external_import_policy='quarantine_until_license_units_mounts_and_clearance_reviewed')


def variant(category,key):
    if category not in COMPONENTS:raise ValueError('Unknown component category.')
    for item in COMPONENTS[category]:
        if item['id']==key:return item
    raise ValueError(f'Unknown {category} variant: {key}')
