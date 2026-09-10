"""Contract checks on real source assets: geometry, wheel parts, UV provenance."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pytest
from mishape.assets_pipeline import ASSET_ROOT, load_catalog, load_asset, validate_asset

@pytest.mark.parametrize('asset_id',['porsche-930','porsche-carrera-4s'])
def test_real_polygon_asset_contract(asset_id):
 model=load_asset(asset_id)
 result=validate_asset(model)
 assert result['vertices']>10000
 assert result['polygons']>1000
 vertices=np.array(model['vertices']).reshape(-1,3)
 assert abs(vertices[:,2].min())<1e-6
 assert model['metadata']['front_axis']=='-X'
 assert model['metadata']['scale_status']=='estimated'
 lm=model['metadata']['landmarks']
 assert lm['front_axle'][0]<0<lm['rear_axle'][0]
 assert lm['windshield_base'][0]<lm['windshield_top'][0]
 assert lm['rear_glass_base'][0]>lm['rear_glass_top'][0]
 wheels=[p for p in model['parts'] if p['key'].startswith('wheel_')]
 assert {p['key'] for p in wheels}=={'wheel_front_left','wheel_front_right','wheel_rear_left','wheel_rear_right'}
 for wheel in wheels:
  assert wheel['rigid']
  assert .25<wheel['wheel_radius_m']<.4
 assert len(model['face_materials'])==len(model['face_labels'])
 assert all(len(poly)>=3 for poly in model['polygons'])
 assert any(len(poly)==4 for poly in model['polygons'])
 assert max(model['source_face_ids'])<len(model['polygons'])

@pytest.mark.parametrize('asset_id',['porsche-930','porsche-carrera-4s'])
def test_source_polygons_and_uvs_preserved(asset_id):
 path=ASSET_ROOT/asset_id/'original_mesh.json.gz'
 with gzip.open(path,'rt') as stream: original=json.load(stream)
 assert any(obj['uv_layers'] for obj in original['objects'])
 assert any(len(poly)==4 for obj in original['objects'] for poly in obj['polygons'])
 assert hashlib.sha256((ASSET_ROOT/asset_id/'source.blend').read_bytes()).hexdigest()==original['source_sha256']
 assert original['excluded_from_design']
 for texture in original['textures']:
  assert (ASSET_ROOT/asset_id/texture['path']).is_file()


def test_no_arbitrary_asset_path_access():
 with pytest.raises(ValueError):load_asset('../../README.md')
