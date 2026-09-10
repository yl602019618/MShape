"""HTTP integration checks for saved polygon projects and deterministic variants."""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import mishape.app as api


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    state = tmp_path_factory.mktemp('mishape-http-state')
    old_state, old_models, old_jobs = api.STATE, api.MODELS, api.JOBS
    api.STATE, api.MODELS, api.JOBS = state, {}, {}
    with TestClient(api.app, raise_server_exceptions=False) as client:
        yield client
    api.STATE, api.MODELS, api.JOBS = old_state, old_models, old_jobs


@pytest.fixture(scope='module')
def opened(client):
    assets = client.get('/api/bootstrap').json()['assets']
    if not assets:
        pytest.skip('Bundled car conversion has not finished.')
    response = client.post('/api/models/open', json={'asset_id': assets[0]['id']})
    assert response.status_code == 200, response.text[:500]
    return response.json()


def test_health_and_bundled_asset_landmarks(client, opened):
    assert client.get('/api/health').json()['product'] == 'MiShape'
    assert len(client.get('/api/bootstrap').json()['parameters']) == 17
    assert len(opened['cage']['points']) == 108
    assert opened['cage']['type'] == 'fitted'
    assert len(opened['model']['polygons']) > 0
    assert opened['analysis']['landmarks']['front_axle'][0] < opened['analysis']['landmarks']['rear_axle'][0]
    assert len(opened['analysis']['wheels']) == 4
    assert opened['analysis']['calibration'] == 'source_landmarks'


def test_zero_deform_then_live_parameter_update(client, opened):
    mid = opened['model_id']
    identity = client.post(f'/api/models/{mid}/deform', json={})
    assert identity.status_code == 200, identity.text[:500]
    assert identity.json()['vertices'] == opened['model']['vertices']
    changed = client.post(f'/api/models/{mid}/deform', json={'parameters': {'total_length': 100, 'ride_height': 20}})
    assert changed.status_code == 200, changed.text[:500]
    data = changed.json()
    assert data['vertices'] != opened['model']['vertices']
    assert data['analysis']['measurements']['total_length']['delta_from_source'] == pytest.approx(100, abs=.001)
    assert data['quality']['topology_preserved']
    assert data['quality']['wheel_rigidity']['max_shape_error_mm'] == 0
    assert client.get(f'/api/models/{mid}').json()['model']['vertices'] == opened['model']['vertices']


def test_obj_export_preserves_polygon_indices(client, opened):
    result = client.post(f"/api/models/{opened['model_id']}/export", json={'format': 'obj', 'parameters': {'vehicle_height': 30}})
    assert result.status_code == 200, result.text[:500]
    lines = result.text.splitlines()
    vertices = [line for line in lines if line.startswith('v ')]
    polygons = [[int(index) - 1 for index in line.split()[1:]] for line in lines if line.startswith('f ')]
    assert len(vertices) == len(opened['model']['vertices']) // 3
    source = opened['model']['polygons']
    if source and isinstance(source[0], dict):
        source = [poly.get('vertices', poly.get('indices')) for poly in source]
    assert polygons == source
    assert any(len(face) == 4 for face in polygons)


def test_project_json_import_replays_same_parameters(client, opened):
    mid = opened['model_id']
    recipe = {'parameters': {'total_length': 80, 'greenhouse_tapering': 20}, 'controls': [{'id': 35, 'delta': [0, 0, .01]}], 'options': {'symmetry': True, 'preserve_wheels': True}}
    expected = client.post(f'/api/models/{mid}/deform', json=recipe).json()['vertices']
    saved = client.post(f'/api/models/{mid}/export', json={'format': 'json', **recipe})
    assert saved.status_code == 200
    imported = client.post('/api/import', files={'file': ('restyled.mishape.json', saved.content, 'application/json')})
    assert imported.status_code == 200, imported.text[:500]
    data = imported.json()
    assert data['recipe'] == recipe
    assert data['model']['polygons'] == opened['model']['polygons']
    replayed = client.post(f"/api/models/{data['model_id']}/deform", json=data['recipe'])
    assert replayed.status_code == 200
    assert replayed.json()['vertices'] == expected


@pytest.mark.parametrize('payload', [b'{}', b'[]', b'{"vertices":null}', b'not JSON'])
def test_invalid_json_upload_returns_validation_error(client, payload):
    response = client.post('/api/import', files={'file': ('broken.json', payload, 'application/json')})
    assert response.status_code == 422, response.text[:500]


def test_invalid_driver_upload_and_identifier(client, opened):
    mid = opened['model_id']
    assert client.post(f'/api/models/{mid}/deform', json={'parameters': {'unrecognized': 1}}).status_code == 422
    assert client.post(f'/api/models/{mid}/deform', json={'parameters': {'vehicle_pitch': 99}}).status_code == 422
    assert client.post('/api/import', files={'file': ('bad.exe', b'MZ', 'application/octet-stream')}).status_code == 422
    assert client.get('/api/models/..%5Coutside').status_code == 400
    assert client.post('/api/models/open', json={'asset_id': '../../outside'}).status_code == 404
    assert client.get('/api/batches/..%5Coutside/download').status_code == 400
    assert client.post('/api/batches', json={'model_id': mid, 'count': 2, 'vary': ['unknown']}).status_code == 422


def test_catalog_path_cannot_escape_assets(client, tmp_path, monkeypatch):
    assets = tmp_path / 'assets'; assets.mkdir()
    outside = tmp_path / 'outside.json'; outside.write_text('{}')
    (assets / 'catalog.json').write_text(json.dumps([{'id': 'escape', 'name': 'escape', 'path': '../outside.json'}]))
    monkeypatch.setattr(api, 'ASSETS', assets)
    response = client.post('/api/models/open', json={'asset_id': 'escape'})
    assert response.status_code == 404


def _completed_batch(client, request):
    started = client.post('/api/batches', json=request)
    assert started.status_code == 200, started.text[:500]
    job_id = started.json()['id']
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = client.get(f'/api/batches/{job_id}').json()
        if status['status'] != 'running':
            assert status['status'] == 'complete', status
            return job_id, status
        time.sleep(.05)
    pytest.fail('Batch did not finish within 30 seconds.')


def test_batch_seed_reproducibility_and_archive(client, opened):
    request = {'model_id': opened['model_id'], 'count': 3, 'seed': 157, 'strength': .1,
               'vary': ['vehicle_length', 'vehicle_width'], 'parameters': {'ride_height': 10}}
    first_id, first = _completed_batch(client, request)
    second_id, second = _completed_batch(client, request)
    assert first['completed'] == second['completed'] == 3
    assert [item['recipe'] for item in first['items']] == [item['recipe'] for item in second['items']]
    assert len({item['recipe']['parameters']['vehicle_length'] for item in first['items']}) == 3
    for index in range(3):
        a = client.get(f'/api/batches/{first_id}/variants/{index}').json()
        b = client.get(f'/api/batches/{second_id}/variants/{index}').json()
        assert a['vertices'] == b['vertices']
    archive_response = client.get(f'/api/batches/{first_id}/download')
    assert archive_response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['count'] == 3
        assert manifest['seed'] == 157
        assert 'VAR-001/vehicle.obj' in archive.namelist()
        assert 'base-model.json' in archive.namelist()
        assert json.loads(archive.read('base-model.json'))['polygons'] == opened['model']['polygons']


@pytest.mark.parametrize('file_format', ['obj', 'ply'])
def test_component_export_contains_only_referenced_component_vertices(client, opened, file_format):
    import trimesh
    model = opened['model']
    wheel = next(p for p in model['parts'] if p.get('key') == 'wheel_front_left')
    response = client.post(f"/api/models/{opened['model_id']}/export", json={'format': file_format, 'part_id': wheel['id']})
    assert response.status_code == 200, response.text[:500]
    if file_format == 'obj':
        lines = response.text.splitlines()
        vertices = np.array([[float(x) for x in line.split()[1:]] for line in lines if line.startswith('v ')])
        polygons = [[int(index) - 1 for index in line.split()[1:]] for line in lines if line.startswith('f ')]
        used = set(index for polygon in polygons for index in polygon)
        assert used == set(range(len(vertices)))
    else:
        mesh = trimesh.load(io.BytesIO(response.content), file_type='ply', process=False)
        vertices = np.asarray(mesh.vertices)
        assert set(mesh.faces.ravel()) == set(range(len(vertices)))
    source_v = np.array(model['vertices']).reshape(-1, 3)
    source_f = np.array(model['faces']).reshape(-1, 3)
    source_ids = np.unique(source_f[np.array(model['face_labels']) == wheel['id']])
    assert len(vertices) == len(source_ids)
    assert vertices.min(axis=0) == pytest.approx(source_v[source_ids].min(axis=0), abs=1e-6)
    assert vertices.max(axis=0) == pytest.approx(source_v[source_ids].max(axis=0), abs=1e-6)


def test_known_length_calibration_preserves_current_shape_and_source(client, opened):
    source = opened['model']
    mid = opened['model_id']
    recipe = {'parameters': {'total_length': 70, 'vehicle_width': 45, 'ride_height': 20},
              'controls': [{'id': 35, 'delta': [0, 0, .01]}],
              'options': {'symmetry': True, 'preserve_wheels': True}}
    preview_response = client.post(f'/api/models/{mid}/deform', json=recipe)
    assert preview_response.status_code == 200
    preview = preview_response.json()
    current_vertices = np.array(preview['vertices']).reshape(-1, 3)
    target_length = 4.65
    scale = target_length / np.ptp(current_vertices[:, 0])
    response = client.post(f'/api/models/{mid}/calibrate', json={'length_m': target_length, **recipe})
    assert response.status_code == 200, response.text[:500]
    calibrated = response.json()
    model = calibrated['model']
    vertices = np.array(model['vertices']).reshape(-1, 3)
    assert calibrated['model_id'] != mid
    assert np.ptp(vertices[:, 0]) == pytest.approx(target_length, abs=1e-12)
    assert calibrated['analysis']['measurements']['total_length']['value'] == pytest.approx(4650, abs=.0001)
    assert model['metadata']['scale_status'] == 'user_calibrated'
    assert model['metadata']['reference_length_m'] == target_length
    assert calibrated['analysis']['scale_calibration']['verified']
    assert not calibrated['analysis']['measurements']['total_length']['estimated']
    np.testing.assert_allclose(vertices, current_vertices * scale, rtol=1e-12, atol=1e-12)
    # A length-only resize of the rest car would lose the width/ride/cage edits.
    source_vertices = np.array(source['vertices']).reshape(-1, 3)
    source_scale = target_length / np.ptp(source_vertices[:, 0])
    assert not np.allclose(vertices, source_vertices * source_scale)
    for name, point in preview['analysis']['landmarks'].items():
        expected = np.array(point) * scale
        np.testing.assert_allclose(model['metadata']['landmarks'][name], expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(calibrated['analysis']['landmarks'][name], expected, rtol=1e-12, atol=1e-12)
    assert model['faces'] == source['faces']
    assert model['polygons'] == source['polygons']
    assert model['face_labels'] == source['face_labels']
    assert calibrated['recipe']['parameters'] == {}
    assert calibrated['recipe']['controls'] == []
    assert calibrated['recipe']['options']['cage']['type'] == 'fitted'
    identity = client.post(f"/api/models/{calibrated['model_id']}/deform", json={})
    assert identity.status_code == 200
    assert identity.json()['vertices'] == model['vertices']
    assert client.get(f'/api/models/{mid}').json()['model'] == source


def test_malformed_project_metadata_parts_and_polygons_are_rejected(client):
    import trimesh
    mesh = trimesh.creation.box(extents=[4, 1.8, 1.4])
    base = {'vertices': mesh.vertices.ravel().tolist(), 'faces': mesh.faces.ravel().tolist()}
    invalid_fields = [
        {'metadata': []},
        {'parts': [{}]},
        {'parts': [{'id': 0, 'name': 'first'}, {'id': 0, 'name': 'duplicate'}]},
        {'polygons': [[0, 999, 2]]},
        {'polygons': [[0, 1]]},
        {'polygons': [[0, 1.5, 2]]},
        {'polygons': 'not-an-index-list'},
        {'polygons': [[0, 1, 2]], 'polygon_labels': []},
    ]
    for extra in invalid_fields:
        response = client.post('/api/import', files={'file': ('invalid.json', json.dumps({**base, **extra}), 'application/json')})
        assert response.status_code == 422, (extra, response.status_code, response.text[:500])
