"""HTTP regressions for fitted cages, density changes and portable recipes."""
from __future__ import annotations

from copy import deepcopy
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import mishape.app as api


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    previous = api.STATE, api.MODELS, api.JOBS
    api.STATE = tmp_path_factory.mktemp('mishape-cage-http')
    api.MODELS, api.JOBS = {}, {}
    try:
        with TestClient(api.app, raise_server_exceptions=False) as session:
            yield session
    finally:
        api.STATE, api.MODELS, api.JOBS = previous


@pytest.fixture(scope='module')
def opened(client):
    """A small polygon coupe makes round-trip and batch tests self-contained."""
    xs = [-2.2, -1.65, -1.05, -.5, .15, .85, 1.4, 1.8, 2.2]
    widths = [.65, .89, .92, .86, .83, .88, .94, .86, .65]
    tops = [.62, .79, .84, 1.31, 1.45, 1.35, 1.04, .78, .65]
    vertices = []
    for x, width, top in zip(xs, widths, tops):
        vertices.extend([[x, -width * .8, .12], [x, width * .8, .12],
                         [x, width, .45], [x, width * .9, top * .8],
                         [x, width * .65, top], [x, -width * .65, top],
                         [x, -width * .9, top * .8], [x, -width, .45]])
    polygons = [list(range(7, -1, -1)), list(range(64, 72))]
    for section in range(len(xs) - 1):
        for side in range(8):
            polygons.append([section * 8 + side, section * 8 + (side + 1) % 8,
                             (section + 1) * 8 + (side + 1) % 8, (section + 1) * 8 + side])
    faces = [[poly[0], poly[i], poly[i + 1]] for poly in polygons for i in range(1, len(poly) - 1)]
    model = dict(name='Cage HTTP coupe', vertices=np.asarray(vertices).ravel().tolist(),
                 faces=np.asarray(faces).ravel().tolist(), polygons=polygons,
                 face_labels=[0] * len(faces), polygon_labels=[0] * len(polygons),
                 parts=[dict(id=0, name='Body', key='body')], metadata={'units': 'm'})
    response = client.post('/api/import', files={'file': ('coupe.json', json.dumps(model), 'application/json')})
    assert response.status_code == 200, response.text[:500]
    return response.json()


def post_json(client, path, payload):
    response = client.post(path, json=payload)
    assert response.status_code == 200, response.text[:500]
    return response.json()


def test_new_models_have_fitted_envelope_and_boundary_edges(opened):
    cage = opened['cage']
    assert cage['type'] == 'fitted'
    assert cage['dimensions'] == [9, 3, 4]
    assert cage['settings']['padding_mm'] == 30
    assert len(cage['points']) == cage['total_points'] == 108
    assert cage['visible_points'] == 108 - 7 * 1 * 2
    assert opened['recipe']['options']['cage'] == cage['settings']
    points = {point['id']: point for point in cage['points']}
    assert len(points) == 108
    for point in points.values():
        expected = any(index in (0, cage['dimensions'][axis] - 1)
                       for axis, index in enumerate(point['index']))
        assert point['visible'] == point['boundary'] == expected
        assert point['rest'] == point['position']
    for start, end in cage['edges']:
        a, b = points[start], points[end]
        assert a['boundary'] and b['boundary']
        changes = [axis for axis in range(3) if a['index'][axis] != b['index'][axis]]
        assert len(changes) == 1
        assert abs(a['index'][changes[0]] - b['index'][changes[0]]) == 1
        assert any(a['index'][axis] == b['index'][axis] and
                   a['index'][axis] in (0, cage['dimensions'][axis] - 1)
                   for axis in range(3) if axis != changes[0])
    roof = [p['rest'][2] for p in points.values() if p['index'][1:] == [1, 3]]
    assert np.ptp(roof) > .4, 'The envelope must follow this coupe roof, hood and tail.'


@pytest.fixture(scope='module')
def refined(client, opened):
    path = f"/api/models/{opened['model_id']}"
    recipe = deepcopy(opened['recipe'])
    recipe['parameters'] = {'vehicle_length': 60, 'vehicle_width': 20}
    recipe['options']['preserve_wheels'] = False
    handle = next(p for p in opened['cage']['points'] if p['index'] == [4, 2, 3])
    recipe['controls'] = [{'id': handle['id'], 'delta': [0, .006, .015]}]
    before = post_json(client, path + '/deform', recipe)
    response = post_json(client, path + '/cage', {
        **recipe, 'cage': {'type': 'fitted', 'dimensions': [13, 5, 5], 'padding_mm': 30}})
    return path, recipe, before, response


def test_density_change_transfers_manual_edits_and_reports_actual_error(client, opened, refined):
    path, source, before, result = refined
    assert len(result['cage']['points']) == 325
    assert result['recipe']['parameters'] == source['parameters']
    assert result['recipe']['options']['symmetry'] is True
    assert result['recipe']['options']['preserve_wheels'] is False
    assert result['recipe']['options']['cage']['dimensions'] == [13, 5, 5]
    assert result['recipe']['controls']
    assert result['vertices'] != opened['model']['vertices']
    errors = np.linalg.norm(np.asarray(result['vertices']).reshape(-1, 3) -
                            np.asarray(before['vertices']).reshape(-1, 3), axis=1) * 1000
    report = result['regrid']
    assert report['max_error_mm'] == pytest.approx(errors.max(), abs=1e-6)
    assert report['rms_error_mm'] == pytest.approx(np.sqrt(np.mean(errors ** 2)), abs=1e-6)
    assert report['evaluated_vertices'] == len(errors)
    assert report['parameters_preserved'] is True
    replay = post_json(client, path + '/deform', result['recipe'])
    assert replay['vertices'] == result['vertices']
    assert replay['cage'] == result['cage']
    assert client.get(path).json()['model'] == opened['model'], 'Regridding must not bake or mutate the source.'


def test_json_project_round_trip_keeps_refined_cage_and_polygon_topology(client, opened, refined):
    path, _, _, result = refined
    response = client.post(path + '/export', json={**result['recipe'], 'format': 'json'})
    assert response.status_code == 200, response.text[:500]
    imported = client.post('/api/import', files={
        'file': ('refined.mishape.json', response.content, 'application/json')})
    assert imported.status_code == 200, imported.text[:500]
    saved = imported.json()
    assert saved['recipe'] == result['recipe']
    assert saved['cage'] == result['cage']
    assert saved['model']['polygons'] == opened['model']['polygons']
    assert saved['model']['faces'] == opened['model']['faces']
    assert saved['model']['polygon_labels'] == opened['model']['polygon_labels']
    replay = post_json(client, f"/api/models/{saved['model_id']}/deform", saved['recipe'])
    assert replay['vertices'] == result['vertices']


def test_recipe_without_cage_options_replays_legacy_unpadded_box(client, opened):
    path = f"/api/models/{opened['model_id']}"
    result = post_json(client, path + '/deform', {'parameters': {'vehicle_length': 40}})
    assert result['cage']['type'] == 'box'
    assert result['cage']['settings'] == {'type': 'box', 'dimensions': [7, 3, 4], 'padding_mm': 0}
    assert len(result['cage']['points']) == 84
    rest = np.asarray([point['rest'] for point in result['cage']['points']])
    vertices = np.asarray(opened['model']['vertices']).reshape(-1, 3)
    np.testing.assert_array_equal(rest.min(0), vertices.min(0))
    np.testing.assert_array_equal(rest.max(0), vertices.max(0))


@pytest.mark.parametrize('settings', [
    {'dimensions': [4, 3, 4]}, {'dimensions': [18, 3, 4]},
    {'dimensions': [9, 4, 4]}, {'dimensions': [9, 3, 8]},
    {'dimensions': [9, 3]}, {'dimensions': [9, 3, 4.5]},
    {'dimensions': [True, 3, 4]}, {'dimensions': ['9', 3, 4]},
    {'type': 'surface-mesh'}, {'padding_mm': -1}, {'padding_mm': 151},
    {'padding_mm': '30'}, {'padding_mm': True},
])
def test_invalid_cage_settings_return_422(client, opened, settings):
    path = f"/api/models/{opened['model_id']}"
    change = client.post(path + '/cage', json={**opened['recipe'], 'cage': settings})
    assert change.status_code == 422, change.text[:500]
    replay = client.post(path + '/deform', json={'options': {'cage': settings}})
    assert replay.status_code == 422, replay.text[:500]


def test_maximum_density_accepts_more_than_200_manual_controls(client, opened):
    controls = [{'id': node, 'delta': [0, 0, .00002]} for node in range(833)]
    recipe = {'controls': controls, 'options': {'symmetry': True, 'preserve_wheels': False,
              'cage': {'type': 'fitted', 'dimensions': [17, 7, 7], 'padding_mm': 30}}}
    result = post_json(client, f"/api/models/{opened['model_id']}/deform", recipe)
    assert len(result['cage']['points']) == 833
    assert result['cage']['visible_points'] == 833 - 15 * 5 * 5
    moved = np.asarray(result['vertices']).reshape(-1, 3)
    original = np.asarray(opened['model']['vertices']).reshape(-1, 3)
    np.testing.assert_allclose(moved - original, np.tile([0, 0, .00002], (len(original), 1)), atol=1e-12)


def test_batch_retains_custom_cage_and_manual_recipe(client, opened):
    recipe = {'parameters': {'ride_height': 10}, 'controls': [{'id': 33, 'delta': [0, 0, .006]}],
              'options': {'symmetry': False, 'preserve_wheels': False,
                          'cage': {'type': 'fitted', 'dimensions': [11, 5, 4], 'padding_mm': 45}}}
    job = post_json(client, '/api/batches', {**recipe, 'model_id': opened['model_id'],
                    'count': 2, 'seed': 711, 'strength': .05, 'vary': ['vehicle_length']})
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = client.get(f"/api/batches/{job['id']}").json()
        if result['status'] != 'running':
            break
        time.sleep(.025)
    assert result['status'] == 'complete', result
    assert result['completed'] == 2
    for item in result['items']:
        assert item['recipe']['options'] == recipe['options']
        assert item['recipe']['controls'] == recipe['controls']
        assert item['recipe']['parameters']['ride_height'] == 10
        variant = client.get(f"/api/batches/{job['id']}/variants/{item['index']}").json()
        assert variant['recipe'] == item['recipe']
        replay = post_json(client, f"/api/models/{opened['model_id']}/deform", item['recipe'])
        assert replay['vertices'] == variant['vertices']
        assert replay['cage']['settings'] == recipe['options']['cage']
