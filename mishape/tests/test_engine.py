"""Geometry contract tests with a polygon body and independently labelled wheels."""
from copy import deepcopy

import numpy as np
import pytest
import trimesh

from mishape.engine import analyze, deform, get_cage, sample_parameters, schema


@pytest.fixture(scope='module')
def vehicle():
    body = trimesh.creation.box(extents=[4.4, 1.8, .7]); body.apply_translation([0, 0, .65])
    cabin = trimesh.creation.box(extents=[1.8, 1.5, .5]); cabin.apply_translation([.1, 0, 1.25])
    meshes = [body, cabin]
    parts = [dict(id=0, name='body', key='body'), dict(id=1, name='cabin', key='roof')]
    for front in (True, False):
        for left in (True, False):
            wheel = trimesh.creation.cylinder(radius=.32, height=.20, sections=32)
            wheel.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
            wheel.apply_translation([-1.35 if front else 1.35, .85 if left else -.85, .32])
            parts.append(dict(id=len(meshes), name='wheel', key='wheel_' + str(len(meshes))))
            meshes.append(wheel)
    v, f, labels, polygons = [], [], [], []
    for i, mesh in enumerate(meshes):
        offset = len(v)
        v.extend(mesh.vertices.tolist()); f.extend((mesh.faces + offset).tolist()); labels.extend([i] * len(mesh.faces))
        polygons.extend((mesh.faces + offset).tolist())
    return dict(vertices=np.array(v).ravel().tolist(), faces=np.array(f).ravel().tolist(), face_labels=labels,
                polygons=polygons, parts=parts, metadata={'name': 'test', 'units': 'm'})


def test_zero_is_exact_identity_and_no_input_mutation(vehicle):
    before = deepcopy(vehicle)
    result = deform(vehicle, {s['id']: 0 for s in schema()})
    assert result['vertices'] == before['vertices']
    assert result['faces'] == before['faces']
    assert result['polygons'] == before['polygons']
    assert result['face_labels'] == before['face_labels']
    assert vehicle == before
    assert result['metadata']['mishape']['quality']['max_displacement_mm'] == 0


def test_affine_dimensions_have_expected_measured_change(vehicle):
    result = deform(vehicle, {'total_length': 100, 'vehicle_width': 50, 'vehicle_height': 75}, options={'preserve_wheels': False})
    measurements = result['metadata']['mishape']['analysis']['measurements']
    assert measurements['total_length']['delta_from_source'] == pytest.approx(100, abs=.0001)
    assert measurements['vehicle_width']['delta_from_source'] == pytest.approx(50, abs=.0001)
    assert measurements['vehicle_height']['delta_from_source'] == pytest.approx(75, abs=.0001)


def test_body_stretch_keeps_overhangs_and_rigid_wheels(vehicle):
    result = deform(vehicle, {'vehicle_length': 200, 'ride_height': 30, 'vehicle_pitch': 1})
    a = result['metadata']['mishape']
    assert a['quality']['wheel_rigidity']['groups'] == 4
    assert a['quality']['wheel_rigidity']['verified']
    assert a['quality']['wheel_rigidity']['max_shape_error_mm'] == 0
    base_v = np.array(vehicle['vertices']).reshape(-1, 3)
    moved = np.array(result['vertices']).reshape(-1, 3)
    f = np.array(vehicle['faces']).reshape(-1, 3); labels = np.array(vehicle['face_labels'])
    for part in range(2, 6):
        ids = np.unique(f[labels == part])
        assert np.ptp(moved[ids], axis=0) == pytest.approx(np.ptp(base_v[ids], axis=0), abs=1e-12)
        assert moved[ids, 2] == pytest.approx(base_v[ids, 2], abs=1e-12)
    stretch = deform(vehicle, {'vehicle_length': 200})['metadata']['mishape']['analysis']['measurements']
    assert stretch['front_overhang']['delta_from_source'] == pytest.approx(0, abs=.0001)
    assert stretch['rear_overhang']['delta_from_source'] == pytest.approx(0, abs=.0001)
    assert stretch['wheelbase']['delta_from_source'] == pytest.approx(200, abs=.0001)


def test_controls_are_local_and_mirrored(vehicle):
    cage = get_cage(vehicle)
    assert len(cage['points']) == 84
    node = next(p for p in cage['points'] if p['index'] == [2, 2, 3])
    changed = get_cage(vehicle, controls=[{'id': node['id'], 'delta': [0, .1, .05]}])
    moved = changed['points'][node['id']]
    pair = changed['points'][node['mirror_id']]
    assert np.array(moved['position']) - moved['rest'] == pytest.approx([0, .1, .05])
    assert np.array(pair['position']) - pair['rest'] == pytest.approx([0, -.1, .05])
    result = deform(vehicle, controls=[{'id': node['id'], 'delta': [0, .1, .05]}])
    assert result['vertices'] != vehicle['vertices']
    assert result['faces'] == vehicle['faces']


def test_each_design_driver_changes_geometry_without_nonfinite_values(vehicle):
    for item in schema():
        result = deform(vehicle, {item['id']: 3 * item['step']})
        q = result['metadata']['mishape']['quality']
        assert q['finite'], item['id']
        assert q['max_displacement_mm'] > 0, item['id']
        assert q['topology_preserved']
        assert q['manufacturing_validation'] == 'not_performed'


def test_invalid_values_fail_loudly(vehicle):
    for parameters in ({'nonsense': 1}, {'vehicle_width': float('nan')}, {'vehicle_pitch': 100}, {'hood_angle': True}):
        with pytest.raises(ValueError):
            deform(vehicle, parameters)
    with pytest.raises(ValueError):
        deform(vehicle, controls=[{'id': 999, 'delta': [0, 0, 0]}])
    with pytest.raises(ValueError):
        deform(vehicle, controls=[{'id': 0, 'delta': [1, 0, 0]}])
    broken = dict(vehicle, vertices=vehicle['vertices'][:-1])
    with pytest.raises(ValueError):
        analyze(broken)


def test_batch_and_evaluation_are_reproducible(vehicle):
    ranges = {'vehicle_width': [-20, 20], 'vehicle_height': [-15, 15]}
    first = sample_parameters(8, seed=71, ranges=ranges)
    assert first == sample_parameters(8, seed=71, ranges=ranges)
    assert first != sample_parameters(8, seed=72, ranges=ranges)
    assert len({p['vehicle_width'] for p in first}) == 8
    assert deform(vehicle, first[0])['vertices'] == deform(vehicle, first[0])['vertices']
    assert all(-20 <= p['vehicle_width'] <= 20 for p in first)


def test_landmark_calibration_is_used_and_validated(vehicle):
    lm = {'front_axle': [-1.4, 0, .32], 'rear_axle': [1.4, 0, .32],
          'windshield_base': [-.9, 0, .9], 'windshield_top': [-.3, 0, 1.5]}
    analysis = analyze(vehicle, lm)
    assert analysis['calibration'] == 'user_landmarks'
    assert analysis['measurements']['windscreen_angle']['value'] == pytest.approx(45)
    assert analysis['landmarks']['front_axle'] == lm['front_axle']
    with pytest.raises(ValueError):
        analyze(vehicle, {'front_axle': [2, 0, .3], 'rear_axle': [-2, 0, .3]})


def test_trim_material_is_not_mistaken_for_a_rim(vehicle):
    model = deepcopy(vehicle)
    model['parts'][0].update(key='trim_and_underbody', name='饰件与底盘 / Trim')
    changed = deform(model, {'ride_height': 30})
    f = np.array(model['faces']).reshape(-1, 3)
    body_ids = np.unique(f[np.array(model['face_labels']) == 0])
    old = np.array(model['vertices']).reshape(-1, 3)
    new = np.array(changed['vertices']).reshape(-1, 3)
    assert new[body_ids, 2] - old[body_ids, 2] == pytest.approx(.03)
    assert changed['metadata']['mishape']['quality']['wheel_rigidity']['groups'] == 4


def test_estimated_asset_scale_is_not_reported_as_verified_dimensions(vehicle):
    model = deepcopy(vehicle)
    model['metadata'].update(scale_status='estimated', reference_length_m=4.4)
    analysis = analyze(model)
    assert not analysis['scale_calibration']['verified']
    assert analysis['measurements']['total_length']['method'] == 'mesh_bounds'
    assert analysis['measurements']['total_length']['estimated']
    assert analysis['measurements']['total_length']['confidence'] == 'low'
    assert any('尺度' in warning for warning in analysis['warnings'])
    result = deform(model, {'total_length': 100})
    assert result['metadata']['mishape']['analysis']['measurements']['total_length']['estimated']


def test_identity_does_not_replace_calibrated_axles_with_wheel_bounds(vehicle):
    landmarks = {'front_axle': [-1.4, 0, .33], 'rear_axle': [1.4, 0, .33]}
    result = deform(vehicle, options={'landmarks': landmarks})
    analysis = result['metadata']['mishape']['analysis']
    assert analysis['landmarks']['front_axle'] == landmarks['front_axle']
    assert analysis['landmarks']['rear_axle'] == landmarks['rear_axle']
    for item in analysis['measurements'].values():
        assert item['delta_from_source'] == 0
