"""Fitted coordinates, real vehicle envelopes, and recipe field transfer."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from mishape import cage as cage_geometry
from mishape.engine import deform, get_cage, regrid, _prepare


ASSETS = Path(__file__).resolve().parents[1] / 'assets'


@pytest.fixture(scope='module', params=['porsche-930', 'porsche-carrera-4s'])
def porsche(request):
    return json.loads((ASSETS / request.param / 'viewport.json').read_text())


def options(dims=(9, 3, 4), **kw):
    return dict(cage=dict(type='fitted', dimensions=list(dims), padding_mm=30), **kw)


def test_real_vehicle_cage_follows_hood_roof_and_cabin_width(porsche):
    cage = get_cage(porsche, options=options())
    grid = np.array([p['rest'] for p in cage['points']]).reshape(9, 3, 4, 3)
    top = grid[:, 1, -1, 2]
    assert top[0] < top.max() - .45
    assert top[1] < top.max() - .3
    assert top[-1] < top.max() - .3
    # Roof side sections taper inward from the shoulder at the same station.
    assert grid[4, 2, -1, 1] < grid[4, 2, 1, 1] - .15
    assert cage['type'] == 'fitted'
    assert cage['method'] == 'curvilinear_section_cubic_displacement'
    assert cage['diagnostics']['excluded_vertices'] > 1000
    assert cage['diagnostics']['source_coverage_percent'] > 95
    assert np.allclose(grid[:, 0, :, 1], -grid[:, 2, :, 1])


def test_dynamic_counts_boundary_visibility_and_mirror_ids(porsche):
    for dims in [(5, 3, 3), (9, 5, 5), (17, 7, 7)]:
        cage = get_cage(porsche, options=options(dims))
        assert len(cage['points']) == np.prod(dims)
        expected_boundary = np.prod(dims) - np.prod(np.array(dims) - 2)
        assert cage['visible_points'] == expected_boundary
        for point in cage['points']:
            i, j, k = point['index']
            expected = any(n in (0, d - 1) for n, d in zip((i, j, k), dims))
            assert point['visible'] == expected
            partner = cage['points'][point['mirror_id']]
            assert partner['index'] == [i, dims[1] - 1 - j, k]
            assert partner['mirror_id'] == point['id']
        for start, end in cage['edges']:
            assert cage['points'][start]['visible'] and cage['points'][end]['visible']
            a, b = np.array(cage['points'][start]['index']), np.array(cage['points'][end]['index'])
            assert np.abs(a - b).sum() == 1


def test_fitted_grid_points_roundtrip_through_real_binding(porsche):
    data = _prepare(porsche, cage_settings=options()['cage'])
    uvw = cage_geometry.coordinates(data['grid'], data['shape'])
    expected = np.stack(np.meshgrid(*[np.linspace(0, 1, d) for d in data['dimensions']], indexing='ij'), axis=-1).reshape(-1, 3)
    assert uvw == pytest.approx(expected, abs=1e-12)
    assert cage_geometry.positions(uvw, data['shape']) == pytest.approx(data['grid'], abs=1e-12)
    # Hood-top vertices occupy the fitted top band, not the box's middle band.
    nose_top = data['grid'].reshape(9, 3, 4, 3)[1, 1, -1]
    assert cage_geometry.coordinates(nose_top[None], data['shape'])[0, 2] == pytest.approx(1)
    box_z = (nose_top[2] - data['lo'][2]) / (data['hi'][2] - data['lo'][2])
    assert box_z < .7


def test_identity_topology_metadata_and_legacy_recipe_replay(porsche):
    before = deepcopy(porsche)
    fitted = deform(porsche, options=options())
    assert fitted['vertices'] == porsche['vertices']
    for key in ('faces', 'polygons', 'face_labels', 'parts'):
        if key in porsche:
            assert fitted[key] == porsche[key]
    assert porsche == before
    parameters = dict(total_length=75, hood_angle=2, greenhouse_tapering=30)
    controls = [dict(id=35, delta=[.01, .02, .03])]
    legacy = deform(porsche, parameters, controls)
    explicit = deform(porsche, parameters, controls, dict(cage=dict(type='box', dimensions=[7, 3, 4], padding_mm=0)))
    assert legacy['vertices'] == explicit['vertices']
    assert legacy['metadata']['mishape']['source_hash'] == explicit['metadata']['mishape']['source_hash']


def test_manual_control_changes_local_surface_and_keeps_wheels_rigid(porsche):
    cage = get_cage(porsche, options=options())
    node = next(p for p in cage['points'] if p['index'] == [2, 2, 3])
    controls = [dict(id=node['id'], delta=[0, .025, .045])]
    edited = deform(porsche, controls=controls, options=options())
    before = np.array(porsche['vertices']).reshape(-1, 3)
    after = np.array(edited['vertices']).reshape(-1, 3)
    distances = np.linalg.norm(after - before, axis=1)
    assert distances.max() > .01
    # Cubic support does not reach the far rear stations.
    assert distances[before[:, 0] > before[:, 0].max() - .15].max() < 1e-10
    changed_cage = edited['metadata']['mishape']['cage']
    primary = changed_cage['points'][node['id']]
    mirror = changed_cage['points'][node['mirror_id']]
    assert np.array(primary['position']) - primary['rest'] == pytest.approx([0, .025, .045])
    assert np.array(mirror['position']) - mirror['rest'] == pytest.approx([0, -.025, .045])
    quality = edited['metadata']['mishape']['quality']
    assert quality['wheel_rigidity']['groups'] == 4
    assert quality['wheel_rigidity']['max_shape_error_mm'] == 0
    assert quality['topology_preserved']


def test_parameters_are_density_independent_and_global_affine_exact(porsche):
    parameters = dict(total_length=90, vehicle_width=40, vehicle_height=25, hood_angle=2, greenhouse_tapering=10)
    low = deform(porsche, parameters, options=options((5, 3, 3)))
    high = deform(porsche, parameters, options=options((17, 7, 7)))
    assert low['vertices'] == high['vertices']
    affine = deform(porsche, dict(total_length=90, vehicle_width=40, vehicle_height=25), options=options(preserve_wheels=False))
    measures = affine['metadata']['mishape']['analysis']['measurements']
    for key, expected in [('total_length', 90), ('vehicle_width', 40), ('vehicle_height', 25)]:
        assert measures[key]['delta_from_source'] == pytest.approx(expected, abs=.0001)


def test_regrid_preserves_parameters_and_reports_measured_surface_error(porsche):
    base = options()
    parameters = dict(vehicle_width=40, hood_angle=1)
    cage = get_cage(porsche, options=base)
    node = next(p['id'] for p in cage['points'] if p['index'] == [4, 2, 3])
    controls = [dict(id=node, delta=[.015, .04, .05])]
    changed = regrid(porsche, parameters, controls, base, options((13, 5, 5))['cage'])
    assert changed['parameters'] == parameters
    assert changed['controls']
    assert changed['options']['cage']['dimensions'] == [13, 5, 5]
    old_v = np.array(deform(porsche, parameters, controls, base)['vertices']).reshape(-1, 3)
    new_v = np.array(deform(porsche, changed['parameters'], changed['controls'], changed['options'])['vertices']).reshape(-1, 3)
    errors = np.linalg.norm(new_v - old_v, axis=1) * 1000
    report = changed['resampling']
    assert report['max_error_mm'] == pytest.approx(errors.max(), abs=1e-6)
    assert report['rms_error_mm'] == pytest.approx(np.sqrt(np.mean(errors ** 2)), abs=1e-6)
    # Non-nested Catmull-Rom knots give an approximate transfer. A 67 mm
    # local edit must stay within 6 mm at every vertex and 0.5 mm RMS.
    assert report['max_error_mm'] < 6
    assert report['rms_error_mm'] < .5
    coarse = regrid(porsche, parameters, controls, base, options((5, 3, 3))['cage'])
    assert coarse['resampling']['requires_review']
    assert coarse['resampling']['warning']
    assert not coarse['resampling']['exact']
    assert coarse['resampling']['evaluated_vertices'] == len(old_v)


def test_parameter_only_density_change_has_zero_error(porsche):
    result = regrid(porsche, dict(vehicle_height=40, hood_angle=3), [], options(), options((5, 3, 3))['cage'])
    assert result['controls'] == []
    assert result['resampling']['max_error_mm'] == 0
    assert result['resampling']['exact']


def test_regrid_refines_rare_surface_peaks_for_roof_center_edit(porsche):
    # Browser regression: uniform sampling missed sparse interior/trim vertices,
    # producing 6.05 / 11.47 mm peaks despite a sub-mm RMS on the two Porsches.
    base = options()
    result = regrid(porsche, dict(total_length=120), [dict(id=55, delta=[0, 0, .06])],
                    base, options((13, 5, 5))['cage'])
    report = result['resampling']
    assert report['max_error_mm'] < 4
    assert report['rms_error_mm'] < .4
    assert 1 <= report['refinement_passes'] <= 3
    assert report['refinement_accepted'] >= 1
    assert report['extra_samples'] > 0
    assert report['max_error_mm'] < report['initial_max_error_mm']
    assert report['parameters_preserved']
    assert report['requires_review']  # Remaining approximation stays explicit.


@pytest.mark.parametrize('bad', [dict(type='sphere'), dict(dimensions=[4, 3, 4]), dict(dimensions=[9, 4, 4]),
    dict(dimensions=[9, 3, 8]), dict(dimensions=[9.5, 3, 4]), dict(padding_mm=-1), dict(padding_mm=151),
    dict(padding_mm=float('nan')), dict(padding_mm=True), dict(dimensions=[True, 3, 4]), dict(typo=1)])
def test_invalid_cage_settings_fail_loudly(bad):
    with pytest.raises(ValueError):
        cage_geometry.settings(bad)
