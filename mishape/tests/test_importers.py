"""OBJ n-gon preservation and triangulation correctness."""
import numpy as np
import pytest

from mishape.importers import import_obj


def _obj(points, faces, prefix=''):
    return (prefix + '\n'.join('v ' + ' '.join(str(value) for value in point) for point in points)
            + '\n' + '\n'.join('f ' + ' '.join(str(i) for i in face) for face in faces)).encode()


def _areas(model):
    v = np.asarray(model['vertices']).reshape(-1, 3)
    f = np.asarray(model['faces']).reshape(-1, 3)
    return np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) / 2


def test_quad_keeps_source_indices_and_unreferenced_vertices():
    model = import_obj(_obj([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0), (9, 9, 9)], [[1, 2, 3, 4]]))
    assert model['polygons'] == [[0, 1, 2, 3]]
    assert len(model['vertices']) == 15
    assert len(model['faces']) == 6
    assert _areas(model).sum() == pytest.approx(2)
    assert model['vertices'][-3:] == [9, 9, 9]


@pytest.mark.parametrize('reverse', [False, True])
def test_concave_polygon_ear_clipping_preserves_area_and_winding(reverse):
    # The notch makes a naive fan from the first vertex cover extra area.
    points = [(0, 0, 0), (3, 0, 0), (3, 3, 0), (2, 3, 0), (2, 1, 0), (1, 1, 0), (1, 3, 0), (0, 3, 0)]
    indices = list(range(1, 9))
    if reverse:
        indices.reverse()
    data = _obj(points, [indices])
    model = import_obj(data)
    assert model['polygons'] == [[i - 1 for i in indices]]
    assert _areas(model).sum() == pytest.approx(7)
    vertices = np.array(model['vertices']).reshape(-1, 3)
    faces = np.array(model['faces']).reshape(-1, 3)
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    assert np.all(normals[:, 2] < 0 if reverse else normals[:, 2] > 0)
    assert model['faces'] == import_obj(data)['faces']


def test_negative_indices_groups_materials_and_uv_references():
    data = b'''mtllib vehicle.mtl
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
o Body
g Hood
usemtl Paint
f -4/-4/-1 -3/-3/-1 -2/-2/-1 -1/-1/-1
g Wheel
usemtl Rubber
f 1//1 3//1 4//1
'''
    model = import_obj(data)
    assert model['polygons'] == [[0, 1, 2, 3], [0, 2, 3]]
    assert model['polygon_labels'] == [0, 1]
    assert model['face_labels'] == [0, 0, 1]
    assert model['parts'][0]['name'] == 'Body / Hood · Paint'
    assert model['parts'][1]['name'] == 'Body / Wheel · Rubber'
    assert model['polygon_texture_indices'] == [[0, 1, 2, 3], [-1, -1, -1]]
    assert model['polygon_normal_indices'] == [[0, 0, 0, 0], [0, 0, 0]]
    assert model['metadata']['material_library_names'] == ['vehicle.mtl']
    assert model['metadata']['external_dependencies_loaded'] is False


def test_tilted_planar_ngon_and_collinear_samples():
    points = [(0, 0, 0), (1, 0, 1), (2, 0, 2), (2, 1, 2), (0, 1, 0)]
    model = import_obj(_obj(points, [[1, 2, 3, 4, 5]]))
    assert model['polygons'] == [[0, 1, 2, 3, 4]]
    assert _areas(model).sum() == pytest.approx(2 * np.sqrt(2))
    assert np.all(_areas(model) > 0)


@pytest.mark.parametrize('points,faces', [
    ([(0, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)], [[1, 2, 3, 4]]),
    ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[0, 2, 3]]),
    ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[1, 2, 9]]),
    ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[-4, -2, -1]]),
    ([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [[1, 2, 3, 1]]),
    ([(0, 0, 0), (float('nan'), 0, 0), (0, 1, 0)], [[1, 2, 3]]),
])
def test_invalid_or_unsupported_polygon_is_rejected(points, faces):
    with pytest.raises(ValueError, match='OBJ line'):
        import_obj(_obj(points, faces))


def test_invalid_uv_reference_and_no_geometry_are_rejected():
    with pytest.raises(ValueError, match='texture index'):
        import_obj(b'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1 2/1 3/1')
    with pytest.raises(ValueError, match='no polygon surface'):
        import_obj(b'v 0 0 0\n')


def test_nonplanar_quad_is_preserved_with_reported_projection_convention():
    model = import_obj(_obj([(0, 0, 0), (1, 0, 0), (1, 1, .1), (0, 1, 0)], [[1, 2, 3, 4]]))
    assert model['polygons'] == [[0, 1, 2, 3]]
    assert len(model['faces']) == 6
    assert np.all(_areas(model) > 0)
    assert model['metadata']['nonplanar_quad_count'] == 1
    assert 'Newell' in model['metadata']['nonplanar_quad_convention']


def test_original_degenerate_triangles_are_retained_and_counted():
    model = import_obj(_obj([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [[1, 2, 3], [1, 1, 2]]))
    assert model['polygons'] == [[0, 1, 2], [0, 0, 1]]
    assert model['faces'] == [0, 1, 2, 0, 0, 1]
    assert model['metadata']['source_degenerate_triangle_count'] == 2


def test_nonplanar_source_ngon_uses_a_reported_simple_projection():
    points = [(0, 0, 0), (1, 0, 0), (1, 1, .1), (.5, 1.5, 0), (0, 1, 0)]
    model = import_obj(_obj(points, [[1, 2, 3, 4, 5]]))
    assert model['polygons'] == [[0, 1, 2, 3, 4]]
    assert model['metadata']['nonplanar_ngon_count'] == 1
    assert np.all(_areas(model) > 0)


def test_porsche_930_obj_round_trip_retains_original_vehicle_polygons():
    import json
    from pathlib import Path
    from mishape.app import mesh_bytes
    path = Path(__file__).resolve().parents[1] / 'assets' / 'porsche-930' / 'viewport.json'
    if not path.is_file():
        pytest.skip('Bundled Porsche 930 has not been converted.')
    source = json.loads(path.read_text())
    exported = mesh_bytes(source, 'obj')
    restored = import_obj(exported)
    assert restored['polygons'] == source['polygons']
    np.testing.assert_allclose(restored['vertices'], source['vertices'], rtol=0, atol=1e-8)
    areas = _areas(restored)
    assert np.all(np.isfinite(areas))
    assert np.all(areas >= 0)
    assert np.count_nonzero(areas > 0) > len(areas) * .99
    assert restored['metadata']['nonplanar_quad_count'] > 0
