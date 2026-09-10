"""Polygon-preserving OBJ reader; output retains the source coordinate system.

Only geometry is evaluated. Material library names, texture coordinates and
source normals are retained as data; no external files or URLs are opened.
"""
from __future__ import annotations

from collections import Counter
import io
import math

import numpy as np

MAX_BYTES = 180 * 1024 * 1024
MAX_VERTICES = 1_500_000
MAX_TRIANGLES = 2_000_000
MAX_POLYGON_VERTICES = 1024


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, point, area_tolerance, length_tolerance):
    return (abs(_cross(a, b, point)) <= area_tolerance
            and min(a[0], b[0]) - length_tolerance <= point[0] <= max(a[0], b[0]) + length_tolerance
            and min(a[1], b[1]) - length_tolerance <= point[1] <= max(a[1], b[1]) + length_tolerance)


def _segments_intersect(a, b, c, d, area_tolerance, length_tolerance):
    if (max(a[0], b[0]) + length_tolerance < min(c[0], d[0])
            or max(c[0], d[0]) + length_tolerance < min(a[0], b[0])
            or max(a[1], b[1]) + length_tolerance < min(c[1], d[1])
            or max(c[1], d[1]) + length_tolerance < min(a[1], b[1])):
        return False
    ab_c, ab_d, cd_a, cd_b = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > area_tolerance and ab_d < -area_tolerance or ab_c < -area_tolerance and ab_d > area_tolerance)
            and (cd_a > area_tolerance and cd_b < -area_tolerance or cd_a < -area_tolerance and cd_b > area_tolerance)):
        return True
    return any((_on_segment(a, b, c, area_tolerance, length_tolerance),
                _on_segment(a, b, d, area_tolerance, length_tolerance),
                _on_segment(c, d, a, area_tolerance, length_tolerance),
                _on_segment(c, d, b, area_tolerance, length_tolerance)))


def _simple_projection(projection, area_tolerance, length_tolerance):
    n = len(projection)
    for i in range(n):
        a, b = projection[i], projection[(i + 1) % n]
        if math.hypot(a[0] - b[0], a[1] - b[1]) <= length_tolerance:
            return False
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_intersect(a, b, projection[j], projection[(j + 1) % n], area_tolerance, length_tolerance):
                return False
    return True


def _triangulate(indices, vertices, statistics=None):
    """Tessellate simple projected boundaries; retain source triangles."""
    if len(indices) == 3:
        # Triangles are already planar. Avoid allocating projection arrays for
        # the common case of large triangle-only OBJ files.
        a, b, c = (vertices[index] for index in indices)
        extent = max(max(a[axis], b[axis], c[axis]) - min(a[axis], b[axis], c[axis]) for axis in range(3))
        if not math.isfinite(extent):
            raise ValueError('Triangle has unsupported coordinate extent.')
        if extent <= 0:
            if statistics is not None:
                statistics['source_degenerate_triangle_count'] += 1
            return [list(indices)]
        u = [(b[axis] - a[axis]) / extent for axis in range(3)]
        w = [(c[axis] - a[axis]) / extent for axis in range(3)]
        normal = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
        if math.hypot(*normal) <= 1e-12:
            if statistics is not None:
                statistics['source_degenerate_triangle_count'] += 1
        return [list(indices)]
    if len(set(indices)) != len(indices):
        raise ValueError('Polygon repeats a vertex; closed faces must not repeat their first index.')
    points = np.asarray([vertices[index] for index in indices], dtype=float)
    extent = float(np.ptp(points, axis=0).max())
    if extent <= 0:
        raise ValueError('Polygon has zero extent.')
    centered = points - points.mean(axis=0)
    normal = np.sum(np.cross(centered, np.roll(centered, -1, axis=0)), axis=0)
    normal_length = float(np.linalg.norm(normal))
    area_tolerance = extent * extent * 1e-12
    length_tolerance = extent * 1e-10
    if normal_length <= area_tolerance:
        raise ValueError('Polygon has zero signed area or is self-intersecting.')
    normal /= normal_length
    plane_tolerance = max(extent * 1e-6, np.finfo(float).eps * float(np.abs(points).max()) * 32)
    nonplanar = float(np.abs(centered @ normal).max()) > plane_tolerance
    axes = np.argsort(-np.abs(normal), kind='stable').tolist()
    # A skew source quad can overlap in its dominant projection while having a
    # simple boundary in another projection. Prefer the Newell dominant plane;
    # try the other two coordinate planes only for non-planar source polygons.
    candidates = axes if nonplanar else axes[:1]
    projection, selected_axis = None, None
    for axis in candidates:
        candidate = np.delete(centered, axis, axis=1).tolist()
        if _simple_projection(candidate, area_tolerance, length_tolerance):
            projection, selected_axis = candidate, axis
            break
    if projection is None:
        raise ValueError('Self-intersecting, self-touching or ambiguously projected polygon is unsupported.')
    n = len(indices)
    area_twice = sum(projection[i][0] * projection[(i + 1) % n][1]
                     - projection[(i + 1) % n][0] * projection[i][1] for i in range(n))
    if abs(area_twice) <= area_tolerance:
        raise ValueError('Polygon has zero projected area.')
    sign = 1 if area_twice > 0 else -1
    # A fan is valid only after proving convexity and a simple boundary.
    convex = all(sign * _cross(projection[i - 1], projection[i], projection[(i + 1) % n]) >= -area_tolerance for i in range(n))
    if convex:
        local_triangles = [(0, i, i + 1) for i in range(1, n - 1)
                           if sign * _cross(projection[0], projection[i], projection[i + 1]) > area_tolerance]
    else:
        remaining = list(range(n))
        local_triangles = []
        while len(remaining) > 3:
            found = False
            for position, middle in enumerate(remaining):
                before, after = remaining[position - 1], remaining[(position + 1) % len(remaining)]
                a, b, c = projection[before], projection[middle], projection[after]
                if sign * _cross(a, b, c) <= area_tolerance:
                    continue
                obstructed = any(sign * _cross(a, b, projection[other]) >= -area_tolerance
                                 and sign * _cross(b, c, projection[other]) >= -area_tolerance
                                 and sign * _cross(c, a, projection[other]) >= -area_tolerance
                                 for other in remaining if other not in (before, middle, after))
                if obstructed:
                    continue
                local_triangles.append((before, middle, after))
                remaining.pop(position)
                found = True
                break
            if not found:
                # Collinear boundary samples remain in the original polygon but
                # need not create zero-area triangles in the rendering mesh.
                for position, middle in enumerate(remaining):
                    before, after = remaining[position - 1], remaining[(position + 1) % len(remaining)]
                    if _on_segment(projection[before], projection[after], projection[middle], area_tolerance, length_tolerance):
                        remaining.pop(position)
                        found = True
                        break
            if not found:
                raise ValueError('Polygon triangulation failed; boundary must be simple and planar.')
        if len(remaining) == 3 and sign * _cross(*(projection[index] for index in remaining)) > area_tolerance:
            local_triangles.append(tuple(remaining))
    covered = sum(abs(_cross(*(projection[index] for index in triangle))) for triangle in local_triangles)
    if not local_triangles or not math.isclose(covered, abs(area_twice), rel_tol=1e-8, abs_tol=area_tolerance * n):
        raise ValueError('Polygon triangulation did not preserve area.')
    if nonplanar and statistics is not None:
        statistics['nonplanar_quad_count' if n == 4 else 'nonplanar_ngon_count'] += 1
        if selected_axis != axes[0]:
            statistics['alternate_projection_count'] += 1
    return [[indices[index] for index in triangle] for triangle in local_triangles]


def _index(token, count, kind):
    try:
        value = int(token)
    except (TypeError, ValueError):
        raise ValueError(f'Invalid OBJ {kind} index {token!r}.') from None
    if value == 0:
        raise ValueError(f'OBJ {kind} indices are one-based; zero is invalid.')
    index = value - 1 if value > 0 else count + value
    if not 0 <= index < count:
        raise ValueError(f'OBJ {kind} index {value} is outside previously declared data.')
    return index


def _numbers(tokens, minimum, maximum, kind):
    if not minimum <= len(tokens) <= maximum:
        raise ValueError(f'Invalid number of coordinates in OBJ {kind}.')
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        raise ValueError(f'Invalid numeric coordinate in OBJ {kind}.') from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'OBJ {kind} coordinates must be finite.')
    return values


def import_obj(data: bytes) -> dict:
    """Read OBJ bytes without normalizing axes, recentering, scaling or welding.

    ``polygons`` stores each original face. ``faces`` is a deterministic triangle
    rendering of those same boundaries. Face indices resolve against the global
    OBJ vertex pool, including negative relative indices. Supported polygons are
    simple in a coordinate-plane projection, at most 1024 corners per face.
    Non-planar boundaries prefer the Newell dominant plane and fall back to the
    other two planes only when needed to obtain a simple projection. Original
    polygons and source triangles, including degenerate ones, remain unchanged.
    """
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        raise ValueError('OBJ input must be nonempty text data no larger than 180 MB.')
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise ValueError('OBJ must be UTF-8 text; binary or undecodable data is unsupported.') from None
    vertices, textures, normals = [], [], []
    polygons, polygon_labels, polygon_uvs, polygon_normals = [], [], [], []
    triangles, labels, parts = [], [], []
    part_lookup, material_libraries = {}, []
    object_name, group_name, material_name = '', '', ''
    ignored, statistics = Counter(), Counter()
    pending = ''
    for line_number, raw_line in enumerate(io.StringIO(text), 1):
        line = raw_line.split('#', 1)[0].strip()
        if line.endswith('\\'):
            pending += line[:-1] + ' '
            continue
        line = pending + line
        pending = ''
        if not line:
            continue
        tokens = line.split()
        record, arguments = tokens[0], tokens[1:]
        try:
            if record == 'v':
                values = _numbers(arguments, 3, 7, 'vertex')
                if len(values) not in (3, 4, 6, 7):
                    raise ValueError('Vertex requires xyz, optional homogeneous w, or optional RGB.')
                xyz = values[:3]
                if len(values) in (4, 7):
                    if values[3] == 0:
                        raise ValueError('Homogeneous vertex w must be nonzero.')
                    xyz = [value / values[3] for value in xyz]
                    if not all(math.isfinite(value) for value in xyz):
                        raise ValueError('Homogeneous vertex produces nonfinite coordinates.')
                vertices.append(xyz)
                if len(vertices) > MAX_VERTICES:
                    raise ValueError('OBJ exceeds the 1.5 million vertex limit.')
            elif record == 'vt':
                textures.append(_numbers(arguments, 1, 3, 'texture coordinate'))
                if len(textures) > MAX_TRIANGLES * 3:
                    raise ValueError('OBJ has too many texture coordinates.')
            elif record == 'vn':
                normals.append(_numbers(arguments, 3, 3, 'normal'))
                if len(normals) > MAX_TRIANGLES * 3:
                    raise ValueError('OBJ has too many normals.')
            elif record == 'o':
                object_name, group_name = ' '.join(arguments), ''
            elif record == 'g':
                group_name = ' '.join(arguments)
            elif record == 'usemtl':
                material_name = ' '.join(arguments)
            elif record == 'mtllib':
                material_libraries.extend(arguments)
            elif record == 'f':
                if not 3 <= len(arguments) <= MAX_POLYGON_VERTICES:
                    raise ValueError(f'OBJ face requires 3–{MAX_POLYGON_VERTICES} vertices.')
                indices, uv_indices, normal_indices = [], [], []
                for reference in arguments:
                    fields = reference.split('/')
                    if not fields[0] or len(fields) > 3:
                        raise ValueError('Invalid OBJ face reference; expected v, v/vt, or v/vt/vn.')
                    indices.append(_index(fields[0], len(vertices), 'vertex'))
                    uv_indices.append(_index(fields[1], len(textures), 'texture') if len(fields) > 1 and fields[1] else -1)
                    normal_indices.append(_index(fields[2], len(normals), 'normal') if len(fields) > 2 and fields[2] else -1)
                face_triangles = _triangulate(indices, vertices, statistics)
                if len(triangles) + len(face_triangles) > MAX_TRIANGLES:
                    raise ValueError('OBJ exceeds the 2 million triangle limit.')
                component = (object_name, group_name, material_name)
                if component not in part_lookup:
                    pid = len(parts)
                    part_lookup[component] = pid
                    name = ' / '.join(dict.fromkeys(item for item in (object_name, group_name) if item)) or 'Imported mesh'
                    if material_name:
                        name += ' · ' + material_name
                    parts.append(dict(id=pid, key=f'obj_part_{pid}', name=name, color='#788e75',
                                      source_object=object_name, source_group=group_name, source_material=material_name,
                                      material=dict(name=material_name or 'OBJ default', base_color='#788e75', roughness=.4, metallic=.15)))
                pid = part_lookup[component]
                polygons.append(indices); polygon_labels.append(pid)
                polygon_uvs.append(uv_indices); polygon_normals.append(normal_indices)
                triangles.extend(face_triangles); labels.extend([pid] * len(face_triangles))
            elif record != 's':
                ignored[record] += 1
        except (ValueError, OverflowError) as error:
            raise ValueError(f'OBJ line {line_number}: {error}') from None
    if pending:
        raise ValueError('OBJ ends with an incomplete continued line.')
    if len(vertices) < 3 or not triangles:
        raise ValueError('OBJ contains no polygon surface.')
    return dict(vertices=np.asarray(vertices, dtype=float).reshape(-1).tolist(),
                faces=np.asarray(triangles, dtype=int).reshape(-1).tolist(), face_labels=labels,
                parts=parts, polygons=polygons, polygon_labels=polygon_labels,
                obj_texture_coordinates=textures, obj_source_normals=normals,
                polygon_texture_indices=polygon_uvs, polygon_normal_indices=polygon_normals,
                metadata=dict(source_format='obj', coordinates='source OBJ coordinates; not normalized',
                              polygon_preservation='original OBJ face boundaries and global vertex indices',
                              triangulation='Newell dominant-plane projection; deterministic ear clipping; validated convex diagonal',
                              nonplanar_quad_count=statistics['nonplanar_quad_count'],
                              nonplanar_ngon_count=statistics['nonplanar_ngon_count'],
                              alternate_projection_count=statistics['alternate_projection_count'],
                              source_degenerate_triangle_count=statistics['source_degenerate_triangle_count'],
                              nonplanar_quad_convention='Original 3D quad boundary is preserved; its display triangles follow a valid diagonal in the first simple axis projection, prioritized by Newell normal magnitude.',
                              nonplanar_ngon_convention='Non-planar source boundaries remain unchanged. Display triangulation uses the first simple axis projection, prioritized by Newell normal magnitude; this is a tessellation convention, not a unique inferred CAD surface.',
                              material_library_names=material_libraries, external_dependencies_loaded=False,
                              texture_note='UV and source normal references retained as data; viewport uses material base colors and recomputed normals.',
                              source_normal_note='obj_source_normals remain in the original OBJ coordinate system and do not drive rendering.',
                              ignored_obj_records=dict(ignored)))
