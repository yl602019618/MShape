from __future__ import annotations
import io
import json
import zipfile
import hashlib
import re
from pathlib import Path
import numpy as np
import trimesh
from .model import Model, default_parts, array_hash
from .normalization import normalize_reference, resolve_units, FACE_ARRAYS

ALLOWED = {'.stl', '.obj', '.ply', '.off', '.glb', '.gltf', '.npz'}
MAX_FACES = 2_000_000
MAX_BYTES = 256 * 1024 * 1024


def npz_bytes(model: Model, vertices=None) -> bytes:
    b = io.BytesIO()
    # Large face-aligned provenance arrays belong in compressed numeric arrays,
    # not millions of JSON numbers. Decode remains compatible with old files.
    metadata = dict(model.metadata)
    arrays = {}
    for key in FACE_ARRAYS:
        if key in metadata:
            arrays['provenance_'+key] = np.asarray(metadata.pop(key))
    np.savez_compressed(b, vertices=model.vertices if vertices is None else vertices,
                        faces=model.faces, face_labels=model.labels,
                        parts_json=json.dumps(model.parts, ensure_ascii=False),
                        metadata_json=json.dumps(metadata, ensure_ascii=False), **arrays)
    return b.getvalue()


def load_npz(data: bytes, *, units='auto', up='+z', longitudinal='+x',
             center=True, weld=True, symmetry='auto') -> Model:
    if len(data) > MAX_BYTES:
        raise ValueError('Input exceeds 256 MiB.')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if sum(x.file_size for x in z.infolist()) > 512*1024**2:
            raise ValueError('Uncompressed NPZ exceeds the 512 MiB limit.')
    with np.load(io.BytesIO(data), allow_pickle=False) as d:
        if not {'vertices','faces'}.issubset(d.files):
            raise ValueError('NPZ requires vertices and faces arrays.')
        v, f = d['vertices'], d['faces']
        if len(f) > MAX_FACES:
            raise ValueError('Mesh exceeds the 2M-face limit.')
        if f.dtype.kind not in 'iu':
            raise ValueError('faces must have integer dtype.')
        # DrivAer annotation archives are source-space meshes, NOT canonical
        # AeroShape snapshots. Their labels are aligned to the contained faces.
        if 'raw_face_label' in d and 'face_labels' not in d:
            labels = d['raw_face_label']
            if 'raw_label_names' not in d:
                raise ValueError('Annotated NPZ requires the raw_label_names codebook.')
            names = d['raw_label_names']
            if names.ndim != 1 or names.dtype.kind not in 'US' or not 1 <= len(names) <= 1000:
                raise ValueError('raw_label_names must be a text codebook with at most 1000 names.')
            if labels.dtype.kind not in 'iu' or labels.shape != (len(f),) or labels.min() < 0 or labels.max() >= len(names):
                raise ValueError('raw_face_label must be face aligned and covered by raw_label_names.')
            names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in names]
            if any(not name or len(name)>200 for name in names):
                raise ValueError('Raw label names must contain between 1 and 200 characters.')
            parts = [dict(id=i, key=str(name).lower(), name=str(name),
                          color=default_parts()[i % len(default_parts())]['color'],
                          locked=any(word in str(name).lower() for word in ('wheel', 'tire', 'tyre')))
                     for i, name in enumerate(names)]
            # Validate geometry before coordinate inference or indexing triangles.
            raw = Model(v, f, labels, parts)
            ex, ez = _axis(longitudinal), _axis(up)
            if np.dot(ex, ez) != 0:
                raise ValueError('Longitudinal and up axes must be orthogonal.')
            rotation = np.stack((ex, np.cross(ez, ex), ez))
            selected_units, scale = resolve_units(raw.vertices, units)
            meta = dict(source_format='drivaernet_seg_npz', source_units=selected_units,
                        source_units_requested=units, source_scale=scale, source_up=up,
                        source_longitudinal=longitudinal, source_to_canonical_rotation=rotation.tolist(),
                        raw_label_names=[str(x) for x in names], raw_face_labels=labels.tolist(),
                        source_face_indices=np.arange(len(f), dtype=np.int32).tolist(),
                        source_annotation_association='contained_npz_faces_exact',
                        source_input_geometry_hash=array_hash(raw.vertices, raw.faces),
                        source_sha256=hashlib.sha256(data).hexdigest(),
                        annotation_status='original_source_labels_preserved',
                        asset_license='user_must_verify')
            for source_key in ('mesh_file', 'source'):
                if source_key in d and d[source_key].ndim == 0 and d[source_key].dtype.kind in 'US':
                    meta['annotation_'+source_key] = str(d[source_key])[:1000]
            out = Model((raw.vertices @ rotation.T)*scale, f, labels, parts, meta)
            return normalize_reference(out, symmetry=symmetry, center=center, weld=weld)
        labels = d['face_labels'] if 'face_labels' in d else np.zeros(len(f), np.int32)
        if labels.dtype.kind not in 'iu':
            raise ValueError('face_labels must have integer dtype.')
        parts = json.loads(str(d['parts_json'])) if 'parts_json' in d else default_parts()
        meta = json.loads(str(d['metadata_json'])) if 'metadata_json' in d else {}
        for key in FACE_ARRAYS:
            if 'provenance_'+key in d:
                a = d['provenance_'+key]
                if a.ndim != 1 or len(a) != len(f) or a.dtype.kind not in 'iuf':
                    raise ValueError(f'Invalid stored face provenance: {key}.')
                meta[key] = a.tolist()
        # Never trust a supplied hash to represent these actual bytes.
        meta['base_geometry_hash'] = array_hash(v, f)
        return Model(v,f,labels,parts,meta)


def _axis(s: str):
    lookup = {'+x':[1,0,0],'-x':[-1,0,0],'+y':[0,1,0],'-y':[0,-1,0],'+z':[0,0,1],'-z':[0,0,-1]}
    if s not in lookup:
        raise ValueError('Axis must be +x, -x, +y, -y, +z, or -z.')
    return np.array(lookup[s], float)


def import_mesh(data: bytes, filename: str, *, units='m', up='+z', longitudinal='+x',
                center=True, weld=True, symmetry='keep') -> Model:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError('Supported: STL, OBJ, PLY, OFF, GLB, self-contained glTF, and AeroShape NPZ.')
    if len(data) > MAX_BYTES:
        raise ValueError('Input exceeds 256 MiB.')
    if suffix == '.npz':
        return load_npz(data, units=units, up=up, longitudinal=longitudinal,
                        center=center, weld=weld, symmetry=symmetry)
    if suffix == '.gltf':
        document = json.loads(data)
        for category in ('buffers','images'):
            for obj in document.get(category,[]):
                if 'uri' in obj and not obj['uri'].startswith('data:'):
                    raise ValueError('External glTF resources are disabled. Export a self-contained GLB instead.')
    if suffix == '.glb':
        # Block external buffers/images in the GLB JSON chunk as well.
        import struct
        if len(data) < 20 or data[:4] != b'glTF':
            raise ValueError('Invalid GLB header.')
        length, kind = struct.unpack('<II', data[12:20])
        if kind != 0x4E4F534A or length > len(data)-20:
            raise ValueError('Invalid GLB JSON chunk.')
        doc = json.loads(data[20:20+length])
        for category in ('buffers','images'):
            for obj in doc.get(category,[]):
                if 'uri' in obj and not obj['uri'].startswith('data:'):
                    raise ValueError('External GLB resources are disabled.')
    if units not in ('auto', 'm', 'cm', 'mm'):
        raise ValueError('Units must be auto, m, cm, or mm.')
    ex, ez = _axis(longitudinal), _axis(up)
    if np.dot(ex,ez) != 0:
        raise ValueError('Longitudinal and up axes must be orthogonal.')
    ey = np.cross(ez,ex)
    rotation = np.stack((ex,ey,ez))
    raw = trimesh.load(io.BytesIO(data), file_type=suffix[1:], process=False, force='scene',
                       resolver={}, skip_materials=True, group_material=False, split_objects=True, split_groups=True)
    vertices, faces, labs, parts = [], [], [], []
    offset = 0
    for node in raw.graph.nodes_geometry:
        transform, geom_name = raw.graph[node]
        mesh = raw.geometry[geom_name]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces)==0:
            continue
        if len(mesh.faces) + sum(len(f) for f in faces) > MAX_FACES:
            raise ValueError('Mesh exceeds the 2M-face limit.')
        v = trimesh.transform_points(mesh.vertices, transform)
        v = v @ rotation.T
        i = len(parts)
        name = str(node)[:150]
        key = name.lower()
        parts.append(dict(id=i, key=key, name=name,
                          color=default_parts()[i % len(default_parts())]['color'],
                          locked=any(x in key for x in ('wheel','tire','tyre'))))
        vertices.append(v); faces.append(np.asarray(mesh.faces) + offset)
        labs.append(np.full(len(mesh.faces), i, np.int32)); offset += len(v)
    if not vertices:
        raise ValueError('No triangle mesh found. Point clouds alone are not editable surfaces.')
    v, f, labels = np.vstack(vertices), np.vstack(faces), np.concatenate(labs)
    if len(parts) > 1000:
        raise ValueError('Too many scene nodes (>1000); simplify the source assembly.')
    selected_units, scale = resolve_units(v, units)
    v *= scale
    meta = dict(source_filename=Path(filename).name, source_sha256=hashlib.sha256(data).hexdigest(),
                source_units=selected_units, source_units_requested=units, source_up=up, source_longitudinal=longitudinal,
                source_to_canonical_rotation=rotation.tolist(), source_scale=scale,
                annotation_status='source_scene_nodes_not_verified_semantics',
                asset_license='user_must_verify', imported_materials_preserved=False)
    return normalize_reference(Model(v,f,labels,parts,meta), symmetry=symmetry, center=center, weld=weld)


def export_bundle(model: Model, vertices: np.ndarray, recipe: dict, report: dict) -> bytes:
    """Export assembled geometry only. No view or explosion transforms enter here."""
    result = io.BytesIO()
    mesh = model.mesh(vertices)
    with zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('reference.npz', npz_bytes(model))
        z.writestr('surface.npz', npz_bytes(model, vertices))
        z.writestr('surface.stl', mesh.export(file_type='stl'))
        z.writestr('surface.ply', mesh.export(file_type='ply'))
        z.writestr('display-y-up.glb', glb_bytes(model, vertices))
        # OBJ g groups are useful; labels remain authoritative in NPZ/JSON.
        obj = ['# AeroShape canonical units m; X front-to-rear, Z up']
        obj.extend(f'v {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}' for p in vertices)
        patch_table = []
        for part in model.parts:
            ids = np.flatnonzero(model.labels == part['id'])
            if not len(ids): continue
            slug = re.sub(r'[^A-Za-z0-9_-]', '_', part.get('key','part'))[:50]
            name = f"part_{part['id']}_{slug}"
            obj.append('g '+name)
            obj.extend(f'f {a+1} {b+1} {c+1}' for a,b,c in model.faces[ids])
            pm = trimesh.Trimesh(vertices=vertices, faces=model.faces[ids], process=False)
            z.writestr(f'patches/{name}.stl', pm.export(file_type='stl'))
            patch_table.append(dict(part_id=part['id'], name=part['name'], file=f'patches/{name}.stl',
                                    face_count=len(ids), note='An open patch is not itself a closed solid.'))
        z.writestr('surface.obj', '\n'.join(obj)+'\n')
        metadata = dict(schema_version='aeroshape-0.3', units='m', source=model.metadata,
                        geometry_hash=array_hash(vertices, model.faces),
                        annotation_hash=model.annotation_hash(), parts=model.parts,
                        patches=patch_table, recipe=recipe, quality=report,
                        cfd_status='not_run', surrogate_status='not_run',
                        operating_conditions=None, frontal_area_reference=None,
                        view_transforms_exported=False)
        z.writestr('design.json', json.dumps(metadata, indent=2, ensure_ascii=False))
        z.writestr('README.txt', 'Geometry is in the assembled canonical frame, in metres.\n'
                   'Exploded view is NEVER exported. NPZ preserves face labels; STL does not.\n'
                   'display-y-up.glb is a named material-bearing display copy in glTF Y-up axes.\n'
                   'NPZ/STL/PLY/OBJ use canonical Z-up coordinates; GLB uses (X,Z,-Y).\n'
                   'No CFD validity, global collision clearance, or manufacturability certificate.\n'
                   'Run geometry repair/clearance checks and external-flow meshing before CFD.\n')
    return result.getvalue()


def glb_bytes(model: Model, vertices=None) -> bytes:
    """Named, material-bearing *display* assembly, in glTF Y-up metres.

    NPZ remains authoritative for shared vertices, labels and the recipe. No
    explosion transforms are applied. Convert (X,Y,Z) to (X,Z,-Y) for glTF.
    """
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals
    v=model.vertices if vertices is None else np.asarray(vertices, dtype=float)
    if v.shape!=model.vertices.shape or not np.isfinite(v).all():
        raise ValueError('Export vertices must match the finite reference shape.')
    rotation=np.array([[1.,0,0],[0,0,1],[0,-1,0]])
    gltf_vertices=v@rotation.T
    scene=trimesh.Scene()
    face_materials=model.metadata.get('face_material_ids')
    palette=model.metadata.get('material_palette',[])
    if face_materials is not None:
        face_materials=np.asarray(face_materials,dtype=int)
        if face_materials.shape!=(len(model.faces),) or not len(palette) or face_materials.min()<0 or face_materials.max()>=len(palette):
            raise ValueError('Face material indices do not match the reference mesh.')
    for part in model.parts:
        selected=np.flatnonzero(model.labels==part['id'])
        if not len(selected): continue
        slots=np.unique(face_materials[selected]) if face_materials is not None else [None]
        for slot in slots:
            face_ids=selected if slot is None else selected[face_materials[selected]==slot]
            faces=model.faces[face_ids]
            used, inv=np.unique(faces, return_inverse=True)
            mesh=trimesh.Trimesh(gltf_vertices[used],inv.reshape(-1,3),process=False)
            mat=part.get('material', {}) if slot is None else palette[int(slot)]
            color=mat.get('base_color',part.get('color','#8296a7'))
            if not re.fullmatch(r'#[0-9a-fA-F]{6}',color):color='#8296a7'
            rgb=[int(color[i:i+2],16) for i in (1,3,5)]
            emission=float(np.clip(mat.get('emission',0.),0.,1.))
            name=f"{part['id']:03d}_{part.get('key','part')}"+(f"_mat{int(slot)}" if slot is not None else '')
            material=PBRMaterial(name=name,
                baseColorFactor=rgb+[255], metallicFactor=float(np.clip(mat.get('metallic',.05),0,1)),
                roughnessFactor=float(np.clip(mat.get('roughness',.65),0,1)),
                emissiveFactor=[emission*x/255 for x in rgb], doubleSided=True,alphaMode='OPAQUE')
            mesh.visual=TextureVisuals(material=material)
            mesh.metadata.update(part_id=part['id'], semantic_name=part['name'], locked=part.get('locked',False))
            scene.add_geometry(mesh,node_name=name,geom_name=name)
    scene.metadata.update(name=model.metadata.get('name','AeroShape design'),
        license=model.metadata.get('asset_license','user_must_verify'),units='metres',
        axis_note='glTF X=canonical X, Y=canonical Z, Z=-canonical Y',
        assembled=True,cfd_ready=False,reference_format='AeroShape NPZ')
    return scene.export(file_type='glb',include_normals=True)
