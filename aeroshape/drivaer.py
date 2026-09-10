"""Local DrivAerNet ingestion, explicit assembly manifests and safe archives.

The adapter targets surface geometries, not OpenFOAM cases, ANSA CAD or CFD
labels. Filename classification is a hint, never authentication of an asset.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import zipfile
from typing import Literal
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from .io import import_mesh, ALLOWED, MAX_BYTES, MAX_FACES
from .model import Model, array_hash
from .semantic import SegmentationOptions, segment30, SCHEMA

DOI='doi:10.7910/DVN/CAWRXI'
MAX_EXPANDED=1024*1024**2
MAX_MEMBERS=5000
MAX_BATCH=128


class ImportOptions(BaseModel):
    model_config=ConfigDict(extra='forbid')
    source:Literal['drivaernet','user','original_demo']='drivaernet'
    dataset_version:str=Field(default='unspecified',max_length=80)
    body_type:Literal['auto','fastback','notchback','estateback','suv','hatchback','unknown']='auto'
    units:Literal['auto','m','cm','mm']='auto'
    up:Literal['+x','-x','+y','-y','+z','-z']='+z'
    longitudinal:Literal['+x','-x','+y','-y','+z','-z']='+x'
    center:bool=True
    weld:bool=True
    symmetry:Literal['auto','keep','mirror']='auto'
    auto_segment:bool=True
    source_map:dict[str,str]=Field(default_factory=dict)
    asset_license:str|None=Field(default=None,max_length=200)
    attribution:str=Field(default='',max_length=2000)
    family_id:str|None=Field(default=None,max_length=160)
    accept_noncommercial:bool=False


def classify_name(filename:str) -> dict:
    stem=Path(filename).stem
    if stem.lower().endswith('.seg'):
        stem=stem[:-4]
    # Documented F/N/E family prefix; configuration tokens deliberately opaque.
    match=re.fullmatch(r'(?:DrivAer_)?([FNE])_([A-Za-z0-9_]+)_(\d+)',stem,re.I)
    if match:
        code,config,design=match.groups();body={'F':'fastback','N':'notchback','E':'estateback'}[code.upper()]
        return dict(case_id=stem,body_type=body,configuration=config.upper(),design_index=design,
                    inferred_from='filename_hint_not_verified_geometry')
    for family in ('fastback','notchback','estateback','suv','hatchback'):
        if family in stem.lower():return dict(case_id=stem,body_type=family,configuration=None,design_index=None,inferred_from='name_hint')
    return dict(case_id=stem,body_type='unknown',configuration=None,design_index=None,inferred_from='unknown')


def safe_relative(name:str) -> str:
    if not isinstance(name,str) or not name or '\x00' in name or '\\' in name or re.match(r'^[A-Za-z]:',name):
        raise ValueError('Unsafe archive/manifest path.')
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.name:raise ValueError('Archive/manifest traversal is not allowed.')
    return str(p)


def read_archive(data:bytes,filename:str) -> dict[str,bytes]:
    """Read regular files only, without extracting any filesystem paths."""
    if len(data)>MAX_BYTES:raise ValueError('Archive upload exceeds 256 MiB. Use the local directory CLI for larger batches.')
    result={};total=0
    def add(name,size,read):
        nonlocal total
        name=safe_relative(name)
        if name in result:raise ValueError(f'Duplicate archive entry: {name}')
        total+=size
        if size>MAX_BYTES or total>MAX_EXPANDED:raise ValueError('Archive expansion exceeds safety limits.')
        # Read only surfaces and small manifest/sidecar JSON, never executable content.
        if Path(name).suffix.lower() not in ALLOWED|{'.json','.csv'}:return
        if Path(name).suffix.lower() in ('.json','.csv') and size>16*1024**2:raise ValueError('Metadata file exceeds 16 MiB.')
        body=read()
        if len(body)!=size:raise ValueError('Truncated archive member.')
        result[name]=body
    if filename.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            entries=z.infolist()
            if len(entries)>MAX_MEMBERS:raise ValueError('Too many archive members.')
            # Validate all paths and entry kinds, not just selected mesh entries.
            for entry in entries:
                safe_relative(entry.filename.rstrip('/'))
                if stat.S_ISLNK(entry.external_attr>>16):raise ValueError('Archive symlinks are forbidden.')
                if entry.flag_bits & 1:raise ValueError('Encrypted archives are unsupported.')
                if entry.is_dir():continue
                add(entry.filename,entry.file_size,lambda e=entry:z.read(e))
    else:
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:*') as tar:
            for i,entry in enumerate(tar):
                if i>=MAX_MEMBERS:raise ValueError('Too many archive members.')
                safe_relative(entry.name.rstrip('/'))
                if entry.isdir():continue
                if not entry.isfile():raise ValueError('Archive links/devices are forbidden.')
                add(entry.name,entry.size,lambda e=entry:tar.extractfile(e).read())
    return result


def _attach_face_sidecar(model:Model, payload:dict) -> Model:
    """Accept exact canonical face labels only with both geometry and order hashes.

    External 29-ID labels are not assumed to match our numbering. Convert them
    with an explicit codebook/association first; never load pickled .pt files.
    """
    if payload.get('association')!='canonical_faces':raise ValueError('Sidecar association must be canonical_faces; point/vertex labels require explicit projection.')
    if payload.get('geometry_hash')!=array_hash(model.vertices,model.faces):raise ValueError('Sidecar geometry hash does not match imported canonical geometry.')
    labels=np.asarray(payload.get('face_labels',[]))
    if labels.dtype.kind not in 'iu' or labels.shape!=(len(model.faces),) or labels.min()<0:raise ValueError('Sidecar face_labels shape/type mismatch.')
    codebook=payload.get('codebook')
    if not isinstance(codebook,dict) or not set(map(str,np.unique(labels))).issubset(codebook):raise ValueError('Explicit source label codebook required.')
    parts=[dict(id=int(i),key=str(k),name=str(k),locked=False,color='#89a9bd') for i,k in codebook.items()]
    out=model.clone(labels=labels,parts=parts)
    out.metadata['sidecar_association']='canonical_faces_geometry_hash_verified'
    return out


def import_vehicle(data:bytes,filename:str,options:ImportOptions,*,case:dict|None=None,
                   sidecar:dict|None=None) -> Model:
    if options.source=='drivaernet' and not options.accept_noncommercial:
        raise ValueError('Confirm DrivAerNet CC BY-NC 4.0 research-use restriction before importing (accept_noncommercial=true).')
    m=import_mesh(data,filename,units=options.units,up=options.up,longitudinal=options.longitudinal,
                  center=options.center,weld=options.weld,symmetry=options.symmetry)
    if sidecar:m=_attach_face_sidecar(m,sidecar)
    info=classify_name(filename);case=case or {}
    declared=case.get('body_type')
    body=declared or (info['body_type'] if options.body_type=='auto' else options.body_type)
    if body not in ('fastback','notchback','estateback','suv','hatchback','unknown'):raise ValueError('Invalid body type in manifest.')
    # The ID denotes a template family, not a promise of shared vertex indices.
    family=case.get('family_id') or options.family_id or f"{options.source}:{body}:{info['configuration'] or 'unspecified'}"
    license_value=options.asset_license or ('CC BY-NC 4.0' if options.source=='drivaernet' else m.metadata.get('asset_license','user_must_verify'))
    if options.source=='drivaernet':license_value='CC BY-NC 4.0'  # A manifest cannot silently relicense source data.
    provenance=dict(source_type=options.source,dataset_doi=DOI if options.source=='drivaernet' else None,
        dataset_version=options.dataset_version,case_id=case.get('case_id') or info['case_id'],
        source_filename=safe_relative(filename),source_sha256=hashlib.sha256(data).hexdigest(),
        dataset_origin_verification='user_supplied_not_remotely_verified',configuration=info['configuration'],
        filename_classification=info,attribution=options.attribution or ('Elrefaie, Morar, Dai and Ahmed; DrivAerNet++' if options.source=='drivaernet' else ''),
        license=license_value,noncommercial_acknowledged=options.accept_noncommercial,
        input_transform=dict(units=options.units,resolved_units=m.metadata.get('source_units','m'),
                             up=options.up,longitudinal=options.longitudinal,center=options.center,
                             weld=options.weld,symmetry=options.symmetry))
    m.metadata.update(name=case.get('name') or case.get('case_id') or info['case_id'],body_type=body,family_id=family,
        asset_license=license_value,provenance=provenance,cfd_ready=False)
    if options.auto_segment:
        m=segment30(m,SegmentationOptions(body_type=body,source_map=case.get('source_map',options.source_map)))
    return m


def assemble_vehicle(items:list[tuple[str,bytes,str|None]],options:ImportOptions,case:dict) -> Model:
    """Shared source assembly frame; never center each component separately."""
    if not items:raise ValueError('Assembly manifest has no surfaces.')
    if options.source=='drivaernet' and not options.accept_noncommercial:raise ValueError('Confirm noncommercial research restriction.')
    vertices=[];faces=[];labs=[];parts=[];offset=0;nf=0;provenance=[]
    for name,data,part_key in items:
        # Source assembly members share one unit system; auto resolve it from
        # their combined bounds, since a small mirror alone is not a car.
        m=import_mesh(data,name,units='m' if options.units=='auto' else options.units,
                      up=options.up,longitudinal=options.longitudinal,center=False,weld=options.weld)
        if Path(name).suffix.lower()=='.npz':raise ValueError('Assembly members must share raw source coordinates; do not mix canonical NPZ transforms.')
        nf+=len(m.faces)
        if nf>MAX_FACES:raise ValueError('Assembled surface exceeds 2M faces.')
        vertices.append(m.vertices);faces.append(m.faces+offset);offset+=len(m.vertices)
        mapping={}
        if part_key:
            pid=len(parts);parts.append(dict(id=pid,key=part_key,name=part_key,color='#93a9b9',locked=False))
            labs.append(np.full(len(m.faces),pid,np.int32))
        else:
            for p in m.parts:
                pid=len(parts);mapping[p['id']]=pid;parts.append({**p,'id':pid})
            labs.append(np.array([mapping[int(i)] for i in m.labels],np.int32))
        provenance.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),part_key=part_key))
    v=np.vstack(vertices);f=np.vstack(faces).astype(np.int32);labels=np.concatenate(labs)
    if options.units=='auto':
        from .normalization import resolve_units
        resolved,scale=resolve_units(v,'auto');v*=scale
    translation=np.zeros(3)
    if options.center:
        translation[:2]=-(v.min(0)[:2]+v.max(0)[:2])/2;translation[2]=-v[:,2].min();v+=translation
    # Do not weld across independently supplied assembly members; preserve intentional gaps.
    raw=Model(v,f,labels,parts,dict(assembly_members=provenance,source_translation_m=translation.tolist(),
        source_units=options.units,source_up=options.up,source_longitudinal=options.longitudinal,
        assembly_policy='explicit_common_frame_no_per_part_center_no_cross_member_weld'))
    from .io import npz_bytes
    internal_options=options.model_copy(update={'auto_segment':False})
    m=import_vehicle(npz_bytes(raw),(case.get('case_id') or 'assembly')+'.npz',internal_options,case=case)
    m.metadata['provenance']['assembly_members']=provenance
    m.metadata['provenance']['source_sha256']=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    m.metadata['provenance']['source_filename']='explicit_assembly_manifest'
    m.metadata['provenance']['input_transform'].update(center=options.center)
    if options.symmetry!='keep':
        from .normalization import normalize_reference
        m=normalize_reference(m,options.symmetry,center=options.center,weld=False)
    if options.auto_segment:m=segment30(m,SegmentationOptions(body_type=m.metadata['body_type'],source_map=case.get('source_map',options.source_map)))
    return m


def _verify_seg_pair(mesh_data:bytes,mesh_name:str,seg_data:bytes) -> dict:
    """Verify triangle coordinates/order before using an NPZ as an STL sidecar."""
    with zipfile.ZipFile(io.BytesIO(seg_data)) as z:
        if sum(info.file_size for info in z.infolist())>512*1024**2:
            raise ValueError('Uncompressed NPZ exceeds the 512 MiB limit.')
    with np.load(io.BytesIO(seg_data),allow_pickle=False) as d:
        if not {'vertices','faces','raw_face_label','raw_label_names'}.issubset(d.files):
            raise ValueError('Paired .seg.npz requires geometry and raw annotation arrays.')
        v,f=d['vertices'],d['faces']
        if len(f)>MAX_FACES:raise ValueError('Mesh exceeds the 2M-face limit.')
        raw=Model(v,f,np.zeros(len(f),np.int32))
    stl=import_mesh(mesh_data,mesh_name,units='m',center=False,weld=False,symmetry='keep')
    if len(stl.faces)!=len(raw.faces):
        raise ValueError('STL / .seg.npz face count mismatch; labels were not attached.')
    tolerance=max(float(np.ptp(raw.vertices,axis=0).max())*2e-7,1e-7)
    maximum=0.
    for start in range(0,len(raw.faces),50000):
        end=start+50000
        delta=np.abs(stl.vertices[stl.faces[start:end]]-raw.vertices[raw.faces[start:end]])
        maximum=max(maximum,float(delta.max()))
        if maximum>tolerance:
            raise ValueError('STL / .seg.npz triangle coordinates or face order mismatch; labels were not attached.')
    return dict(association='stl_triangle_order_verified_against_npz',face_count=len(raw.faces),
                maximum_coordinate_difference_source_units=maximum,tolerance_source_units=tolerance,
                stl_sha256=hashlib.sha256(mesh_data).hexdigest(),
                annotation_sha256=hashlib.sha256(seg_data).hexdigest(),
                geometry_used='seg_npz_contained_geometry',stl_filename=mesh_name)


def import_collection(files:dict[str,bytes],options:ImportOptions,save,*,manifest:dict|None=None) -> dict:
    files={safe_relative(k):v for k,v in files.items()}
    if manifest is None and 'vehicles.json' in files:
        manifest=json.loads(files['vehicles.json'])
    if manifest is not None:
        if manifest.get('schema')!='aeroshape.vehicle-import.v1':raise ValueError('Unsupported vehicle manifest schema.')
        cases=manifest.get('vehicles')
        if not isinstance(cases,list):raise ValueError('Manifest needs a vehicles list.')
    else:
        # A paired annotation archive is a sidecar for ONE vehicle, not a
        # second library record. Standalone .seg.npz is authoritative itself.
        paired={name[:-8]+'.stl':name for name in files if name.lower().endswith('.seg.npz')
                and name[:-8]+'.stl' in files}
        cases=[dict(mesh=name,**({'seg_npz':paired[name]} if name in paired else {}))
               for name in sorted(files) if Path(name).suffix.lower() in ALLOWED
               and name not in paired.values()]
    if not cases:raise ValueError('No supported surface meshes found.')
    if len(cases)>MAX_BATCH:raise ValueError('At most 128 vehicles per import; use multiple local CLI batches.')
    accepted=[];failed=[]
    for case in cases:
        if not isinstance(case,dict):
            failed.append(dict(input=repr(case)[:160],error='Manifest vehicle must be an object.'));continue
        name=case.get('mesh') or case.get('case_id','assembly')
        try:
            if 'meshes' in case:
                items=[]
                for item in case['meshes']:
                    path=safe_relative(item['path']);data=files[path]
                    if item.get('sha256') and item['sha256']!=hashlib.sha256(data).hexdigest():raise ValueError('Assembly member checksum mismatch.')
                    items.append((path,data,item.get('part_key')))
                model=assemble_vehicle(items,options,case)
            else:
                path=safe_relative(case['mesh']);data=files[path]
                if manifest is None and re.search(r'(^|[_/ -])(wheels?|tyres?|tires?|body_only|front_wheels|rear_wheels)([_/ .-]|$)',path,re.I):
                    raise ValueError('Possible component-only file. Use vehicles.json to declare an assembly or an intentional whole-vehicle asset.')
                if case.get('sha256') and case['sha256']!=hashlib.sha256(data).hexdigest():raise ValueError('Mesh checksum mismatch.')
                label_path=case.get('seg_npz') or (case.get('labels') if str(case.get('labels','')).lower().endswith('.seg.npz') else None)
                if label_path:
                    label_path=safe_relative(label_path);seg_data=files[label_path]
                    verification=_verify_seg_pair(data,path,seg_data)
                    model=import_vehicle(seg_data,label_path,options,case=case)
                    model.metadata['source_annotation_pair']=verification
                    model.metadata['provenance']['source_filename']=path
                    model.metadata['provenance']['source_sha256']=verification['stl_sha256']
                else:
                    sidecar=json.loads(files[safe_relative(case['labels'])]) if case.get('labels') else None
                    model=import_vehicle(data,path,options,case=case,sidecar=sidecar)
            result=save(model);accepted.append(dict(input=name,**result))
        except Exception as e:
            # Continue with the remaining entries; never fabricate a replacement vehicle.
            failed.append(dict(input=str(name),error=f'{type(e).__name__}: {e}'))
    return dict(schema='aeroshape.import-report.v1',requested=len(cases),accepted_count=len(accepted),failed_count=len(failed),
                accepted=accepted,failed=failed,cfd_ready=False,remote_data_verified=False)


def directory_files(root:Path,manifest:dict|None=None) -> dict[str,bytes]:
    root=root.resolve();result={};total=0
    candidates=[p for p in root.rglob('*') if p.is_file()]
    if len(candidates)>MAX_MEMBERS:raise ValueError('Too many files: select a smaller directory.')
    for p in sorted(candidates):
        if p.is_symlink() or root not in p.resolve().parents:raise ValueError('Symlinks outside input root are not allowed.')
        if p.suffix.lower() not in ALLOWED|{'.json','.csv'}:continue
        size=p.stat().st_size;total+=size
        if size>MAX_BYTES or total>MAX_EXPANDED:raise ValueError('Batch exceeds memory safety limits; split it into smaller directories.')
        result[p.relative_to(root).as_posix()]=p.read_bytes()
    return result
