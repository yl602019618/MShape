"""Local, persistent vehicle catalog: immutable content-addressed references.

SQLite indexes summaries; compressed NPZ and thumbnails live in a private
application directory. No arbitrary server filesystem paths are accepted by
HTTP routes. Annotations and mesh validity are separate review dimensions.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import uuid
import numpy as np
from PIL import Image, ImageDraw
from .io import npz_bytes, load_npz
from .model import Model, array_hash
from .quality import baseline_report
from .semantic import SCHEMA, UNKNOWN, refresh_summary


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def thumbnail_png(model:Model,eye_vector=(-1.6,-2.8,1.3)) -> bytes:
    """Orthographic rendering of actual triangles, no invented vehicle photos."""
    from .geometry import preview_indices
    # A coherent clustered surface covers the silhouette; skipping triangles in
    # a dense source produces visible holes. Only this display copy is reduced.
    vertex_ids,faces,source_faces=preview_indices(model,target=90000)
    v=model.vertices[vertex_ids].copy();v-=(v.min(0)+v.max(0))/2
    labels=model.labels[source_faces]
    eye=np.array(eye_vector,dtype=float);eye/=np.linalg.norm(eye)
    right=np.cross(np.array([0.,0,1.]),eye);right/=np.linalg.norm(right);up=np.cross(eye,right)
    xy=np.c_[v@right,v@up];scale=min(338/max(np.ptp(xy[:,0]),1e-9),170/max(np.ptp(xy[:,1]),1e-9))
    xy*=scale;xy[:,0]+=190;xy[:,1]=111-xy[:,1]
    t=v[faces];cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);norm=np.linalg.norm(cross,axis=1)
    normals=cross/np.maximum(norm[:,None],1e-30)
    # Source CFD surfaces may be inward-wound. Draw both orientations in
    # far-to-near order and face the shading normal toward the camera.
    normals*=np.where(normals@eye>=0,1.,-1.)[:,None]
    ids=np.argsort(t.mean(1)@eye)
    supersample=2
    img=Image.new('RGB',(380*supersample,220*supersample),(245,247,249));draw=ImageDraw.Draw(img)
    draw.line(tuple(x*supersample for x in (24,196,356,196)),fill=(223,228,232),width=supersample)
    palette={}
    for p in model.parts:
        color=p.get('color','#8babbf')
        if not re.fullmatch(r'#[a-fA-F0-9]{6}',color):color='#8babbf'
        palette[p['id']]=np.array([int(color[k:k+2],16) for k in (1,3,5)])
    light=np.array([-.3,-.4,1]);light/=np.linalg.norm(light)
    shade=.50+.50*np.maximum(normals@light,0)
    coords=np.rint(xy[faces]*supersample).astype(int)
    materials=model.metadata.get('material_palette',[]);fm=model.metadata.get('face_material_ids',[])
    fm=np.asarray(fm)[source_faces] if len(fm)==len(model.faces) else []
    for i in ids:
        base=palette[int(labels[i])]
        if len(fm)==len(faces) and 0<=fm[i]<len(materials):
            col=materials[fm[i]].get('base_color','#8babbf')
            if re.fullmatch(r'#[a-fA-F0-9]{6}',col):base=np.array([int(col[k:k+2],16) for k in (1,3,5)])
        color=tuple(np.clip(base*shade[i],0,255).astype(int))
        draw.polygon([tuple(x) for x in coords[i]],fill=color)
    img=img.resize((380,220),Image.Resampling.LANCZOS)
    b=io.BytesIO();img.save(b,format='PNG');return b.getvalue()


class VehicleLibrary:
    def __init__(self,root:Path):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'objects').mkdir(exist_ok=True);(self.root/'thumbnails').mkdir(exist_ok=True)
        self.db=self.root/'catalog.sqlite3'
        with self.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS vehicles(
                id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL, blob TEXT NOT NULL,
                name TEXT NOT NULL, source TEXT NOT NULL, body_type TEXT NOT NULL,
                family_id TEXT NOT NULL, review_status TEXT NOT NULL, created TEXT NOT NULL,
                summary TEXT NOT NULL)''')
            conn.execute('CREATE INDEX IF NOT EXISTS filter_idx ON vehicles(source,body_type,review_status)')
            conn.execute('CREATE TABLE IF NOT EXISTS archived_vehicles(id TEXT PRIMARY KEY,reason TEXT NOT NULL)')
            # Preserve historical v0.7 objects but remove the retired source-mesh
            # deformation results from the active parametric vehicle library.
            conn.execute("INSERT OR IGNORE INTO archived_vehicles SELECT id,'retired_reference_mesh_deformation_v08' FROM vehicles WHERE source='generated' AND family_id LIKE 'reference-calibrated-%'")
            # Only known original-demo family IDs are retired, never name-based user assets.
            conn.execute("INSERT OR IGNORE INTO archived_vehicles SELECT id,'retired_internal_fixture_v06' FROM vehicles WHERE source='original_demo' AND family_id IN ('original-demo-fastback','original-demo-notchback','original-demo-estateback')")
    def connect(self):
        conn=sqlite3.connect(self.db,timeout=30);conn.row_factory=sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL');return conn
    def _id(self,vid):
        if not isinstance(vid,str) or not re.fullmatch(r'[a-f0-9]{32}',vid):raise ValueError('Invalid library vehicle ID.')
        return vid
    def add(self,model:Model,*,name:str|None=None,parent_id:str|None=None) -> dict:
        m=model.clone()
        for key in ('library_id','library_fingerprint'):m.metadata.pop(key,None)
        if name:
            if not 0<len(name)<=160:raise ValueError('Vehicle name must be 1–160 characters.')
            m.metadata['name']=name
        if parent_id:
            self.get(parent_id);m.metadata['library_parent_id']=parent_id
        # Force CFD readiness false; imported claims cannot certify our geometry.
        m.metadata['cfd_ready']=False
        geometry_hash=array_hash(m.vertices,m.faces);annotation_hash=m.annotation_hash()
        review=m.metadata.get('segmentation',{}).get('review',{})
        if review.get('status') in ('reviewed','reviewed_with_unassigned') and (review.get('geometry_hash')!=geometry_hash or review.get('annotation_hash')!=annotation_hash):
            m.metadata['segmentation']['invalidated_review']=review
            m.metadata['segmentation']['review']=dict(status='pending',reviewer=None,reason='review_hashes_no_longer_match')
        finger_metadata={k:v for k,v in m.metadata.items() if k!='library_parent_id'}
        finger=hashlib.sha256((geometry_hash+annotation_hash+_json(finger_metadata)).encode()).hexdigest()
        with self.connect() as conn:
            existing=conn.execute('SELECT summary FROM vehicles WHERE fingerprint=?',(finger,)).fetchone()
            if existing:return {**json.loads(existing['summary']),'deduplicated':True}
        baseline=baseline_report(m);source=m.metadata.get('provenance',{}).get('source_type','user')
        if source not in ('drivaernet','user','original_demo','generated'):source='user'
        seg=m.metadata.get('segmentation',{})
        review=seg.get('review',{}).get('status','pending')
        status='labels_reviewed' if review=='reviewed' else 'labels_reviewed_with_unassigned' if review=='reviewed_with_unassigned' else 'needs_review'
        vid=uuid.uuid4().hex;created=datetime.now(timezone.utc).isoformat()
        is_annotation=m.metadata.get('annotation_schema')=='aeroshape.annotation.v1'
        has_unknown=m.metadata.get('semantic_schema')==SCHEMA or is_annotation
        dim=baseline['dimensions_m'];parts=int(sum(np.any(m.labels==p['id']) for p in m.parts if p['id']!=UNKNOWN or not has_unknown))
        record=dict(id=vid,name=m.metadata.get('name','Imported vehicle'),source=source,
            body_type=m.metadata.get('body_type','unknown'),family_id=m.metadata.get('family_id','unknown'),
            schema=m.metadata.get('annotation_schema') if is_annotation else m.metadata.get('semantic_schema'),present_regions=parts,target_regions=35 if is_annotation else 30 if m.metadata.get('semantic_schema')==SCHEMA else None,
            review_status=status,unknown_faces=int((m.labels==UNKNOWN).sum()) if has_unknown else None,
            low_score_faces=seg.get('low_score_faces'),geometry_hash=geometry_hash,annotation_hash=annotation_hash,
            fingerprint=finger,faces=len(m.faces),vertices=len(m.vertices),dimensions_m=dim,
            geometry_status='edge_checks_passed' if not baseline['boundary_edges'] and not baseline['nonmanifold_edges'] and not baseline['degenerate_faces'] else 'issues_found',
            boundary_edges=baseline['boundary_edges'],nonmanifold_edges=baseline['nonmanifold_edges'],degenerate_faces=baseline['degenerate_faces'],
            asset_license=m.metadata.get('asset_license','user_must_verify'),dataset_version=m.metadata.get('provenance',{}).get('dataset_version'),
            attribution=m.metadata.get('provenance',{}).get('attribution',''),
            origin_verification=m.metadata.get('provenance',{}).get('dataset_origin_verification','not_verified'),
            symmetry=seg.get('symmetry'),thumbnail_url=f'/api/library/vehicles/{vid}/thumbnail',
            created=created,parent_id=parent_id,cfd_ready=False,
            physics_labels_status='not_attached_to_editable_geometry',generator_parameters=m.metadata.get('generator_parameters'),component_choices=m.metadata.get('component_choices'),generator_measurements=m.metadata.get('generator_measurements'))
        blob=npz_bytes(m);blob_sha=hashlib.sha256(blob).hexdigest();path=self.root/'objects'/(blob_sha+'.npz')
        temp=path.with_name(uuid.uuid4().hex+'.tmp');temp.write_bytes(blob);temp.replace(path)
        thumb=self.root/'thumbnails'/(vid+'.png');tmp=thumb.with_suffix('.tmp');tmp.write_bytes(thumbnail_png(m));tmp.replace(thumb)
        try:
            with self.connect() as conn:
                conn.execute('INSERT INTO vehicles VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (vid,finger,blob_sha,record['name'],source,record['body_type'],record['family_id'],status,created,_json(record)))
        except sqlite3.IntegrityError:
            with self.connect() as conn:
                existing=conn.execute('SELECT summary FROM vehicles WHERE fingerprint=?',(finger,)).fetchone()
            if existing:
                thumb.unlink(missing_ok=True);return {**json.loads(existing['summary']),'deduplicated':True}
            raise
        if source=='original_demo' and record['family_id'] in ('original-demo-fastback','original-demo-notchback','original-demo-estateback'):
            with self.connect() as conn:conn.execute('INSERT OR IGNORE INTO archived_vehicles VALUES(?,?)',(vid,'retired_internal_fixture_v06'))
        return {**record,'deduplicated':False}
    def list(self,*,query='',body_type='',source='',review_status='',offset=0,limit=24):
        if offset<0 or not 1<=limit<=128:raise ValueError('Invalid pagination.')
        clauses=['id NOT IN (SELECT id FROM archived_vehicles)'];params=[]
        for field,value in [('body_type',body_type),('source',source),('review_status',review_status)]:
            if value:clauses.append(field+'=?');params.append(value)
        if query:
            # LIKE wildcard characters are escaped; user input is not SQL.
            term='%'+query.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%'
            clauses.append("(name LIKE ? ESCAPE '\\' OR family_id LIKE ? ESCAPE '\\')");params.extend([term,term])
        where=(' WHERE '+' AND '.join(clauses)) if clauses else ''
        with self.connect() as conn:
            total=conn.execute('SELECT count(*) FROM vehicles'+where,params).fetchone()[0]
            rows=conn.execute('SELECT summary FROM vehicles'+where+' ORDER BY created DESC,id LIMIT ? OFFSET ?',params+[limit,offset]).fetchall()
            facets={field:[dict(value=r[0],count=r[1]) for r in conn.execute(f'SELECT {field},count(*) FROM vehicles WHERE id NOT IN (SELECT id FROM archived_vehicles) GROUP BY {field}')] for field in ('body_type','source','review_status')}
        return dict(items=[json.loads(r['summary']) for r in rows],total=total,offset=offset,limit=limit,
                    next_offset=offset+limit if offset+limit<total else None,facets=facets)
    def get(self,vid):
        with self.connect() as conn:row=conn.execute('SELECT summary FROM vehicles WHERE id=?',(self._id(vid),)).fetchone()
        if row is None:raise ValueError('Vehicle not found in library.')
        return json.loads(row['summary'])
    def load(self,vid) -> Model:
        with self.connect() as conn:row=conn.execute('SELECT blob,fingerprint FROM vehicles WHERE id=?',(self._id(vid),)).fetchone()
        if row is None:raise ValueError('Vehicle not found in library.')
        path=self.root/'objects'/(row['blob']+'.npz');data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['blob']:raise ValueError('Stored library object checksum mismatch.')
        m=load_npz(data);m.metadata['library_id']=vid;m.metadata['library_fingerprint']=row['fingerprint'];return m
    def thumbnail(self,vid):
        self.get(vid);return (self.root/'thumbnails'/(self._id(vid)+'.png')).read_bytes()
    def manifest(self):
        with self.connect() as conn:rows=conn.execute('SELECT summary FROM vehicles ORDER BY id').fetchall()
        return dict(schema='aeroshape.vehicle-library.v1',vehicles=[json.loads(r[0]) for r in rows],cfd_ready=False)


def review_labels(model:Model,*,geometry_hash:str,annotation_hash:str,reviewer:str,notes:str,
                  allow_unassigned:bool=False) -> Model:
    if geometry_hash!=array_hash(model.vertices,model.faces) or annotation_hash!=model.annotation_hash():
        raise ValueError('Review is stale: geometry or annotation hash changed.')
    if model.metadata.get('semantic_schema')!=SCHEMA:raise ValueError('Run canonical segmentation before label review.')
    if not reviewer.strip() or not notes.strip():raise ValueError('Reviewer and review notes are required.')
    unknown=int((model.labels==UNKNOWN).sum())
    if unknown and not allow_unassigned:raise ValueError('Unassigned faces remain; correct them or explicitly acknowledge their exclusion.')
    m=model.clone();refresh_summary(m)
    m.metadata['segmentation']['review']=dict(status='reviewed_with_unassigned' if unknown else 'reviewed',
        reviewer=reviewer.strip(),notes=notes.strip(),geometry_hash=geometry_hash,annotation_hash=annotation_hash,
        reviewed_at=datetime.now(timezone.utc).isoformat(),scope='semantic_labels_only_not_mesh_or_CFD_validation')
    m.metadata['annotation_status']='human_reviewed_design30';return m
