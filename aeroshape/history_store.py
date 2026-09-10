"""Server-side immutable design snapshots; parent is the checked-out head."""
from pathlib import Path
import hashlib,json,uuid,sqlite3
from datetime import datetime,timezone
from .model import array_hash
from .io import npz_bytes,load_npz

class HistoryStore:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);(self.root/'objects').mkdir(exist_ok=True)
        self.db=self.root/'history.sqlite3'
        with sqlite3.connect(self.db) as c:c.execute('CREATE TABLE IF NOT EXISTS versions(id TEXT PRIMARY KEY,project TEXT NOT NULL,record TEXT NOT NULL)')
    def project(self,m):return hashlib.sha256((array_hash(m.vertices,m.faces)+m.annotation_hash()).encode()).hexdigest()
    def _put(self,data):
        key=hashlib.sha256(data).hexdigest();p=self.root/'objects'/f'{key}.npz'
        if not p.exists():
            temp=p.with_name(uuid.uuid4().hex+'.tmp');temp.write_bytes(data);temp.replace(p)
        return key
    def list(self,m):
        with sqlite3.connect(self.db) as c:rows=c.execute('SELECT record FROM versions WHERE project=? ORDER BY rowid',(self.project(m),)).fetchall()
        return [json.loads(row[0]) for row in rows]
    def get(self,vid):
        with sqlite3.connect(self.db) as c:r=c.execute('SELECT record FROM versions WHERE id=?',(vid,)).fetchone()
        if r is None:raise ValueError('Unknown history revision.')
        return json.loads(r[0])
    def commit(self,m,vertices,design,quality,name,parent=None,kind='design'):
        project=self.project(m)
        if parent and self.get(parent)['project']!=project:raise ValueError('History parent belongs to a different geometry/annotation reference.')
        rid=self._put(npz_bytes(m));d=m.clone(vertices=vertices);d.metadata['base_geometry_hash']=array_hash(vertices,m.faces)
        did=self._put(npz_bytes(d))
        record=dict(id=uuid.uuid4().hex,project=project,parent=parent,name=name or '设计版本',created_at=datetime.now(timezone.utc).isoformat(),
            base_geometry_hash=array_hash(m.vertices,m.faces),annotation_hash=m.annotation_hash(),
            result_geometry_hash=array_hash(vertices,m.faces),reference_object=rid,result_object=did,
            design=design,quality=quality,generator_parameters=m.metadata.get('generator_parameters'),
            kind=kind,immutable=True)
        with sqlite3.connect(self.db) as c:c.execute('INSERT INTO versions VALUES(?,?,?)',(record['id'],project,json.dumps(record,ensure_ascii=False,allow_nan=False)))
        return record
    def bytes(self,key):
        if not isinstance(key,str) or len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Invalid object hash.')
        data=(self.root/'objects'/f'{key}.npz').read_bytes()
        if hashlib.sha256(data).hexdigest()!=key:raise ValueError('History object checksum mismatch.')
        return data
