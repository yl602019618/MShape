from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np
import trimesh

PARTS = [
    ("unassigned", "未标注", "#8294a6"),
    ("front", "前保险杠 / 车鼻", "#82b7cb"),
    ("hood", "发动机盖", "#a1ced5"),
    ("windshield", "前风挡", "#456273"),
    ("roof", "车顶", "#8ed1c5"),
    ("rear_glass", "后风挡 / 尾部过渡", "#54778b"),
    ("tail", "尾门 / 后保险杠", "#c0ab89"),
    ("side", "侧围 / 车门", "#9ab2cb"),
    ("underbody", "底盘外表面", "#5a7084"),
    ("wheels", "轮胎 / 车轮（默认固定）", "#303d50"),
    ("mirrors", "后视镜", "#d2bca4"),
    ("diffuser", "后底部 / 扩散器区域", "#b393ac"),
]

def default_parts() -> list[dict]:
    return [dict(id=i, key=k, name=n, color=c, locked=k == "wheels")
            for i, (k, n, c) in enumerate(PARTS)]

def array_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vertices, dtype='<f8').tobytes())
    h.update(np.ascontiguousarray(faces, dtype='<i4').tobytes())
    return h.hexdigest()

@dataclass
class Model:
    vertices: np.ndarray
    faces: np.ndarray
    labels: np.ndarray
    parts: list[dict] = field(default_factory=default_parts)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64)
        # Check BEFORE narrowing indices: prevent overflow or truncation of uploaded arrays.
        for name in ('faces', 'labels'):
            a = np.asarray(getattr(self, name))
            if a.dtype.kind not in 'iu' or (a.size and (a.min() < 0 or a.max() > np.iinfo(np.int32).max)):
                raise ValueError(f'{name} must contain nonnegative int32-compatible integers.')
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int32)
        self.labels = np.ascontiguousarray(self.labels, dtype=np.int32)
        v, f = self.vertices, self.faces
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("Expected vertices[N,3] and triangle faces[M,3].")
        if not len(v) or not len(f) or not np.isfinite(v).all():
            raise ValueError("Empty geometry or non-finite coordinates.")
        if f.min() < 0 or f.max() >= len(v) or self.labels.shape != (len(f),):
            raise ValueError("Invalid face indices or face label array.")
        ids = [int(p['id']) for p in self.parts]
        if len(set(ids)) != len(ids) or not set(np.unique(self.labels)).issubset(ids):
            raise ValueError("Part IDs must be unique and cover all labels.")
        if max(np.ptp(v, axis=0)) <= 1e-10:
            raise ValueError("Geometry extent is zero.")
        self.metadata.setdefault('base_geometry_hash', array_hash(v, f))
        self.metadata.setdefault('family_id', self.metadata['base_geometry_hash'][:16])
        self.metadata.setdefault('units', 'm')
        self.metadata.setdefault('coordinates', 'X front-to-rear; Y lateral; Z up; reflection plane Y=0')

    def mesh(self, vertices=None) -> trimesh.Trimesh:
        return trimesh.Trimesh(vertices=self.vertices if vertices is None else vertices,
                               faces=self.faces, process=False)

    def annotation_hash(self) -> str:
        h = hashlib.sha256(self.labels.astype('<i4').tobytes())
        h.update(json.dumps(self.parts, sort_keys=True, ensure_ascii=False).encode())
        constraints={k:self.metadata[k] for k in ('driver_vertices','rigid_followers','seam_ties') if k in self.metadata}
        if constraints:h.update(json.dumps(constraints,sort_keys=True,separators=(',',':')).encode())
        return h.hexdigest()

    def clone(self, **changes) -> 'Model':
        args = dict(vertices=self.vertices.copy(), faces=self.faces.copy(), labels=self.labels.copy(),
                    parts=json.loads(json.dumps(self.parts)), metadata=json.loads(json.dumps(self.metadata)))
        args.update(changes)
        return Model(**args)
