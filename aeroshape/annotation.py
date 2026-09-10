"""Non-destructive face annotation: connected selections, constrained cuts, history.

All operations address original imported triangle IDs. No display LOD is edited.
Label boundaries follow existing triangle edges; vertices are never displaced.
"""
from __future__ import annotations
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
import io
import json
import re
import uuid
import zipfile
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, maximum_flow
from scipy.spatial import cKDTree
from .model import Model, array_hash
from .io import load_npz, npz_bytes
from .semantic import schema_parts, segment30, SegmentationOptions, face_topology, UNKNOWN
from .geometry import face_geometry, _invalidate_bindings_after_relabel

SCHEMA = 'aeroshape.annotation.v1'


def parts():
    rows = schema_parts()
    for p in rows:
        p['locked'] = False
    extra = [('front_intakes', '进气口 / 格栅', '#42696c'),
             ('mirror_glass_r', '右后视镜镜片', '#82cee5'),
             ('mirror_glass_l', '左后视镜镜片', '#a5dff0'),
             ('body_other', '其他车身外板', '#c4aa87'),
             ('wheelhouse', '轮拱内衬', '#7c8299')]
    for i, (key, name, color) in enumerate(extra, 31):
        rows.append(dict(id=i, key=key, name=name, color=color, locked=False))
    rows[32]['pair_id'] = 33
    rows[33]['pair_id'] = 32
    for p in rows:
        if p['key'].startswith('mirror_') and 'glass' not in p['key']:
            p['name'] = p['name'].replace('总成', '外壳 / 支架')
    return rows


def workflow():
    groups = [
        ('机盖', ['hood'], 'top', '沿机盖接缝，避免带入前风挡和翼子板。'),
        ('前风挡', ['windshield'], 'iso', '只选玻璃；A 柱和压边可以留给侧围。'),
        ('车顶', ['roof'], 'top', '确认车顶外板与玻璃的界限。'),
        ('后风挡', ['rear_glass'], 'rear', '两厢车也可以有后风挡；没有则标记不存在。'),
        ('前围', ['front_bumper'], 'front', '进气口和前灯在后续步骤单独提取。'),
        ('后围', ['rear_bumper'], 'back', '区分后保险杠与尾门。'),
        ('尾门 / 后甲板', ['tailgate'], 'rear', '兼容三厢行李箱盖、两厢尾门和 SUV 尾门。'),
        ('前翼子板', ['front_fender_r', 'front_fender_l'], 'side', '贴着轮拱外板标注，内衬可以保留到可选项。'),
        ('前车门', ['front_door_r', 'front_door_l'], 'side', '沿真实门缝选择，左右分别确认。'),
        ('后车门', ['rear_door_r', 'rear_door_l'], 'side', '两门车型可标记不存在，不必强行补齐。'),
        ('后翼子板', ['rear_quarter_r', 'rear_quarter_l'], 'side', '区分后门、后围和轮拱外板。'),
        ('侧窗 / 柱框', ['side_glass_r', 'side_glass_l'], 'side', '沿用平台侧窗与柱框分区；可保留不确定面。'),
        ('前灯', ['front_lights'], 'front', 'STL 没有颜色，接缝不明显时用前景 / 背景辅助。'),
        ('尾灯', ['rear_lights'], 'back', '没有独立灯面时可跳过，避免凭空划出细节。'),
        ('后视镜外壳', ['mirror_r', 'mirror_l'], 'front', '外壳和支架一起选，镜片另标。'),
        ('后视镜镜片', ['mirror_glass_r', 'mirror_glass_l'], 'back', '对应原始 Mirrors_Glass 标签；允许缺失。'),
        ('底板', ['underbody'], 'bottom', '底板外表面；保留轮组和内衬。'),
        ('进气口', ['front_intakes'], 'front', '对应 front_intakes；没有明显边界时允许暂跳过。'),
    ]
    by_key = {p['key']: p['id'] for p in parts()}
    return [dict(index=i+1, name=n, parts=[by_key[k] for k in keys], view=view, hint=hint)
            for i, (n, keys, view, hint) in enumerate(groups)]


def ids_checked(values, count):
    a = np.asarray(values)
    if a.size == 0:
        return np.empty(0, np.int32)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or a.min() < 0 or a.max() >= count:
        raise ValueError('选区包含无效三角面编号，请重新载入当前草稿。')
    return np.unique(a.astype(np.int32))


class SurfaceGraph:
    def __init__(self, model):
        self.model = model
        self.centers, self.normals, self.area = face_geometry(model.vertices, model.faces)
        # STL triangle soups can contain identical positions with different vertex IDs.
        # Weld only the graph's indices at exact coordinates, never modify the model.
        _, remap = np.unique(model.vertices, axis=0, return_inverse=True)
        self.a, self.b, self.islands = face_topology(remap[model.faces])
        dot = np.einsum('ij,ij->i', self.normals[self.a], self.normals[self.b])
        self.angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        self.distance = np.maximum(np.linalg.norm(self.centers[self.a]-self.centers[self.b], axis=1), 1e-9)
        self.weight = np.exp(-(self.angle / 25.)**2)
        self.count = len(model.faces)
        self.tree = None

    def graph(self, edge_values, valid=None):
        keep = np.ones(len(self.a), bool) if valid is None else valid
        a, b, v = self.a[keep], self.b[keep], edge_values[keep]
        return coo_matrix((np.r_[v, v], (np.r_[a, b], np.r_[b, a])),
                          shape=(self.count, self.count)).tocsr()

    def grow(self, seeds, radius=.45, angle=28., blocked=None):
        seeds = ids_checked(seeds, self.count)
        if not len(seeds):
            return seeds
        valid = self.angle <= angle
        if blocked is not None:
            valid &= ~blocked[self.a] & ~blocked[self.b]
            seeds = seeds[~blocked[seeds]]
        if not len(seeds):
            return seeds
        g = self.graph(self.distance, valid)
        dist = dijkstra(g, directed=False, indices=seeds, min_only=True, limit=radius)
        return np.flatnonzero(np.isfinite(dist)).astype(np.int32)

    def mirror(self, ids):
        # Near-exact symmetric triangles only; no blind nearest-side painting.
        if self.tree is None:
            self.tree = cKDTree(self.centers)
        q = self.centers[ids].copy(); q[:, 1] *= -1
        distance, other = self.tree.query(q)
        back = self.centers[other].copy(); back[:, 1] *= -1
        _, reverse = self.tree.query(back)
        tol = float(np.ptp(self.model.vertices, axis=0).max()) * 2e-6
        rn = self.normals[ids].copy(); rn[:, 1] *= -1
        valid = (distance < tol) & (reverse == ids)
        valid &= np.abs(np.einsum('ij,ij->i', rn, self.normals[other])) > .98
        valid &= np.abs(self.area[other]-self.area[ids]) <= np.maximum(self.area[ids]*1e-4, 1e-12)
        return ids[valid], other[valid].astype(np.int32)

    def cut(self, foreground, background, *, radius=.6, strength=2., blocked=None):
        fg = ids_checked(foreground, self.count); bg = ids_checked(background, self.count)
        if not len(fg) or not len(bg):
            raise ValueError('请分别涂至少一处前景（保留）和背景（排除），再运行智能分割。')
        if np.intersect1d(fg, bg).size:
            raise ValueError('同一三角面不能同时是前景和背景。')
        if blocked is not None and np.any(blocked[fg]):
            raise ValueError('前景位于已锁定组件内，请先解锁该组件。')
        roi = self.grow(fg, radius, 90, blocked)
        if len(roi) > 150000:
            raise ValueError('局部图割超过 15 万面，请减小作用范围或先用套索缩小选区。')
        if not len(roi):
            return roi
        in_roi = np.zeros(self.count, bool); in_roi[roi] = True
        bg_local = bg[in_roi[bg]]
        if not len(bg_local):
            raise ValueError('背景涂划不在当前作用范围内，请靠近目标边缘涂背景或增大范围。')
        valid = in_roi[self.a] & in_roi[self.b]
        # Penalized geodesics give unary evidence; narrow creases encourage a cut.
        g = self.graph(self.distance * (1 + 8*(1-self.weight)), valid)
        df = dijkstra(g, indices=fg, min_only=True)[roi]
        db = dijkstra(g, indices=bg_local, min_only=True)[roi]
        df = np.nan_to_num(df, posinf=radius*100)
        db = np.nan_to_num(db, posinf=radius*100)
        denom = np.maximum(df+db, 1e-9)
        cost_fg, cost_bg = df/denom, db/denom
        selected = self._cut_roi(roi, fg, bg_local, cost_fg, cost_bg, strength)
        return selected

    def _cut_roi(self, roi, fg, bg, cost_fg, cost_bg, strength):
        n = len(roi)
        local = np.full(self.count, -1, np.int32); local[roi] = np.arange(n)
        edges = (local[self.a] >= 0) & (local[self.b] >= 0)
        a, b = local[self.a[edges]], local[self.b[edges]]
        weights = np.maximum(1, (self.weight[edges]*strength*1000).astype(np.int64))
        source, sink = n, n+1
        # Source-side nodes are foreground. Hard seeds exceed all finite costs.
        to_fg = np.maximum(1, (cost_fg*1000).astype(np.int64))
        to_bg = np.maximum(1, (cost_bg*1000).astype(np.int64))
        total = int(2*weights.sum()+to_fg.sum()+to_bg.sum())
        # SciPy's flow kernel uses bounded integer capacities. Keep hard terminals
        # above the entire finite energy without overflowing on dense local ROIs.
        scale = max(1, int(np.ceil(total / 100_000_000)))
        weights = np.maximum(1, weights // scale)
        to_fg = to_fg // scale; to_bg = to_bg // scale
        hard = int(2*weights.sum()+to_fg.sum()+to_bg.sum()+1)
        to_bg[local[fg]] = hard; to_fg[local[fg]] = 0
        to_fg[local[bg]] = hard; to_bg[local[bg]] = 0
        rows = np.r_[a, b, np.full(n, source), np.arange(n)]
        cols = np.r_[b, a, np.arange(n), np.full(n, sink)]
        cap = coo_matrix((np.r_[weights, weights, to_bg, to_fg], (rows, cols)), shape=(n+2, n+2)).tocsr()
        flow = maximum_flow(cap, source, sink)
        residual = cap-flow.flow
        residual.data = (residual.data > 0).astype(np.int64); residual.eliminate_zeros()
        from scipy.sparse.csgraph import breadth_first_order
        reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
        result = roi[reachable[reachable < n]]
        if not np.isin(fg, result).all() or np.isin(bg, result).any():
            raise ValueError('图割约束校验失败，未修改选区。请缩小范围重试。')
        return np.sort(result).astype(np.int32)

    def smooth(self, selected, strength=2., iterations=3, blocked=None):
        selected = ids_checked(selected, self.count)
        inside = np.zeros(self.count, bool); inside[selected] = True
        if not inside.any():
            return selected
        boundary = inside[self.a] != inside[self.b]
        band = np.zeros(self.count, bool)
        band[self.a[boundary]] = True; band[self.b[boundary]] = True
        for _ in range(iterations):
            touch = band[self.a] | band[self.b]
            band[self.a[touch]] = True; band[self.b[touch]] = True
        if blocked is not None:
            band &= ~blocked
        roi = np.flatnonzero(band)
        if not len(roi):
            return selected
        if len(roi) > 150000:
            raise ValueError('边界范围过大，请先分组件选择后再平滑。')
        edge_band = band[self.a] != band[self.b]
        anchors = np.unique(np.r_[self.a[edge_band], self.b[edge_band]])
        anchors = anchors[band[anchors]]
        fg = anchors[inside[anchors]]; bg = anchors[~inside[anchors]]
        # Narrow-band Potts cut: data term keeps original mask, interior anchors fixed.
        kept = self._cut_roi(roi, fg, bg, (~inside[roi]).astype(float)*1.5,
                             inside[roi].astype(float)*1.5, strength)
        inside[roi] = False; inside[kept] = True
        if blocked is not None:
            inside[blocked] = False
        return np.flatnonzero(inside).astype(np.int32)


class AnnotationStore:
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock(); self.cache = OrderedDict()

    def folder(self, sid):
        if not re.fullmatch(r'[a-f0-9]{32}', sid):
            raise ValueError('无效标注草稿编号。')
        return self.root/sid

    def load(self, sid):
        folder = self.folder(sid)
        if not (folder/'head.json').exists():
            raise ValueError('标注草稿不存在。')
        head = json.loads((folder/'head.json').read_text())
        with np.load(folder/(head['history'][head['cursor']]+'.npz'), allow_pickle=False) as z:
            snap = dict(labels=z['labels'], manual=z['manual'], reviews=json.loads(str(z['reviews'])))
        return head, snap

    def surface(self, sid):
        if sid not in self.cache:
            self.cache[sid] = SurfaceGraph(load_npz((self.folder(sid)/'geometry.npz').read_bytes()))
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(sid)
        return self.cache[sid]

    def _head(self, sid, head):
        folder = self.folder(sid); temp = folder/'head.tmp'
        temp.write_text(json.dumps(head, ensure_ascii=False, indent=2)); temp.replace(folder/'head.json')

    def save(self, sid, head, snap, action):
        token = uuid.uuid4().hex
        np.savez_compressed(self.folder(sid)/(token+'.npz'), labels=snap['labels'], manual=snap['manual'],
                            reviews=json.dumps(snap['reviews'], ensure_ascii=False))
        head['history'] = head['history'][:head['cursor']+1] + [token]
        head['cursor'] = len(head['history'])-1
        head['revision'] += 1; head['action'] = action
        head['updated'] = datetime.now(timezone.utc).isoformat()
        self._head(sid, head)

    def create(self, model, name, body_type='unknown', propose=True):
        with self.lock:
            sid = uuid.uuid4().hex; folder = self.folder(sid); folder.mkdir()
            (folder/'geometry.npz').write_bytes(npz_bytes(model))
            head = dict(id=sid, name=name[:160], body_type=body_type, revision=0, history=[], cursor=-1,
                        geometry_hash=array_hash(model.vertices, model.faces), faces=len(model.faces), vertices=len(model.vertices))
            labels = self.proposals(model, body_type) if propose else np.full(len(model.faces), UNKNOWN, np.int32)
            if model.metadata.get('annotation_schema') == SCHEMA:
                labels = model.labels.copy()
            snap = dict(labels=labels, manual=np.zeros(len(labels), bool), reviews={})
            if model.metadata.get('annotation_schema') == SCHEMA:
                snap['manual'][ids_checked(model.metadata.get('annotation_manual_faces', []), len(labels))] = True
                snap['reviews'] = {str(k): v for k, v in model.metadata.get('annotation_reviews', {}).items()
                                   if str(k) in {str(p['id']) for p in parts()} and v in ('pending', 'confirmed', 'absent', 'skipped')}
            self.save(sid, head, snap, '自动初分割' if propose else '空白标注')
            return self.info(sid)

    @staticmethod
    def proposals(model, body_type):
        result = segment30(model, SegmentationOptions(body_type=body_type, smooth_iterations=2))
        labels = result.labels.copy()
        source_labels = np.asarray(model.metadata.get('semantic_source_labels', model.labels))
        source_parts = model.metadata.get('semantic_source_parts', model.parts)
        c = None
        for p in source_parts:
            name = p['key'].lower()
            if name in ('front_intakes', 'mirrors_glass', 'mirror_glass_l', 'mirror_glass_r'):
                mask = source_labels == p['id']
                if name == 'front_intakes': labels[mask] = 31
                else:
                    if c is None: c = model.vertices[model.faces].mean(1)
                    labels[mask] = np.where(c[mask, 1] >= 0, 33, 32)
        return labels

    def info(self, sid, inspect=False):
        with self.lock:
            head, snap = self.load(sid); g = self.surface(sid)
            labels = snap['labels']; count = np.bincount(labels, minlength=len(parts()))
            area = np.bincount(labels, weights=g.area, minlength=len(parts()))
            rows = []
            for p in parts():
                row = dict(p, faces=int(count[p['id']]), area_m2=float(area[p['id']]),
                           status=snap['reviews'].get(str(p['id']), 'pending'))
                if inspect and row['faces']:
                    mask = labels == p['id']; valid = mask[g.a] & mask[g.b]
                    graph = g.graph(np.ones(len(g.a)), valid)
                    _, islands = connected_components(graph[np.flatnonzero(mask)][:, np.flatnonzero(mask)], directed=False)
                    sizes = np.bincount(islands); row['islands'] = len(sizes)
                    row['tiny_islands'] = int((sizes < 8).sum())
                rows.append(row)
            boundary = labels[g.a] != labels[g.b]
            return dict(id=sid, name=head['name'], revision=head['revision'], updated=head['updated'],
                        geometry_hash=head['geometry_hash'], face_count=len(labels), vertex_count=head['vertices'],
                        body_type=head['body_type'], parts=rows, action=head['action'],
                        undo=head['cursor'] > 0, redo=head['cursor'] < len(head['history'])-1,
                        coverage=float(1-area[UNKNOWN]/max(area.sum(), 1e-12)),
                        unassigned_faces=int(count[UNKNOWN]), manual_faces=int(snap['manual'].sum()),
                        boundary_edges=int(boundary.sum()), geometry_unchanged=True,
                        bounds=[g.model.vertices.min(0).tolist(), g.model.vertices.max(0).tolist()])

    @staticmethod
    def check_revision(head, revision):
        if head['revision'] != revision:
            raise ValueError('草稿已在其他窗口更新；请重新载入，避免覆盖新标注。')

    @staticmethod
    def blocked(snap, current=None):
        locked = [int(k) for k, status in snap['reviews'].items() if status == 'confirmed' and int(k) != current]
        return np.isin(snap['labels'], locked)

    def selection(self, sid, req):
        with self.lock:
            head, snap = self.load(sid); self.check_revision(head, req.revision)
            g = self.surface(sid); blocked = self.blocked(snap)
            if req.tool == 'wand':
                ids = g.grow(req.seeds, req.radius, req.angle, blocked)
            elif req.tool == 'cut':
                ids = g.cut(req.seeds, req.background, radius=req.radius, strength=req.strength, blocked=blocked)
            elif req.tool == 'smooth':
                ids = g.smooth(req.selected, req.strength, req.rings, blocked)
            else:
                ids = ids_checked(req.selected, g.count)
                mask = np.zeros(g.count, bool); mask[ids] = True
                for _ in range(req.rings):
                    if req.tool == 'expand':
                        edge = mask[g.a] | mask[g.b]; mask[g.a[edge]] = True; mask[g.b[edge]] = True
                    else:
                        edge = mask[g.a] != mask[g.b]; mask[g.a[edge]] = False; mask[g.b[edge]] = False
                    mask[blocked] = False
                ids = np.flatnonzero(mask).astype(np.int32)
            return dict(face_ids=ids.tolist(), count=len(ids), revision=head['revision'])

    def edit(self, sid, req):
        with self.lock:
            head, snap = self.load(sid); self.check_revision(head, req.revision)
            g = self.surface(sid)
            if req.action in ('undo', 'redo'):
                cursor = head['cursor'] + (-1 if req.action == 'undo' else 1)
                if cursor < 0 or cursor >= len(head['history']):
                    raise ValueError('没有可撤销 / 重做的步骤。')
                head['cursor'] = cursor; head['revision'] += 1; head['action'] = req.action
                head['updated'] = datetime.now(timezone.utc).isoformat(); self._head(sid, head)
                return self.info(sid)
            if req.action == 'assign':
                if req.part not in [p['id'] for p in parts()]: raise ValueError('未知组件。')
                if snap['reviews'].get(str(req.part)) == 'confirmed': raise ValueError('当前组件已锁定，请先解锁再编辑。')
                ids = ids_checked(req.faces, g.count)
                ids = ids[~self.blocked(snap)[ids]]
                if req.only_unassigned: ids = ids[snap['labels'][ids] == UNKNOWN]
                if req.source_part is not None: ids = ids[snap['labels'][ids] == req.source_part]
                if req.replace_part and (req.part == UNKNOWN or req.only_unassigned or req.source_part is not None or req.mirror):
                    raise ValueError('修整标签需明确替换当前组件，不能同时使用镜像、来源过滤或仅填未标注。')
                prior = snap['labels'].copy()
                if req.replace_part:
                    retain = np.zeros(g.count, bool); retain[ids] = True
                    removed = (prior == req.part) & ~retain
                    snap['labels'][removed] = UNKNOWN; snap['manual'][removed] = True
                snap['labels'][ids] = req.part; snap['manual'][ids] = True
                if req.mirror and len(ids):
                    _, other = g.mirror(ids)
                    pair = next(p for p in parts() if p['id'] == req.part).get('pair_id', req.part)
                    if snap['reviews'].get(str(pair)) == 'confirmed': other = other[:0]
                    other = other[~self.blocked(snap)[other]]
                    if req.only_unassigned: other = other[prior[other] == UNKNOWN]
                    # Two-sided selection is explicit; do not overwrite it with reflection.
                    other = other[~np.isin(other, ids)]
                    snap['labels'][other] = pair; snap['manual'][other] = True
                affected = np.unique(np.r_[prior[prior != snap['labels']], snap['labels'][prior != snap['labels']]])
                for pid in affected: snap['reviews'].pop(str(pid), None)
                if not len(ids) and not np.any(prior != snap['labels']):
                    # An empty/fully protected stroke must not consume an undo step
                    # or discard an existing redo branch.
                    return dict(self.info(sid), changed_faces=0, no_change=True)
            elif req.action == 'review':
                if req.part == UNKNOWN: raise ValueError('未标注区域不能确认为组件。')
                exists = np.any(snap['labels'] == req.part)
                if req.status == 'confirmed' and not exists: raise ValueError('空组件不能确认，请标记不存在或暂跳过。')
                if req.status == 'absent' and exists: raise ValueError('组件仍有标注面，不能标记不存在；请先改标签或暂跳过。')
                snap['reviews'][str(req.part)] = req.status
            elif req.action == 'propose':
                proposed = self.proposals(g.model, head['body_type'])
                keep = snap['manual'] | self.blocked(snap)
                snap['labels'][~keep] = proposed[~keep]
                # All automatic proposals remain unreviewed.
                for key, status in list(snap['reviews'].items()):
                    if status != 'confirmed': snap['reviews'].pop(key)
            self.save(sid, head, snap, req.action)
            return self.info(sid)

    def export_model(self, sid):
        with self.lock:
            head, snap = self.load(sid); g = self.surface(sid)
            model = g.model.clone(labels=snap['labels'], parts=parts())
            _invalidate_bindings_after_relabel(model)
            for key in ('semantic_schema', 'segmentation', 'design_sets', 'generator_parameters', 'reference_fit'):
                model.metadata.pop(key, None)
            if 'raw_face_labels' in model.metadata:
                model.metadata.setdefault('annotation_import_raw_face_labels', model.metadata['raw_face_labels'])
                model.metadata.setdefault('annotation_import_raw_label_names', model.metadata.get('raw_label_names', []))
            model.metadata.update(annotation_schema=SCHEMA, annotation_status='partial_manual_annotation',
                                  annotation_session=sid, annotation_revision=head['revision'],
                                  annotation_reviews=snap['reviews'], annotation_geometry_hash=head['geometry_hash'],
                                  annotation_manual_faces=np.flatnonzero(snap['manual']).tolist(), name=head['name'],
                                  # New segmentation/fitting must consume these reviewed source labels.
                                  semantic_source_labels=snap['labels'].tolist(), semantic_source_parts=parts(),
                                  raw_face_labels=snap['labels'].tolist(), raw_label_names=[p['key'] for p in parts()])
            assert array_hash(model.vertices, model.faces) == head['geometry_hash']
            return model

    def export(self, sid):
        with self.lock:
            model = self.export_model(sid); info = self.info(sid, inspect=True)
            output = io.BytesIO()
            with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
                z.writestr('annotated.npz', npz_bytes(model))
                data = io.BytesIO()
                np.savez_compressed(data, face_labels=model.labels,
                                    source_face_indices=np.asarray(model.metadata.get('source_face_indices', np.arange(len(model.faces))), np.int32),
                                    geometry_hash=info['geometry_hash'], part_keys=np.asarray([p['key'] for p in parts()]))
                z.writestr('face-labels.npz', data.getvalue())
                z.writestr('annotation.json', json.dumps(dict(schema=SCHEMA, workflow=workflow(), **info), ensure_ascii=False, indent=2))
                for p in parts():
                    mask = model.labels == p['id']
                    if mask.any(): z.writestr('components/'+p['key']+'.stl', model.mesh().submesh([np.flatnonzero(mask)], append=True, repair=False).export(file_type='stl'))
                z.writestr('README.txt', 'Partial vehicle annotation. Master vertices and triangle order are unchanged after import.\n'
                           'annotated.npz is the authoritative mesh + labels. STL does not store labels.\n'
                           'Unassigned faces are retained; pending/skipped components are not reviewed.\n'
                           'Label boundaries follow existing triangle edges; surface parts need not be closed solids.\n')
            return output.getvalue()
