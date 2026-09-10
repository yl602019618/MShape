/* MiShape studio renderer. The public geometry remains in metres, X nose -X/Z up.
 * GPU buffers and corner smoothing islands persist through live deformations.
 * No renderer dependency, texture download or external environment map. */
import {Viewer} from './renderer.js';

const color = hex => {
  const text = /^#[0-9a-f]{6}$/i.test(hex || '') ? hex : '#81958e';
  return [parseInt(text.slice(1, 3), 16) / 255, parseInt(text.slice(3, 5), 16) / 255, parseInt(text.slice(5, 7), 16) / 255];
};
const normalise = value => {
  const length = Math.hypot(value[0], value[1], value[2]) || 1;
  return [value[0] / length, value[1] / length, value[2] / length];
};
const meshBounds = values => {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < values.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const value = values[i + k];
      if (value < lo[k]) lo[k] = value;
      if (value > hi[k]) hi[k] = value;
    }
  }
  return [lo, hi];
};

export class StudioViewer extends Viewer {
  constructor(host) {
    super(host);
    this.backgroundColor = color('#f2f4ee');
    this.gridColor = color('#e3e7de');
    this.wireColor = color('#546b61');
    this.cageLineColor = color('#688761');
    this.cagePointColor = color('#b59b57');
    this.yaw = -2.2;
    this.pitch = 1.32;
    this.fov = .65;
    this.creaseAngle = 42;
    this.wireMode = 'triangle-edges';
    this.performance = {topologyMs: 0, updateMs: 0, drawMs: 0};
    this.draw();
  }

  makeGrid() {
    const positions = [];
    for (let x = -6; x <= 6; x += .5) {
      positions.push(x, -6, -.024, x, 6, -.024, -6, x, -.024, 6, x, -.024);
    }
    return positions;
  }

  fit() {
    if (!this.vertices?.length) return;
    const [lo, hi] = meshBounds(this.vertices);
    this.target = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    const aspect = Math.max(.2, this.host.clientWidth / Math.max(1, this.host.clientHeight));
    this.radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * 1.35 * Math.max(1, 1.16 / aspect);
    this.yaw = -2.2;
    this.pitch = 1.32;
    this.draw();
  }

  setModel(model) {
    if (!model?.vertices?.length || !model?.faces?.length) throw Error('Vehicle geometry is empty.');
    const faceCount = model.faces.length / 3;
    if (model.vertices.length % 3 || !Number.isInteger(faceCount)) throw Error('Expected flat XYZ vertices and triangle indices.');
    if (model.face_labels?.length !== faceCount) throw Error('Face labels must match triangle count.');
    this._releaseTopology();
    // Keep the original imported materials and source IDs; defaults are renderer-local.
    const prepared = {...model, metadata: model.metadata || {}, parts: model.parts || [],
      face_materials: model.face_materials || model.face_material_ids,
      source_face_ids: model.source_face_ids || Uint32Array.from({length: faceCount}, (_, i) => i)};
    this.scalar = null;
    this._cageData = null;
    this.cage = null;
    super.setModel(prepared);
  }

  setVertices(values) {
    if (!this.vertices || values.length !== this.vertices.length) throw Error('Deformation must preserve the current vertex count.');
    const start = performance.now();
    if (values !== this.vertices) this.vertices.set(values);
    this.buildGroups();
    this.draw();
    this.performance.updateMs = performance.now() - start;
  }

  _releaseTopology() {
    if (!this.gl) return;
    for (const group of this.groups || []) {
      if (group.buffer) this.gl.deleteBuffer(group.buffer);
      if (group.edgeBuffer) this.gl.deleteBuffer(group.edgeBuffer);
    }
    this.groups = [];
    this._facesIdentity = null;
    this._cornerSmooth = null;
  }

  _prepareTopology() {
    const started = performance.now(), gl = this.gl, faces = this.faces;
    const cornerCount = faces.length, faceCount = cornerCount / 3, vertexCount = this.vertices.length / 3;
    const reference = this.reference, labels = this.labels, groups = new Map(), parts = new Map(this.model.parts.map(p => [p.id, p]));
    const faceGroup = new Uint32Array(faceCount), cornerOffset = new Uint32Array(cornerCount), partCenters = new Map();
    const faceNormals = new Float32Array(cornerCount), parent = new Uint32Array(cornerCount);
    const [lo, hi] = meshBounds(reference);
    this.center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    for (let i = 0; i < cornerCount; i++) parent[i] = i;
    for (let fi = 0; fi < faceCount; fi++) {
      const id = labels[fi], materialID = this.materialIDs?.[fi] ?? -1, key = `${id}:${materialID}`;
      let group = groups.get(key);
      if (!group) {
        const part = parts.get(id) || {id, color: '#81958e'};
        group = {id, materialID, groupIndex: groups.size, faceIDs: [], wireCorners: [], color: color(part.color),
          material: this.model.metadata.material_palette?.[materialID] || this.model.materials?.[materialID] || part.material || {base_color: part.color || '#81958e', roughness: .42, metallic: .30},
          direction: part.explode_direction};
        if (!group.material.base_color) group.material = {...group.material, base_color: part.color || '#81958e'};
        groups.set(key, group);
      }
      faceGroup[fi] = group.groupIndex;
      const localCorner = group.faceIDs.length * 3;
      group.faceIDs.push(fi);
      const ca = fi * 3, ia = faces[ca] * 3, ib = faces[ca + 1] * 3, ic = faces[ca + 2] * 3;
      cornerOffset[ca] = localCorner * 7;
      cornerOffset[ca + 1] = (localCorner + 1) * 7;
      cornerOffset[ca + 2] = (localCorner + 2) * 7;
      const ux = reference[ib] - reference[ia], uy = reference[ib + 1] - reference[ia + 1], uz = reference[ib + 2] - reference[ia + 2];
      const vx = reference[ic] - reference[ia], vy = reference[ic + 1] - reference[ia + 1], vz = reference[ic + 2] - reference[ia + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, length = Math.hypot(nx, ny, nz) || 1;
      faceNormals[ca] = nx / length; faceNormals[ca + 1] = ny / length; faceNormals[ca + 2] = nz / length;
      let center = partCenters.get(id);
      if (!center) {center = [0, 0, 0, 0]; partCenters.set(id, center);}
      center[0] += reference[ia] + reference[ib] + reference[ic];
      center[1] += reference[ia + 1] + reference[ib + 1] + reference[ic + 1];
      center[2] += reference[ia + 2] + reference[ib + 2] + reference[ic + 2]; center[3] += 3;
    }
    this.groups = [...groups.values()];
    const root = value => {
      while (parent[value] !== value) {parent[value] = parent[parent[value]]; value = parent[value];}
      return value;
    };
    const join = (a, b) => {const ra = root(a), rb = root(b); if (ra !== rb) parent[rb] = ra;};
    const edges = new Map(), threshold = Math.cos((this.creaseAngle || 42) * Math.PI / 180);
    for (let fi = 0; fi < faceCount; fi++) {
      const offset = fi * 3, group = this.groups[faceGroup[fi]];
      for (let j = 0; j < 3; j++) {
        const ca = offset + j, cb = offset + (j + 1) % 3, a = faces[ca], b = faces[cb];
        const key = Math.min(a, b) * vertexCount + Math.max(a, b), found = edges.get(key);
        if (found === undefined) {edges.set(key, ca); group.wireCorners.push(ca, cb); continue;}
        const first = found >= 0 ? found : ~found, neighbourFace = Math.floor(first / 3), neighbourOffset = neighbourFace * 3;
        if (faceGroup[neighbourFace] !== faceGroup[fi]) group.wireCorners.push(ca, cb);
        if (found < 0) continue;
        edges.set(key, ~first);
        const cosine = faceNormals[offset] * faceNormals[neighbourOffset] + faceNormals[offset + 1] * faceNormals[neighbourOffset + 1] + faceNormals[offset + 2] * faceNormals[neighbourOffset + 2];
        if (cosine < threshold) continue;
        const next = neighbourOffset + (first % 3 + 1) % 3;
        if (faces[first] === a) {join(ca, first); join(cb, next);} else {join(ca, next); join(cb, first);}
      }
    }
    // Hard geometric edges retain separate normal islands even at shared vertices.
    const smooth = new Uint32Array(cornerCount), compact = new Int32Array(cornerCount).fill(-1);
    let smoothCount = 0;
    for (let corner = 0; corner < cornerCount; corner++) {
      const value = root(corner);
      if (compact[value] < 0) compact[value] = smoothCount++;
      smooth[corner] = compact[value] * 3;
    }
    this._cornerSmooth = smooth;
    this._normalSums = new Float64Array(smoothCount * 3);
    this._vertexShifts = new Float32Array(vertexCount);
    this._cornerOffset = cornerOffset;
    for (const group of this.groups) {
      group.faceIDs = Uint32Array.from(group.faceIDs);
      group.wireCorners = Uint32Array.from(group.wireCorners);
      group.count = group.faceIDs.length * 3;
      group.edgeCount = group.wireCorners.length;
      group.data = new Float32Array(group.count * 7);
      group.buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, group.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, group.data.byteLength, gl.DYNAMIC_DRAW);
      const center = partCenters.get(group.id);
      group.centroid = [center[0] / center[3], center[1] / center[3], center[2] / center[3]];
      const direction = [group.centroid[0] - this.center[0], group.centroid[1] - this.center[1], (group.centroid[2] - this.center[2]) * 1.5];
      group.direction = group.direction || (Math.hypot(...direction) < .05 ? [0, 0, 1] : normalise(direction));
    }
    this._facesIdentity = faces;
    this.performance ||= {};
    this.performance.topologyMs = performance.now() - started;
  }

  buildGroups() {
    if (!this.faces || !this.vertices) return;
    if (this._facesIdentity !== this.faces) this._prepareTopology();
    const vertices = this.vertices, faces = this.faces, sums = this._normalSums, smooth = this._cornerSmooth;
    sums.fill(0);
    for (let fi = 0; fi < faces.length; fi += 3) {
      const a = faces[fi] * 3, b = faces[fi + 1] * 3, c = faces[fi + 2] * 3;
      const ux = vertices[b] - vertices[a], uy = vertices[b + 1] - vertices[a + 1], uz = vertices[b + 2] - vertices[a + 2];
      const vx = vertices[c] - vertices[a], vy = vertices[c + 1] - vertices[a + 1], vz = vertices[c + 2] - vertices[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      for (let j = 0; j < 3; j++) {
        const target = smooth[fi + j]; sums[target] += nx; sums[target + 1] += ny; sums[target + 2] += nz;
      }
    }
    for (let i = 0; i < sums.length; i += 3) {
      const length = Math.hypot(sums[i], sums[i + 1], sums[i + 2]) || 1;
      sums[i] /= length; sums[i + 1] /= length; sums[i + 2] /= length;
    }
    const shifts = this._vertexShifts, reference = this.reference;
    let maxShift = 0;
    for (let i = 0, vertex = 0; i < vertices.length; i += 3, vertex++) {
      const shift = Math.hypot(vertices[i] - reference[i], vertices[i + 1] - reference[i + 1], vertices[i + 2] - reference[i + 2]);
      shifts[vertex] = shift; if (shift > maxShift) maxShift = shift;
    }
    this.maxShift = maxShift;
    const field = this.scalar || (this.mask ? this.influence : null), review = this.reviewMode && this.reviewScores;
    for (const group of this.groups) {
      const data = group.data, faceIDs = group.faceIDs;
      let out = 0;
      for (let f = 0; f < faceIDs.length; f++) {
        const fi = faceIDs[f], start = fi * 3;
        for (let j = 0; j < 3; j++) {
          const corner = start + j, vertex = faces[corner], position = vertex * 3, n = smooth[corner];
          data[out++] = vertices[position]; data[out++] = vertices[position + 1]; data[out++] = vertices[position + 2];
          data[out++] = sums[n]; data[out++] = sums[n + 1]; data[out++] = sums[n + 2];
          data[out++] = field ? field[vertex] : review ? review[fi] : shifts[vertex];
        }
      }
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, group.buffer);
      this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, data);
      group.edgeDirty = true;
    }
  }

  _updateWire() {
    for (const group of this.groups) {
      if (!group.edgeDirty) continue;
      if (!group.edgeBuffer) {
        group.edgeBuffer = this.gl.createBuffer();
        group.edgeData = new Float32Array(group.edgeCount * 7);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, group.edgeBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, group.edgeData.byteLength, this.gl.DYNAMIC_DRAW);
      }
      const target = group.edgeData, source = group.data, corners = group.wireCorners;
      let offset = 0;
      for (let i = 0; i < corners.length; i++) {
        const start = this._cornerOffset[corners[i]];
        for (let k = 0; k < 7; k++) target[offset++] = source[start + k];
      }
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER, group.edgeBuffer);
      this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, target);
      group.edgeDirty = false;
    }
  }

  setCage(data) {
    if (!data) {this._cageData = null; this.cage = null; this.activeHandle = null; this.draw(); return;}
    const dimensions = data.dimensions || data.grid;
    if (!Array.isArray(dimensions) || dimensions.length !== 3 || dimensions.some(n => !Number.isInteger(n) || n < 2)) throw Error('Cage dimensions must contain three integers of at least two.');
    if (!Array.isArray(data.points) || data.points.length !== dimensions[0] * dimensions[1] * dimensions[2]) throw Error('Cage point count must match its dimensions.');
    const points = data.points.map((point, order) => {
      const index = point.index || [Math.floor(order / (dimensions[1] * dimensions[2])), Math.floor(order / dimensions[2]) % dimensions[1], order % dimensions[2]];
      if (!Array.isArray(point.position) || point.position.length !== 3 || point.position.some(v => !Number.isFinite(v))) throw Error('Cage positions must be finite XYZ points.');
      return {...point, id: point.id ?? order, index: [...index], position: [...point.position], rest: [...(point.rest || point.position)]};
    });
    this._cageData = {...data, dimensions: [...dimensions], points};
    this.cage = {...data, kind: 'ffd', grid: [...dimensions], points};
    this.draw();
  }

  cagePoints() {
    if (!this.cage) return [];
    if (this._cageData) return this._cageData.points.filter(point => point.visible !== false);
    return super.cagePoints();
  }

  cameraState() {
    return {yaw: this.yaw, pitch: this.pitch, radius: this.radius, target: [...this.target], fov: this.fov};
  }

  setCamera(sourceViewer) {
    const source = sourceViewer?.cameraState ? sourceViewer.cameraState() : sourceViewer;
    if (!source || ![source.yaw, source.pitch, source.radius, ...(source.target || [])].every(Number.isFinite) || source.target?.length !== 3 || source.radius <= 0) return;
    this.yaw = source.yaw; this.pitch = source.pitch; this.radius = source.radius; this.target = [...source.target];
    if (Number.isFinite(source.fov)) this.fov = source.fov;
    this._cameraSyncing = true;
    try {this.draw();} finally {this._cameraSyncing = false;}
  }

  draw() {
    const started = performance.now();
    if (this.wire) this._updateWire();
    super.draw();
    this.performance ||= {};
    this.performance.drawMs = performance.now() - started;
    const signature = [this.yaw, this.pitch, this.radius, ...this.target, this.fov].join('|');
    if (signature !== this._cameraSignature) {
      this._cameraSignature = signature;
      if (!this._cameraSyncing) this.onCameraChange?.(this, this.cameraState());
    }
  }
}
