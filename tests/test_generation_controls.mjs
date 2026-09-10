// Run: node --experimental-default-type=module --test tests/test_generation_controls.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {reconcileGeneration} from '../mishape/web/generation-controls.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const localPython = fileURLToPath(new URL('../.venv/bin/python', import.meta.url));
const python = process.env.PYTHON || (existsSync(localPython) ? localPython : 'python3');
const schema = JSON.parse(execFileSync(python, ['-c', 'import json;from mishape.generation import generation_schema;print(json.dumps(generation_schema()))'], {cwd: root, encoding: 'utf8'}));
const suv = schema.presets.suv;

test('SUV 5.1 m minimally extends cabin, preserving wheelbase and requested length', () => {
  const result = reconcileGeneration(suv, 'length', 5.1, schema);
  assert.equal(result.error, undefined);
  assert.equal(result.values.length, 5.1);
  assert.equal(result.values.wheelbase, 2.91);
  assert.equal(result.values.front_overhang_ratio, suv.front_overhang_ratio);
  assert.equal(result.values.cabin_length, 2.301);
  assert.deepEqual(result.changes, [{key: 'cabin_length', from: 2.21, to: 2.301}]);
});

test('input objects are immutable and linked changes exclude the explicit edit', () => {
  const original = structuredClone(suv), frozen = Object.freeze({...suv});
  const result = reconcileGeneration(frozen, 'width', 1.85, schema);
  assert.equal(result.error, undefined);
  assert.equal(result.values.track, 1.70);
  assert.equal(result.values.width, 1.85);
  assert.deepEqual(result.changes.map(change => change.key), ['track']);
  assert.deepEqual(suv, original);
});

test('wider track increases body width only when required', () => {
  const result = reconcileGeneration(suv, 'track', 1.88, schema);
  assert.equal(result.error, undefined);
  assert.equal(result.values.track, 1.88);
  assert.equal(result.values.width, 2.03);
  assert.deepEqual(result.changes.map(change => change.key), ['width']);
});

test('lower roof can extend cabin while retaining all other structure controls', () => {
  const result = reconcileGeneration(suv, 'height', 1.65, schema);
  assert.equal(result.error, undefined);
  assert.ok(result.values.cabin_length > suv.cabin_length);
  assert.deepEqual(result.changes.map(change => change.key), ['cabin_length']);
  for (const key of ['decklid_height', 'rear_belt_height', 'tailgate_lower_height', 'windscreen_angle', 'hood_height']) assert.equal(result.values[key], suv[key]);
});

test('too-low SUV roof rejects the edit instead of rewriting deck and glass', () => {
  const result = reconcileGeneration(suv, 'height', 1.32, schema);
  assert.match(result.error, /车高至少/);
  assert.deepEqual(result.values, suv);
  assert.deepEqual(result.changes, []);
});

test('direct cabin edit is never replaced by a projected value', () => {
  const result = reconcileGeneration(suv, 'cabin_length', 1.10, schema);
  assert.match(result.error, /车顶控制段须为/);
  assert.deepEqual(result.values, suv);
  assert.deepEqual(result.changes, []);
});

test('overhangs adjust balance only when there is a feasible solution', () => {
  const nearFrontLimit = {...suv, front_overhang_ratio: .6};
  const result = reconcileGeneration(nearFrontLimit, 'length', 5.1, schema);
  assert.equal(result.error, undefined);
  const total = result.values.length - result.values.wheelbase;
  assert.ok(total * result.values.front_overhang_ratio <= 1.12);
  assert.ok(total * (1 - result.values.front_overhang_ratio) <= 1.18);
  assert.equal(result.values.wheelbase, suv.wheelbase);
  assert.ok(result.changes.some(change => change.key === 'front_overhang_ratio'));
});

test('impossible overhang total is rejected without silently increasing wheelbase', () => {
  const result = reconcileGeneration(suv, 'length', 5.35, schema);
  assert.match(result.error, /前后悬合计/);
  assert.deepEqual(result.values, suv);
  assert.deepEqual(result.changes, []);
});

test('direct overhang ratio edit outside the physical interval is rejected', () => {
  const long = reconcileGeneration(suv, 'length', 5.1, schema).values;
  const result = reconcileGeneration(long, 'front_overhang_ratio', .6, schema);
  assert.match(result.error, /前悬占比须为/);
  assert.deepEqual(result.values, long);
});

test('full schema and its fields array produce identical results', () => {
  assert.deepEqual(reconcileGeneration(suv, 'length', 5.1, schema), reconcileGeneration(suv, 'length', 5.1, schema.fields));
});

test('component choice and no-op edits do not introduce unrelated dimensional changes', () => {
  const result = reconcileGeneration(suv, 'mirror_style', 'sport', schema);
  assert.equal(result.error, undefined);
  assert.equal(result.values.mirror_style, 'sport');
  assert.deepEqual(result.changes, []);
  for (const [style, preset] of Object.entries(schema.presets)) {
    const unchanged = reconcileGeneration(preset, 'length', preset.length, schema);
    assert.equal(unchanged.error, undefined, style);
    assert.deepEqual(unchanged.values, preset, style);
    assert.deepEqual(unchanged.changes, [], style);
  }
});

test('invalid keys, NaN, invalid choices and invalid schemas are rejected safely', () => {
  for (const [key, value, definitions] of [['absent', 1, schema], ['height', NaN, schema], ['mirror_style', 'absent', schema], ['height', 1.7, {}]]) {
    const result = reconcileGeneration(suv, key, value, definitions);
    assert.equal(typeof result.error, 'string');
    assert.deepEqual(result.values, suv);
    assert.deepEqual(result.changes, []);
  }
});

test('representative linked edits also pass the real Python geometry generator', () => {
  // Deterministic small boundary sample, independently checked by the real engine.
  const edits = [['length', 5.1], ['length', 4.70], ['width', 1.85], ['track', 1.88], ['height', 1.65], ['height', 1.60]];
  const requests = edits.map(([key, value]) => {
    const result = reconcileGeneration(suv, key, value, schema);
    assert.equal(result.error, undefined, `${key}=${value}`);
    return {body_style: 'suv', parameters: result.values};
  });
  const report = JSON.parse(execFileSync(python, ['-c', [
    'import sys,json',
    'from mishape.generation import generate',
    'out=[]',
    'for request in json.load(sys.stdin):',
    ' model=generate(request)',
    ' quality=model["metadata"]["quality"]',
    ' out.append({"pass":quality["screen_pass"],"errors":quality["errors"]})',
    'print(json.dumps(out))',
  ].join('\n')], {cwd: root, encoding: 'utf8', input: JSON.stringify(requests), maxBuffer: 1024 * 1024}));
  assert.equal(report.length, edits.length);
  for (let i = 0; i < report.length; i++) assert.equal(report[i].pass, true, `${edits[i]}: ${report[i].errors.join('; ')}`);
});
