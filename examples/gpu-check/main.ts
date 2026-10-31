// The GPU solver against the CPU one, on the scenes of tests/physics.test.ts:
// the same cables and settings, stepped for the same time on both, then
// measured the same way (how much the cords stretch, how deep they go into
// one another and into the floor, where they hang). `?only=` a scene's name,
// `?stretch=` nodes per stretch of cord on the GPU.

import { CABLES, CableSim, closest, segSeg, type CableInit, type SimParams, type StaticShape } from '@drschizzo/avbd-cables';
import { GPUCableSim } from '@drschizzo/avbd-cables/gpu';

const status = document.getElementById('status')!, out = document.getElementById('out') as HTMLTableElement;
const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) { status.textContent = 'no WebGPU here'; throw new Error('no WebGPU'); }
const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [] });
device.addEventListener('uncapturederror', (e) => console.error((e as GPUUncapturedErrorEvent).error.message));

const PATCH = CABLES.patchCord.cable, R = PATCH.radius;
const floor: StaticShape = { kind: 'box', min: [-1, -0.1, -1], max: [1, 0, 1] };

function cord(a: number[], b: number[], seg = 0.025): number[] {
  const n = Math.max(2, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / seg) + 1);
  const p: number[] = [];
  for (let k = 0; k < n; k++) { const t = k / (n - 1); p.push(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t); }
  return p;
}

/** what both solvers have, as the measures read it */
interface View { gpu?: boolean; n: number; x: ArrayLike<number>; rest: ArrayLike<number>; segOK: ArrayLike<number>; cableOf: ArrayLike<number>; radius: ArrayLike<number> }

function stretch(s: View): number {
  let w = 0;
  for (let i = 0; i < s.n; i++) {
    if (!s.segOK[i]) continue;
    const d = Math.hypot(s.x[3 * i] - s.x[3 * i + 3], s.x[3 * i + 1] - s.x[3 * i + 4], s.x[3 * i + 2] - s.x[3 * i + 5]);
    w = Math.max(w, Math.abs(d - s.rest[i]) / s.rest[i]);
  }
  return w;
}
function overlap(s: View): number {
  let w = 0;
  const x = s.x;
  for (let a = 0; a < s.n; a++) {
    if (!s.segOK[a]) continue;
    for (let b = a + 1; b < s.n; b++) {
      if (!s.segOK[b] || s.cableOf[a] === s.cableOf[b]) continue;
      segSeg(x[3 * a], x[3 * a + 1], x[3 * a + 2], x[3 * a + 3], x[3 * a + 4], x[3 * a + 5], x[3 * b], x[3 * b + 1], x[3 * b + 2], x[3 * b + 3], x[3 * b + 4], x[3 * b + 5]);
      w = Math.max(w, s.radius[a] + s.radius[b] - Math.sqrt(closest.d2));
    }
  }
  return w;
}
function lowest(s: View): number { let y = Infinity; for (let i = 0; i < s.n; i++) y = Math.min(y, s.x[3 * i + 1]); return y; }
/** how far apart the two solvers' nodes are, at most (m) */
function apart(a: View, b: View): number { let w = 0; for (let i = 0; i < a.n; i++) w = Math.max(w, Math.hypot(a.x[3 * i] - b.x[3 * i], a.x[3 * i + 1] - b.x[3 * i + 1], a.x[3 * i + 2] - b.x[3 * i + 2])); return w; }

interface Scene {
  name: string;
  params?: Partial<SimParams>;
  defs: CableInit[];
  statics?: StaticShape[];
  seconds: number;
  setup?: (s: CableSim | GPUCableSim) => void;
  /** called every frame before the step */
  frame?: (s: CableSim | GPUCableSim, f: number) => void;
  measures: [string, (s: View) => number, string, (v: number) => boolean][];
}

/** what the hands of the scene below felt, CPU and GPU, and the weight of the cord */
const held = [0, 0];
const weight = (s: View) => { let w = 0; for (let i = 0; i < s.n; i++) w += massOf[i] * 9.81; return w; };
let massOf: ArrayLike<number> = [];

const scenes: Scene[] = [
  {
    name: 'cord on the floor', defs: [{ points: cord([-0.3, 0.2, 0], [0.3, 0.2, 0]), ...PATCH }], statics: [floor], seconds: 3,
    measures: [['lowest (mm)', (s) => 1000 * lowest(s), `${1000 * (R - 0.001)}…${1000 * (R + 0.002)}`, (v) => v > 1000 * (R - 0.001) && v < 1000 * (R + 0.002)], ['stretch %', (s) => 100 * stretch(s), '< 1', (v) => v < 1]],
  },
  {
    name: 'fast cord, thin plate', defs: [{ points: cord([-0.3, 0.1, 0], [0.3, 0.1, 0]), ...PATCH }], statics: [{ kind: 'box', min: [-1, -0.004, -1], max: [1, 0, 1] }], seconds: 1,
    setup: (s) => { if (s instanceof CableSim) for (let i = 0; i < s.n; i++) s.v[3 * i + 1] = -3; else s.setVelocities(Array.from({ length: 3 * s.n }, (_, k) => (k % 3 === 1 ? -3 : 0))); },
    measures: [['lowest (mm)', (s) => 1000 * lowest(s), '> 0', (v) => v > 0]],
  },
  {
    name: 'cord pinned at both ends', defs: [{ points: cord([-0.375, 0.5, 0], [0.375, 0.5, 0]), ...PATCH }], seconds: 4,
    setup: (s) => { s.pin(0, -0.3, 0.5, 0); s.pin(s.n - 1, 0.3, 0.5, 0); },
    measures: [['middle y (m)', (s) => s.x[3 * (s.n >> 1) + 1], '< 0.4', (v) => v < 0.4], ['stretch %', (s) => 100 * stretch(s), '< 1', (v) => v < 1]],
  },
  {
    name: 'cord across a rod', defs: [{ points: cord([-0.2, 0.1, 0], [0.2, 0.1, 0]), ...PATCH }], statics: [{ kind: 'capsule', ax: 0, ay: 0, az: -0.2, bx: 0, by: 0, bz: 0.2, r: 0.01 }], seconds: 3,
    measures: [['middle y (mm)', (s) => 1000 * s.x[3 * (s.n >> 1) + 1], '> 10', (v) => v > 10], ['ends y (mm)', (s) => 1000 * Math.max(s.x[1], s.x[3 * (s.n - 1) + 1]), '< −50', (v) => v < -50], ['stretch %', (s) => 100 * stretch(s), '< 1', (v) => v < 1]],
  },
  {
    name: 'six cords piled', statics: [floor], seconds: 3,
    defs: Array.from({ length: 6 }, (_, k) => { const a = (k * Math.PI) / 6, c = Math.cos(a) * 0.25, s = Math.sin(a) * 0.25; return { points: cord([-c, 0.02 + 0.03 * k, -s], [c, 0.02 + 0.03 * k, s]), ...PATCH }; }),
    measures: [['overlap (mm)', (s) => 1000 * overlap(s), '< 0.5', (v) => v < 0.5], ['lowest (mm)', (s) => 1000 * lowest(s), `> ${1000 * (R - 0.001)}`, (v) => v > 1000 * (R - 0.001)], ['stretch %', (s) => 100 * stretch(s), '< 1', (v) => v < 1]],
  },
  {
    name: 'rope hanging straight', params: CABLES.rope.params, defs: [{ ...CABLES.rope.cable, points: cord([0, 1, 0], [0, 0.05, 0], CABLES.rope.segment) }], seconds: 3,
    setup: (s) => s.pin(0, 0, 1, 0),
    measures: [['stretch %', (s) => 100 * stretch(s), '< 2', (v) => v < 2], ['end y (m)', (s) => s.x[3 * (s.n - 1) + 1], '≈ 0.05', (v) => Math.abs(v - 0.05) < 0.02]],
  },
  {
    name: 'rope draped over a thick bar (16 substeps)', params: { substeps: 16, margin: 0.003, marginCap: 0.02 }, seconds: 4,
    defs: [{ points: (() => { const r = 0.05, y = 0.9, p: number[] = []; for (let h = y - 0.4; h < y; h += 0.02) p.push(-r, h, 0); for (let k = 0; k <= 12; k++) { const th = Math.PI * (1 - k / 12); p.push(r * Math.cos(th), y + r * Math.sin(th), 0); } for (let h = y - 0.02; h >= y - 0.4; h -= 0.02) p.push(r, h, 0); return p; })(), radius: 0.01, mass: 0.002, bend: 0.5 }],
    statics: [{ kind: 'capsule', ax: 0, ay: 0.9, az: -0.3, bx: 0, by: 0.9, bz: 0.3, r: 0.04 }],
    measures: [['ends moved (mm)', (s) => 1000 * Math.max(Math.abs(s.x[1] - 0.5), Math.abs(s.x[3 * (s.n - 1) + 1] - 0.5)), '< 20', (v) => v < 20]],
  },
  {
    name: 'a target leads a node', defs: [{ points: cord([-0.3, 0.1, 0], [0.3, 0.1, 0]), ...PATCH }], statics: [floor], seconds: 3,
    frame: (s, f) => { if (f === 60) s.setTarget(s.n - 1, 0.25, 0.2, 0.05, 2, 0.05); },
    measures: [['to its point (mm)', (s) => 1000 * Math.hypot(s.x[3 * (s.n - 1)] - 0.25, s.x[3 * (s.n - 1) + 1] - 0.2, s.x[3 * (s.n - 1) + 2] - 0.05), '< 10', (v) => v < 10], ['stretch %', (s) => 100 * stretch(s), '< 1', (v) => v < 1]],
  },
  {
    name: 'a heavy target out of reach', defs: [{ points: cord([0, 0.5, 0], [0.5, 0.5, 0]), ...PATCH }], seconds: 4,
    setup: (s) => { s.pin(0, 0, 0.5, 0); s.setTarget(s.n - 1, 2, 0.5, 0, 1.6, 0.06, true); },
    measures: [['reach (m)', (s) => s.x[3 * (s.n - 1)], '> 0.49', (v) => v > 0.49], ['stretch %', (s) => 100 * stretch(s), '< 3', (v) => v < 3]],
  },
  {
    // (solved to convergence, as the test of targetPull: at 3 substeps the hand overshoots)
    name: 'a hand holding a hanging cord (12 substeps)', params: { substeps: 12, sleepTime: 1e9 }, seconds: 6,
    defs: [{ points: cord([0, 2, 0], [0, 1, 0]), ...PATCH }],
    setup: (s) => s.setTarget(0, 0, 2, 0, 2, 1),
    frame: (s, f) => { if (f >= 300) held[s instanceof CableSim ? 0 : 1] += s.targetPull(0) / 60; },
    measures: [['pull / weight', (s) => held[s.gpu ? 1 : 0] / weight(s), '≈ 1 (± 5 %)', (v) => Math.abs(v - 1) < 0.05]],
  },
  {
    name: 'a hundred cords in a heap (10 s)', params: CABLES.patchCord.params, statics: [{ kind: 'box', min: [-3, -0.1, -3], max: [3, 0, 3] }], seconds: 10,
    defs: (() => { let seed = 7; const r = () => (seed = (seed * 16807) % 2147483647) / 2147483647; return Array.from({ length: 100 }, (_, c) => { const a = r() * Math.PI, h = 0.1 + c * 0.03, cx = (r() - 0.5) * 1.6, cz = (r() - 0.5) * 1.6; return { ...PATCH, points: cord([cx - 0.6 * Math.cos(a), h, cz - 0.6 * Math.sin(a)], [cx + 0.6 * Math.cos(a), h, cz + 0.6 * Math.sin(a)]) }; }); })(),
    measures: [['overlap (mm)', (s) => 1000 * overlap(s), '', () => true], ['lowest (mm)', (s) => 1000 * lowest(s), '', () => true], ['stretch %', (s) => 100 * stretch(s), '< 3', (v) => v < 3], ['highest (m)', (s) => { let y = -Infinity; for (let i = 0; i < s.n; i++) y = Math.max(y, s.x[3 * i + 1]); return y; }, '', () => true]],
  },
];

function row(cells: string[], cls: string[] = []): void {
  const tr = out.insertRow();
  cells.forEach((c, k) => { const td = tr.insertCell(); td.textContent = c; if (cls[k]) td.className = cls[k]; });
}

const query = new URLSearchParams(location.search), only = query.get('only'), stretchLength = Number(query.get('stretch')) || undefined;
const results: Record<string, unknown>[] = [];
for (const sc of scenes) {
  if (only && !sc.name.includes(only)) continue;
  status.textContent = sc.name;
  const cpu = new CableSim(sc.params);
  const gpu = new GPUCableSim(device, sc.params, { stretch: stretchLength });
  const frames = Math.round(sc.seconds / cpu.P.dt);
  let cpuMs = 0, gpuMs = 0, encMs = 0;
  held[0] = held[1] = 0;
  for (const s of [cpu, gpu]) {
    s.build(sc.defs);
    if (sc.statics) s.setStatics(sc.statics);
    sc.setup?.(s);
  }
  massOf = cpu.mass;
  for (let f = 0; f < frames; f++) {
    sc.frame?.(cpu, f); cpu.step(); cpuMs += cpu.stats.ms;
    sc.frame?.(gpu, f); gpu.step(); encMs += gpu.stats.ms; gpuMs += gpu.stats.gpuMs;
    if (f % 30 === 29) await gpu.settled();
  }
  await gpu.settled();
  const x = await gpu.readPositions();
  const gv: View = { gpu: true, n: gpu.n, x, rest: gpu.rest, segOK: gpu.segOK, cableOf: gpu.cableOf, radius: gpu.radius };
  const r: Record<string, unknown> = { scene: sc.name, finite: x.every(Number.isFinite), apart: apart(cpu, gv), cpuMs: cpuMs / frames, gpuMs: gpuMs / frames, encMs: encMs / frames, stats: { ...gpu.stats } };
  for (const [name, f, expect, ok] of sc.measures) {
    const c = f(cpu), g = f(gv);
    r[name] = [c, g];
    row([sc.name, name, c.toFixed(3), g.toFixed(3), expect], ['', '', ok(c) ? 'ok' : 'bad', ok(g) ? 'ok' : 'bad', '']);
  }
  row([sc.name, 'CPU and GPU apart (mm)', '', (1000 * (r.apart as number)).toFixed(1), '']);
  row([sc.name, 'ms per frame (CPU / GPU, encoding)', (cpuMs / frames).toFixed(2), `${(gpuMs / frames).toFixed(2)} (${(encMs / frames).toFixed(2)})`, '']);
  results.push(r);
  gpu.dispose();
}
status.textContent = 'done';
Object.assign(window, { results, debug: { device, scenes, CableSim, GPUCableSim } });
