// The four scenes of the knots example, as the solver sees them: the ropes
// they start from, the static shapes, the settings, and what moves the ropes
// (hands pulling a knot tight, braiding three strands). Nothing here draws.

import type { CableSim, SimParams, StaticShape } from '@drschizzo/avbd-cables';
import { BRAIDED, JACKET, LAID } from './rope';
import { figureEight, openKnot, torusKnot, wound, type KnotCurve, type Vec } from './shapes';

export interface RopeDef {
  points: number[];
  radius: number;
  color: string;
  /** LAID, BRAIDED or JACKET (see rope.ts) */
  make: number;
}

export interface Scene {
  help: string;
  params: Partial<SimParams>;
  ropes: RopeDef[];
  statics: StaticShape[];
  /** the static shapes not to draw (the floor: the stage draws it) */
  hidden: number;
  camera: { position: Vec; target: Vec };
  /** text pinned to points of the scene */
  labels?: { text: string; at: Vec }[];
  /** once built: pins */
  setup?(sim: CableSim, starts: number[]): void;
  /** every frame, before the step: where the hands are; the nodes they hold */
  drive?(sim: CableSim, frame: number): number[];
  /** buttons of the scene's own */
  actions?: { label: string; run(sim: CableSim): void }[];
}

/** a thick rope: 2 cm, 100 g/m, a node every 2 cm (no longer than its diameter, so that it can be tied tight) */
const ROPE = { radius: 0.01, mass: 0.002, bend: 0.5, segment: 0.02 };
/** tied tight, ropes press hard on one another: more substeps than the presets */
const PARAMS: Partial<SimParams> = { substeps: 10, margin: 0.003, marginCap: 0.02 };
const FLOOR: StaticShape = { kind: 'box', min: [-4, -0.2, -4], max: [4, 0, 4] };

export const KNOTS: Record<string, { name: string; curve: KnotCurve }> = {
  overhand: { name: 'overhand (3₁)', curve: torusKnot(3) },
  eight: { name: 'figure-eight (4₁)', curve: figureEight() },
  cinquefoil: { name: 'cinquefoil (5₁)', curve: torusKnot(5) },
  seven: { name: 'septafoil (7₁)', curve: torusKnot(7) },
};

/**
 * A knot tied loose on a table, pulled tight by two hands, then let go: it
 * holds, by friction alone.
 */
export function knotScene(kind: keyof typeof KNOTS): Scene {
  const points = openKnot(KNOTS[kind].curve, 0.04, 0.35, [0, ROPE.radius + 0.05, 0.12], ROPE.segment, true);
  let a0: Vec = [0, 0, 0], b0: Vec = [0, 0, 0], ends = [0, 0];
  let letGo = 540, pulling = true;
  const PULL = 300;
  return {
    help: 'Two hands pull a loose knot tight across the table, then let go: it holds by friction alone. Grab the rope to loosen it.',
    params: PARAMS,
    ropes: [{ points, radius: ROPE.radius, color: '#b48a4e', make: LAID }],
    statics: [FLOOR],
    hidden: 1,
    camera: { position: [0.15, 0.75, 0.75], target: [0, 0.02, 0.02] },
    setup(sim, starts) {
      ends = [starts[0], starts[0] + sim.cables[0].count - 1];
      const at = (i: number): Vec => [sim.x[3 * i], sim.x[3 * i + 1], sim.x[3 * i + 2]];
      a0 = at(ends[0]); b0 = at(ends[1]);
    },
    drive(sim, f) {
      if (!pulling) return [];
      if (f >= letGo) {
        pulling = false;
        for (const e of ends) sim.clearTarget(e);
        return [];
      }
      // the hands come to the front and draw apart, as far as the knot lets them
      const t = Math.min(1, f / PULL), ease = t * t * (3 - 2 * t);
      sim.setTarget(ends[0], a0[0] - 0.9 * ease, ROPE.radius, 0.05 + (a0[2] - 0.05) * (1 - ease), 1.6, 0.03, true);
      sim.setTarget(ends[1], b0[0] + 0.9 * ease, ROPE.radius, 0.05 + (b0[2] - 0.05) * (1 - ease), 1.6, 0.03, true);
      return ends;
    },
    actions: [{ label: 'pull again', run: () => { pulling = true; letGo = Number.MAX_SAFE_INTEGER; } },
      { label: 'let go', run: () => { letGo = 0; } }],
  };
}

/**
 * Three ropes over three bars, wound half a turn, one and a half and two and a
 * half: a short end on the left, eight times as much rope on the right. The
 * capstan equation says a rope holds on a bar while T₁/T₂ < e^(μθ): with μ = 0.4,
 * 3.5 for half a turn, 43 and 535 for the others.
 */
export function capstanScene(): Scene {
  const bar = 0.04, y = 0.95, short = 0.08, long = 0.85, mu = 0.4;
  const xs = [-0.45, 0, 0.45];
  const ropes: RopeDef[] = [], statics: StaticShape[] = [FLOOR], labels: { text: string; at: Vec }[] = [];
  const colors = ['#d0452f', '#2f7fd0', '#3aa35a'];
  xs.forEach((x, k) => {
    ropes.push({ points: wound([x, y, 0], bar + ROPE.radius, k, 0.03, short, long, ROPE.segment), radius: ROPE.radius, color: colors[k], make: BRAIDED });
    statics.push({ kind: 'capsule', ax: x, ay: y, az: -0.15, bx: x, by: y, bz: 0.15, r: bar, mu });
    const theta = Math.PI * (1 + 2 * k), grip = Math.exp(mu * theta);
    labels.push({ text: `${k + 0.5} turn${k ? 's' : ''}<br>e<sup>μθ</sup> = ${grip.toFixed(1)} ${grip > long / short ? '>' : '<'} ${(long / short).toFixed(0)}`, at: [x, y + 0.12, 0] });
  });
  return {
    help: `Each rope has ${(long / short).toFixed(0)} times as much rope hanging on the right as on the left. Half a turn cannot hold it; one and a half turns can (the capstan equation, μ = ${mu}). Pull the ropes about.`,
    params: PARAMS,
    ropes, statics, hidden: 1, labels,
    camera: { position: [0, 0.75, 2.5], target: [0, 0.5, 0] },
  };
}

/**
 * Three strands hanging from a bar, braided by two hands: the right strand
 * over the middle one, then the left one, and again.
 */
export function braidScene(): Scene {
  const R = 0.007, seg = 0.015, top = 1.05, len = 0.85, w = 0.07, swap = 50, swaps = 14;
  const colors = ['#e0b53a', '#b8402c', '#2c6fb8'];
  const ropes = [-1, 0, 1].map((k, c): RopeDef => {
    const p: number[] = [];
    for (let i = 0; i * seg <= len + 1e-9; i++) p.push(k * w * 0.4, top - i * seg, 0);
    return { points: p, radius: R, color: colors[c], make: BRAIDED };
  });
  const xs = [-w, 0, w];
  let ends: number[] = [];
  const slot = [0, 1, 2];
  // below the ends: the solver brings a target out of reach back within it, so
  // the hands keep the strands taut and rise as the braid takes up rope
  const yHand = top - len - 0.2;
  return {
    help: 'Two hands braid three strands: right over middle, left over middle, again. The strands slide over one another and never pass through.',
    params: { ...PARAMS, kCapCableHeld: 4 },
    ropes,
    statics: [FLOOR, { kind: 'capsule', ax: -0.15, ay: top + 0.012, az: 0, bx: 0.15, by: top + 0.012, bz: 0, r: 0.012 }],
    hidden: 1,
    camera: { position: [0.3, 0.75, 1.3], target: [0, 0.62, 0] },
    setup(sim, starts) {
      ends = starts.map((s, c) => s + sim.cables[c].count - 1);
      starts.forEach((s) => sim.pin(s, sim.x[3 * s], sim.x[3 * s + 1], sim.x[3 * s + 2]));
    },
    drive(sim, f) {
      const k = Math.floor(f / swap), u = (f % swap) / swap;
      // right over middle, then left over middle
      const side = k % 2 === 0 ? 2 : 0;
      for (let c = 0; c < 3; c++) {
        let x = xs[slot[c]], z = 0;
        const outer = slot[c] === side, mid = slot[c] === 1;
        if (k < swaps && (outer || mid)) {
          const to = outer ? 0 : xs[side], e = 0.5 - 0.5 * Math.cos(Math.PI * u);
          x += (to - x) * e;
          z = (outer ? 1 : -1) * Math.sin(Math.PI * u) * w * 0.6;
        }
        sim.setTarget(ends[c], x, yHand, z, 0.6, 0.02, true);
      }
      if (k < swaps && f % swap === swap - 1) {
        const o = slot.indexOf(side), m = slot.indexOf(1);
        slot[o] = 1; slot[m] = side;
      }
      return ends;
    },
  };
}

/**
 * Pegs, a ring on a stand, ropes falling on them, and a loose knot on the
 * table: for the hand.
 */
export function playgroundScene(): Scene {
  const pegs: [number, number][] = [[-0.38, 0.5], [-0.16, 0.72], [0.12, 0.56], [0.36, 0.78]];
  const statics: StaticShape[] = [FLOOR];
  for (const [x, y] of pegs) statics.push({ kind: 'capsule', ax: x, ay: y, az: -0.2, bx: x, by: y, bz: 0.2, r: 0.016 });
  // a ring facing the camera, of 20 short capsules, on a stand
  const rc: Vec = [0.02, 0.3, 0.05], rr = 0.09, n = 20;
  for (let k = 0; k < n; k++) {
    const a = (2 * Math.PI * k) / n, b = (2 * Math.PI * (k + 1)) / n;
    statics.push({ kind: 'capsule', ax: rc[0] + rr * Math.cos(a), ay: rc[1] + rr * Math.sin(a), az: rc[2], bx: rc[0] + rr * Math.cos(b), by: rc[1] + rr * Math.sin(b), bz: rc[2], r: 0.011 });
  }
  statics.push({ kind: 'box', min: [rc[0] - 0.012, 0, rc[2] - 0.012], max: [rc[0] + 0.012, rc[1] - rr - 0.008, rc[2] + 0.012] });
  const line = (a: Vec, b: Vec, seg: number) => {
    const p: number[] = [], m = Math.round(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / seg);
    for (let k = 0; k <= m; k++) p.push(a[0] + ((b[0] - a[0]) * k) / m, a[1] + ((b[1] - a[1]) * k) / m, a[2] + ((b[2] - a[2]) * k) / m);
    return p;
  };
  const ropes: RopeDef[] = [
    { points: line([-0.6, 1.0, -0.08], [0.55, 0.95, -0.08], ROPE.segment), radius: ROPE.radius, color: '#d0452f', make: BRAIDED },
    { points: line([-0.5, 1.15, 0.06], [0.6, 1.1, 0.0], ROPE.segment), radius: ROPE.radius, color: '#c9a86a', make: LAID },
    { points: line([-0.55, 1.3, 0.12], [0.5, 1.35, 0.1], ROPE.segment), radius: 0.008, color: '#2a2d33', make: JACKET },
    { points: openKnot(figureEight(), 0.035, 0.3, [-0.05, ROPE.radius + 0.045, 0.45], ROPE.segment, true), radius: ROPE.radius, color: '#3aa35a', make: BRAIDED },
  ];
  return {
    help: 'Throw ropes over the pegs, thread them through the ring, untie the knot on the table: drag a rope with the mouse, the background to orbit.',
    params: PARAMS,
    ropes, statics, hidden: 1,
    camera: { position: [0.1, 0.9, 1.9], target: [0, 0.45, 0.1] },
  };
}
