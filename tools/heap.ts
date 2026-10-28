// A big heap of cables dropped on a floor, headless: does it come to rest?
// Every second: cables awake, how many fell asleep and woke up again, the
// fastest node, the median of how far a cable moved in that second, and the
// deepest overlap. Then the time everything was asleep (or not).
//   N=100 KIND=patchCord PARAMS='{"iterations":12}' npm run heap
declare const process: { env: Record<string, string | undefined> };
import { CABLES, CableSim, closest, segSeg, type CableInit } from '@drschizzo/avbd-cables';

const N = Number(process.env.N ?? 100), AREA = Number(process.env.AREA ?? 1.6), SECONDS = Number(process.env.SECONDS ?? 15);
const kind = CABLES[(process.env.KIND ?? 'patchCord') as keyof typeof CABLES];
const sim = new CableSim({ ...kind.params, ...JSON.parse(process.env.PARAMS ?? '{}') });
let s = 7;
const r = () => (s = (s * 16807) % 2147483647) / 2147483647;
const defs: CableInit[] = [];
for (let c = 0; c < N; c++) {
  const a = r() * Math.PI, h = 0.1 + c * 0.03, cx = (r() - 0.5) * AREA, cz = (r() - 0.5) * AREA;
  const n = Math.round(1.2 / kind.segment) + 1, p: number[] = [];
  for (let j = 0; j < n; j++) { const t = (j / (n - 1) - 0.5) * 1.2; p.push(cx + Math.cos(a) * t, h, cz + Math.sin(a) * t); }
  defs.push({ ...kind.cable, points: p });
}
sim.build(defs);
sim.setStatics([{ kind: 'box', min: [-3, -0.2, -3], max: [3, 0, 3] }]);

function overlap(): number {
  let w = 0;
  const x = sim.x;
  for (let i = 0; i < sim.n; i++) w = Math.max(w, sim.radius[i] - x[3 * i + 1]);
  for (let a = 0; a < sim.n; a++) {
    if (!sim.segOK[a]) continue;
    for (let b = a + 1; b < sim.n; b++) {
      if (!sim.segOK[b] || sim.cableOf[a] === sim.cableOf[b]) continue;
      segSeg(x[3 * a], x[3 * a + 1], x[3 * a + 2], x[3 * a + 3], x[3 * a + 4], x[3 * a + 5], x[3 * b], x[3 * b + 1], x[3 * b + 2], x[3 * b + 3], x[3 * b + 4], x[3 * b + 5]);
      w = Math.max(w, sim.radius[a] + sim.radius[b] - Math.sqrt(closest.d2));
    }
  }
  return w;
}

const was = sim.cables.map((c) => c.asleep);
let slept = 0, woke = 0, ms = 0, done = -1;
let ref = Float64Array.from(sim.x);
const fps = Math.round(1 / sim.P.dt);
for (let f = 1; f <= SECONDS * fps; f++) {
  sim.step();
  ms += sim.stats.ms;
  sim.cables.forEach((c, k) => { if (c.asleep && !was[k]) slept++; if (!c.asleep && was[k]) woke++; was[k] = c.asleep; });
  if (done < 0 && sim.cables.every((c) => c.asleep)) done = f / fps;
  if (f % fps === 0) {
    let v = 0;
    for (let i = 0; i < sim.n; i++) v = Math.max(v, Math.hypot(sim.v[3 * i], sim.v[3 * i + 1], sim.v[3 * i + 2]));
    const moved = sim.cables.map((cb) => { let m = 0; for (let i = cb.start; i < cb.start + cb.count; i++) m = Math.max(m, Math.hypot(sim.x[3 * i] - ref[3 * i], sim.x[3 * i + 1] - ref[3 * i + 1], sim.x[3 * i + 2] - ref[3 * i + 2])); return m; }).sort((a, b) => a - b);
    ref = Float64Array.from(sim.x);
    if (!process.env.QUIET) console.log(`${f / fps}s awake ${sim.stats.awake} slept ${slept} woke ${woke} vmax ${v.toFixed(3)} moved median ${(1000 * moved[moved.length >> 1]).toFixed(1)} mm max ${(1000 * moved[moved.length - 1]).toFixed(1)} mm overlap ${(1000 * overlap()).toFixed(1)} mm contacts ${sim.stats.contacts} ${(ms / fps).toFixed(1)} ms`);
    slept = woke = 0; ms = 0;
  }
}
console.log(`${N} × ${kind === CABLES.patchCord ? 'patchCord' : 'cable'}: ${done < 0 ? 'never all asleep' : `all asleep at ${done.toFixed(1)} s`}`);
