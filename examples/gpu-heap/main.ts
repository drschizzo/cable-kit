// Thousands of ropes falling in a heap over a few bars, on WebGPU. The solver
// runs in compute shaders (GPUCableSim), and the tubes read its nodes on the
// GPU (GPUTubeFeed): they never come back to the CPU, which only asks for a
// step and an image each frame. With ?solver=cpu, the same scene runs on the
// CPU solver in a worker (as in the `falling` example), to compare. On both,
// the ropes fall asleep once at rest, and the physics costs little more.
//   ?ropes=  how many (2000), ?solver=cpu, ?stretch= nodes per stretch of cord on the GPU

import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CABLES, CableSim, threaded, type CableInit, type StaticShape } from '@drschizzo/avbd-cables';
import { GPUCableSim } from '@drschizzo/avbd-cables/gpu';
import { GPUTubeFeed, GPUTubeNodes, TubeMesh, TubeNodes, TubeSync } from '@drschizzo/three-cable-tubes/webgpu';

class CPUSim extends threaded(CableSim, () => new Worker(new URL('../falling/sim.worker.ts', import.meta.url), { type: 'module' })) {}

const query = new URLSearchParams(location.search);
const onCPU = query.get('solver') === 'cpu';
const ROPES = Number(query.get('ropes')) || (onCPU ? 300 : 2000), LENGTH = 1.2;
const kind = CABLES.rope;
const statsEl = document.getElementById('stats')!;

// the renderer, on the adapter's largest buffers (the contacts of thousands of ropes take more than the default 128 MB)
const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) { statsEl.textContent = 'this example needs WebGPU, which this browser does not offer'; throw new Error('no WebGPU'); }
const renderer = new THREE.WebGPURenderer({
  antialias: true,
  requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
await renderer.init();
const device: GPUDevice = (renderer.backend as any).device;

// --- the ropes, in a tall stack of crossed layers above the bars; the bars and the floor
function random(seed = 7): () => number {
  let s = seed;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}
/** cells of 1.7 m, as many as the ropes need (400 each at most) */
const GRID = Math.ceil(Math.sqrt(ROPES / 400)), CELL = 1.7, SIDE = GRID * CELL;
/**
 * The ropes, in every cell a stack of layers of parallel ropes 4 cm apart,
 * each layer turned any way, 1.5 cm above the one below: none starts in
 * another.
 */
function ropes(): CableInit[] {
  const r = random(), out: CableInit[] = [];
  const n = Math.round(LENGTH / kind.segment) + 1, per = Math.ceil(ROPES / (GRID * GRID)), row = 25;
  let layerAngle = 0;
  for (let c = 0; c < ROPES; c++) {
    const cell = Math.floor(c / per), k = c % per, layer = Math.floor(k / row);
    const ox = ((cell % GRID) + 0.5) * CELL - SIDE / 2, oz = (Math.floor(cell / GRID) + 0.5) * CELL - SIDE / 2;
    const a = k % row === 0 ? (layerAngle = r() * Math.PI) : layerAngle, h = 0.4 + layer * 0.015;
    const off = ((k % row) / (row - 1) - 0.5) * 0.96;
    const ux = Math.cos(a), uz = Math.sin(a);
    const points: number[] = [];
    for (let j = 0; j < n; j++) { const t = (j / (n - 1) - 0.5) * LENGTH; points.push(ox + ux * t - uz * off, h, oz + uz * t + ux * off); }
    out.push({ ...kind.cable, points });
  }
  return out;
}
const statics: StaticShape[] = [{ kind: 'box', min: [-8, -0.2, -8], max: [8, 0, 8] }];
for (let k = -2; k <= 2; k++) statics.push({ kind: 'capsule', ax: k * SIDE / 5, ay: 0.25, az: -SIDE / 2, bx: k * SIDE / 5, by: 0.25, bz: SIDE / 2, r: 0.03 });

// --- the solver, and the nodes the tubes read
const defs = ropes();
let nodes: GPUTubeNodes | TubeNodes;
let step: () => void, stats: () => string, again: () => void;
if (!onCPU) {
  const sim = new GPUCableSim(device, kind.params, { stretch: Number(query.get('stretch')) || undefined });
  sim.build(defs);
  sim.setStatics(statics);
  const gpuNodes = new GPUTubeNodes(sim.n);
  const feed = new GPUTubeFeed(renderer, gpuNodes, sim);
  nodes = gpuNodes;
  step = () => { sim.step(); feed.update(); };
  again = () => { sim.build(defs); sim.setStatics(statics); };
  stats = () => {
    const s = sim.stats;
    return `${sim.n.toLocaleString('en')} nodes · ${s.contacts.toLocaleString('en')} contacts · ${s.awake}/${ROPES} awake · ${s.colors.length} colours · physics on the GPU ${s.gpuMs.toFixed(1)} ms (CPU ${s.ms.toFixed(2)} ms to ask for it)${s.overflow.some((o) => o > 0) ? ` · ${s.overflow.join('/')} contacts dropped` : ''}`;
  };
  Object.assign(window, { example: { sim, renderer } });
} else {
  const sim = new CPUSim(kind.params);
  sim.build(defs);
  sim.setStatics(statics);
  sim.startWorker();
  const cpuNodes = new TubeNodes(sim.n);
  const sync = new TubeSync(cpuNodes, kind.cable.radius);
  nodes = cpuNodes;
  step = () => { sim.advance(1); sync.update(sim); };
  again = () => sim.teleport(defs.flatMap((d) => Array.from(d.points)));
  stats = () => {
    const s = sim.stats;
    return `${sim.n.toLocaleString('en')} nodes · ${s.contacts.toLocaleString('en')} contacts · ${s.awake}/${ROPES} awake · physics on the CPU ${s.ms.toFixed(1)} ms (worker)`;
  };
  Object.assign(window, { example: { sim, renderer } });
}
nodes.box.set(new THREE.Vector3(-SIDE, -0.1, -SIDE), new THREE.Vector3(SIDE, 1, SIDE));
nodes.radius = kind.cable.radius;

const material = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.7 });
const per = defs[0].points.length / 3;
const tubes = new TubeMesh(ROPES * (per - 1), material, nodes, { radial: 8, sub: 4 });
const r = random(3), color = new THREE.Color();
for (let c = 0; c < ROPES; c++) tubes.lay(c * (per - 1), c * per, per, color.setHSL(r(), 0.5, 0.5), 0, kind.segment, 0);

// --- the room
const scene = new THREE.Scene();
scene.background = new THREE.Color('#101214');
scene.add(tubes);
scene.add(new THREE.HemisphereLight('#cfd8e3', '#202326', 1.4));
const sun = new THREE.DirectionalLight('#ffffff', 2);
sun.position.set(1.5, 3, 1);
scene.add(sun);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(16, 16), new THREE.MeshStandardNodeMaterial({ color: '#3a3f45', roughness: 0.9 }));
floor.rotation.x = -Math.PI / 2;
scene.add(floor);
const metal = new THREE.MeshStandardNodeMaterial({ color: '#8a9099', roughness: 0.35, metalness: 0.7 });
for (const s of statics) {
  if (s.kind !== 'capsule') continue;
  const bar = new THREE.Mesh(new THREE.CylinderGeometry(s.r, s.r, Math.hypot(s.bx - s.ax, s.by - s.ay, s.bz - s.az), 24), metal);
  bar.position.set((s.ax + s.bx) / 2, (s.ay + s.by) / 2, (s.az + s.bz) / 2);
  bar.rotation.x = Math.PI / 2;
  scene.add(bar);
}

const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.02, 60);
camera.position.set(SIDE * 1.3, SIDE * 0.9, SIDE * 1.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.15, 0);
controls.update();
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
document.getElementById('again')!.addEventListener('click', () => again());

// --- every frame: a step, the image; twice a second, the counters
let last = performance.now(), frames = 0;
renderer.setAnimationLoop((now) => {
  step();
  renderer.render(scene, camera);
  frames++;
  if (now - last > 500) {
    const fps = (1000 * frames) / (now - last);
    last = now; frames = 0;
    statsEl.textContent = `${ROPES.toLocaleString('en')} ropes · ${stats()} · ${fps.toFixed(0)} frames/s`;
  }
});
