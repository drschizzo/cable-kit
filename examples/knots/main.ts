// Knots, a capstan, a braid and a playground: ropes pressing hard on one
// another and on the bars they are wound round, where a rope that passes
// through another (or itself) for a single frame unties the knot. Each scene
// is its own simulation (scenes.ts); the ropes are drawn with strands of
// their own (rope.ts). The line at the bottom shows how well the solver holds:
// the worst stretch of a segment, and how deep two stretches of rope go into
// one another, as a fraction of their thickness.

import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { CableSim, closest, segSeg } from '@drschizzo/avbd-cables';
import { TubeMesh, TubeNodes, TubeSync } from '@drschizzo/three-cable-tubes';
import { stage } from '../common';
import { ropeMaterial } from './rope';
import { KNOTS, braidScene, capstanScene, knotScene, playgroundScene, type Scene } from './scenes';

const { renderer, scene, camera, controls } = stage();
renderer.toneMapping = THREE.ACESFilmicToneMapping;
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.35;
// the shadows are drawn with coarser tubes than the image, whose chords stand
// out of the fine tube inside a tight bend: a little bias keeps them off it
scene.traverse((o) => { if (o instanceof THREE.DirectionalLight) o.shadow.normalBias = 0.004; });
const labels = new CSS2DRenderer();
labels.setSize(innerWidth, innerHeight);
Object.assign(labels.domElement.style, { position: 'absolute', top: '0', pointerEvents: 'none' });
document.body.appendChild(labels.domElement);
window.addEventListener('resize', () => labels.setSize(innerWidth, innerHeight));

const material = ropeMaterial();
const metal = new THREE.MeshStandardMaterial({ color: '#8a939c', roughness: 0.35, metalness: 0.8 });
const gripMaterial = new THREE.MeshStandardMaterial({ color: '#ffb347', roughness: 0.4, emissive: '#7a3d00' });

const SCENES: Record<string, () => Scene> = {
  knot: () => knotScene(knotKind),
  capstan: capstanScene,
  braid: braidScene,
  playground: playgroundScene,
};
let knotKind: keyof typeof KNOTS = 'eight';

// --- what one scene is made of; rebuilt when another is chosen
let sim: CableSim, current: Scene, tubes: TubeMesh, nodes: TubeNodes, sync: TubeSync;
let frame = 0, held = -1;
const props = new THREE.Group();
const grips: THREE.Mesh[] = [];
scene.add(props);

function load(name: string): void {
  current = SCENES[name]();
  sim = new CableSim(current.params);
  const starts = sim.build(current.ropes.map((r) => ({ points: r.points, radius: r.radius, mass: 0.002 * (r.radius / 0.01) ** 2, bend: 0.5 })));
  sim.setStatics(current.statics);
  current.setup?.(sim, starts);
  frame = 0;
  held = -1;

  // (disposed, the old tubes leave the material to the new ones)
  if (tubes) { scene.remove(tubes); tubes.dispose(); }
  nodes = new TubeNodes(sim.n);
  tubes = new TubeMesh(sim.n - current.ropes.length, material, nodes, { radial: 24, sub: 8 });
  const color = new THREE.Color();
  let at = 0;
  current.ropes.forEach((r, c) => {
    const cb = sim.cables[c];
    tubes.lay(at, cb.start, cb.count, color.set(r.color).convertSRGBToLinear(), 0, sim.rest[cb.start], r.make);
    at += cb.count - 1;
  });
  tubes.castShadow = tubes.receiveShadow = true;
  scene.add(tubes);
  sync = new TubeSync(nodes, sim.radius);
  sync.update(sim, true);

  // the bars, pegs and ring; their labels
  for (const o of [...props.children]) {
    props.remove(o);
    if (o instanceof THREE.Mesh) o.geometry.dispose();
    if (o instanceof CSS2DObject) o.element.remove();
  }
  current.statics.slice(current.hidden).forEach((s) => {
    let mesh: THREE.Mesh;
    if (s.kind === 'capsule') {
      const a = new THREE.Vector3(s.ax, s.ay, s.az), b = new THREE.Vector3(s.bx, s.by, s.bz);
      mesh = new THREE.Mesh(new THREE.CapsuleGeometry(s.r, a.distanceTo(b), 6, 24), metal);
      mesh.position.copy(a).add(b).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.sub(a).normalize());
    } else {
      const size = s.max.map((v, k) => v - s.min[k]) as [number, number, number];
      mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), metal);
      mesh.position.set((s.min[0] + s.max[0]) / 2, (s.min[1] + s.max[1]) / 2, (s.min[2] + s.max[2]) / 2);
    }
    mesh.castShadow = mesh.receiveShadow = true;
    props.add(mesh);
  });
  for (const l of current.labels ?? []) {
    const div = document.createElement('div');
    div.className = 'label';
    div.innerHTML = l.text;
    const o = new CSS2DObject(div);
    o.position.set(...l.at);
    props.add(o);
  }

  camera.position.set(...current.camera.position);
  controls.target.set(...current.camera.target);
  controls.update();
  help.textContent = current.help;
  actions.replaceChildren(knotPicker, ...(current.actions ?? []).map((a) => button(a.label, () => a.run(sim))));
  knotPicker.style.display = name === 'knot' ? '' : 'none';
  for (const b of picker.children) b.classList.toggle('on', (b as HTMLElement).dataset.scene === name);
}

// --- the page around it
const picker = document.getElementById('scenes')!;
const actions = document.getElementById('actions')!;
const help = document.getElementById('help')!;
const stats = document.getElementById('stats')!;
const knotPicker = document.getElementById('knot') as HTMLSelectElement;
function button(label: string, run: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.textContent = label;
  b.onclick = run;
  return b;
}
let name = new URLSearchParams(location.search).get('scene') ?? 'knot';
if (!(name in SCENES)) name = 'knot';
for (const [key, label] of [['knot', 'knot'], ['capstan', 'capstan'], ['braid', 'braid'], ['playground', 'playground']]) {
  const b = button(label, () => { name = key; load(key); });
  b.dataset.scene = key;
  picker.appendChild(b);
}
picker.appendChild(button('restart', () => load(name)));
for (const [key, k] of Object.entries(KNOTS)) knotPicker.add(new Option(k.name, key, key === knotKind, key === knotKind));
knotPicker.onchange = () => { knotKind = knotPicker.value as keyof typeof KNOTS; load('knot'); };

// --- the hand: the nearest node to the pointer ray, led across a plane facing the camera
const pointer = new THREE.Vector2(), ray = new THREE.Raycaster(), plane = new THREE.Plane(), hit = new THREE.Vector3();
const p = new THREE.Vector3();
function aim(e: PointerEvent): void {
  pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(pointer, camera);
}
renderer.domElement.addEventListener('pointerdown', (e) => {
  aim(e);
  let best = 0.03;
  for (let i = 0; i < sim.n; i++) {
    if (sim.pinned[i]) continue;
    p.set(sim.x[3 * i], sim.x[3 * i + 1], sim.x[3 * i + 2]);
    const d = ray.ray.distanceToPoint(p) - sim.radius[i];
    if (d < best) { best = d; held = i; }
  }
  if (held < 0) return;
  controls.enabled = false;
  p.set(sim.x[3 * held], sim.x[3 * held + 1], sim.x[3 * held + 2]);
  plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(hit).negate(), p);
  hit.copy(p);
});
window.addEventListener('pointermove', (e) => {
  if (held < 0) return;
  aim(e);
  ray.ray.intersectPlane(plane, hit);
});
window.addEventListener('pointerup', () => {
  if (held < 0) return;
  sim.clearTarget(held);
  held = -1;
  controls.enabled = true;
});

// --- how well it holds: the worst stretch, the deepest overlap of two stretches of rope (of their thickness)
let worstStretch = 0, worstOverlap = 0, ms = 0, measured = 0;
function measure(): void {
  const x = sim.x;
  let st = 0, ov = 0;
  for (let a = 0; a < sim.n; a++) {
    if (!sim.segOK[a]) continue;
    const d = Math.hypot(x[3 * a + 3] - x[3 * a], x[3 * a + 4] - x[3 * a + 1], x[3 * a + 5] - x[3 * a + 2]);
    st = Math.max(st, d / sim.rest[a] - 1);
    for (let b = a + 3; b < sim.n; b++) {
      if (!sim.segOK[b]) continue;
      const rs = sim.radius[a] + sim.radius[b];
      const mx = x[3 * a] - x[3 * b], my = x[3 * a + 1] - x[3 * b + 1], mz = x[3 * a + 2] - x[3 * b + 2];
      if (mx * mx + my * my + mz * mz > (rs + 0.06) ** 2) continue;
      segSeg(x[3 * a], x[3 * a + 1], x[3 * a + 2], x[3 * a + 3], x[3 * a + 4], x[3 * a + 5],
        x[3 * b], x[3 * b + 1], x[3 * b + 2], x[3 * b + 3], x[3 * b + 4], x[3 * b + 5]);
      ov = Math.max(ov, 1 - Math.sqrt(closest.d2) / rs);
    }
  }
  worstStretch = Math.max(worstStretch * 0.97, st);
  worstOverlap = Math.max(worstOverlap * 0.97, ov);
}

load(name);
Object.assign(window, { example: { get sim() { return sim; }, get tubes() { return tubes; }, camera, load } });
renderer.setAnimationLoop(() => {
  const driven = current.drive?.(sim, frame) ?? [];
  if (held >= 0) {
    hit.y = Math.max(hit.y, sim.radius[held]);
    sim.setTarget(held, hit.x, hit.y, hit.z, 1.6, 0.04, true);
  }
  sim.step();
  frame++;
  ms += sim.stats.ms;
  sync.update(sim);
  // the hands: a grip on each node something holds
  const hands = held >= 0 ? [...driven, held] : driven;
  while (grips.length < hands.length) {
    const g = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), gripMaterial);
    g.castShadow = true;
    grips.push(g);
    scene.add(g);
  }
  grips.forEach((g, k) => {
    g.visible = k < hands.length;
    if (!g.visible) return;
    const i = hands[k];
    g.position.set(sim.x[3 * i], sim.x[3 * i + 1], sim.x[3 * i + 2]);
    g.scale.setScalar(sim.radius[i] * 1.6);
  });
  if (frame % 10 === 0) {
    measure();
    measured++;
    if (measured % 3 === 0) {
      stats.textContent = `${sim.n} nodes · ${(ms / 30).toFixed(2)} ms a step · worst stretch ${(100 * worstStretch).toFixed(1)} % · deepest overlap ${(100 * worstOverlap).toFixed(0)} % of the thickness`;
      ms = 0;
    }
  }
  renderer.render(scene, camera);
  labels.render(scene, camera);
});
