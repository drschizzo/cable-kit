// Ropes you take with the mouse. The node under the pointer is led by a heavy
// target (a spring on a leash, as a hand): the rope follows it where it can,
// held back by the rod, the floor and the other ropes, and never stretches
// like an elastic. One end of each rope is pinned to the post.

import * as THREE from 'three';
import { CABLES, CableSim, type CableInit, type StaticShape } from '@drschizzo/avbd-cables';
import { TubeMesh, TubeNodes, TubeSync } from '@drschizzo/three-cable-tubes';
import { line, random, stage } from '../common';

const kind = CABLES.rope;
const { renderer, scene, camera, controls } = stage();

// --- the ropes: pinned to a post at one end, lying over a rod
const sim = new CableSim(kind.params);
const defs: CableInit[] = [];
for (let c = 0; c < 8; c++) {
  const z = -0.28 + c * 0.08;
  defs.push({ ...kind.cable, points: line([-0.6, 0.5, z], [0.6, 0.5, z], kind.segment) });
}
const starts = sim.build(defs);
starts.forEach((s) => sim.pin(s, sim.x[3 * s], sim.x[3 * s + 1], sim.x[3 * s + 2]));
const rod: StaticShape = { kind: 'capsule', ax: 0, ay: 0.3, az: -0.5, bx: 0, by: 0.3, bz: 0.5, r: 0.015 };
const post: StaticShape = { kind: 'box', min: [-0.66, 0, -0.35], max: [-0.62, 0.5, 0.35] };
sim.setStatics([{ kind: 'box', min: [-3, -0.2, -3], max: [3, 0, 3] }, rod, post]);

// --- what you see of them: the tubes, the rod, the post
const nodes = new TubeNodes(sim.n);
const tubes = new TubeMesh(sim.n - defs.length, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7 }), nodes);
const r = random(), color = new THREE.Color();
let at = 0;
starts.forEach((s, c) => {
  tubes.lay(at, s, sim.cables[c].count, color.setHSL(r(), 0.55, 0.5), 0, kind.segment, 0);
  at += sim.cables[c].count - 1;
});
tubes.castShadow = tubes.receiveShadow = true;
scene.add(tubes);
const metal = new THREE.MeshStandardMaterial({ color: '#8a939c', roughness: 0.4, metalness: 0.6 });
const rodMesh = new THREE.Mesh(new THREE.CylinderGeometry(rod.r, rod.r, 1, 24), metal);
rodMesh.rotation.x = Math.PI / 2;
rodMesh.position.set(0, 0.3, 0);
const postMesh = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.5, 0.7), metal);
postMesh.position.set(-0.64, 0.25, 0);
for (const m of [rodMesh, postMesh]) { m.castShadow = m.receiveShadow = true; scene.add(m); }
const sync = new TubeSync(nodes, kind.cable.radius);

// --- the hand: the nearest node to the pointer ray, led across a plane facing the camera
const pointer = new THREE.Vector2(), ray = new THREE.Raycaster(), plane = new THREE.Plane(), hit = new THREE.Vector3();
let held = -1;
const glow = new THREE.Color(0.35, 0.15, 0), dark = new THREE.Color(0, 0, 0);
function aim(e: PointerEvent): void {
  pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(pointer, camera);
}
function tint(i: number, c: THREE.Color): void {
  const cb = sim.cables[sim.cableOf[i]];
  for (let k = cb.start; k < cb.start + cb.count; k++) nodes.tint(k, c);
}
renderer.domElement.addEventListener('pointerdown', (e) => {
  aim(e);
  let best = 0.03, p = new THREE.Vector3();
  for (let i = 0; i < sim.n; i++) {
    if (sim.pinned[i]) continue;
    p.set(sim.x[3 * i], sim.x[3 * i + 1], sim.x[3 * i + 2]);
    const d = ray.ray.distanceToPoint(p);
    if (d < best) { best = d; held = i; }
  }
  if (held < 0) return;
  controls.enabled = false;
  p = new THREE.Vector3(sim.x[3 * held], sim.x[3 * held + 1], sim.x[3 * held + 2]);
  plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()).negate(), p);
  tint(held, glow);
});
window.addEventListener('pointermove', (e) => {
  if (held < 0) return;
  aim(e);
  if (!ray.ray.intersectPlane(plane, hit)) return;
  hit.y = Math.max(hit.y, kind.cable.radius);
  sim.setTarget(held, hit.x, hit.y, hit.z, 1.6, 0.06, true);
});
window.addEventListener('pointerup', () => {
  if (held < 0) return;
  sim.clearTarget(held);
  tint(held, dark);
  held = -1;
  controls.enabled = true;
});

Object.assign(window, { example: { sim, tubes, camera } });
renderer.setAnimationLoop(() => {
  sim.step();
  sync.update(sim);
  renderer.render(scene, camera);
});
