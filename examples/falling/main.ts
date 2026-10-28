// A hundred ropes falling in a heap. The solver runs in a Web Worker
// (`threaded`): the page keeps a mirror of it, sends it the changes and asks
// for a step every frame, and draws the last positions it got back. Rendering
// never waits for the physics. Ropes at rest fall asleep and cost nothing.

import * as THREE from 'three';
import { CABLES, CableSim, threaded, type CableInit } from '@drschizzo/avbd-cables';
import { TubeMesh, TubeNodes, TubeSync } from '@drschizzo/three-cable-tubes';
import { line, random, stage } from '../common';

class Sim extends threaded(CableSim, () => new Worker(new URL('./sim.worker.ts', import.meta.url), { type: 'module' })) {}

const ROPES = 120, LENGTH = 1.2;
const kind = CABLES.rope;
const { renderer, scene, camera } = stage([0, 0.1, 0]);
camera.position.set(2.2, 1.7, 2.6);

/** the ropes, crossed above the floor in a tall stack */
function ropes(): number[] {
  const r = random(), out: number[] = [];
  for (let c = 0; c < ROPES; c++) {
    const a = r() * Math.PI, h = 0.1 + c * 0.03, cx = (r() - 0.5) * 2.4, cz = (r() - 0.5) * 2.4;
    const d = [Math.cos(a) * LENGTH / 2, Math.sin(a) * LENGTH / 2];
    out.push(...line([cx - d[0], h, cz - d[1]], [cx + d[0], h, cz + d[1]], kind.segment));
  }
  return out;
}

const sim = new Sim(kind.params);
const start = ropes(), per = start.length / 3 / ROPES;
const defs: CableInit[] = [];
for (let c = 0; c < ROPES; c++) defs.push({ ...kind.cable, points: start.slice(3 * per * c, 3 * per * (c + 1)) });
sim.build(defs);
sim.setStatics([{ kind: 'box', min: [-3, -0.2, -3], max: [3, 0, 3] }]);
sim.startWorker();

const nodes = new TubeNodes(sim.n);
const tubes = new TubeMesh(sim.n - ROPES, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7 }), nodes);
const r = random(3), color = new THREE.Color();
sim.cables.forEach((cb, c) => tubes.lay(c * (per - 1), cb.start, cb.count, color.setHSL(r(), 0.5, 0.5), 0, kind.segment, 0));
tubes.castShadow = tubes.receiveShadow = true;
scene.add(tubes);
const sync = new TubeSync(nodes, kind.cable.radius);

document.getElementById('again')!.addEventListener('click', () => sim.teleport(start));
const stats = document.getElementById('stats')!;
Object.assign(window, { example: { sim, tubes, camera } });

let last = performance.now();
renderer.setAnimationLoop((now) => {
  // one frame of physics per frame drawn, at most two behind (time is dropped rather than queued)
  sim.advance(1);
  sync.update(sim);
  renderer.render(scene, camera);
  if (now - last > 500) {
    last = now;
    const s = sim.stats;
    stats.textContent = `${sim.n} nodes · ${s.awake}/${ROPES} awake · ${s.contacts} contacts · physics ${s.ms.toFixed(1)} ms${sim.threaded ? ' (worker)' : ''}`;
  }
});
