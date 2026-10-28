// The tubes on their own: thousands of curves animated by hand on the CPU,
// only their nodes sent to the GPU every frame, which sweeps the tubes and
// thins the far ones (levels of detail). No physics at all.

import * as THREE from 'three';
import { TubeMesh, TubeNodes } from '@drschizzo/three-cable-tubes';
import { random, stage } from '../common';

const CURVES = 3000, NODES = 24, SEG = 0.04;
const { renderer, scene, camera } = stage([0, 0.4, 0]);
camera.position.set(3.4, 2.3, 3.9);

const nodes = new TubeNodes(CURVES * NODES);
const tubes = new TubeMesh(CURVES * (NODES - 1), new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 }), nodes);
const r = random(), color = new THREE.Color();
// each curve: where it stands, how it waves (phase, speed), how thick it is
const base = new Float32Array(3 * CURVES), wave = new Float32Array(2 * CURVES), thick = new Float32Array(CURVES);
for (let c = 0; c < CURVES; c++) {
  const a = r() * Math.PI * 2, d = Math.sqrt(r()) * 1.4;
  base.set([Math.cos(a) * d, 0, Math.sin(a) * d], 3 * c);
  wave.set([r() * Math.PI * 2, 0.5 + r()], 2 * c);
  thick[c] = 0.002 + r() * 0.004;
  tubes.lay(c * (NODES - 1), c * NODES, NODES, color.setHSL(0.5 + r() * 0.3, 0.5, 0.45 + r() * 0.2), 0, SEG, 0);
}
tubes.castShadow = tubes.receiveShadow = true;
scene.add(tubes);

// a radius per node: each curve tapers to its tip
const radius = new Float32Array(CURVES * NODES);
for (let c = 0; c < CURVES; c++) for (let k = 0; k < NODES; k++) radius[c * NODES + k] = thick[c] * (1 - 0.7 * (k / NODES));
const x = new Float32Array(3 * CURVES * NODES);
const stats = document.getElementById('stats')!;

renderer.setAnimationLoop((now) => {
  const t = now / 1000;
  // grass in the wind: each curve rises from its base and bends more towards its tip
  for (let c = 0; c < CURVES; c++) {
    const ph = wave[2 * c], sp = wave[2 * c + 1];
    let px = base[3 * c], py = 0, pz = base[3 * c + 2];
    for (let k = 0; k < NODES; k++) {
      const i = 3 * (c * NODES + k), u = k / NODES;
      x[i] = px; x[i + 1] = py; x[i + 2] = pz;
      const bend = u * 0.9 * Math.sin(t * sp + ph + px * 1.5);
      px += SEG * Math.sin(bend);
      pz += SEG * 0.3 * Math.sin(bend * 0.7 + ph);
      py += SEG * Math.cos(bend);
    }
  }
  nodes.write(x, 0, CURVES * NODES, radius);
  renderer.render(scene, camera);
  stats.textContent = `${CURVES} curves · ${CURVES * NODES} nodes · level of detail ${tubes.level}`;
});
