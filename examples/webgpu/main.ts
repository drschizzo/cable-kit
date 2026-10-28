// A fibre optic lamp, on WebGPU: 16 000 strands that rise from the lamp, bend
// under their weight and sway. Their nodes are computed every frame by a
// compute shader and written straight into the tubes' node map
// (`GPUTubeNodes`, `storeTubeNode`): they never leave the GPU, and the CPU
// does nothing per frame but ask for the compute and the image. Pulses of
// light run up the strands: the material makes them in TSL, from `tubeCoord`
// (where a fragment is along its strand, and the strand's tag: its phase) and
// the time, and the glow of each node (`tubeTint`) gives the strand its colour.
// A bloom pass, in TSL too, lets them shine.

import * as THREE from 'three/webgpu';
import { Fn, Loop, cos, exp, float, fract, hash, instanceIndex, mix, pass, sin, smoothstep, time, vec3 } from 'three/tsl';
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GPUTubeNodes, TubeMesh, storeTubeNode, tubeCoord, tubeTint } from '@drschizzo/three-cable-tubes/webgpu';

const STRANDS = 16000, NODES = 24, LENGTH = 0.55, SEG = LENGTH / (NODES - 1), RADIUS = 0.0006, BASE = 0.3;

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);
await renderer.init();
const info = document.getElementById('backend')!;
if (!(renderer.backend as unknown as { isWebGPUBackend?: boolean }).isWebGPUBackend) {
  info.textContent = 'this example needs WebGPU (compute shaders), which this browser does not offer';
  throw new Error('no WebGPU');
}

// --- the strands, computed on the GPU: one invocation per strand, walking up it
const nodes = new GPUTubeNodes(STRANDS * NODES);
nodes.box.set(new THREE.Vector3(-0.8, -0.1, -0.8), new THREE.Vector3(0.8, 1, 0.8));
nodes.radius = RADIUS;
const strands = Fn(() => {
  const id = instanceIndex;
  const h1 = hash(id), h2 = hash(id.add(7919)), h3 = hash(id.add(104729));
  // where it heads (round the lamp), how steeply it rises, how much it bends
  const az = h1.mul(Math.PI * 2).add(sin(time.mul(0.5).add(h2.mul(6.283))).mul(0.06)).toVar();
  const rise = mix(0.55, 1.45, h2.sqrt()).add(sin(time.mul(0.9).add(h3.mul(6.283))).mul(0.04));
  const bend = mix(2.5, 6.0, h3);
  const out = vec3(cos(az), 0, sin(az));
  // square to every tangent of the strand, which stays in the plane of `out` and up
  const side = vec3(sin(az).negate(), 0, cos(az));
  const colour = mix(vec3(0.05, 0.35, 1.0), vec3(1.0, 0.2, 0.75), h1).mul(mix(0.6, 1.0, h2));
  const p = vec3(0, BASE, 0).toVar();
  Loop(NODES, ({ i }: { i: any }) => {
    const u = float(i).div(NODES - 1);
    const angle = rise.sub(bend.mul(u).mul(u).mul(LENGTH));
    const dir = out.mul(cos(angle)).add(vec3(0, 1, 0).mul(sin(angle)));
    // dim along the strand, bright at its tip, as fibre optics are
    const glow = colour.mul(smoothstep(0.9, 1, u).mul(2.5).add(0.012));
    storeTubeNode(nodes, id.mul(NODES).add(i), p, float(RADIUS).mul(u.mul(-0.4).add(1)), side, glow);
    p.addAssign(dir.mul(SEG));
  });
})().compute(STRANDS);

// --- the tubes: a dark fibre, its own glow, and pulses of light running up it
const material = new THREE.MeshStandardNodeMaterial({ color: '#0a0b0d', roughness: 0.4 });
const along = tubeCoord.x.div(LENGTH);
const pulse = exp(along.sub(fract(time.mul(0.35).add(tubeCoord.w))).mul(14).pow(2).negate());
material.emissiveNode = tubeTint.add(tubeTint.mul(pulse).mul(40));
const tubes = new TubeMesh(STRANDS * (NODES - 1), material, nodes, { radial: 8, sub: 4 });
const dark = new THREE.Color(0, 0, 0);
for (let s = 0; s < STRANDS; s++) tubes.lay(s * (NODES - 1), s * NODES, NODES, dark, 0, SEG, Math.random());

// --- the room: the lamp's base, a floor, little light
const scene = new THREE.Scene();
scene.background = new THREE.Color('#050506');
scene.add(tubes);
const metal = new THREE.MeshStandardNodeMaterial({ color: '#30343a', roughness: 0.35, metalness: 0.8 });
const base = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.09, BASE, 48), metal);
base.position.y = BASE / 2;
scene.add(base);
const floor = new THREE.Mesh(new THREE.CircleGeometry(3, 64), new THREE.MeshStandardNodeMaterial({ color: '#141518', roughness: 0.8 }));
floor.rotation.x = -Math.PI / 2;
scene.add(floor);
scene.add(new THREE.HemisphereLight('#8090a0', '#101010', 0.4));
const glowLight = new THREE.PointLight('#8a6cff', 0.6, 3);
glowLight.position.set(0, 0.6, 0);
scene.add(glowLight);

const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.02, 30);
camera.position.set(1.05, 0.75, 1.25);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.42, 0);
controls.update();
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// --- the image, and its bloom
const pipeline = new THREE.RenderPipeline(renderer);
const image = pass(scene, camera).getTextureNode('output');
pipeline.outputNode = image.add(bloom(image, 0.6, 0.3, 0.55));

info.textContent = `${STRANDS.toLocaleString('en')} strands, ${(STRANDS * NODES).toLocaleString('en')} nodes computed on the GPU`;
const cpu = document.getElementById('cpu')!;
let worst = 0, last = performance.now();
Object.assign(window, { example: { renderer, scene, camera, tubes, nodes, strands, material, tubeTint } });
renderer.setAnimationLoop(() => {
  const t0 = performance.now();
  renderer.compute(strands);
  pipeline.render();
  worst = Math.max(worst, performance.now() - t0);
  if (t0 - last > 500) { cpu.textContent = `CPU per frame ${worst.toFixed(2)} ms`; worst = 0; last = t0; }
});
