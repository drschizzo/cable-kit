// A fibre optic lamp, on WebGPU: 50 000 strands that rise from the lamp, bend
// under their weight and sway. Their nodes are computed every frame by a
// compute shader and written straight into the tubes' node map
// (`GPUTubeNodes`, `storeTubeNode`): they never leave the GPU, and the CPU
// does nothing per frame but ask for the compute and the image. Pulses of
// light run up the strands: the material makes them in TSL, from `tubeCoord`
// (where a fragment is along its strand, and the strand's tag: its phase) and
// the time, and the glow of each node (`tubeTint`) gives the strand its colour.
// A bloom pass, in TSL too, lets them shine.

import * as THREE from 'three/webgpu';
import { Fn, Loop, cos, exp, float, fract, hash, instanceIndex, mix, pass, sin, smoothstep, time, uint, vec3 } from 'three/tsl';
import { bloom } from 'three/examples/jsm/tsl/display/BloomNode.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GPUTubeNodes, TubeMesh, storeTubeNode, storeTubeNodeTint, tubeCoord, tubeTint } from '@drschizzo/three-cable-tubes/webgpu';

// ?strands=, ?nodes= (per strand), ?radial= (vertices round a ring), ?lod= (radius on screen, px, under which a fibre keeps a sixteenth of its vertices), ?bloom=0
const query = new URLSearchParams(location.search);
const STRANDS = Number(query.get('strands')) || 50000, NODES = Number(query.get('nodes')) || 24, LENGTH = 0.55, SEG = LENGTH / (NODES - 1), RADIUS = 0.0006, BASE = 0.3;

const renderer = new THREE.WebGPURenderer({ antialias: true, trackTimestamp: true });
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

// --- the strands, computed on the GPU: one invocation per node, so that
// neighbouring invocations write neighbouring texels; each finds where it is
// by walking up its strand from the lamp
const nodes = new GPUTubeNodes(STRANDS * NODES);
nodes.box.set(new THREE.Vector3(-0.8, -0.1, -0.8), new THREE.Vector3(0.8, 1, 0.8));
nodes.radius = RADIUS;

/** what is the strand's own: where it heads (round the lamp), how steeply it rises, how much it bends, how it sways, its colour */
function strand(s: any) {
  const h1 = hash(s), h2 = hash(s.add(7919)), h3 = hash(s.add(104729));
  return {
    heading: h1.mul(Math.PI * 2), rise: mix(0.55, 1.45, h2.sqrt()), bend: mix(2.5, 6.0, h3),
    phase: h2.mul(6.283), speed: mix(0.8, 1.4, h3),
    colour: mix(vec3(0.05, 0.35, 1.0), vec3(1.0, 0.2, 0.75), h1).mul(mix(0.6, 1.0, h2)),
  };
}

/** heading round the lamp and slope of a strand at u (0 at the lamp, 1 at the tip): waves run up it, growing towards the tip */
function bearing(st: ReturnType<typeof strand>, u: any): { az: any; angle: any } {
  const az = st.heading.add(sin(time.mul(st.speed).sub(u.mul(5)).add(st.phase)).mul(u).mul(0.3));
  const angle = st.rise.sub(st.bend.mul(u).mul(u).mul(LENGTH)).add(sin(time.mul(st.speed.mul(0.8)).sub(u.mul(4)).add(st.phase.mul(1.7))).mul(u).mul(0.18));
  return { az, angle };
}

const strands = Fn(() => {
  const s = instanceIndex.div(NODES), i = instanceIndex.mod(NODES);
  const st = strand(s);
  const p = vec3(0, BASE, 0).toVar();
  Loop({ start: uint(0), end: i, type: 'uint', condition: '<' }, ({ i: k }: { i: any }) => {
    const { az, angle } = bearing(st, float(k).div(NODES - 1));
    p.addAssign(vec3(cos(az).mul(cos(angle)), sin(angle), sin(az).mul(cos(angle))).mul(SEG));
  });
  const u = float(i).div(NODES - 1);
  const { az } = bearing(st, u);
  // square to the strand there, which lies in the plane of its heading and up
  const side = vec3(sin(az).negate(), 0, cos(az));
  storeTubeNode(nodes, instanceIndex, p, float(RADIUS).mul(u.mul(-0.4).add(1)), side);
})().compute(STRANDS * NODES);

// the glow does not change: written once. Dim along the strand, bright at its tip, as fibre optics are
const glows = Fn(() => {
  const s = instanceIndex.div(NODES), u = float(instanceIndex.mod(NODES)).div(NODES - 1);
  storeTubeNodeTint(nodes, instanceIndex, strand(s).colour.mul(smoothstep(0.9, 1, u).mul(1.2).add(0.006)));
})().compute(STRANDS * NODES);
renderer.compute(glows);

// --- the tubes: a dark fibre, its own glow, and pulses of light running up it
const material = new THREE.MeshStandardNodeMaterial({ color: '#3a3f48', roughness: 0.5 });
const along = tubeCoord.x.div(LENGTH);
const pulse = exp(along.sub(fract(time.mul(0.12).add(tubeCoord.w))).mul(10).pow(2).negate());
material.emissiveNode = tubeTint.add(tubeTint.mul(pulse).mul(12));
const tubes = new TubeMesh(STRANDS * (NODES - 1), material, nodes, { radial: Number(query.get('radial')) || 8, sub: 4, lodQuarter: Number(query.get('lod')) || 1.5, lodHalf: 4 * (Number(query.get('lod')) || 1.5) });
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
scene.add(new THREE.HemisphereLight('#8090a0', '#101010', 1.0));
const glowLight = new THREE.PointLight('#8a6cff', 1.0, 3);
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
pipeline.outputNode = query.get('bloom') === '0' ? image : image.add(bloom(image, 0.25, 0.2, 0.9));

info.textContent = `${STRANDS.toLocaleString('en')} strands, ${(STRANDS * NODES).toLocaleString('en')} nodes computed on the GPU`;
const cpu = document.getElementById('cpu')!;
let worst = 0, last = performance.now();
Object.assign(window, { example: { renderer, scene, camera, tubes, nodes, strands, material, tubeTint } });
renderer.setAnimationLoop(() => {
  const t0 = performance.now();
  renderer.compute(strands);
  pipeline.render();
  worst = Math.max(worst, performance.now() - t0);
  if (t0 - last > 500) {
    const w = worst;
    worst = 0; last = t0;
    // what the GPU spent, where the browser can time it
    Promise.all([renderer.resolveTimestampsAsync('compute'), renderer.resolveTimestampsAsync('render')]).then(([c, r]) => {
      const gpu = typeof c === 'number' && typeof r === 'number' ? ` · GPU: strands ${c.toFixed(2)} ms, image ${r.toFixed(2)} ms` : '';
      cpu.textContent = `CPU per frame ${w.toFixed(2)} ms${gpu}`;
    });
  }
});
