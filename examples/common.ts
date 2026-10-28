// What the WebGL examples share: a renderer, a camera you orbit, two lights
// with shadows, a floor, and cables laid along a straight line.

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export function stage(target: [number, number, number] = [0, 0.15, 0]) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#101214');
  scene.add(new THREE.HemisphereLight('#cfd8e3', '#202326', 1.2));
  const sun = new THREE.DirectionalLight('#ffffff', 2.5);
  sun.position.set(1.5, 3, 1);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -1.5, right: 1.5, top: 1.5, bottom: -1.5, near: 0.5, far: 6 });
  scene.add(sun);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshStandardMaterial({ color: '#3a3f45', roughness: 0.9 }));
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, 0.02, 40);
  camera.position.set(1.3, 1, 1.5);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(...target);
  controls.update();
  window.addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });
  return { renderer, scene, camera, controls };
}

/** nodes from a to b, `seg` apart (xyz interleaved) */
export function line(a: number[], b: number[], seg: number): number[] {
  const n = Math.max(2, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / seg) + 1);
  const pts: number[] = [];
  for (let k = 0; k < n; k++) {
    const t = k / (n - 1);
    pts.push(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t);
  }
  return pts;
}

/** a small seeded random generator, so that every run looks the same */
export function random(seed = 7): () => number {
  let s = seed;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}
