// The tubes for WebGPURenderer (three/webgpu), in TSL: the same nodes, the
// same stretch of tube instanced on every segment, the same levels of detail
// as tube.ts, which does it for WebGLRenderer in GLSL. WebGPURenderer runs it
// on WebGPU, or on WebGL 2 where there is none.
//
//   import { TubeMesh, TubeNodes, tubeCoord, tubeTint } from '@drschizzo/three-cable-tubes/webgpu';
//   const nodes = new TubeNodes(nodeCount);
//   const material = new THREE.MeshStandardNodeMaterial();
//   material.emissiveNode = tubeTint;                 // the glow of the nodes
//   const tubes = new TubeMesh(segmentCount, material, nodes);
//
// Any node material draws them: the mesh sets its `positionNode`, which also
// sets the normal. `tubeCoord` (where the fragment is along the cord at rest,
// cos and sin of its angle round it, the cord's tag) and `tubeTint` are
// varyings its other nodes may read. The shadows and the passes of the post
// processing draw the same material, so nothing else needs it.
//
// The nodes may also be computed on the GPU, and never leave it: a compute
// shader writes them into `GPUTubeNodes` with `storeTubeNode`, and the mesh
// reads them from there (WebGPU only: WebGL 2 has no compute shaders).

import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { uniform, varyingProperty } from 'three/tsl';
import { PER, W, resolve, stretch, type TubeOptions } from './shared.js';

export { TubeNodes, TubeSync, type CableSource, type TubeOptions } from './shared.js';

/** where a fragment is along its cord at rest (m), cos and sin of its angle round it, the cord's tag (see `lay`) */
export const tubeCoord = varyingProperty('vec4', 'tubeCoord');
/** the glow of the node a fragment is nearest to (linear RGB) */
export const tubeTint = varyingProperty('vec3', 'tubeTint');

// (three's TSL types reject the mixes of int and float the shader needs: it is checked when it compiles)
const {
  Fn, Loop, abs, attribute, cameraPosition, cos, cross, dot, float, instanceIndex, int, ivec2, max, mix, modelViewMatrix,
  modelWorldMatrixInverse, normalLocal, normalize, select, sin, textureLoad, textureStore, uint, vec3, vec4,
} = TSL as any;

/** what a TubeMesh reads its nodes from: TubeNodes (written by the CPU) or GPUTubeNodes */
export interface TubeNodeSource {
  readonly texture: THREE.Texture;
  /** texels per row of the texture (TubeNodes: 1024) */
  readonly width?: number;
  /** where the tubes may be, and their largest radius: the levels of detail go by them */
  readonly box: THREE.Box3;
  radius: number;
  dispose(): void;
}

/**
 * Nodes the GPU writes itself, from a compute shader (`storeTubeNode`), in
 * the layout of TubeNodes. The CPU never sees them: set `box` and `radius` to
 * where the tubes may be and how thick they are at most, for the levels of
 * detail.
 */
export class GPUTubeNodes implements TubeNodeSource {
  readonly texture: THREE.StorageTexture;
  /** texels per row: wider when there are many nodes, so that the texture stays within 8192 rows (WebGPU's default limit) */
  readonly width: number;
  readonly box = new THREE.Box3();
  radius = 0;

  constructor(nodes: number) {
    let w = W;
    while (Math.ceil((PER * nodes) / w) > 8192 && w < 8192) w *= 2;
    this.width = w;
    const h = Math.max(1, Math.ceil((PER * nodes) / w));
    this.texture = new THREE.StorageTexture(w, h);
    this.texture.type = THREE.FloatType;
    this.texture.magFilter = this.texture.minFilter = THREE.NearestFilter;
  }

  dispose(): void {
    this.texture.dispose();
  }
}

/** where texel `t` of a node map `w` texels wide is */
const texel = (t: any, w: number) => ivec2(t.mod(w), t.div(w));

/**
 * In a compute shader (TSL), writes node `i` of `nodes`: its position and
 * the radius of the tube there (0 hides it), the normal of its frame (square
 * to the cord, and carried along it so that the rings do not twist), and,
 * if given, its glow (linear RGB, `tubeTint`). A glow that does not change
 * is better written once, with `storeTubeNodeTint`: the writes are most of
 * what such a compute shader costs.
 */
export function storeTubeNode(nodes: GPUTubeNodes, i: any, position: any, radius: any, normal: any, tint?: any): void {
  const k = int(i).mul(PER);
  textureStore(nodes.texture, texel(k, nodes.width), vec4(position, radius)).toWriteOnly();
  textureStore(nodes.texture, texel(k.add(1), nodes.width), vec4(normal, 0)).toWriteOnly();
  if (tint !== undefined) storeTubeNodeTint(nodes, i, tint);
}

/** In a compute shader (TSL), writes the glow of node `i` of `nodes` (linear RGB, `tubeTint`). */
export function storeTubeNodeTint(nodes: GPUTubeNodes, i: any, tint: any): void {
  textureStore(nodes.texture, texel(int(i).mul(PER).add(2), nodes.width), vec4(tint, 0)).toWriteOnly();
}

/**
 * Strands for `sweepStrands`: `count` of them, of `nodes` nodes `segment`
 * metres apart. The functions are TSL, of the strand's index `s` (uint) and
 * of where along it `u` is (0 at its root, 1 at its tip).
 */
export interface Strands {
  count: number;
  nodes: number;
  segment: number;
  /** where strand s starts */
  root(s: any): any;
  /** which way strand s goes at u (a unit vector): it is laid along it, `segment` by `segment` */
  direction(s: any, u: any): any;
  /** its radius at u (m) */
  radius(s: any, u: any): any;
  /** its glow at u (linear RGB), written once (`glow`), not every frame */
  tint?(s: any, u: any): any;
}

/**
 * Compute shaders that write strands into `nodes` (strand s from node
 * s × strands.nodes): `update`, to run every frame (or whenever they move),
 * and `glow`, to run once when there is a tint. One invocation per node, so
 * that neighbouring invocations write neighbouring texels; each walks up its
 * strand from the root to find where it is, and carries the frame of the
 * tube along it (square to the strand, without twisting) on the way.
 */
export function sweepStrands(nodes: GPUTubeNodes, strands: Strands): { update: THREE.ComputeNode; glow: THREE.ComputeNode | null } {
  const N = strands.nodes, total = strands.count * N;
  const update = Fn(() => {
    const s = instanceIndex.div(N), i = instanceIndex.mod(N);
    const p = strands.root(s).toVar();
    const t0 = strands.direction(s, float(0));
    // a first normal square to the strand at its root, then carried up it
    const n = normalize(cross(t0, select(abs(t0.y).lessThan(0.9), vec3(0, 1, 0), vec3(1, 0, 0)))).toVar();
    Loop({ start: uint(0), end: i, type: 'uint', condition: '<' }, ({ i: k }: { i: any }) => {
      const t = strands.direction(s, float(k).div(N - 1));
      p.addAssign(t.mul(strands.segment));
      const next = strands.direction(s, float(k.add(1)).div(N - 1));
      n.assign(normalize(n.sub(next.mul(dot(n, next)))));
    });
    storeTubeNode(nodes, instanceIndex, p, strands.radius(s, float(i).div(N - 1)), n);
  })().compute(total);
  const tint = strands.tint;
  const glow = tint
    ? Fn(() => {
      const s = instanceIndex.div(N), u = float(instanceIndex.mod(N)).div(N - 1);
      storeTubeNodeTint(nodes, instanceIndex, tint(s, u));
    })().compute(total)
    : null;
  return { update, glow };
}

/** lays every strand of `strands` on `mesh` (see TubeMesh.lay): its colour and tag, by strand */
export function layStrands(mesh: TubeMesh, strands: Pick<Strands, 'count' | 'nodes' | 'segment'>, color: (s: number) => THREE.Color, tag: (s: number) => number = () => 0): void {
  const N = strands.nodes;
  for (let s = 0; s < strands.count; s++) mesh.lay(s * (N - 1), s * N, N, color(s), 0, strands.segment, tag(s));
}

const sphere = new THREE.Sphere();
const eye = new THREE.Vector3();
const size = new THREE.Vector2();

/** the mesh each material draws the tubes of (its `positionNode` reads that mesh's nodes) */
const drawing = new WeakMap<THREE.NodeMaterial, TubeMesh>();

/**
 * The mesh of the tubes: `segments` instances of the stretch of tube, laid
 * along the cords by lay(). `options.aliases` does not apply here: read
 * `tubeCoord` and `tubeTint`.
 *
 * Its material draws its nodes only: given a material that another mesh
 * draws (and was not disposed), it draws a copy of it instead (its
 * `material`). Once disposed, a mesh leaves its material free for the next one.
 */
export class TubeMesh extends THREE.Mesh {
  readonly isTubeMesh = true;
  declare geometry: THREE.InstancedBufferGeometry;
  /** the triangle lists of the stretch, finest first */
  private lists: THREE.BufferAttribute[];
  /** per segment: first node, first and last of its cord; colour; start along the cord (m), length, tag */
  private seg: THREE.InstancedBufferAttribute;
  private color: THREE.InstancedBufferAttribute;
  private arc: THREE.InstancedBufferAttribute;
  /** pixels on screen per metre at a depth of 1 m, for the camera being drawn from */
  private pxPerMetre = uniform(500);
  /** the finest list the nearest node needs from where the camera stands (0, 1, 2), as of the last image */
  level = 0;
  private options: Required<TubeOptions>;
  /** a shadow is being drawn: its renderObject calls onBeforeRender too, after onBeforeShadow */
  private shadowing = false;
  /** the material it draws with, and the copy it made of the one it was given, if it made one */
  private main: THREE.NodeMaterial;
  private ownCopy: THREE.NodeMaterial | null;

  constructor(segments: number, material: THREE.NodeMaterial, private nodes: TubeNodeSource, options: TubeOptions = {}) {
    const o = resolve(options);
    const geo = new THREE.InstancedBufferGeometry();
    const { rings, lists } = stretch(o.radial, o.sub);
    geo.setAttribute('normal', rings);
    const seg = new THREE.InstancedBufferAttribute(new Float32Array(3 * segments), 3);
    const color = new THREE.InstancedBufferAttribute(new Float32Array(3 * segments), 3);
    const arc = new THREE.InstancedBufferAttribute(new Float32Array(3 * segments), 3);
    geo.setAttribute('tubeSeg', seg);
    geo.setAttribute('color', color);
    geo.setAttribute('tubeArc', arc);
    geo.setIndex(lists[0]);
    geo.instanceCount = segments;
    const taken = drawing.has(material);
    const main = taken ? material.clone() : material;
    super(geo, main);
    drawing.set(main, this);
    this.main = main;
    this.ownCopy = taken ? main : null;
    this.options = o;
    this.lists = lists;
    this.seg = seg;
    this.color = color;
    this.arc = arc;
    main.positionNode = this.sweep();
    // (a material another mesh drew before: built again for these nodes)
    main.needsUpdate = true;
    this.frustumCulled = false;
    // (typed for WebGLRenderer by three's types: here, a WebGPURenderer)
    this.onBeforeRender = (r, _s, c) => {
      if (this.shadowing) { this.shadowing = false; return; }
      this.measure(r as unknown as THREE.Renderer, c);
      this.level = this.levelFor(c);
      this.lod(this.level);
    };
    this.onBeforeShadow = (r, _o, _c, shadowCamera) => {
      this.measure(r as unknown as THREE.Renderer, shadowCamera);
      this.lod(this.level + 1);
      this.shadowing = true;
    };
    // (the shadow is drawn in the middle of drawing the image: give it back its list)
    this.onAfterShadow = () => this.lod(this.level);
  }

  /** the point of the vertex on its tube, setting its normal, `tubeCoord` and `tubeTint` on the way (see tube.ts) */
  private sweep() {
    const { sub: SUB, radial: RADIAL, lodHalf, lodQuarter } = this.options;
    const tex = this.nodes.texture, px = this.pxPerMetre;
    const w = this.nodes.width ?? W;
    const at = (k: any): any => textureLoad(tex, ivec2(k.mod(w), k.div(w)));
    return Fn(() => {
      const s = attribute('tubeSeg', 'vec3'), ring = attribute('normal', 'vec3'), a = attribute('tubeArc', 'vec3');
      const i1 = int(s.x.add(0.5)).toVar(), iA = int(s.y.add(0.5)), iB = int(s.z.add(0.5));
      // (not min/max: TSL makes them float, which GLSL does not mix with int)
      const below = (n: any, lo: any) => select(n.lessThan(lo), lo, n), above = (n: any, hi: any) => select(n.greaterThan(hi), hi, n);
      const i0 = below(i1.sub(1), iA), i2 = above(i1.add(1), iB), i3 = above(i1.add(2), iB);
      const k = int(ring.x.add(0.5)).toVar(), r = int(ring.y.add(0.5)).toVar();
      const n1 = at(i1.mul(PER)), n2 = at(i2.mul(PER));
      // radius on screen (px), from the node of the ring at either end (the ring both segments
      // draw folds the same way in both), from the first node elsewhere
      const end = select(k.equal(SUB), n2, n1);
      const depth = max(float(1e-4), modelViewMatrix.mul(vec4(end.xyz, 1)).z.negate());
      const onScreen = end.w.mul(px).div(depth);
      const step = select(onScreen.lessThan(lodQuarter), int(4), select(onScreen.lessThan(lodHalf), int(2), int(1)));
      k.assign(k.sub(k.mod(step)));
      r.assign(r.sub(r.mod(step)));
      const u = float(k).div(SUB), u2 = u.mul(u), u3 = u2.mul(u);
      const ang = float(r).mul((2 * Math.PI) / RADIAL);
      const p0 = at(i0.mul(PER)).xyz, p1 = n1.xyz, p2 = n2.xyz, p3 = at(i3.mul(PER)).xyz;
      const c1 = p2.sub(p0);
      const c2 = p0.mul(2).sub(p1.mul(5)).add(p2.mul(4)).sub(p3);
      const c3 = p0.negate().add(p1.mul(3)).sub(p2.mul(3)).add(p3);
      const centre = p1.mul(2).add(c1.mul(u)).add(c2.mul(u2)).add(c3.mul(u3)).mul(0.5);
      const t0 = c1.add(c2.mul(u.mul(2))).add(c3.mul(u2.mul(3)));
      const t = normalize(select(dot(t0, t0).lessThan(1e-14), p2.sub(p0), t0)).toVar();
      const f0 = mix(at(i1.mul(PER).add(1)).xyz, at(i2.mul(PER).add(1)).xyz, u);
      const f1 = f0.sub(t.mul(dot(f0, t)));
      const any = normalize(cross(t, select(abs(t.y).lessThan(0.9), vec3(0, 1, 0), vec3(1, 0, 0))));
      const f = select(dot(f1, f1).lessThan(1e-12), any, normalize(f1));
      // as thin as that, the ring turns to face the camera (two vertices: a ribbon), so that a tube seen
      // edge-on does not vanish; its normals lean towards the camera, so that it shades as a round tube
      const toCam = modelWorldMatrixInverse.mul(vec4(cameraPosition, 1)).xyz.sub(centre);
      const across = toCam.sub(t.mul(dot(toCam, t)));
      const facing = step.equal(int(4)).and(dot(across, across).greaterThan(1e-12));
      const up = select(facing, normalize(across), cross(t, f)), side = select(facing, cross(normalize(across), t), f);
      const off = side.mul(cos(ang)).add(up.mul(sin(ang)));
      const nrm = select(step.equal(int(4)), normalize(off.add(up.mul(0.6))), off);
      normalLocal.assign(nrm);
      tubeCoord.assign(vec4(a.x.add(u.mul(a.y)), cos(ang), sin(ang), a.z));
      tubeTint.assign(at(select(u.lessThan(0.5), i1, i2).mul(PER).add(2)).xyz);
      return centre.add(off.mul(end.w));
    })();
  }

  /** pixels per metre at 1 m from this camera, in what is being drawn into */
  private measure(renderer: THREE.Renderer, camera: THREE.Camera): void {
    const t = renderer.getRenderTarget();
    const half = (t ? t.height : renderer.getDrawingBufferSize(size).y) / 2;
    this.pxPerMetre.value = Math.abs((camera as THREE.PerspectiveCamera).projectionMatrix.elements[5]) * half;
  }

  /**
   * Lays the cord through the n nodes from `start` on the segments from `at`
   * (n − 1 of them): its colour, and along it, where its first segment starts
   * and the length at rest of a segment (m), and a tag of the caller's (the
   * last of `tubeCoord`).
   */
  lay(at: number, start: number, n: number, color: THREE.Color, arc0: number, seg: number, tag: number): void {
    const s = this.seg.array as Float32Array, c = this.color.array as Float32Array, a = this.arc.array as Float32Array;
    for (let j = 0; j < n - 1; j++) {
      const o = 3 * (at + j);
      s[o] = start + j; s[o + 1] = start; s[o + 2] = start + n - 1;
      c[o] = color.r; c[o + 1] = color.g; c[o + 2] = color.b;
      a[o] = arc0 + j * seg; a[o + 1] = seg; a[o + 2] = tag;
    }
    for (const attr of [this.seg, this.color, this.arc]) {
      attr.addUpdateRange(3 * at, 3 * (n - 1));
      attr.needsUpdate = true;
    }
  }

  /** draws only the first segments */
  setDrawn(segments: number): void {
    this.geometry.instanceCount = segments;
  }

  /** the finest list the image needs (see tube.ts) */
  private levelFor(camera: THREE.Camera): number {
    const b = this.nodes.box;
    if (b.isEmpty() || !(camera as THREE.PerspectiveCamera).isPerspectiveCamera) return 0;
    b.getBoundingSphere(sphere).applyMatrix4(this.matrixWorld);
    eye.setFromMatrixPosition(camera.matrixWorld);
    const d = Math.max(1e-3, 0.9 * (eye.distanceTo(sphere.center) - sphere.radius));
    const px = (this.nodes.radius * this.pxPerMetre.value) / d;
    return px < this.options.lodQuarter ? 2 : px < this.options.lodHalf ? 1 : 0;
  }

  private lod(level: number): void {
    this.geometry.index = this.lists[Math.min(level, this.lists.length - 1)];
  }

  /** frees what is its own beyond its geometry and material */
  dispose(): void {
    this.geometry.index = this.lists[0];
    if (drawing.get(this.main) === this) drawing.delete(this.main);
    this.ownCopy?.dispose();
    this.nodes.dispose();
  }
}
