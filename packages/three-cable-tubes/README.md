# @drschizzo/three-cable-tubes

Thousands of smooth tubes for three.js, swept on the GPU from their nodes alone. The CPU sends only the nodes (a texture, three texels each); the vertex shader finds every point of every tube: the Catmull-Rom curve through the nodes, a frame carried along it so the rings do not twist, the ring round it. Every segment of every tube is an instance of the same short stretch of tube, so all of them are one mesh and one draw call. Tubes a few pixels wide on screen lose vertices, and far ones draw coarser lists of triangles.

It draws cables from [`@drschizzo/avbd-cables`](../avbd-cables), or any curves you compute: ropes, hoses, wires, grass, hair.

There are two versions with the same API: `@drschizzo/three-cable-tubes` for `WebGLRenderer` (GLSL), and `@drschizzo/three-cable-tubes/webgpu` for `WebGPURenderer` (TSL), which runs on WebGPU or, without it, on WebGL 2.

## WebGLRenderer

```ts
import * as THREE from 'three';
import { TubeMesh, TubeNodes } from '@drschizzo/three-cable-tubes';

const nodes = new TubeNodes(nodeCount);
const tubes = new TubeMesh(segmentCount, new THREE.MeshStandardMaterial({ vertexColors: true }), nodes);
scene.add(tubes);

// a tube through the n nodes from node 0, drawn on the segments from 0 (n − 1 of them)
tubes.lay(0, 0, n, new THREE.Color('teal'), 0, segmentLength, 0);

// whenever it moves
nodes.write(positions, 0, n, radius);
```

Any material draws them: `swept` makes its vertex shader see the point of the tube as `position` and `normal`, whoever reads them (three.js, your own chunks, the passes of the post-processing). The mesh also hands a swept copy of its material to the passes that draw the scene with a material of their own (normals, depth), and draws its shadows with swept depth materials.

## WebGPURenderer

```ts
import * as THREE from 'three/webgpu';
import { TubeMesh, TubeNodes, tubeCoord, tubeTint } from '@drschizzo/three-cable-tubes/webgpu';

const nodes = new TubeNodes(nodeCount);
const material = new THREE.MeshStandardNodeMaterial();
material.emissiveNode = tubeTint;
const tubes = new TubeMesh(segmentCount, material, nodes);
```

The mesh sets the material's `positionNode`, which also sets the normal. Shadows and post-processing passes draw the same material, so nothing else needs it.

### Nodes computed on the GPU

With WebGPU, the nodes can be computed by a compute shader and never leave the GPU: write them into `GPUTubeNodes` with `storeTubeNode`, and give those to the mesh instead of `TubeNodes`.

```ts
import { Fn, instanceIndex } from 'three/tsl';
import { GPUTubeNodes, TubeMesh, storeTubeNode } from '@drschizzo/three-cable-tubes/webgpu';

const nodes = new GPUTubeNodes(count);
nodes.box.set(min, max);       // where the tubes may be, for the levels of detail
nodes.radius = 0.001;          // and how thick they are at most

const update = Fn(() => {
  const i = instanceIndex;
  // position, radius, a normal square to the cord (carried along it), glow
  storeTubeNode(nodes, i, positionOf(i), radiusOf(i), normalOf(i), glowOf(i));
})().compute(count);

const tubes = new TubeMesh(segments, material, nodes);
renderer.setAnimationLoop(() => {
  renderer.compute(update);
  renderer.render(scene, camera);
});
```

For strands (grass, hair, fibres, decorative cables), `sweepStrands` writes that compute shader for you: give it where each strand starts and which way it goes along its length, and it lays the nodes, carries the frame of the tube up each strand without twisting, and writes them with one invocation per node. `layStrands` lays them on the mesh.

```ts
const strands = {
  count: 50_000, nodes: 24, segment: 0.02,
  root: (s) => rootOf(s),                  // TSL, of the strand's index
  direction: (s, u) => directionOf(s, u),  // a unit vector, u from 0 (root) to 1 (tip)
  radius: (s, u) => float(0.0006),
  tint: (s, u) => glowOf(s, u),            // optional: written once
};
const { update, glow } = sweepStrands(nodes, strands);
renderer.compute(glow);                    // once
layStrands(tubes, strands, (s) => colourOf(s));
// every frame: renderer.compute(update)
```

The CPU then does nothing per frame but ask for the compute and the image, however many nodes there are. Write what does not change (often the glow) once, with `storeTubeNodeTint`, rather than every frame, and let neighbouring invocations write neighbouring nodes (one invocation per node rather than per cord): the writes are most of what such a compute shader costs. A `GPUTubeNodes` of millions of nodes widens its texture to stay within WebGPU's default limit of 8192 rows. Compute shaders need WebGPU: this does not run on the WebGL 2 fallback.

## The API

- **`TubeNodes(count)`**: the nodes, as the GPU reads them. `write(x, start, n, radius)` sends n nodes from `start` (positions xyz interleaved), with one radius for all of them or one per node (`radius[i]`); a radius of 0 hides the tube. `hide(start, n)` hides it. `tint(i, color)` sets the glow at node i.
- **`TubeMesh(segments, material, nodes, options)`**: the mesh. `lay(at, start, n, color, arc0, seg, tag)` lays a tube through n nodes from `start` on the segments from `at`: its colour, where its first segment starts along it and how long a segment is at rest (for `tubeCoord`), and a tag of yours. `setDrawn(k)` draws only the first k segments. `level` is the level of detail of the last image (0 finest).
- **`TubeSync(nodes, radius)`**: writes the nodes of the cables that moved, from anything with `x` and `cables: { start, count, asleep }[]` (an avbd-cables simulation). A sleeping cable is written again only if some of its nodes moved anyway.
- **Options**: `radial` (vertices round a ring, 16) and `sub` (rings per segment, 8), both multiples of 4; `lodHalf` and `lodQuarter`, the radius on screen in pixels under which a tube keeps half, then a quarter of its vertices (6 and 1.5). At a quarter, its rings turn to face the camera (with `radial: 8`, two vertices: a ribbon), so that a thin tube never vanishes edge-on, and its normals lean towards the camera so that it still shades as a round tube.
- **Values for your material**: `tubeCoord` (vec4: where the vertex is along its tube at rest, in metres; cos and sin of its angle round it; the tag) and `tubeTint` (vec3: the glow of the nearest node). In GLSL, they are names in the vertex shader; `options.aliases` also gives them names your material already reads, such as `{ emissiveTint: 'tubeTint' }`. In TSL, they are varyings the other nodes of the material can read.

## Limits

- The WebGL version needs WebGL 2 (it fetches texels in the vertex shader from a float texture).
- The WebGPU version has been checked in Chromium, on WebGPU and on its WebGL 2 fallback; nodes computed on the GPU need WebGPU itself.
- Every tube of a mesh shares its material; colour, tag and glow vary per tube or per node.
- A material draws the tubes of one mesh at a time. Given one that another mesh draws, a mesh draws a copy of it (its `material`; in GLSL, the copy keeps your `onBeforeCompile`), which does not follow later changes to the original. A disposed mesh leaves its material free: a mesh built again to replace another one draws the same material.

## License

MIT
