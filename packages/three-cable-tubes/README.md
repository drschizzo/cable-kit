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

## The API

- **`TubeNodes(count)`**: the nodes, as the GPU reads them. `write(x, start, n, radius)` sends n nodes from `start` (positions xyz interleaved), with one radius for all of them or one per node (`radius[i]`); a radius of 0 hides the tube. `hide(start, n)` hides it. `tint(i, color)` sets the glow at node i.
- **`TubeMesh(segments, material, nodes, options)`**: the mesh. `lay(at, start, n, color, arc0, seg, tag)` lays a tube through n nodes from `start` on the segments from `at`: its colour, where its first segment starts along it and how long a segment is at rest (for `tubeCoord`), and a tag of yours. `setDrawn(k)` draws only the first k segments. `level` is the level of detail of the last image (0 finest).
- **`TubeSync(nodes, radius)`**: writes the nodes of the cables that moved, from anything with `x` and `cables: { start, count, asleep }[]` (an avbd-cables simulation). A sleeping cable is written again only if some of its nodes moved anyway.
- **Options**: `radial` (vertices round a ring, 16) and `sub` (rings per segment, 8), both multiples of 4; `lodHalf` and `lodQuarter`, the radius on screen in pixels under which a tube keeps half, then a quarter of its vertices (6 and 1.5).
- **Values for your material**: `tubeCoord` (vec4: where the vertex is along its tube at rest, in metres; cos and sin of its angle round it; the tag) and `tubeTint` (vec3: the glow of the nearest node). In GLSL, they are names in the vertex shader; `options.aliases` also gives them names your material already reads, such as `{ emissiveTint: 'tubeTint' }`. In TSL, they are varyings the other nodes of the material can read.

## Limits

- The WebGL version needs WebGL 2 (it fetches texels in the vertex shader from a float texture).
- The WebGPU version has been checked in Chromium, on WebGPU and on its WebGL 2 fallback.
- Every tube of a mesh shares its material; colour, tag and glow vary per tube or per node.

## License

MIT
