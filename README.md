# cable-kit

Thousands of cables for the web, in two packages that work together or apart:

| Package | What it does | Depends on |
|---|---|---|
| [`@drschizzo/avbd-cables`](packages/avbd-cables) | simulates cables: inextensible, bending, colliding with each other and with static shapes, with friction, falling asleep at rest; in a Web Worker if you like | nothing |
| [`@drschizzo/three-cable-tubes`](packages/three-cable-tubes) | draws them: smooth tubes swept on the GPU from their nodes alone, one draw call for all of them, with levels of detail; for `WebGLRenderer` (GLSL) and `WebGPURenderer` (TSL), whose nodes may also come from a compute shader | three.js |

The solver uses Augmented Vertex Block Descent (Giles, Diaz and Yuksel, SIGGRAPH 2025). Both packages were written for [Brassage](https://github.com/drschizzo/cablemanagement), a cable management game in which a room of network racks holds thousands of patch cords ([play it](https://drschizzo.github.io/cablemanagement/)).

## Examples

```bash
npm install
npm run dev
```

Then open http://localhost:5173:

- **grab**: ropes you take and drag with the mouse, held back by a rod and the floor;
- **falling**: hundreds of ropes falling in a heap, the physics in a Web Worker;
- **tubes only**: thousands of tubes animated by hand, no physics;
- **webgpu**: a fibre optic lamp: 16 000 strands whose nodes a compute shader moves every frame, never leaving the GPU, with pulses of light made by a TSL material and a bloom (needs WebGPU).

## In a few lines

```ts
import * as THREE from 'three';
import { CABLES, CableSim } from '@drschizzo/avbd-cables';
import { TubeMesh, TubeNodes, TubeSync } from '@drschizzo/three-cable-tubes';

const kind = CABLES.rope;                       // radius, mass, bending, solver settings
const sim = new CableSim(kind.params);
sim.build([{ ...kind.cable, points: [/* x, y, z of each node, 5 cm apart */] }]);
sim.setStatics([{ kind: 'box', min: [-2, -0.1, -2], max: [2, 0, 2] }]);   // a floor

const nodes = new TubeNodes(sim.n);
const tubes = new TubeMesh(sim.n - 1, new THREE.MeshStandardMaterial({ vertexColors: true }), nodes);
tubes.lay(0, 0, sim.n, new THREE.Color('orange'), 0, kind.segment, 0);
scene.add(tubes);
const sync = new TubeSync(nodes, kind.cable.radius);

renderer.setAnimationLoop(() => {
  sim.step();                                   // 1/60 s
  sync.update(sim);                             // only the cables that moved
  renderer.render(scene, camera);
});
```

Each package's README explains how to use it; [docs/how-it-works.md](docs/how-it-works.md) explains how they work.

## Installing

The packages are not on npm yet. To use them in a project:

- as a git submodule, with npm workspaces pointing at `packages/*` of it (this is how Brassage uses them), and the `source` export condition if you would rather build them with your app than use their `dist/`;
- or as tarballs: `npm run pack` here builds them and shows what they hold, `npm pack -w @drschizzo/avbd-cables` writes one, and `npm install ./drschizzo-avbd-cables-0.1.0.tgz` installs it.

## Working on it

| Command | What it does |
|---|---|
| `npm run dev` | the examples, served with the packages' sources |
| `npm run build` | the packages' `dist/` (JavaScript, types, source maps) |
| `npm run build:examples` | the examples, built into `dist/` |
| `npm run check` | lint, types, tests |
| `npm run presets` | each kind of cable on a floor, over a rod, in a heap, dragged through the heap |
| `npm run heap` | a big heap (`N=200 KIND=rope`): every second, cables awake, falling asleep and waking, how much they move |
| `npm run pack` | builds the packages and lists what their tarballs would hold |

The repository reads the packages from their sources (`src/`, through the `source` export condition); what is published is their `dist/`.

## License

[MIT](LICENSE).
