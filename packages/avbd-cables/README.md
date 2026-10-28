# @drschizzo/avbd-cables

A cable solver for the web. Cables are chains of nodes that do not stretch, resist bending, collide with each other and with static shapes (capsules and boxes), with friction, and fall asleep when they come to rest, so that a room full of them costs nothing until something touches them. It uses Augmented Vertex Block Descent (Giles, Diaz and Yuksel, SIGGRAPH 2025), runs in a Web Worker if you like, and has no dependency: it computes positions, and you draw them however you want (see [`@drschizzo/three-cable-tubes`](../three-cable-tubes) for three.js).

Units are metres, kilograms and seconds; gravity pulls along −y.

## Quick start

```ts
import { CABLES, CableSim } from '@drschizzo/avbd-cables';

const kind = CABLES.patchCord;
const sim = new CableSim(kind.params);

// one cable of 1 m, straight, a node every `kind.segment` metres
const points: number[] = [];
for (let x = -0.5; x <= 0.5 + 1e-9; x += kind.segment) points.push(x, 0.3, 0);
const [start] = sim.build([{ ...kind.cable, points }]);

sim.setStatics([{ kind: 'box', min: [-2, -0.1, -2], max: [2, 0, 2] }]);   // a floor
sim.pin(start, -0.5, 0.3, 0);                                               // one end held

for (let f = 0; f < 120; f++) sim.step();   // two seconds
console.log(sim.x);                          // the nodes, xyz interleaved
```

## The model

- **Cables and nodes.** `build(defs)` makes the cables from their nodes (`points`, xyz interleaved): the length at rest of each segment is its length there. All nodes live in shared arrays: `x` (positions), `v` (velocities), `mass`, `radius`, `cableOf`; cable `c` has `cables[c].count` nodes from `cables[c].start`.
- **A step.** `step()` advances `P.dt` (1/60 s) in `P.substeps` substeps. Contacts are found once a step, with a margin that grows with speed, so that fast cables do not pass through each other.
- **Static shapes.** `setStatics()` takes capsules (`kind: 'capsule'`, a segment and a radius) and axis-aligned boxes (`kind: 'box'`), each with an optional friction `mu`.
- **Pins.** `pin(i, x, y, z)` holds a node in place; `unpin(i)` lets it go.
- **Targets.** `setTarget(i, x, y, z, k, lead, heavy)` leads a node towards a point with a spring on a leash: it pulls from at most `lead` metres away, so a cable caught on something is held back rather than stretched like an elastic. A `heavy` target is a hand: the cable follows it rather than the opposite, and it does not fall asleep. A target out of reach of the cable (further from a pinned node than the length of cable between them) is brought back within reach. `clearTarget(i)` lets go.
- **Straps.** `addBundle(nodes, r)` keeps the given nodes within `r` of their centre (a strap round several cables); `removeBundle(id)` takes it off.
- **Pinches.** A pinch keeps the shape a few consecutive nodes have (shape matching, Müller et al. 2005): see `setPinches`, `firmPinch`, `unpinch`.
- **Ghosts.** A node whose `ghost` is −1 passes through the other cables (never through the static shapes); set a number of frames instead and it becomes solid again where it is clear of them, or pushes them out once they run out.
- **Sleep.** A cable whose nodes all stay within `P.sleepDist` for `P.sleepTime` falls asleep: its nodes are frozen and act as static geometry for the others, until something awake touches it hard enough. `wakeCable(c)` and `wakeAll()` wake them by hand.
- **Moving things by hand.** `moveNodes(start, pos)` puts nodes somewhere else, at rest; `teleport(pos)` does it for every free node.

## Kinds of cable

`CABLES` holds kinds of cable with the solver settings that suit them:

| Kind | Cable | Settings |
|---|---|---|
| `patchCord` | 4.4 mm thick, 40 g/m, a connector and its boot at each end, a node every 2.5 cm | the defaults |
| `rope` | 10 mm, 70 g/m, supple, a node every 5 cm | 6 substeps, wider contact margins |
| `hose` | 20 mm, 250 g/m, stiff, a node every 6 cm | 6 substeps, wider contact margins |

Each has `cable` (spread it into a `CableInit`), `segment` (the distance between nodes to build it with) and `params` (pass them to the constructor). Thicker cables with longer segments need more substeps: with too few, they sink into each other and into the floor. Mixing very different kinds in one simulation takes the settings of the finest.

All settings are in `SimParams`, documented one by one; `defaultSimParams(partial)` fills in the rest. Stiffnesses are given for a reference substep (`refSubsteps`) and converted, so that more substeps make the same cables more accurate rather than stiffer.

## In a Web Worker

```ts
// sim.worker.ts
import { CableSim, hostSimWorker } from '@drschizzo/avbd-cables';
hostSimWorker((params) => new CableSim(params));
```

```ts
// main.ts
import { CableSim, threaded } from '@drschizzo/avbd-cables';

class Sim extends threaded(CableSim, () => new Worker(new URL('./sim.worker.ts', import.meta.url), { type: 'module' })) {}

const sim = new Sim(params);
sim.build(defs);
sim.setStatics(shapes);
sim.startWorker();

renderer.setAnimationLoop(() => {
  sim.advance(1);          // asks the worker for a step, without waiting
  draw(sim.x);             // the last positions it sent back
});
```

The page keeps a full copy of the simulation and reads `x` as if it ran there. Every change (the methods a class lists in its static `mirrored`) is applied in the page and repeated in the worker, in the same order. One step request is in flight at a time; if the worker falls behind, time is dropped rather than queued (`dropped` counts it). `settle(frames)` runs many frames at once and resolves when done; `hold` stops stepping during scripted motion; `version` bumps when new positions arrive.

The worker starts from `snapshot()`: positions, velocities, pins, targets, pinches, straps. Not the multipliers of the constraints and contacts, nor which cables sleep: the worker rebuilds them, so a simulation moved to a worker while things move goes on a little differently than it would have in place. Start the worker before things move to keep it exact.

## Your own simulation

Subclass `CableSim` for what drives your cables (a hand, sockets, channels that keep cables in them):

- `prepare(dt)` runs at the start of every step: set your targets and flags there (`firm`: a cable firmer against the static shapes; `noStatics`: a segment that passes through them; `heldBy`: a node held by something of yours);
- `nodeTerm(i, holder, mh, system)` adds your own pull to the 3×3 system of a node you hold (`heldBy[i] >= 0`), in the solve itself;
- `keepHeld(h)` runs before and after the length passes of each substep, to put held nodes back;
- `stepInput()` / `setStepInput()` carry what changes every frame without a call (the pointer) to the worker;
- `snapshot()` / `restore()` carry your own state to it; list your mirrored methods in `static mirrored`.

## Limits

- Large heaps of cables lying on one another (more than about a hundred long cables on a floor) may keep shivering instead of falling asleep. Smaller heaps, and cables hanging or lying apart, settle.
- The settings are checked for the three kinds above; other sizes need their own (more substeps and wider margins for thicker cables with longer segments). `npm run presets` in the repository shows how a kind behaves.
- Static shapes are capsules and axis-aligned boxes.
- The solver is single-threaded: a worker keeps it off the page's thread, it does not split it.

## License

MIT
