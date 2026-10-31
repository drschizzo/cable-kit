# The solver on the GPU

`GPUCableSim` (`@drschizzo/avbd-cables/gpu`) runs the cable solver of `CableSim` in WebGPU compute shaders (WGSL). It is an addition: `CableSim`, its worker and its tests are untouched, and the GPU solver reads the same settings (`SimParams`, `CABLES`) and takes the same calls (`build`, `setStatics`, `pin`, `setTarget`, `step`, `targetPull`). The positions stay on the GPU: the tubes of `three-cable-tubes` read them from there (`GPUTubeFeed`), and the CPU reads back only what it asks for, a frame or two late.

This document says how the CPU solver maps onto the GPU, what it costs, and where the two differ. [How it works](how-it-works.md) explains the solver itself. `examples/gpu-heap` drops thousands of ropes with either solver; `examples/gpu-check` runs the scenes of `tests/physics.test.ts` on both and compares them.

## What is ported

Per frame, `CableSim.step()` does the following; each line says where it went.

| CPU | What it is | On the GPU |
| --- | --- | --- |
| `reach` | targets brought within reach of the pins | CPU: the pins do not move, so the CPU knows all it reads; it uploads where each target pulls |
| `collide` | broad phase, narrow phase, warm start from last frame's contacts | compute: a spatial hash, then one invocation per segment finds its contacts and warm starts them from last frame's contacts of the same segment |
| `buildAdjacency` | node → its contacts | compute: a list per node, built once a frame |
| — | | colouring of the stretches of cord (below) |
| `track` | the closest point of crossing segments follows them | one invocation per contact, at the start of every substep |
| `cc0`, `λ ×= lamWarm` | stabilisation, warm start of the contacts | same invocation |
| stretch warm start | `λ ← αγλ`, `k ← γk`, `C₀` | same dispatch, one invocation per segment |
| prediction | x̃ = x + h v + h² g | one invocation per node |
| sweeps × iterations | `solveNode`: the 3×3 block of each node (stretch with λ, bending, contacts and friction, target on its leash) | colour by colour (below) |
| `dualUpdate` | λ and k of the stretch and of the contacts, friction (λt, anchors) | one dispatch: one invocation per segment and per contact |
| `inextensible` × passes | length projection along each cord, alternating directions | one invocation per cable, walking it as the CPU does |
| velocities | v = (x − x₀)/h, drag | one invocation per node |
| `contactVelocities` × 2 | separating normal velocity removed, sliding damped | one invocation per contact (its impulse), then one per node (the mean of its contacts') |
| rest friction | a pressed node barely moving stops | with the second of those |
| `refreshFixed` | the nodes of a sleeping cable are fixed | one invocation per node, first thing in the frame |
| `updateSleep`, `wakeNext` | a cable still for `sleepTime` falls asleep; a sleeper touched hard enough wakes | three dispatches at the end of the frame: per node, per cable, per node (below) |
| `dormant` | the contacts of sleepers, kept as they fell asleep | carried from slot to slot by the contact search (below) |

Not ported either: straps (`addBundle`), pinches, ghosts, holders (`heldBy`, `nodeTerm`, `keepHeld`), `noStatics`, `moveNodes`, `teleport` (build again instead), `threaded` (there is nothing to move off the page: the CPU only encodes the commands, 0.1 to 0.4 ms a frame).

## Data

Everything is `f32` (the CPU uses `f64`).

- **Per node**: `pos`, a `vec4` (x, y, z and the radius of the cable there: what the tubes read); x₀, x̃ and v; mass, bending stiffness, length at rest of the segment to the next node, and flags (pinned, segment, heavy target, cable led by a heavy target, firm, cable asleep) with the cable's index; where its target pulls, its stiffness and leash; where it was when its cable's sleep window started. Per segment, λ, k and C₀ of its stretch.
- **Per cable**: first node and count, whether it sleeps (and what happened to it this frame), how long it has been still.
- **Static shapes**: one record each (capsule or box, μ, bounds), uploaded by `setStatics`.
- **Contacts**, double-buffered (this frame's and last frame's). A segment `a` owns `MS` = 4 slots for its contacts with the static shapes and `MC` = 12 for its contacts with segments `b > a` (the CPU keys its pairs by the lower segment too). A contact holds its normal, s and t, its static point, rs, C at the start of the substep, k, its ceiling, μ, M/(h·h_ref), λ, its friction anchor and tangential λ, its key and whether it is tracked (112 bytes). A segment also lists the contacts it is the other side of (`MH` = 20 references), appended with atomics and then sorted, so that their order does not depend on the scheduling. The contacts found are also appended to a list, so that the kernels per contact run over them only (indirect dispatch).
- **Per node again**, its contacts and its weight in each (up to `MA` = 64), built once a frame from the above: the block solve reads one list instead of walking its segments.

## Contacts

**Broad phase.** A spatial hash of the segment midpoints, cells as large as on the CPU (longest segment, two radii and the margin cap), rebuilt every frame as linked lists: one atomic exchange per segment on the head of its bucket. Each segment then walks the 27 cells around its own and tests the segments after it, with the CPU's rejections (midpoint distance, same-cable neighbours, then the exact closest points) and its narrow phase (`pairContacts`: two contacts at the ends of the overlap of nearly parallel segments, the closest point otherwise; against a box, one per node and the closest point near an edge). The contacts a segment finds are sorted by key before they are written.

**Static shapes.** Built once by the CPU (`setStatics`). Each segment tests all of them, rejected by their bounds first: tens of shapes, not thousands. (A grid, as on the CPU, if a level ever has thousands.)

**Warm start.** A contact looks for the same key (the other segment or static shape, and which contact of the pair) among last frame's contacts of its segment, and takes its k (× γ), λ, friction anchor (shifted by how far its point moved along the segments) and tangential λ.

**Moving buckets.** Each bucket also notes whether it holds a segment that moves (not both nodes fixed), as on the CPU: a segment that does not move skips the buckets that hold none, so a heap asleep hardly searches.

**Overflow.** A segment with more contacts than its slots, or a node with more than its list holds, drops the rest; `stats.overflow` counts them. In the heaps of the example: none once the heap has settled, a few hundred out of 100 000 or more while thousands of ropes fall at once.

## Sleep

As on the CPU (see `CableSim.updateSleep`): a cable falls asleep when none of its nodes has moved more than `sleepDist` from where it was when its window started, for `sleepTime`; a cable led by a heavy target, or firm, does not. A sleeping cable's nodes are fixed: they act as static geometry for the others, and wake when something awake touches them faster than `wakeSpeed`, or creeps into them deeper than `wakeSink`.

- **State.** The CPU's `asleep` and `calm` of each cable are on the GPU (the two spare words of its record), and so is where its window started (`sleepRef`, per node). At the start of a frame, one invocation per node sets the asleep flag of its node from its cable: the CPU writes the nodes' flags whenever a pin or target changes, without it, and this puts it back.
- **End of a frame.** Per node: a node of a cable awake that is further than `sleepDist` from its start marks its cable as moved (an atomic or). Per cable: one that moved starts its window again, one that did not adds a frame to its calm and falls asleep beyond `sleepTime`, a sleeper asked to wake wakes. Per node again: the window's start moved to where the node is, or the node of a cable fallen asleep left at rest where it lies (v = 0, x₀ = x̃ = x), as the CPU leaves it.
- **Waking.** The contact search asks: a segment that moves, touching a sleeping one hard enough (the CPU's test), writes its cable into a list read at the end of the frame, as the CPU's `wakeNext` (a cable woken in the middle of the search would miss its contacts with the segments already searched). `wakeCable`, `wakeAll`, and the calls that wake a cable on the CPU (`pin`, `unpin`, `setTarget`, `clearTarget`, `setStatics`), write its state on the GPU before the next step.
- **What is skipped.** A sleeping cable's stretches take no colour: they are in none of the lists, never dispatched. Its segments look for no static contact, nor for other sleepers (both fixed), and only in buckets where something moves. Its stretch constraints keep their λ (the CPU's: it wakes with the tension it fell asleep with). The per-node kernels still run over every node, returning at once for the fixed ones.
- **Dormant contacts.** A cable that woke with no contacts rebuilt every force from nothing and sank into what held it (on the CPU, fast enough to wake its neighbours, which did the same: see `CableSim.dormant`). A sleeping segment carries its contacts with the static shapes and with other sleepers from last frame's slots into this frame's, as they were, marked dormant: they keep their slots and their references, but are left out of the list of contacts found, so nothing solves or updates them. Once awake, the segment finds them again among last frame's contacts, by key, and warm starts from them. Woken all at once, six cords asleep in a pile sink 0.37 mm into one another with them (the CPU: 0.37), 0.82 without; a cord asleep across a rod 0.05 mm (CPU 0.29), 1.9 without.
- **Read back.** Every cable's state comes back with the counters: `cables[c].asleep` and `calm` are a frame or two old, but for a cable woken since, which is awake at once; `stats.awake` counts them.

## The sweeps: stretches of cord, by colour

The CPU solves one node after another (Gauss–Seidel), each node seeing the moves of those before, cord after cord, one way then the other. On the GPU, what runs in parallel must not be coupled: the AVBD paper colours the vertices, so that the vertices of a colour share no constraint, and solves one colour after another.

**Colouring single nodes is not enough for cords.** Tried first: a cord dropped across a thin rod slid off it, and a draped cord sank, where the CPU holds them. With 3 or 4 colours along a chain, a sweep carries the tension only a few nodes along the cord; the CPU carries it from one end to the other in each sweep. With 30 iterations instead of 6, the cord still sank. Solved in the CPU's order (one colour per node), the GPU held the cord exactly as the CPU does (15.9 mm against 15.6 mm), so the rest of the port was right.

**So cords are cut into stretches** of 5 nodes (`options.stretch`), and the stretches are coloured: within a stretch, the nodes are solved one after the other in the CPU's order, one way then the other; stretches of a colour run in parallel. A stretch is coupled to the stretches within two nodes of its ends (stretch and bending) and to those its nodes touch.

How long is a matter of balance, and not a settled one. A single node lets the cord slide off the rod. Beyond that, two scenes of the tests sit on a knife edge (a cord balanced across a thin rod; a cord pinned 7.5 cm inside where it was built, which folds near its pins in the first frames and can stay folded, held by friction on itself), and which stretch lengths pass them changed with unrelated changes to the kernels: at the end, 5 and 12 pass both, 4, 6 and 8 fail one. 5 was chosen for that, which is fitting the default to two scenes: other scenes may prefer another length. Longer stretches cost more: each colour waits for its longest stretch.

**Colouring.** Every frame, after the contacts, in rounds of two passes (speculative colouring, Gebremedhin–Manne, also what the paper's "parallel Jacobi" greedy colouring amounts to): every uncoloured stretch picks the smallest colour none of its coloured neighbours has, then keeps it unless a neighbour of higher rank picked the same. Each pass reads only what the one before wrote, so the colouring does not depend on the scheduling. 16 rounds; a stretch still uncoloured goes into the last colour dispatched (`stats.uncoloured`: none in the heaps, after the first frames of a fall). A heap takes 10 to 17 colours.

**Lists and dispatches.** The stretches are sorted by colour into lists, with the number of workgroups each colour needs written for an indirect dispatch: a sweep is one dispatch per colour, and nothing is read back. How many colours to dispatch comes from the colours read back a frame or two earlier, plus one; a stretch beyond goes into the last one dispatched (`stats.clamped`), with what that colour holds. With `options.deterministic`, all 32 colours are dispatched, empty or not: how many no longer depends on when a read back came, at the cost of the empty dispatches (a quarter more at the size of loose-ends, nothing measurable on 2 000 ropes).

**A stretch per 8 invocations.** Each node of a stretch is solved by 8 invocations, which share out its contacts and add their gradients and Hessians in workgroup memory; the first then adds the node's own terms (inertia, stretch, bending, target), solves the 3×3 system and moves the node. A workgroup of 64 holds 8 stretches.

## A frame on the GPU

One command buffer, one compute pass, about 30 dispatches for the contacts and the colouring, then per substep 2 + iterations × (colours + 1) + 6 or so: 600 to 900 for a rope heap (6 substeps, 6 iterations).

1. The CPU writes the settings of the step (uniforms), where the targets pull, and what changed (pins, targets, flags, cables woken).
2. Sleeping nodes fixed; hash, contacts (and dormant ones carried, wakes asked for), sorted references, contact lists per node, list of contacts found, colouring, colour lists.
3. Per substep: contacts and stretch warm started, prediction; per iteration, one dispatch per colour, then the dual updates; length passes; velocities; contact velocities, twice, then rest friction.
4. Sleep: cables that moved, cables falling asleep or waking, their nodes.
5. `GPUTubeFeed` (same queue, before the image) writes the tubes' node map from `pos`: one invocation per cable, which carries the frame of the tube along it as `TubeNodes.write` does.
6. The timings, counters, the state of every cable and the nodes the CPU watches (those with a target) are copied into a buffer mapped later.

## Reading back

`step()` never waits for the GPU. After each step, the nodes with a target (and `watch`) are copied into a small buffer mapped asynchronously; `x` holds them as last read, and `targetPull(i)` computes the pull from them with the CPU's formula: a frame or two late. `readAll` reads every position back each step (tests, comparisons); `readPositions()` returns a promise of all of them now. `profile()` times the next step kernel by kernel.

## What it costs

Measured in Chromium (WebGPU) on the machine this was written on, `examples/gpu-heap` (ropes: 25 nodes, 6 substeps, 6 iterations) and a heap of patch cords. "CPU" is `CableSim` in a worker, on the same scene, while every cable is awake.

| Scene | Nodes | Contacts | CPU per step | GPU per step |
| --- | --- | --- | --- | --- |
| 500 ropes falling on 5 bars | 12 500 | 25 000–50 000 | 400–620 ms | 20–22 ms (stretches of 4) |
| 2 000 ropes | 50 000 | 120 000 | (≈ 2 s, not measured) | 50 ms (42 with stretches of 4) |
| 4 000 ropes | 100 000 | 250 000 | — | 80 ms |
| 48 patch cords and a bar, 12 substeps (the size of loose-ends) | 1 200 | 1 100 | 20 ms | 21 ms (26 deterministic) |
| one cord, 3 substeps | 25 | 0 | 0.1 ms | 2–4 ms |

- **Large scenes**: 20 to 25 times the CPU on the 500 ropes, and it grows about linearly from there. Of a 2 000-rope step, the sweeps take three quarters (`profile()`: 36 ms of 50), then the contacts (4.5 ms), the colouring (5 ms) and the dual updates (2.7 ms).
- **Small scenes**: the GPU is no faster than the CPU at the size of loose-ends (1 200 nodes, 12 substeps). It frees the CPU, though: encoding a step takes 0.1 to 0.4 ms. What it costs there is not so much the dispatches (about 900, 10 to 15 µs each when nearly empty) as the latency of the sweeps: each iteration solves, one after the other, a node of each stretch of each colour (6 colours × 5 nodes), each node a chain of dependent reads (its contact list, the contact, the nodes of the other cable) of about 10 µs on this GPU, whatever the number of cables. Running all the iterations of a substep in one dispatch of one workgroup, with workgroup barriers instead of dispatches, was tried: 3 % faster with one invocation per stretch, 25 % slower with eight sharing each node's contacts (32 colours × 2 barriers per node in one workgroup). It was taken out.
- **At rest**: the 2 000 ropes fall asleep about 10 s after they were let go, and a step then takes 1.6 to 3 ms instead of 50 (`examples/gpu-heap` shows how many are awake). What is left is fixed cost: the per-node kernels over every node, each sleeping segment checking its 27 buckets for something that moves, and the empty dispatches of the substeps (a colour or so per iteration, still dispatched).

## Differences with the CPU

Measured by `examples/gpu-check` on the scenes of `tests/physics.test.ts` (same cables, settings and time on both).

| Scene | CPU | GPU |
| --- | --- | --- |
| a cord dropped on the floor: lowest node, stretch | 2.2 mm, 0 % | 2.2 mm, 0 %; same positions |
| a fast cord and a thin plate | stays above | stays above |
| a rope hanging straight from a pin: stretch, its end | 0.19 %, 0.049 m | 0.22 %, 0.049 m |
| a cord dropped across a rod: its middle, its ends | 16.0 mm, −173 mm | 15.5 mm, −171 mm (falls off with some stretch lengths) |
| a rope draped over a thick bar, 16 substeps: how far its ends move | 20.1 mm | 20.0 mm |
| six cords piled: deepest overlap | 0.09 mm | 0.09 mm (0.2–0.55 before sleep was ported: awake, the pile kept shifting) |
| a target leads a node: distance to its point | 1.3 mm | 1.4 mm |
| a heavy target out of reach of a pinned cord: stretch | 1.14 % | 1.16 % |
| a hand holding a hanging cord, 12 substeps: pull / weight | 1.04 | 1.09–1.14 |
| a hundred cords in a heap after 10 s: deepest overlap, lowest node | 2.7 mm, −8.1 mm | 1.2 mm, −1.5 mm |
| a heap of 12 cords asleep, a cord dropped on it: asleep before, woken, asleep at the end; overlap then | 12, 10, 13; 0.44 mm | 12, 10, 13; 0.25–1.1 mm (below) |
| six cords asleep, all woken: how far they sink in the second after | 0.37 mm | 0.32 mm |
| a cord pinned at both ends: its middle | 0.296 m | 0.294 m |

- **Order of the sweeps.** By stretches and colours instead of node after node along every cord: the same fixed points, reached along other paths. Where friction can hold more than one shape, the two can end in different ones: a cord pinned at both ends 7.5 cm inside where it was built folds near its pins in the first frames; with some stretch lengths the GPU found two self-contacts there, which held a fold for good (its middle at 0.34 m instead of 0.30), where the CPU found none (see the stretches, above).
- **Precision.** `f32` instead of `f64`; the tolerance for nearly parallel segments in the closest points is relative (1e-6 of the product of the lengths) rather than 1e-14.
- **Contacts between cords** hold a little less tightly while they move (a pile woken: 0.2 mm of overlap against 0.15), and the pull of a hand is higher (1.09 to 1.14 of the weight against 1.04; `targetPull` reads positions a frame or two old, which does not matter for a cord at rest).
- **Contact velocities.** On the CPU, each contact changes the velocities of its nodes in turn (Gauss–Seidel); on the GPU, all contacts compute their impulse from the same velocities, and a node takes the mean of its contacts' (Jacobi with mass splitting).
- **Sleep** follows the CPU's rules, decided on the GPU's positions: the same cables do not necessarily fall asleep on the same frame. `cables[c].asleep` is a frame or two late. Dormant contacts take slots: a sleeping segment's carried contacts count against its 12, as its live ones do. One behaviour is the CPU's, kept: a cord dropped from 0.5 m on twelve sleeping cords ends 3.7 mm into one of them on both (0.7 mm when they are awake), the sleepers holding as static geometry until the end of the frame that wakes them (`examples/gpu-check` drops it from 0.35 m). There, the overlap left once all sleep again depends on the frames the sleepers wake on, which vary from run to run on the GPU: 0.25 to 1.1 mm over five runs (0.52 mm every time with sleep off; the CPU 0.44), so the check's 0.5 mm fails now and then.
- **Overflow**: the slots are fixed (above); the CPU's arrays grow.
- **Determinism.** By default, no: how many colours are dispatched follows read backs, whose timing varies, and a stretch beyond them shares a colour. With `options.deterministic`, a run gives the same result again, bit for bit, on the same GPU and browser (checked: a hundred cords for 300 frames, twice): the contacts of a segment (and those it keeps when its slots are full), its references and the lists per node are sorted or built in a fixed order, and the colouring depends on nothing but the contacts. One exception: a segment referenced by more contacts than its `MH` slots keeps those that came first. Never the CPU's bits, nor necessarily another GPU's.
- **Latency.** `targetPull` and `x` are a frame or two old.
- **Limits.** The contacts take 2 × 16 slots × 112 bytes per node: 180 MB for 50 000 nodes, more than WebGPU's default 128 MB per binding; `examples/gpu-heap` asks the adapter for its largest buffers.

## Against the paper

The paper simulates 110 000 blocks in 3.5 ms (9.8 ms with collision detection) and 510 000 in 10 ms, on an RTX 4090. Here, 50 000 nodes take 50 ms. Most of the gap is not the port:

- **Sweeps per frame.** The paper runs 4 iterations, one step a frame. The ropes here run 6 substeps × 6 iterations, 36 sweeps a frame, the settings `CableSim` uses for them (fewer substeps and the CPU solver's ropes sink into one another): 9 times the work per node.
- **Chains.** A block's constraints link it to its neighbours in the pile, which a few iterations settle; the hard stretch constraints of a cord link its nodes from one end to the other, and the tension must cross it within a frame. Colouring single nodes did not carry it (above): stretches solved in order do, at the price of a serial chain of 5 nodes per colour.
- **Per node.** In the solve, the paper's throughput is 440 000 body solves in 3.5 ms, about 125 million a second (6 degrees of freedom each); here 1.8 million node solves a frame in some 36 ms, about 50 million a second (3 degrees of freedom, but some 2.5 contacts each in a dense heap, with friction): the same order, on a GPU much smaller than a 4090.
- **Collision detection** is part of the 50 ms here (4.5 ms), not of the paper's 3.5.

## What next

- **Small scenes**: shorter chains of dependent reads in the node solve (contacts packed with what the solve reads of their other side), or fewer iterations or substeps where the scene allows; fusing the dispatches alone was not enough (above).
- **At rest**: skip the per-node kernels and the substeps' dispatches for the sleeping cables entirely (a list of the nodes awake, dispatched indirectly), for a heap asleep to cost next to nothing rather than 2 ms.
- Fewer colours (the sweeps are three quarters of a large step): a colouring that tries the colours of the stretch's neighbours along its cord first, or stretches cut where the contacts are fewest.
