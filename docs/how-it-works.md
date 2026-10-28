# How it works

This document explains the mechanisms of the two packages: how `avbd-cables` moves the cables, and how `three-cable-tubes` draws them. Each package's README explains how to use it.

## The solver

### Cables are chains of nodes

A real cable is continuous. The solver replaces it by **nodes** (point masses) linked by **segments** of fixed length. For collisions, each segment is a capsule (a cylinder with rounded ends) of the cable's radius.

Nodes are not objects: every property is a flat typed array (*structure of arrays*), all cables together.

```ts
x    = new Float64Array(3 * N);  // positions: x0, y0, z0, x1, y1, z1, …
v    = new Float64Array(3 * N);  // velocities
mass = new Float64Array(N);
rest = new Float64Array(N);      // length at rest of the segment i → i + 1
```

Nothing is allocated while the simulation runs, so the garbage collector never pauses it; the data of one property are contiguous, which keeps the loops fast; and a typed array crosses to a worker without a copy.

### What each node must respect

| Contribution | Kind | Effect |
| --- | --- | --- |
| Inertia and gravity | prediction | a free node keeps going and falls |
| Stretch | hard constraint | each segment keeps its length at rest |
| Bending | soft energy | the cable resists curving (more near a connector's boot) |
| Contact, cable–cable and cable–static | hard constraint, one-sided | two capsules do not go into one another |
| Friction | hard constraint, bounded | a cable lying on something does not slide while the tangential force stays under μ × the normal force |
| Target | spring on a leash | a node is led towards a point |
| Strap | one-sided spring | strapped nodes stay within a radius of their centre |
| Pinch | spring | pinched nodes keep the shape they had together |
| Pin | fixed node | it does not move |

A **hard constraint** must hold exactly (to the precision of the solve). A **soft energy** is a spring: it pulls without imposing anything.

### A step, in three ideas

Augmented Vertex Block Descent (Giles, Diaz and Yuksel, SIGGRAPH 2025) computes each substep in three moves:

1. **Predict.** Every free node moves as if alone: it keeps its velocity, and gravity pulls it down. That position is its *prediction*, x̃ = x + h v + h² g.
2. **Solve one node at a time.** For each node, the solver finds the small move that best respects its constraints (the length of its two segments, its contacts, the bending of the cable) while moving it as little as possible from its prediction, its neighbours held still. That is a 3×3 system,

   (M/h² + Σ k ∇C∇Cᵀ) Δx = −(M/h² (x − x̃) + Σ (k C + λ) ∇C),

   solved directly. The move is applied at once, so the next node already sees it (Gauss-Seidel).
3. **Sweep again.** Moving a node disturbs its neighbours, so the solver sweeps every node several times per substep, in alternate directions so that neither end of a cable is favoured.

What makes the constraints hard is the **dual update** after each sweep: a multiplier λ per constraint absorbs the force its penalty is still supplying, and the penalty stiffness k of a contact rises where it is violated. The multipliers are carried from substep to substep (warm starting), so a loaded contact or a taut cable keeps its force instead of rebuilding it by sinking. After the sweeps, a few **length passes** project the segments back to their length along each cable, taking out what stretch the solve left under load. The velocity is then what the node moved, divided by h.

### Contacts

Contacts are found once per step, with a **speculative margin** that grows with the speed of the segments: two segments that will meet during the step already have their contact, so a fast cable does not pass through a thin one.

- **Broad phase.** Every segment goes into a spatial hash by its midpoint, with cells as big as the longest segment plus two radii and the margin: two segments can only touch if their cells are neighbours. A still segment only looks into cells that hold a moving one. Static shapes sit in a grid of their own, built once by `setStatics()`.
- **Narrow phase.** For each pair, the closest points of the two segments give the contact and its normal. Two nearly parallel segments (a cable lying along another) get two contacts, at the ends of their overlap: one, in the middle, would let the cable rock from side to side. Against a box, every node gets its own contact.
- **Tracking.** The contact at the closest point of two crossing segments moves with them at every substep (and keeps its normal's side: a cable pushed past a bar's axis is still pushed out the side it came from). Against a bar, only for a cable a hand draws: a cable resting on a thick bar rocked as its closest point went back and forth along each segment, and crept off.
- **Friction** is a hard constraint too, with an **anchor** kept from step to step: a tangential multiplier holds the contact on its anchor, clamped to the Coulomb cone, and the anchor only moves when the contact slides. Without it, a loaded cable crept forever.
- **Stiffness ceilings.** The penalty of a contact rises where it is violated, up to a ceiling. Between two cables at rest it stays low: a contact stiffer than the inertia of a node over a substep overshoots, and in a heap where every cable presses on several others the overshoots kept the heap shivering, waking its cables as fast as they fell asleep. A cable drawn by a hand gets a stiffer ceiling, so that it does not pass through the others it is dragged across.
- **Overlaps** already there are undone gently, a bounded amount per substep: a cable released across a bar slides out of it instead of being kicked away.

### Sleep

A cable whose nodes all stay within a few millimetres for half a second falls asleep: its nodes are frozen and the per-node passes skip it, so a room full of cables at rest costs nothing. Its contacts are kept aside as they were (*dormant*): when it wakes, it starts with the forces that held it, instead of sinking into what supports it while it rebuilds them and waking its neighbours in turn. Something awake that touches it fast enough, or sinks into it, wakes it at the end of the step.

### Targets

A target leads a node towards a point with a spring that pulls from at most its *leash* away: a cable caught on something is held back, not stretched like an elastic, and does not snap back when let go. A **heavy** target is a hand: in the length passes its node weighs more, so the cable follows it rather than the opposite, and its cable does not fall asleep.

A target is never further from a pinned node of its cable than the length of cable between them: at the start of every step, it is brought back onto that sphere (or where the spheres of the pins on both sides meet). Out of reach, the spring pulled the cable taut against the pin with all its strength, between a heavy node and a fixed one, and the length passes could not take the stretch out of the light nodes between: the cable whipped about.

### Substeps and settings

The settings are given for a reference substep (`refSubsteps`). With more substeps, they are converted so that they act as much per second: springs keep their stiffness, the penalties of the hard constraints keep k·h, and what is done "per substep" is spread over the shorter substeps. More substeps make the same cables more accurate, not stiffer. Thicker cables with longer segments need them: with too few, a segment moves further in a substep than the contacts can hold.

### In a worker

`threaded()` keeps the whole simulation in the page and moves its stepping to a worker. The methods that change it are listed by each class (`static mirrored`); a call is applied in the page, then repeated in the worker in the same order, and only the outermost call is sent (it makes the inner ones there too). Every frame, the page asks for a step (sending what changed without a call, such as where the pointer has taken a hand) and draws the last positions it got back: one request is in flight at a time, and if the worker falls behind, time is dropped rather than queued. A counter discards the answers computed before a teleport.

## The tubes

### Only the nodes go to the GPU

Turning every node into rings of vertices on the CPU, and sending them all each frame a cable moved, made the tubes cost more the bigger they were. Instead, the CPU sends only the nodes, three texels each in a float texture: position and radius, the normal of a frame carried along the cable (parallel transport from its first node, so the rings do not twist), and the glow of the node.

### One stretch of tube, instanced on every segment

The geometry holds no vertex per point of any tube. Every segment of every cable is one instance of the same stretch of tube, `sub + 1` rings of `radial` vertices, with a few numbers of its own: its first node, the ends of its cable, its colour, where it starts along the cable. The vertex shader finds its point: the Catmull-Rom curve through the four nodes around the segment, its tangent, the frame between those of the two nodes set square to the tangent, the ring round it. It then stands for `position` and `normal` in the rest of the shader, whoever wrote it.

### Levels of detail

A tube a few pixels wide needs no 16 sides. As its radius on screen shrinks, every other vertex (then three in four) folds onto its neighbour, along the cable and round it, and their triangles, flat, are dropped before they cost a pixel. Folded vertices still run the shader, though: when even the nearest node is that far, the mesh draws coarser lists of triangles instead, a quarter or a sixteenth of the vertices. Passes that do not show the outline (a shadow, the normals of a post-processing pass) draw a coarser list than the image. At the coarsest, the few vertices left of a ring turn to face the camera: with two of them, a ring that kept the frame of the cable was a flat ribbon, which vanished when seen edge-on. Their normals lean towards the camera, so that the ribbon shades as the round tube it stands for.

### WebGL and WebGPU

For `WebGLRenderer`, the shader is GLSL, inserted at the start of the material's vertex shader; the mesh hands swept copies of its material to the passes that draw with their own. For `WebGPURenderer`, the same computation is written in TSL and given to the material as its `positionNode`, which also sets the normal; shadows and post-processing draw the same material. `WebGPURenderer` runs it on WebGPU, or on WebGL 2 where there is none.

On WebGPU, the node map may also be a storage texture that a compute shader writes (`GPUTubeNodes`, `storeTubeNode`), in the same layout: the nodes are then computed and drawn without ever crossing to the CPU, which only asks for the compute and the image each frame.
