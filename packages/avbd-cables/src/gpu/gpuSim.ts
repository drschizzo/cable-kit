// The cable solver on the GPU (WebGPU compute shaders): the same cables, the
// same settings and calls as CableSim, the cables cut into short stretches
// solved colour by colour in parallel (each stretch in the order of the CPU)
// instead of one node after another. See docs/gpu.md for how it maps onto
// the GPU and how it differs from CableSim, and kernels.ts for the shaders.
//
//   const sim = new GPUCableSim(device, CABLES.rope.params);
//   sim.build(defs);
//   sim.setStatics(shapes);
//   sim.step();                   // encodes and submits a frame, never waits
//   sim.positions                 // the nodes on the GPU: vec4 (x, y, z, radius) each
//
// It reads nothing back unless asked: `targetPull` and `x` are those of a
// frame or two ago, `readPositions()` a promise of all of them. Cables fall
// asleep and wake on the GPU, as in CableSim; `cables[c].asleep` is read
// back with the rest.

/// <reference types="@webgpu/types" />

import { defaultSimParams, type CableInfo, type CableInit, type SimParams, type StaticShape } from '../cableSim.js';
import { COL, COLORS, KERNELS, M, MA, PARAMS, REF, RESOURCES, ROUNDS, STAT, WG, kernelSource } from './kernels.js';

/** the boot of a connector at the ends of a cable (as in cableSim.ts) */
const BOOT = { stiff: 3, firm: 1.8 };
const F = { pinned: 1, seg: 2, heavy: 4, led: 8, firm: 16 };
/** bytes of a contact, a static shape, of the stride of the per-pass uniforms */
const CONTACT_BYTES = 112, STATIC_BYTES = 64, PASS_STRIDE = 256;
/** timestamps of a profiled step (two per dispatch) */
const PROFILE_QUERIES = 4096;

export interface GPUSimStats {
  /** contacts found at the last frame read back */
  contacts: number;
  /** colours the stretches of cord took, and how many stretches each */
  colors: number[];
  /** contacts dropped for want of slots (static, cable, references of a segment or a node) */
  overflow: [number, number, number];
  /** stretches left uncoloured after the rounds of the colouring, and stretches of a colour not dispatched: both share the last colour dispatched */
  uncoloured: number;
  clamped: number;
  /** cables awake */
  awake: number;
  /** time to encode a frame on the CPU (ms), and the GPU's time for it where the browser can time it */
  ms: number;
  gpuMs: number;
}

export class GPUCableSim {
  readonly device: GPUDevice;
  P: SimParams;
  defs: CableInit[] = [];
  n = 0;
  cables: CableInfo[] = [];
  mass = new Float64Array(0);
  rest = new Float64Array(0);
  radius = new Float64Array(0);
  cableOf = new Int32Array(0);
  segOK = new Uint8Array(0);
  pinned = new Uint8Array(0);
  /** cables firmer against the static shapes (drawn by a hand: see `kCapHeld`) */
  firm = new Uint8Array(0);
  /**
   * The positions as last read back (xyz interleaved), a frame or two old: of
   * the watched nodes only (those with a target, and `watch`), unless
   * `readAll` is set; as built until then.
   */
  x = new Float64Array(0);
  /** reads every position back after each step (costly: for tests and comparisons) */
  readAll = false;
  /** nodes read back after each step, beyond those with a target */
  watch: number[] = [];
  target = new Float64Array(0);
  targetK = new Float64Array(0);
  targetLead = new Float64Array(0);
  targetHeavy = new Uint8Array(0);
  stats: GPUSimStats = { contacts: 0, colors: [], overflow: [0, 0, 0], uncoloured: 0, clamped: 0, awake: 0, ms: 0, gpuMs: 0 };
  /** frames stepped, and that frame of the last read back */
  frame = 0;
  readFrame = -1;

  private statics: StaticShape[] = [];
  private targetAt = new Float64Array(0);
  private led = new Uint8Array(0);
  /** CPU copies of `info` and `tgt`, and what changed since they were last sent */
  private infoData = new ArrayBuffer(0);
  private tgtData = new Float32Array(0);
  private infoDirty = [Infinity, -Infinity];
  private tgtDirty = [Infinity, -Infinity];
  /** cables woken since the last step (all of them: `wakeAll`), and the frame each was last woken for */
  private toWake = new Set<number>();
  private wakeEvery = false;
  private wokeFor = new Int32Array(0);
  private flip = false;
  private cur = 0;
  private hashCells = 0;
  private invCell = 1;
  /** nodes per stretch of cord, and how many stretches */
  private stretchLength: number;
  private nChunks = 0;
  /**
   * colours dispatched per sweep: those the stretches took a frame or two
   * ago, and one more (more if some had to share the last); all of them when
   * `deterministic`, so that how many does not depend on when a read back came
   */
  private nColors = 12;
  private deterministic: boolean;

  private buffers: Record<string, GPUBuffer> = {};
  private pipelines: Record<string, GPUComputePipeline> = {};
  private layouts: Record<string, GPUBindGroupLayout> = {};
  private groups: Record<string, GPUBindGroup> = {};
  private passLayout: GPUBindGroupLayout;
  private passBuffer: GPUBuffer;
  private passGroup: GPUBindGroup;
  private paramData = new ArrayBuffer(4 * PARAMS.length);
  private query: GPUQuerySet | null = null;
  private queryBuffer: GPUBuffer | null = null;
  /** buffers the read backs are mapped from, those not in use */
  private staging: GPUBuffer[] = [];
  private watched: number[] = [];

  /**
   * `device`: the renderer's (three.js: `renderer.backend.device`), for the
   * tubes to read the nodes where they are. `options.stretch`: nodes per
   * stretch of cord, solved one after the other in the order of the CPU (see
   * docs/gpu.md): longer stretches carry the tension along a cord in fewer
   * sweeps, shorter ones give the GPU more to do at once (5 by default).
   * `options.deterministic`: sweeps every colour, empty or not (slower for
   * small scenes), rather than as many as the stretches took a frame or two
   * ago: then a run gives the same result again (see docs/gpu.md).
   */
  constructor(device: GPUDevice, params: Partial<SimParams> = {}, options: { stretch?: number; deterministic?: boolean } = {}) {
    this.device = device;
    this.P = defaultSimParams(params);
    this.stretchLength = options.stretch ?? 5;
    this.deterministic = options.deterministic ?? false;
    if (this.deterministic) this.nColors = COLORS;
    this.passLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 16 } }],
    });
    // per colour: its index, and whether its stretches are swept forward
    // (the first COLORS forward, the next backward)
    this.passBuffer = device.createBuffer({ size: PASS_STRIDE * 2 * COLORS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pass = new Uint32Array(PASS_STRIDE / 4 * 2 * COLORS);
    for (let e = 0; e < 2 * COLORS; e++) { pass[e * PASS_STRIDE / 4] = e % COLORS; pass[e * PASS_STRIDE / 4 + 1] = e < COLORS ? 1 : 0; }
    device.queue.writeBuffer(this.passBuffer, 0, pass);
    this.passGroup = device.createBindGroup({ layout: this.passLayout, entries: [{ binding: 0, resource: { buffer: this.passBuffer, size: 16 } }] });
    for (const [name, k] of Object.entries(KERNELS)) {
      const layout = device.createBindGroupLayout({
        entries: k.uses.map((r, b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: RESOURCES[r].type } })),
      });
      this.layouts[name] = layout;
      const module = device.createShaderModule({ label: name, code: kernelSource(k) });
      module.getCompilationInfo().then((info) => {
        for (const m of info.messages) if (m.type === 'error') console.error(`GPUCableSim: ${name}:${m.lineNum}:${m.linePos} ${m.message}`);
      });
      this.pipelines[name] = device.createComputePipeline({
        label: name,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout, this.passLayout] }),
        compute: { module, entryPoint: 'main' },
      });
    }
    if (device.features.has('timestamp-query')) {
      this.query = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.queryBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
  }

  /** the nodes on the GPU, a vec4 each: x, y, z and the radius of the cable there (what GPUTubeFeed copies to the tubes) */
  get positions(): GPUBuffer { return this.buffers.pos; }
  /** the cables on the GPU, a vec4<u32> each: first node, count */
  get cableBuffer(): GPUBuffer { return this.buffers.cables; }

  // ------------------------------------------------------------ building

  /** Makes the cables, replacing any there were (see CableSim.build). Returns the first node of each cable. */
  build(defs: CableInit[]): number[] {
    this.defs = defs;
    let N = 0;
    for (const d of defs) N += d.points.length / 3;
    this.n = N;
    this.x = new Float64Array(3 * N);
    this.mass = new Float64Array(N);
    this.rest = new Float64Array(N);
    this.radius = new Float64Array(N);
    this.cableOf = new Int32Array(N);
    this.segOK = new Uint8Array(N);
    this.pinned = new Uint8Array(N);
    this.target = new Float64Array(3 * N);
    this.targetAt = new Float64Array(3 * N);
    this.targetK = new Float64Array(N);
    this.targetLead = new Float64Array(N);
    this.targetHeavy = new Uint8Array(N);
    this.firm = new Uint8Array(defs.length);
    this.led = new Uint8Array(defs.length);
    this.wokeFor = new Int32Array(defs.length);
    this.toWake.clear();
    this.wakeEvery = false;
    this.cables = [];
    const bend = new Float64Array(N);
    const starts: number[] = [];
    let at = 0;
    defs.forEach((d, ci) => {
      const cnt = d.points.length / 3;
      starts.push(at);
      for (let k = 0; k < cnt; k++) {
        const i = at + k;
        for (let e = 0; e < 3; e++) this.x[3 * i + e] = d.points[3 * k + e];
        const m = d.mass ?? 0.001;
        this.mass[i] = k < 2 || k >= cnt - 2 ? d.endMass ?? m : m;
        this.cableOf[i] = ci;
        this.radius[i] = d.radius;
        const edge = Math.min(k, cnt - 1 - k), b = d.bend ?? 1;
        bend[i] = !d.boot ? b : edge <= 2 ? b * BOOT.stiff : edge <= 4 ? b * BOOT.firm : b;
        if (k < cnt - 1) {
          this.segOK[i] = 1;
          this.rest[i] = Math.hypot(d.points[3 * k + 3] - d.points[3 * k], d.points[3 * k + 4] - d.points[3 * k + 1], d.points[3 * k + 5] - d.points[3 * k + 2]);
        }
      }
      this.cables.push({ start: at, count: cnt, asleep: false, calm: 0 });
      at += cnt;
    });

    // the cells of the hash: two segments touch only if their midpoints are at most a cell apart (see CableSim.collide)
    let rMax = 0, lMax = 0;
    for (let i = 0; i < N; i++) { rMax = Math.max(rMax, this.radius[i]); if (this.segOK[i]) lMax = Math.max(lMax, this.rest[i]); }
    this.invCell = 1 / (lMax + 2 * rMax + this.P.marginCap + 1e-4);
    this.hashCells = 1 << Math.ceil(Math.log2(Math.max(64, N * 2)));

    // the stretches of cord: each cable cut into stretches of stretchLength nodes or fewer
    const chunks: number[] = [];
    for (const cb of this.cables) {
      const parts = Math.ceil(cb.count / this.stretchLength);
      for (let p = 0; p < parts; p++) {
        const a = Math.floor((p * cb.count) / parts), b = Math.floor(((p + 1) * cb.count) / parts);
        chunks.push(cb.start + a, b - a);
      }
    }
    const K = (this.nChunks = chunks.length / 2);

    for (const b of Object.values(this.buffers)) b.destroy();
    const dev = this.device, S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (name: string, bytes: number, usage = S) => (this.buffers[name] = dev.createBuffer({ label: name, size: Math.max(16, bytes), usage }));
    make('params', 4 * PARAMS.length, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    make('pos', 16 * N);
    make('state', 48 * N);
    make('sleepRef', 16 * N);
    make('info', 16 * N);
    make('segs', 16 * N);
    make('tgt', 32 * N);
    make('cables', 16 * defs.length);
    make('hash', 4 * (2 * this.hashCells + N));
    make('contacts', 2 * N * M * CONTACT_BYTES);
    make('refs', 4 * 2 * N * REF);
    make('colors', 4 * (COL.node + 5 * K + N), S | GPUBufferUsage.INDIRECT);
    // the contacts found this frame: their count and counters, then their
    // slots, then the indirect arguments of the kernels per contact, then the
    // cables to wake
    make('found', 4 * (4 + N * M + 3 + defs.length), S | GPUBufferUsage.INDIRECT);
    make('adj', 4 * (N + 2 * N * MA));
    make('stats', 4 * STAT.size);
    this.statics = [];
    this.uploadStatics();

    const q = dev.queue;
    const pos = new Float32Array(4 * N), state = new Float32Array(12 * N);
    for (let i = 0; i < N; i++) {
      for (let e = 0; e < 3; e++) { pos[4 * i + e] = this.x[3 * i + e]; state[12 * i + e] = this.x[3 * i + e]; state[12 * i + 4 + e] = this.x[3 * i + e]; }
      pos[4 * i + 3] = this.radius[i];
    }
    q.writeBuffer(this.buffers.pos, 0, pos);
    q.writeBuffer(this.buffers.sleepRef, 0, pos);
    q.writeBuffer(this.buffers.state, 0, state);
    this.infoData = new ArrayBuffer(16 * N);
    const fi = new Float32Array(this.infoData);
    for (let i = 0; i < N; i++) { fi[4 * i] = this.mass[i]; fi[4 * i + 1] = bend[i]; fi[4 * i + 2] = this.rest[i]; }
    for (let i = 0; i < N; i++) this.setFlags(i);
    this.infoDirty = [0, N];
    this.tgtData = new Float32Array(8 * N);
    this.tgtDirty = [0, N];
    const chunkOf = new Uint32Array(N);
    for (let k = 0; k < K; k++) for (let i = chunks[2 * k]; i < chunks[2 * k] + chunks[2 * k + 1]; i++) chunkOf[i] = k;
    q.writeBuffer(this.buffers.colors, 4 * (COL.node + 2 * K), Uint32Array.from(chunks));
    q.writeBuffer(this.buffers.colors, 4 * (COL.node + 5 * K), chunkOf);
    q.writeBuffer(this.buffers.cables, 0, Uint32Array.from(this.cables.flatMap((c) => [c.start, c.count, 0, 0])));
    this.cur = 0;
    this.frame = 0;
    this.readFrame = -1;
    this.bindAll();
    return starts;
  }

  /** the static shapes the cables collide with, in place of the previous ones */
  setStatics(shapes: StaticShape[]): void {
    this.statics = shapes;
    this.uploadStatics();
    this.bindAll();
    this.wakeAll();
  }

  private uploadStatics(): void {
    const S = this.statics, d = new Float32Array(16 * Math.max(1, S.length));
    S.forEach((s, k) => {
      const o = 16 * k, mu = s.mu ?? this.P.muStatic;
      if (s.kind === 'capsule') {
        d.set([s.ax, s.ay, s.az, s.r, s.bx, s.by, s.bz, mu], o);
        d.set([Math.min(s.ax, s.bx) - s.r, Math.min(s.ay, s.by) - s.r, Math.min(s.az, s.bz) - s.r, 0, Math.max(s.ax, s.bx) + s.r, Math.max(s.ay, s.by) + s.r, Math.max(s.az, s.bz) + s.r, 0], o + 8);
      } else {
        d.set([0, 0, 0, 0, 0, 0, 0, mu, ...s.min, 1, ...s.max, 0], o);
      }
    });
    this.buffers.statics?.destroy();
    this.buffers.statics = this.device.createBuffer({ label: 'statics', size: Math.max(STATIC_BYTES, d.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.buffers.statics, 0, d);
  }

  private bindAll(): void {
    if (!this.buffers.pos) return;
    for (const [name, k] of Object.entries(KERNELS)) {
      this.groups[name] = this.device.createBindGroup({
        layout: this.layouts[name],
        entries: k.uses.map((r, b) => ({ binding: b, resource: { buffer: this.buffers[RESOURCES[r].buffer] } })),
      });
    }
  }

  private setFlags(i: number): void {
    const c = this.cableOf[i];
    const f = (this.pinned[i] ? F.pinned : 0) | (this.segOK[i] ? F.seg : 0) | (this.targetHeavy[i] && this.targetK[i] > 0 ? F.heavy : 0)
      | (this.led[c] ? F.led : 0) | (this.firm[c] ? F.firm : 0) | (c << 8);
    new Uint32Array(this.infoData, 16 * i + 12, 1)[0] = f;
    this.infoDirty[0] = Math.min(this.infoDirty[0], i);
    this.infoDirty[1] = Math.max(this.infoDirty[1], i + 1);
  }

  // ------------------------------------------------------------- control

  /** holds node i at (x, y, z), at rest, until `unpin` */
  pin(i: number, x: number, y: number, z: number): void {
    this.pinned[i] = 1;
    this.x[3 * i] = x; this.x[3 * i + 1] = y; this.x[3 * i + 2] = z;
    const q = this.device.queue;
    q.writeBuffer(this.buffers.pos, 16 * i, new Float32Array([x, y, z, this.radius[i]]));
    q.writeBuffer(this.buffers.state, 48 * i, new Float32Array([x, y, z, 0, x, y, z, 0, 0, 0, 0, 0]));
    this.setFlags(i);
    this.wakeCable(this.cableOf[i]);
  }

  /** sets the velocity of every node (xyz interleaved, m/s), the nodes where they were built: right after `build` */
  setVelocities(v: ArrayLike<number>): void {
    const d = new Float32Array(12 * this.n);
    for (let i = 0; i < this.n; i++) {
      for (let e = 0; e < 3; e++) { d[12 * i + e] = this.x[3 * i + e]; d[12 * i + 4 + e] = this.x[3 * i + e]; d[12 * i + 8 + e] = v[3 * i + e]; }
    }
    this.device.queue.writeBuffer(this.buffers.state, 0, d);
  }

  unpin(i: number): void {
    this.pinned[i] = 0;
    this.setFlags(i);
    this.wakeCable(this.cableOf[i]);
  }

  /** leads node i towards (x, y, z), a spring on a leash (see CableSim.setTarget) */
  setTarget(i: number, x: number, y: number, z: number, k: number, lead: number, heavy = false): void {
    this.target[3 * i] = x; this.target[3 * i + 1] = y; this.target[3 * i + 2] = z;
    this.targetK[i] = k;
    this.targetLead[i] = lead;
    this.targetHeavy[i] = heavy ? 1 : 0;
    this.setFlags(i);
    this.wakeCable(this.cableOf[i]);
  }

  clearTarget(i: number): void {
    this.targetK[i] = 0;
    this.targetHeavy[i] = 0;
    this.setFlags(i);
    this.tgtData[8 * i + 3] = 0;
    this.tgtDirty[0] = Math.min(this.tgtDirty[0], i);
    this.tgtDirty[1] = Math.max(this.tgtDirty[1], i + 1);
    this.wakeCable(this.cableOf[i]);
  }

  /** cable c is drawn by a hand: firmer against the static shapes (see `kCapHeld`) */
  setFirm(c: number, on: boolean): void {
    this.firm[c] = on ? 1 : 0;
    const cb = this.cables[c];
    for (let i = cb.start; i < cb.start + cb.count; i++) this.setFlags(i);
  }

  /** wakes cable ci, from the next step (`cables[ci].asleep` at once) */
  wakeCable(ci: number): void {
    const c = this.cables[ci];
    c.asleep = false;
    c.calm = 0;
    this.toWake.add(ci);
    this.wokeFor[ci] = this.frame;
  }

  /** wakes every cable */
  wakeAll(): void {
    for (const c of this.cables) { c.asleep = false; c.calm = 0; }
    this.wakeEvery = true;
    this.wokeFor.fill(this.frame);
  }

  // ------------------------------------------------------------- reading

  /**
   * How hard the target of node i pulls on it (N), and which way into `out`
   * (see CableSim.targetPull): from where the node was at the last read back,
   * a frame or two ago.
   */
  targetPull(i: number, out?: { x: number; y: number; z: number }): number {
    const k = this.targetK[i];
    if (!(k > 0)) { if (out) { out.x = 0; out.y = 0; out.z = 0; } return 0; }
    const j = 3 * i, t = this.targetAt;
    let ex = this.x[j] - t[j], ey = this.x[j + 1] - t[j + 1], ez = this.x[j + 2] - t[j + 2];
    const el = Math.sqrt(ex * ex + ey * ey + ez * ez), lead = this.targetLead[i];
    if (el > lead) { const f = lead / el; ex *= f; ey *= f; ez *= f; }
    const hr = this.P.dt / this.P.refSubsteps;
    const f = (k * this.mass[i]) / (hr * hr);
    if (out) { out.x = -f * ex; out.y = -f * ey; out.z = -f * ez; }
    return f * Math.sqrt(ex * ex + ey * ey + ez * ez);
  }

  /** every position as it is after the frames submitted so far (xyz interleaved) */
  async readPositions(): Promise<Float32Array> {
    const N = this.n;
    const buf = this.device.createBuffer({ size: Math.max(16, 16 * N), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.buffers.pos, 0, buf, 0, 16 * N);
    this.device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(buf.getMappedRange()), out = new Float32Array(3 * N);
    for (let i = 0; i < N; i++) { out[3 * i] = v[4 * i]; out[3 * i + 1] = v[4 * i + 1]; out[3 * i + 2] = v[4 * i + 2]; }
    buf.unmap();
    buf.destroy();
    return out;
  }

  // ---------------------------------------------------------------- step

  /** one frame of `P.dt` seconds in `P.substeps` substeps, encoded and submitted (the GPU runs it later) */
  step(): void {
    const t0 = performance.now();
    const P = this.P, N = this.n, dev = this.device;
    if (N === 0) return;
    const h = P.dt / P.substeps;
    this.reach();
    this.cur = 1 - this.cur;
    this.writeParams(h);
    this.flush();

    const enc = dev.createCommandEncoder();
    const B = this.buffers;
    enc.clearBuffer(B.hash, 0, 4 * this.hashCells);
    enc.clearBuffer(B.hash, 4 * (this.hashCells + N), 4 * this.hashCells);
    enc.clearBuffer(B.refs, 4 * this.cur * N * REF, 4 * N * REF);
    enc.clearBuffer(B.colors, 0, 4 * COL.node);
    enc.clearBuffer(B.stats);
    enc.clearBuffer(B.found, 0, 16);
    enc.clearBuffer(B.found, 4 * (7 + N * M), 4 * this.cables.length);
    // one compute pass, timed as a whole; when profiling, one per dispatch, each timed (see `profile`)
    const prof = this.profiling ? { names: [] as string[], set: this.profileSet! } : null;
    const main = prof ? null : enc.beginComputePass(this.query ? { timestampWrites: { querySet: this.query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    const go = (name: string, offset: number, dispatch: (pass: GPUComputePassEncoder) => void) => {
      let pass = main!;
      if (prof) {
        // (beyond the queries there are, untimed)
        const j = prof.names.length, timed = 2 * j + 2 <= PROFILE_QUERIES;
        if (timed) prof.names.push(name);
        pass = enc.beginComputePass(timed ? { timestampWrites: { querySet: prof.set, beginningOfPassWriteIndex: 2 * j, endOfPassWriteIndex: 2 * j + 1 } } : {});
      }
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(0, this.groups[name]);
      pass.setBindGroup(1, this.passGroup, [offset]);
      dispatch(pass);
      if (prof) pass.end();
    };
    const run = (name: string, count: number, offset = 0) => go(name, offset, (p) => p.dispatchWorkgroups(Math.ceil(count / WG)));
    /** per contact found (and per segment): as many workgroups as `foundArgs` counted */
    const perContact = (name: string) => go(name, 0, (p) => p.dispatchWorkgroupsIndirect(B.found, 4 * (4 + N * M)));
    const nc = this.cables.length;
    // contacts, colours
    run('sleepFlags', N);
    run('hashInsert', N);
    run('detect', N);
    run('sortRefs', N);
    run('adjacency', N);
    run('foundArgs', 1);
    const K = this.nChunks;
    run('colorInit', K);
    for (let r = 0; r < ROUNDS; r++) { run('colorPick', K); run('colorKeep', K); }
    run('colorCount', K);
    run('colorScan', 1);
    run('colorFill', K);
    const R = this.R(h);
    for (let s = 0; s < P.substeps; s++) {
      perContact('contactStart');
      run('predict', N);
      for (let it = 0; it < P.iterations; it++) {
        this.flip = !this.flip;
        // one dispatch per colour, as many workgroups as colorScan counted
        for (let k = 0; k < this.nColors; k++) {
          const c = this.flip ? k : this.nColors - 1 - k;
          go('solve', (this.flip ? c : COLORS + c) * PASS_STRIDE, (p) => p.dispatchWorkgroupsIndirect(B.colors, 4 * (COL.args + 3 * c)));
        }
        perContact('dualContact');
      }
      for (let k = 0; k < R.pbdPasses; k++) run('pbd', nc, k % 2 === 0 ? 0 : COLORS * PASS_STRIDE);
      run('velocity', N);
      perContact('cvImpulse'); run('cvApply', N);
      perContact('cvImpulse'); run('cvApplyRest', N);
    }
    run('sleepMoved', N);
    run('sleepCable', nc);
    run('sleepNodes', N);
    main?.end();
    if (prof) this.readProfile(enc, prof.names);
    this.readBack(enc);
    dev.queue.submit([enc.finish()]);
    this.frame++;
    this.stats.ms = performance.now() - t0;
  }

  /** the settings of substeps of length h (see CableSim.convert) */
  private R(h: number) {
    const P = this.P, hr = P.dt / P.refSubsteps, r = h / hr;
    return { r, hr, pbdPasses: Math.max(1, Math.round(P.pbdPasses * r)) };
  }

  private writeParams(h: number): void {
    const P = this.P, { r, hr } = this.R(h), kept = (f: number) => f ** r;
    const v: Record<string, number> = {
      n: this.n, nCables: this.cables.length, nStatics: this.statics.length, cur: this.cur, hashMask: this.hashCells - 1, nColors: this.nColors,
      h, ih: 1 / h, gy: P.gravity * h * h, damp: 1 / (1 + P.drag * h), mh: 1 / (hr * hr), mhk: 1 / (h * hr),
      lamWarm: kept(P.lamWarm), lamWarmS: kept(P.lamWarmS), gamma: kept(P.gamma), alpha: kept(P.alpha), alphaS: kept(P.alphaS),
      maxDepen: P.maxDepen * r, restDecel: P.restFriction * r, maxMove: P.maxMove * r,
      stretch: P.stretch, kMin: P.kMin, beta: P.beta, frictionK: P.frictionK, contactDamping: P.contactDamping, restFriction: P.restFriction,
      handWeight: P.handWeight, kCap: P.kCap, kCapCable: P.kCapCable, kCapCableHeld: P.kCapCableHeld, kCapHeld: P.kCapHeld,
      margin: P.margin, marginLead: P.marginLead, marginCap: P.marginCap, marginStatic: P.marginStatic, frame: P.dt,
      invCell: this.invCell, nChunks: this.nChunks, gammaRaw: P.gamma, muCable: P.muCable, muStatic: P.muStatic,
      sleepDist: P.sleepDist, sleepTime: P.sleepTime, wakeSpeed: P.wakeSpeed, wakeCreep: P.wakeCreep, wakeSink: P.wakeSink,
    };
    const dv = new DataView(this.paramData);
    PARAMS.forEach((p, k) => {
      const x = v[p.name] ?? 0;
      if (p.type === 'u32') dv.setUint32(4 * k, x, true); else dv.setFloat32(4 * k, x, true);
    });
    this.device.queue.writeBuffer(this.buffers.params, 0, this.paramData);
  }

  /** sends what changed of `info` and `tgt` */
  private flush(): void {
    const q = this.device.queue;
    const [a, b] = this.infoDirty;
    if (a < b) q.writeBuffer(this.buffers.info, 16 * a, this.infoData, 16 * a, 16 * (b - a));
    const [c, d] = this.tgtDirty;
    if (c < d) q.writeBuffer(this.buffers.tgt, 32 * c, this.tgtData, 8 * c, 8 * (d - c));
    this.infoDirty = [Infinity, -Infinity];
    this.tgtDirty = [Infinity, -Infinity];
    // woken: awake, still for no time (the GPU's state and calm, see sleepCable)
    if (this.wakeEvery) q.writeBuffer(this.buffers.cables, 0, Uint32Array.from(this.cables.flatMap((cb) => [cb.start, cb.count, 0, 0])));
    else for (const ci of this.toWake) q.writeBuffer(this.buffers.cables, 16 * ci + 8, new Uint32Array(2));
    this.toWake.clear();
    this.wakeEvery = false;
  }

  /**
   * Where each target pulls this step, within reach of the pins of its cord
   * (see CableSim.reach): the pins do not move, so the CPU knows all it needs.
   */
  private reach(): void {
    const x = this.x, t = this.target, at = this.targetAt, K = this.targetK, rest = this.rest, pinned = this.pinned;
    for (let ci = 0; ci < this.cables.length; ci++) {
      const cb = this.cables[ci], end = cb.start + cb.count;
      let any = false, heavy = 0;
      for (let i = cb.start; i < end; i++) if (K[i] > 0) { any = true; heavy |= this.targetHeavy[i]; }
      if (this.led[ci] !== heavy) { this.led[ci] = heavy; for (let i = cb.start; i < end; i++) this.setFlags(i); }
      if (!any) continue;
      // the nearest pinned node before each target, and the length of cord to it
      const before = new Map<number, number>(), lenOf = new Map<number, number>();
      let pin = -1, arc = 0;
      for (let i = cb.start; i < end; i++) {
        if (i > cb.start) arc += rest[i - 1];
        if (pinned[i]) { pin = i; arc = 0; }
        if (K[i] <= 0) continue;
        at[3 * i] = t[3 * i]; at[3 * i + 1] = t[3 * i + 1]; at[3 * i + 2] = t[3 * i + 2];
        before.set(i, pinned[i] ? -1 : pin);
        lenOf.set(i, arc);
      }
      pin = -1; arc = 0;
      for (let i = end - 1; i >= cb.start; i--) {
        if (i < end - 1) arc += rest[i];
        if (pinned[i]) { pin = i; arc = 0; }
        if (K[i] <= 0 || pinned[i]) continue;
        const a = before.get(i)!, ra = lenOf.get(i)!;
        for (let it = 0; it < (a >= 0 && pin >= 0 ? 4 : 1); it++) {
          if (a >= 0) within(at, 3 * i, x, 3 * a, ra);
          if (pin >= 0) within(at, 3 * i, x, 3 * pin, arc);
        }
      }
      for (let i = cb.start; i < end; i++) {
        if (K[i] <= 0) continue;
        this.tgtData.set([at[3 * i], at[3 * i + 1], at[3 * i + 2], K[i], this.targetLead[i], 0, 0, 0], 8 * i);
        this.tgtDirty[0] = Math.min(this.tgtDirty[0], i);
        this.tgtDirty[1] = Math.max(this.tgtDirty[1], i + 1);
      }
    }
  }

  /** copies what the CPU reads back (timings, counters, watched nodes) into a buffer it maps when the GPU is done */
  private readBack(enc: GPUCommandEncoder): void {
    const N = this.n;
    const nodes: number[] = this.readAll ? [] : [...this.watch];
    if (!this.readAll) for (let i = 0; i < N; i++) if (this.targetK[i] > 0) nodes.push(i);
    const head = 16 + 4 * STAT.size, nc = this.cables.length;
    const size = head + 16 * nc + 16 * (this.readAll ? N : nodes.length);
    let buf = this.staging.pop();
    if (buf && buf.size < size) { buf.destroy(); buf = undefined; }
    if (!buf) {
      // (no more than a few in flight: a GPU that falls behind is read less often)
      if (this.watched.length > 3) return;
      buf = this.device.createBuffer({ size: Math.max(size, 256), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    if (this.query && this.queryBuffer) {
      enc.resolveQuerySet(this.query, 0, 2, this.queryBuffer, 0);
      enc.copyBufferToBuffer(this.queryBuffer, 0, buf, 0, 16);
    }
    enc.copyBufferToBuffer(this.buffers.stats, 0, buf, 16, 4 * STAT.size);
    enc.copyBufferToBuffer(this.buffers.cables, 0, buf, head, 16 * nc);
    const at = head + 16 * nc;
    if (this.readAll) enc.copyBufferToBuffer(this.buffers.pos, 0, buf, at, 16 * N);
    else nodes.forEach((i, k) => enc.copyBufferToBuffer(this.buffers.pos, 16 * i, buf, at + 16 * k, 16));
    const frame = this.frame, b = buf, n = this.n;
    this.watched.push(frame);
    this.device.queue.onSubmittedWorkDone().then(() => b.mapAsync(GPUMapMode.READ)).then(() => {
      this.watched.splice(this.watched.indexOf(frame), 1);
      if (n === this.n && nc === this.cables.length && frame > this.readFrame) {
        const r = b.getMappedRange();
        if (this.query) { const t = new BigUint64Array(r, 0, 2); this.stats.gpuMs = Number(t[1] - t[0]) / 1e6; }
        const s = new Uint32Array(r, 16, STAT.size);
        this.stats.contacts = s[STAT.contacts];
        this.stats.overflow = [s[STAT.overflowS], s[STAT.overflowC], s[STAT.overflowH]];
        this.stats.uncoloured = s[STAT.uncoloured];
        this.stats.clamped = s[STAT.clamped];
        this.stats.awake = s[STAT.awake];
        // (not those woken since: they are awake)
        const cs = new Uint32Array(r, head, 4 * nc), calm = new Float32Array(r, head, 4 * nc);
        this.cables.forEach((cb, ci) => {
          if (frame < this.wokeFor[ci]) return;
          cb.asleep = (cs[4 * ci + 2] & 1) !== 0;
          cb.calm = calm[4 * ci + 3];
        });
        if (!this.deterministic) this.nColors = Math.min(COLORS, Math.max(s[STAT.clamped] > 0 ? this.nColors + 4 : 0, s[STAT.maxColor] + 2, 4));
        const colors = Array.from(s.subarray(STAT.colors, STAT.colors + COLORS));
        let used = 0;
        colors.forEach((c, k) => { if (c > 0) used = k + 1; });
        this.stats.colors = colors.slice(0, used);
        const p = new Float32Array(r, at);
        if (this.readAll) for (let i = 0; i < n; i++) { this.x[3 * i] = p[4 * i]; this.x[3 * i + 1] = p[4 * i + 1]; this.x[3 * i + 2] = p[4 * i + 2]; }
        else nodes.forEach((i, k) => { this.x[3 * i] = p[4 * k]; this.x[3 * i + 1] = p[4 * k + 1]; this.x[3 * i + 2] = p[4 * k + 2]; });
        this.readFrame = frame;
      }
      b.unmap();
      this.staging.push(b);
    }).catch(() => {});
  }

  private profiling = false;
  private profileSet: GPUQuerySet | null = null;
  private profileDone: ((times: Record<string, number>) => void) | null = null;

  /**
   * Times the next step kernel by kernel (ms on the GPU, summed by kernel),
   * each dispatch in a compute pass of its own: the sum is more than a step
   * takes otherwise. Needs 'timestamp-query'.
   */
  profile(): Promise<Record<string, number>> {
    if (!this.query) return Promise.reject(new Error('profile needs the timestamp-query feature'));
    this.profileSet ??= this.device.createQuerySet({ type: 'timestamp', count: PROFILE_QUERIES });
    this.profiling = true;
    return new Promise((resolve) => (this.profileDone = resolve));
  }

  private readProfile(enc: GPUCommandEncoder, names: string[]): void {
    this.profiling = false;
    const n = 2 * names.length, dev = this.device;
    const resolveBuf = dev.createBuffer({ size: 8 * n, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = dev.createBuffer({ size: 8 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.resolveQuerySet(this.profileSet!, 0, n, resolveBuf, 0);
    enc.copyBufferToBuffer(resolveBuf, 0, read, 0, 8 * n);
    const done = this.profileDone;
    this.profileDone = null;
    dev.queue.onSubmittedWorkDone().then(() => read.mapAsync(GPUMapMode.READ)).then(() => {
      const t = new BigUint64Array(read.getMappedRange()), times: Record<string, number> = {};
      names.forEach((name, j) => (times[name] = (times[name] ?? 0) + Number(t[2 * j + 1] - t[2 * j]) / 1e6));
      read.unmap(); read.destroy(); resolveBuf.destroy();
      done?.(times);
    });
  }

  /** a promise that resolves when the GPU has done every frame submitted, and the last read back has come */
  async settled(): Promise<void> {
    await this.device.queue.onSubmittedWorkDone();
    while (this.watched.length) await new Promise((r) => setTimeout(r, 1));
  }

  dispose(): void {
    for (const b of Object.values(this.buffers)) b.destroy();
    for (const b of this.staging) b.destroy();
    this.passBuffer.destroy();
    this.query?.destroy();
    this.queryBuffer?.destroy();
    this.buffers = {};
  }
}

/** brings p[o..o+2] within r of c[oc..oc+2] */
function within(p: Float64Array, o: number, c: ArrayLike<number>, oc: number, r: number): void {
  const dx = p[o] - c[oc], dy = p[o + 1] - c[oc + 1], dz = p[o + 2] - c[oc + 2];
  const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (d <= r) return;
  const f = r / d;
  p[o] = c[oc] + dx * f; p[o + 1] = c[oc + 1] + dy * f; p[o + 2] = c[oc + 2] + dz * f;
}
