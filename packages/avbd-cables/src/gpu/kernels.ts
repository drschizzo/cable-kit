// The compute shaders of GPUCableSim, in WGSL: the solver of cableSim.ts,
// kernel by kernel (see docs/gpu.md). Each kernel names the buffers it uses;
// their declarations are written from that list, and so are the layouts of
// the bind groups (gpuSim.ts), so the two never disagree.

/** invocations per workgroup */
export const WG = 64;
/** contact slots of a segment: against the static shapes, against the segments after it */
export const MS = 4;
export const MC = 12;
export const M = MS + MC;
/** contacts a segment may be the other side of */
export const MH = 20;
/** contacts a node may have (beyond, they are dropped) */
export const MA = 64;
/** u32 per segment in `refs`: static count, cable count, references count, unused, then the references */
export const REF = 4 + MH;
/** invocations that share the contacts of a node in the block solve */
export const LANES = 8;
/** colours at most */
export const COLORS = 32;
/** rounds of the colouring */
export const ROUNDS = 16;
/**
 * layout of `colors` (u32): count, start, indirect arguments, fill cursor per
 * colour; from `node`, per stretch of cord: its colour, the lists of the
 * stretches by colour, its first node and count (2), the colour it picked in
 * this round of the colouring; then the stretch of each node
 */
export const COL = { count: 0, start: COLORS, args: 2 * COLORS, fill: 5 * COLORS, node: 6 * COLORS };
/** layout of `stats` (u32) */
export const STAT = { overflowS: 0, overflowC: 1, overflowH: 2, uncoloured: 3, clamped: 4, contacts: 5, maxColor: 6, colors: 8, size: 8 + COLORS };

/** the uniforms of a step, in order (all 4 bytes) */
export const PARAMS = [
  'n:u32', 'nCables:u32', 'nStatics:u32', 'cur:u32', 'hashMask:u32', 'nChunks:u32', 'nColors:u32', 'pad1:u32',
  'h', 'ih', 'gy', 'damp', 'mh', 'mhk', 'lamWarm', 'lamWarmS',
  'gamma', 'alpha', 'alphaS', 'maxDepen', 'restDecel', 'maxMove', 'stretch', 'kMin',
  'beta', 'frictionK', 'contactDamping', 'restFriction', 'handWeight', 'kCap', 'kCapCable', 'kCapCableHeld',
  'kCapHeld', 'margin', 'marginLead', 'marginCap', 'marginStatic', 'frame', 'invCell', 'gammaRaw',
  'muCable', 'muStatic', 'pad2', 'pad3',
].map((f) => { const [name, type] = f.split(':'); return { name, type: type ?? 'f32' }; });

/** the buffers, as a kernel declares them */
export const RESOURCES: Record<string, { type: 'uniform' | 'storage' | 'read-only-storage'; wgsl: string; buffer: string }> = {
  P: { type: 'uniform', wgsl: 'var<uniform> P: Params', buffer: 'params' },
  pos: { type: 'storage', wgsl: 'var<storage, read_write> pos: array<vec4f>', buffer: 'pos' },
  state: { type: 'storage', wgsl: 'var<storage, read_write> state: array<State>', buffer: 'state' },
  info: { type: 'storage', wgsl: 'var<storage, read_write> info: array<Info>', buffer: 'info' },
  segs: { type: 'storage', wgsl: 'var<storage, read_write> segs: array<vec4f>', buffer: 'segs' },
  tgt: { type: 'storage', wgsl: 'var<storage, read_write> tgt: array<vec4f>', buffer: 'tgt' },
  cables: { type: 'storage', wgsl: 'var<storage, read_write> cables: array<vec4u>', buffer: 'cables' },
  statics: { type: 'storage', wgsl: 'var<storage, read_write> statics: array<Static>', buffer: 'statics' },
  hash: { type: 'storage', wgsl: 'var<storage, read_write> hash: array<atomic<u32>>', buffer: 'hash' },
  contacts: { type: 'storage', wgsl: 'var<storage, read_write> contacts: array<Contact>', buffer: 'contacts' },
  refs: { type: 'storage', wgsl: 'var<storage, read_write> refs: array<u32>', buffer: 'refs' },
  refsA: { type: 'storage', wgsl: 'var<storage, read_write> refsA: array<atomic<u32>>', buffer: 'refs' },
  colors: { type: 'storage', wgsl: 'var<storage, read_write> colors: array<u32>', buffer: 'colors' },
  colorsR: { type: 'read-only-storage', wgsl: 'var<storage, read> colors: array<u32>', buffer: 'colors' },
  colorsA: { type: 'storage', wgsl: 'var<storage, read_write> colorsA: array<atomic<u32>>', buffer: 'colors' },
  found: { type: 'read-only-storage', wgsl: 'var<storage, read> found: array<u32>', buffer: 'found' },
  foundA: { type: 'storage', wgsl: 'var<storage, read_write> foundA: array<atomic<u32>>', buffer: 'found' },
  adj: { type: 'storage', wgsl: 'var<storage, read_write> adj: array<u32>', buffer: 'adj' },
  stats: { type: 'storage', wgsl: 'var<storage, read_write> stats: array<atomic<u32>>', buffer: 'stats' },
};

const COMMON = /* wgsl */ `
struct Params { ${PARAMS.map((p) => `${p.name}: ${p.type},`).join(' ')} }
struct Pass { color: u32, forward: u32, a: u32, b: u32 }
/** start of the substep, prediction, velocity (xyz) */
struct State { x0: vec4f, xt: vec4f, v: vec4f }
/** flags: 1 pinned, 2 segment to the next node, 4 heavy target, 8 its cable is led by one, 16 firm; the cable from bit 8 */
struct Info { mass: f32, bendK: f32, rest: f32, flags: u32 }
/** a capsule (a, b, radius a.w) or a box (lo, hi; lo.w = 1), friction b.w */
struct Static { a: vec4f, b: vec4f, lo: vec4f, hi: vec4f }
struct Contact {
  n: vec4f,    // normal, s along segment a
  q: vec4f,    // point on the static shape, t along segment b
  anc: vec4f,  // friction anchor, rs (sum of the radii)
  lt: vec4f,   // tangential multiplier, normal multiplier
  k: vec4f,    // stiffness, its ceiling, mu, C at the start of the substep
  id: vec4u,   // segment b (NONE: a static shape), key, M/(h·h_ref) of the contact (its bits), static shape | tracked << 31
  dv: vec4f,   // scratch of the velocity pass
}
@group(1) @binding(0) var<uniform> ps: Pass;
fn outerProduct(u: vec3f, v: vec3f) -> mat3x3f { return mat3x3f(u * v.x, u * v.y, u * v.z); }

const NONE = 0xffffffffu;
const MS = ${MS}u; const MC = ${MC}u; const M = ${M}u; const MH = ${MH}u; const REF = ${REF}u;
const COLORS = ${COLORS}u; const MA = ${MA}u;
const PARALLEL = 0.06; const OVERLAP = 0.05; const DIP = 5e-4; const EDGE = 0.05;
`;

/** helpers that read the nodes (pos, info) */
const NODES = /* wgsl */ `
fn flags(i: u32) -> u32 { return info[i].flags; }
fn fixedNode(i: u32) -> bool { return (info[i].flags & 1u) != 0u; }
fn segOK(i: u32) -> bool { return i < P.n && (info[i].flags & 2u) != 0u; }
fn cableOf(i: u32) -> u32 { return info[i].flags >> 8u; }
fn cbase() -> u32 { return P.cur * P.n * M; }
fn rbase() -> u32 { return P.cur * P.n * REF; }
`;

/** helpers on contacts (pos, info, contacts) */
const CONTACTS = /* wgsl */ `
/** pA − pB at the contact (pA against a static shape), of the positions */
fn crel(c: Contact, a: u32) -> vec3f {
  let pa = mix(pos[a].xyz, pos[a + 1u].xyz, c.n.w);
  if (c.id.x == NONE) { return pa; }
  let b = c.id.x;
  return pa - mix(pos[b].xyz, pos[b + 1u].xyz, c.q.w);
}
/** C = n·(pA − pB) − rs */
fn contactC(c: Contact, a: u32) -> f32 { return cOf(c, crel(c, a)); }
fn cOf(c: Contact, rel: vec3f) -> f32 {
  var r = rel;
  if (c.id.x == NONE) { r -= c.q.xyz; }
  return dot(c.n.xyz, r) - c.anc.w;
}
fn forgive(c0: f32) -> f32 {
  if (c0 >= 0.0) { return 0.0; }
  return min(P.alpha * c0, c0 + P.maxDepen);
}
/** M/(h·h_ref) of the contact (its heavier node), the unit of its stiffnesses */
fn cmh(c: Contact) -> f32 { return bitcast<f32>(c.id.z); }
struct Fric { F: vec3f, k: f32, s: f32, slides: bool }
/** the friction force λt + kt·d, clamped to the cone of radius lim (see CableSim.friction); rel: pA − pB now */
fn friction(c: Contact, rel: vec3f, lim: f32) -> Fric {
  let n = c.n.xyz;
  var d = rel - c.anc.xyz;
  d -= dot(d, n) * n;
  let kt = P.frictionK * cmh(c);
  let l = c.lt.xyz;
  var F = kt * d + l - dot(l, n) * n;
  let FL = length(F);
  var r: Fric;
  r.k = kt; r.s = 1.0; r.slides = FL > lim;
  if (r.slides) { r.s = lim / FL; F *= r.s; }
  r.F = F;
  return r;
}
`;

/**
 * The contacts of node \`i\`: those of its two segments (it is the second
 * node of i − 1, the first of i), as their segment a or as their segment b.
 * Runs \`body\` with \`g\` (the contact) and \`w\` (the node's weight in it).
 */
function forContacts(i: string, body: string): string {
  return /* wgsl */ `
  for (var side = 0u; side < 2u; side++) {
    if (side == 0u && ${i} == 0u) { continue; }
    let sg = ${i} + side - 1u;
    if (!segOK(sg)) { continue; }
    let rb = rbase() + sg * REF;
    let nS = refs[rb]; let nC = refs[rb + 1u]; let nH = refs[rb + 2u];
    for (var kk = 0u; kk < nS + nC + nH; kk++) {
      var g: u32; var w: f32; var own = true;
      if (kk < nS + nC) {
        g = cbase() + sg * M + select(MS + kk - nS, kk, kk < nS);
        let s = contacts[g].n.w;
        w = select(s, 1.0 - s, side == 1u);
      } else {
        own = false;
        g = cbase() + refs[rb + 4u + kk - nS - nC];
        let t = contacts[g].q.w;
        w = select(-t, -(1.0 - t), side == 1u);
      }
      ${body}
    }
  }`;
}

const GEOM = /* wgsl */ `
struct Closest { s: f32, t: f32, d2: f32 }
/** closest points of segments p0p1 and q0q1 (Ericson §5.1.9; see geom.ts) */
fn segSeg(p0: vec3f, p1: vec3f, q0: vec3f, q1: vec3f) -> Closest {
  let d1 = p1 - p0; let d2 = q1 - q0; let r = p0 - q0;
  let a = dot(d1, d1); let e = dot(d2, d2); let f = dot(d2, r);
  var s = 0.0; var t = 0.0;
  let EPS = 1e-14;
  if (a <= EPS && e <= EPS) {
  } else if (a <= EPS) {
    t = clamp(f / e, 0.0, 1.0);
  } else {
    let c = dot(d1, r);
    if (e <= EPS) {
      s = clamp(-c / a, 0.0, 1.0);
    } else {
      let b = dot(d1, d2);
      let denom = a * e - b * b;
      // (relative to a·e: in f32 the cancellation leaves ~1e-7 of it)
      if (denom > 1e-6 * a * e) { s = clamp((b * f - c * e) / denom, 0.0, 1.0); } else { s = 0.5; }
      t = (b * s + f) / e;
      if (t < 0.0) { t = 0.0; s = clamp(-c / a, 0.0, 1.0); }
      else if (t > 1.0) { t = 1.0; s = clamp((b - c) / a, 0.0, 1.0); }
    }
  }
  let d = r + d1 * s - d2 * t;
  return Closest(s, t, dot(d, d));
}
struct BoxHit { s: f32, q: vec3f, n: vec3f, dist: f32 }
/** closest approach of segment p0p1 and the box lo–hi (see geom.ts) */
fn segBox(p0: vec3f, p1: vec3f, lo: vec3f, hi: vec3f) -> BoxHit {
  let d = p1 - p0; let dd = dot(d, d);
  var s = 0.5;
  for (var k = 0; k < 4; k++) {
    let q = clamp(p0 + d * s, lo, hi);
    if (dd < 1e-14) { break; }
    s = clamp(dot(q - p0, d) / dd, 0.0, 1.0);
  }
  let p = p0 + d * s;
  let q = clamp(p, lo, hi);
  let e = p - q; let dl = length(e);
  var h: BoxHit;
  h.s = s;
  if (dl > 1e-9) { h.q = q; h.n = e / dl; h.dist = dl; return h; }
  // inside: out through the nearest face
  let g0 = p - lo; let g1 = hi - p;
  var best = g0.x; h.n = vec3f(-1.0, 0.0, 0.0); h.q = vec3f(lo.x, p.y, p.z);
  if (g1.x < best) { best = g1.x; h.n = vec3f(1.0, 0.0, 0.0); h.q = vec3f(hi.x, p.y, p.z); }
  if (g0.y < best) { best = g0.y; h.n = vec3f(0.0, -1.0, 0.0); h.q = vec3f(p.x, lo.y, p.z); }
  if (g1.y < best) { best = g1.y; h.n = vec3f(0.0, 1.0, 0.0); h.q = vec3f(p.x, hi.y, p.z); }
  if (g0.z < best) { best = g0.z; h.n = vec3f(0.0, 0.0, -1.0); h.q = vec3f(p.x, p.y, lo.z); }
  if (g1.z < best) { best = g1.z; h.n = vec3f(0.0, 0.0, 1.0); h.q = vec3f(p.x, p.y, hi.z); }
  h.dist = -best;
  return h;
}
fn cellOf(p: vec3f) -> vec3i { return vec3i(floor(p * P.invCell)); }
fn bucketOf(c: vec3i) -> u32 {
  return ((bitcast<u32>(c.x) * 73856093u) ^ (bitcast<u32>(c.y) * 19349663u) ^ (bitcast<u32>(c.z) * 83492791u)) & P.hashMask;
}
`;

/** the dual updates (see CableSim.dualUpdate, frictionDual): of the stretch of segment s, of contact g of segment a */
const DUAL = /* wgsl */ `
fn dualSeg(s: u32) {
  if (!segOK(s) || (fixedNode(s) && fixedNode(s + 1u))) { return; }
  var g = segs[s];
  let C = length(pos[s].xyz - pos[s + 1u].xyz) - info[s].rest - P.alphaS * g.z;
  g.x += g.y * C;
  let kmax = P.stretch * max(info[s].mass, info[s + 1u].mass) * P.mhk;
  g.y = min(kmax, g.y + P.beta * abs(C) * kmax / P.stretch);
  segs[s] = g;
}
fn dualOf(g: u32, a: u32) {
  var c = contacts[g];
  let rel = crel(c, a);
  let C = cOf(c, rel) - forgive(c.k.w);
  if (abs(C) >= 1e-7) {
    let lam = min(0.0, c.lt.w + c.k.x * C);
    c.lt.w = lam;
    if (lam < 0.0) { c.k.x = min(c.k.y, c.k.x + P.beta * abs(C) * cmh(c)); }
  }
  var slides = true;
  if (c.lt.w < 0.0 && c.k.z > 0.0) {
    let fr = friction(c, rel, c.k.z * -c.lt.w);
    slides = fr.slides;
    c.lt = vec4f(fr.F, c.lt.w);
  } else {
    c.lt = vec4f(0.0, 0.0, 0.0, c.lt.w);
  }
  if (slides) { c.anc = vec4f(rel, c.anc.w); }
  contacts[g] = c;
}
`;

/** the block solve of batches of stretches by a workgroup of `size` invocations (see solveBatch) */
function batch(size: number): string {
  return /* wgsl */ `
const LANES = ${LANES}u;
const SLOTS = ${size / LANES}u;
var<workgroup> part: array<array<f32, 9>, ${size}>;
var<workgroup> heads: array<vec2u, ${size / LANES}>;
var<workgroup> longest: atomic<u32>;
var<workgroup> most: u32;
${SOLVE}
/**
 * Solves the stretches at0 to at0 + SLOTS − 1 of colour color (those there are),
 * LANES invocations each: li is the invocation in the workgroup. Called by every
 * invocation of the workgroup (it has barriers).
 */
fn solveBatch(at0: u32, color: u32, fwd: bool, li: u32) {
  let l = li % LANES; let slot = li / LANES;
  let at = at0 + slot;
  if (li == 0u) { atomicStore(&longest, 0u); }
  workgroupBarrier();
  if (l == 0u) {
    var hd = vec2u(0u, 0u);
    if (at < colors[${COL.count}u + color]) {
      let k = colors[${COL.node}u + P.nChunks + colors[${COL.start}u + color] + at];
      hd = vec2u(colors[${COL.node}u + 2u * P.nChunks + 2u * k], colors[${COL.node}u + 2u * P.nChunks + 2u * k + 1u]);
    }
    heads[slot] = hd;
    atomicMax(&longest, hd.y);
  }
  workgroupBarrier();
  if (li == 0u) { most = atomicLoad(&longest); }
  let steps = workgroupUniformLoad(&most);
  let hd = heads[slot];
  let first = hd.x; let count = hd.y;
  // in the order of the CPU, one way then the other
  for (var e = 0u; e < steps; e++) {
    let here = e < count;
    let i = select(first + count - 1u - e, first + e, fwd);
    let free = here && !fixedNode(i);
    var xi = vec3f(0.0);
    if (free) { xi = pos[i].xyz; }
    // this invocation's share of the contacts, and their friction
    var gr = vec3f(0.0);
    var H = mat3x3f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (free) { contactTerms(i, l, LANES, &gr, &H); }
    part[li] = array<f32, 9>(gr.x, gr.y, gr.z, H[0].x, H[1].x, H[2].x, H[1].y, H[2].y, H[2].z);
    workgroupBarrier();
    if (l == 0u && free) {
      var g = vec3f(0.0);
      var h = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      for (var q = li; q < li + LANES; q++) {
        let p = part[q];
        g += vec3f(p[0], p[1], p[2]);
        h[0] += p[3]; h[1] += p[4]; h[2] += p[5]; h[3] += p[6]; h[4] += p[7]; h[5] += p[8];
      }
      place(i, xi, g, mat3x3f(h[0], h[1], h[2], h[1], h[3], h[4], h[2], h[4], h[5]));
    }
    storageBarrier();
    workgroupBarrier();
  }
}
`;
}

/** what the colouring passes share (see colorPick) */
const SOLVE = /* wgsl */ `
/** what is the node's own: inertia, stretch, bending, target (gradient, then the Hessian's upper half) */
fn ownTerms(i: u32, xi: vec3f, gr: ptr<function, vec3f>, H: ptr<function, mat3x3f>) {
  let m = info[i].mass;
  let a = m * P.ih * P.ih;
  let mh = m * P.mh;
  *gr += a * (xi - state[i].xt.xyz);
  *H += mat3x3f(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
  // stretch (hard: λ carries the tension)
  for (var side = 0u; side < 2u; side++) {
    if (side == 0u && i == 0u) { continue; }
    let s = i + side - 1u;
    if (!segOK(s)) { continue; }
    let o = select(i - 1u, i + 1u, side == 1u);
    let d = xi - pos[o].xyz;
    let len = length(d);
    if (len < 1e-9) { continue; }
    let u = d / len;
    let g = segs[s];
    let C = len - info[s].rest - P.alphaS * g.z;
    let f = g.y * C + g.x;
    *gr += f * u;
    // Gauss-Newton term + |f|/len (I − uuᵀ), kept positive
    let geo = abs(f) / len;
    *H += (g.y - geo) * outerProduct(u, u) + mat3x3f(geo, 0.0, 0.0, 0.0, geo, 0.0, 0.0, 0.0, geo);
  }
  // bending: the three stencils around i
  for (var k = 0u; k < 3u; k++) {
    if (i + k < 2u) { continue; }
    let c = i + k - 1u;
    if (c + 1u >= P.n || !segOK(c - 1u) || !segOK(c)) { continue; }
    let kb = info[c].bendK;
    let w = select(1.0, -2.0, c == i);
    let dd = pos[c - 1u].xyz - 2.0 * pos[c].xyz + pos[c + 1u].xyz;
    *gr += kb * w * dd;
    let hh = kb * w * w;
    *H += mat3x3f(hh, 0.0, 0.0, 0.0, hh, 0.0, 0.0, 0.0, hh);
  }
  // target: a spring on a leash
  let t = tgt[2u * i];
  if (t.w > 0.0) {
    let k = t.w * mh;
    var e = xi - t.xyz;
    let el = length(e);
    let lead = tgt[2u * i + 1u].x;
    if (el > lead) { e *= lead / el; }
    *gr += k * e;
    *H += mat3x3f(k, 0.0, 0.0, 0.0, k, 0.0, 0.0, 0.0, k);
  }
}

fn inverse3(m: mat3x3f, det: f32) -> mat3x3f {
  let a = m[0]; let b = m[1]; let c = m[2];
  return transpose(mat3x3f(cross(b, c), cross(c, a), cross(a, b))) * (1.0 / det);
}

/** the contacts of node i from the start-th, every step-th (and their friction), into gr and H */
fn contactTerms(i: u32, start: u32, step: u32, gr: ptr<function, vec3f>, H: ptr<function, mat3x3f>) {
  let T = adj[i];
  for (var kk = start; kk < T; kk += step) {
    let e = adj[P.n + 2u * (i * MA + kk)];
    let c = contacts[e];
    let w = bitcast<f32>(adj[P.n + 2u * (i * MA + kk) + 1u]);
    let n = c.n.xyz;
    let rel = crel(c, (e - cbase()) / M);
    let C = cOf(c, rel) - forgive(c.k.w);
    let k = c.k.x;
    let f = k * C + c.lt.w;
    if (f >= 0.0) { continue; }
    *gr += f * w * n;
    *H += (k * w * w) * outerProduct(n, n);
    let mu = c.k.z;
    if (mu <= 0.0) { continue; }
    let fr = friction(c, rel, mu * -f);
    let ke = select(fr.k, fr.k * fr.s, fr.slides);
    *gr += fr.F * w;
    *H += (ke * w * w) * (mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) - outerProduct(n, n));
  }
}

/** adds the node's own terms to the sums g, H of its contacts, solves the 3×3 system and moves node i */
fn place(i: u32, xi: vec3f, g0: vec3f, H0: mat3x3f) {
  var g = g0; var Hs = H0;
  ownTerms(i, xi, &g, &Hs);
  let det = determinant(Hs);
  if (abs(det) < 1e-30) { return; }
  var dx = -(inverse3(Hs, det) * g);
  let L = length(dx);
  if (!(L < 1e9)) { return; }
  if (L > P.maxMove) { dx *= P.maxMove / L; }
  pos[i] = vec4f(xi + dx, pos[i].w);
}
`;

/** what the colouring passes share (see colorPick) */
const COLORING = /* wgsl */ `
fn prio(i: u32) -> u32 {
  var h = i * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}
var<private> me: u32;
var<private> phase: u32;
var<private> mask: u32;
var<private> mine: u32;
var<private> lose: bool;
/** the stretch of a node, seen from stretch \`me\` */
fn look(node: u32) {
  let j = colors[${COL.node}u + 5u * P.nChunks + node];
  if (j == me) { return; }
  if (phase == 0u) {
    let c = colors[${COL.node}u + j];
    if (c < COLORS) { mask |= 1u << c; }
  } else if (colors[${COL.node}u + 4u * P.nChunks + j] == mine) {
    let pj = prio(j); let pm = prio(me);
    if (pj > pm || (pj == pm && j > me)) { lose = true; }
  }
}
fn neighbours(k: u32) {
  let first = colors[${COL.node}u + 2u * P.nChunks + 2u * k];
  let count = colors[${COL.node}u + 2u * P.nChunks + 2u * k + 1u];
  if (first >= 2u) { look(first - 2u); }
  if (first >= 1u) { look(first - 1u); }
  if (first + count < P.n) { look(first + count); }
  if (first + count + 1u < P.n) { look(first + count + 1u); }
  for (var i = first; i < first + count; i++) {
    ${forContacts('i', /* wgsl */ `
      if (own) {
        let b = contacts[g].id.x;
        if (b != NONE) { look(b); look(b + 1u); }
      } else {
        let a = (g - cbase()) / M;
        look(a); look(a + 1u);
      }`)}
  }
}
`;

/** a kernel: its buffers, its code (\`main\` gets \`gid\`), its threads */
export interface Kernel { uses: string[]; code: string }

const main = (body: string) => /* wgsl */ `
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3u) {
  let gid = id.x;
  ${body}
}`;

export const KERNELS: Record<string, Kernel> = {
  // ---------------------------------------------------------------- contacts, once a frame

  /** every segment into the bucket of its midpoint (a linked list per bucket) */
  hashInsert: {
    uses: ['P', 'pos', 'info', 'hash'],
    code: NODES + GEOM + main(/* wgsl */ `
      let a = gid;
      if (!segOK(a)) { return; }
      let b = bucketOf(cellOf(0.5 * (pos[a].xyz + pos[a + 1u].xyz)));
      let old = atomicExchange(&hash[b], a + 1u);
      atomicStore(&hash[P.hashMask + 1u + a], old);
    `),
  },

  /** the contacts of every segment: against the static shapes, and against the segments after it */
  detect: {
    uses: ['P', 'pos', 'state', 'info', 'statics', 'hash', 'contacts', 'refsA', 'foundA'],
    code: NODES + GEOM + /* wgsl */ `
var<private> loc: array<Contact, ${MC}>;
var<private> nloc: u32;
var<private> nstat: u32;
var<private> segA: u32;
/** contacts dropped for want of slots (static, cable, references), summed into the list's header */
var<private> overflow: array<u32, 3>;

/** the previous frame's contact of segment a with the same key, or NONE */
fn previous(a: u32, b: u32, key: u32) -> u32 {
  let prev = 1u - P.cur;
  let rb = prev * P.n * REF + a * REF;
  let base = prev * P.n * M + a * M;
  if (b == NONE) {
    let nS = atomicLoad(&refsA[rb]);
    for (var k = 0u; k < nS; k++) { let c = contacts[base + k]; if (c.id.x == NONE && c.id.y == key) { return base + k; } }
  } else {
    let nC = atomicLoad(&refsA[rb + 1u]);
    for (var k = 0u; k < nC; k++) { let c = contacts[base + MS + k]; if (c.id.x == b && c.id.y == key) { return base + MS + k; } }
  }
  return NONE;
}

fn relOf(a: u32, b: u32, s: f32, t: f32) -> vec3f {
  let pa = mix(pos[a].xyz, pos[a + 1u].xyz, s);
  if (b == NONE) { return pa; }
  return pa - mix(pos[b].xyz, pos[b + 1u].xyz, t);
}

/** a new contact, warm started from the previous frame's of the same key (see CableSim.pushContact) */
fn push(which: u32, b: u32, s: f32, t: f32, n: vec3f, q: vec3f, rs: f32, mu: f32, track: u32, sid: u32) {
  let a = segA;
  var key = which;
  if (b == NONE) { key = which + 4u * sid; }
  var c: Contact;
  c.n = vec4f(n, s);
  c.q = vec4f(q, t);
  c.id = vec4u(b, key, 0u, sid | (track << 31u));
  c.dv = vec4f(0.0);
  var m = max(info[a].mass, info[a + 1u].mass);
  let fa = flags(a);
  var kcap = select(P.kCap, P.kCapHeld, (fa & 16u) != 0u);
  if (b != NONE) {
    m = max(m, max(info[b].mass, info[b + 1u].mass));
    kcap = select(P.kCapCable, P.kCapCableHeld, ((fa | flags(b)) & 24u) != 0u);
  }
  let mh2 = m * P.mhk;
  c.id.z = bitcast<u32>(mh2);
  let kmax = kcap * mh2; let kmin = P.kMin * mh2;
  let old = previous(a, b, key);
  if (old != NONE) {
    let o = contacts[old];
    c.k = vec4f(max(kmin, min(kmax, o.k.x * P.gammaRaw)), kmax, mu, 0.0);
    c.lt = o.lt;
    // the points moved along the segments since last frame (see shiftAnchor)
    var shift = (pos[a + 1u].xyz - pos[a].xyz) * (s - o.n.w);
    if (b != NONE) { shift -= (pos[b + 1u].xyz - pos[b].xyz) * (t - o.q.w); }
    c.anc = vec4f(o.anc.xyz + shift, rs);
  } else {
    c.k = vec4f(kmin, kmax, mu, 0.0);
    c.lt = vec4f(0.0);
    c.anc = vec4f(relOf(a, b, s, t), rs);
  }
  if (b == NONE) {
    if (nstat < MS) { contacts[cbase() + a * M + nstat] = c; nstat++; } else { overflow[0]++; }
  } else {
    if (nloc < MC) { loc[nloc] = c; nloc++; return; }
    // full: the one of the largest key goes, whatever the order they were found in
    overflow[1]++;
    var worst = 0u;
    for (var k = 1u; k < MC; k++) { if (after(loc[k], loc[worst])) { worst = k; } }
    if (after(loc[worst], c)) { loc[worst] = c; }
  }
}

/** is contact p after contact c in the order of their keys (other segment, then which)? */
fn after(p: Contact, c: Contact) -> bool { return p.id.x > c.id.x || (p.id.x == c.id.x && p.id.y > c.id.y); }

struct OnSeg { p: vec3f, t: f32 }
fn pointOn(s: f32, q0: vec3f, w: vec3f, ww: f32) -> OnSeg {
  let a = segA;
  let p = mix(pos[a].xyz, pos[a + 1u].xyz, s);
  return OnSeg(p, clamp(dot(p - q0, w) / ww, 0.0, 1.0));
}

fn emitAt(which: u32, b: u32, s: f32, q0: vec3f, w: vec3f, ww: f32, rs: f32, lim: f32, mu: f32, sid: u32) {
  let o = pointOn(s, q0, w, ww);
  let q = q0 + w * o.t;
  var n = o.p - q;
  let d = length(n);
  if (d > lim) { return; }
  if (d < 1e-9) { n = vec3f(0.0, 1.0, 0.0); } else { n /= d; }
  push(which, b, s, o.t, n, q, rs, mu, 0u, sid);
}

/** contacts of segment a with segment b (or the axis of capsule sid: b = NONE), their closest points cl (see CableSim.pairContacts) */
fn pairContacts(b: u32, q0: vec3f, q1: vec3f, rs: f32, lim: f32, mu: f32, sid: u32, cl: Closest) {
  let a = segA;
  let p0 = pos[a].xyz; let u = pos[a + 1u].xyz - p0; let w = q1 - q0;
  let uu = dot(u, u); let ww = dot(w, w);
  let cr = cross(u, w); let crs = dot(cr, cr);
  var which = 0u;
  if (crs < PARALLEL * uu * ww && uu > 1e-12 && ww > 1e-12) {
    let s0 = dot(q0 - p0, u) / uu; let s1 = dot(q1 - p0, u) / uu;
    let lo = max(0.0, min(s0, s1)); let hi = min(1.0, max(s0, s1));
    if (hi - lo > OVERLAP) {
      // do the ends of the overlap lie on the same side of B?
      let e0 = pointOn(lo, q0, w, ww); let e1 = pointOn(hi, q0, w, ww);
      let g0 = e0.p - q0 - w * e0.t; let g1 = e1.p - q0 - w * e1.t;
      if (dot(g0, g1) > 0.0) {
        emitAt(0u, b, lo, q0, w, ww, rs, lim, mu, sid);
        emitAt(2u, b, hi, q0, w, ww, rs, lim, mu, sid);
        // and the closest point, when it dips well below both ends
        if (sqrt(cl.d2) > min(length(g0), length(g1)) - DIP || cl.s <= lo || cl.s >= hi) { return; }
        which = 1u;
      }
    }
  }
  let pa = p0 + u * cl.s;
  let q = q0 + w * cl.t;
  var n = pa - q;
  var d = length(n);
  if (d < 1e-9) {
    let l = sqrt(crs);
    if (l > 1e-12) { n = cr / l; } else { n = vec3f(0.0, 0.0, 1.0); }
    d = 0.0;
  } else { n /= d; }
  push(which, b, cl.s, cl.t, n, q, rs, mu, select(0u, 1u, crs >= PARALLEL * uu * ww), sid);
}

fn speed(i: u32) -> f32 { return length(state[i].v.xyz); }
fn segSpeed(i: u32) -> f32 { if (i + 1u < P.n) { return max(speed(i), speed(i + 1u)); } return speed(i); }
` + main(/* wgsl */ `
      let a = gid;
      if (!segOK(a)) { return; }
      segA = a; nloc = 0u; nstat = 0u;
      let p0 = pos[a].xyz; let p1 = pos[a + 1u].xyz;
      let r = pos[a].w;
      let va = segSpeed(a);
      let aFixed = fixedNode(a) && fixedNode(a + 1u);
      let ca = cableOf(a);

      // --- against the static shapes
      if (!aFixed) {
        let margin = min(P.marginStatic, P.margin + va * P.frame * P.marginLead);
        let g = r + margin;
        let lo = min(p0, p1) - g; let hi = max(p0, p1) + g;
        for (var sid = 0u; sid < P.nStatics; sid++) {
          let sh = statics[sid];
          if (any(lo > sh.hi.xyz) || any(hi < sh.lo.xyz)) { continue; }
          let mu = sh.b.w;
          if (sh.lo.w == 0.0) {
            let cl = segSeg(p0, p1, sh.a.xyz, sh.b.xyz);
            let rs = r + sh.a.w;
            let lim = rs + margin;
            if (cl.d2 > lim * lim) { continue; }
            pairContacts(NONE, sh.a.xyz, sh.b.xyz, rs, lim, mu, sid, cl);
          } else {
            // a contact for every node, and the segment's closest point near an edge
            let lim = r + margin;
            let last = !segOK(a + 1u);
            var dEnds = 1e30;
            for (var e = 0u; e < 2u; e++) {
              let pe = pos[a + e].xyz;
              let hit = segBox(pe, pe, sh.lo.xyz, sh.hi.xyz);
              dEnds = min(dEnds, hit.dist);
              if ((e == 1u && !last) || hit.dist > lim) { continue; }
              push(e, NONE, f32(e), 0.0, hit.n, hit.q, r, mu, 0u, sid);
            }
            let hit = segBox(p0, p1, sh.lo.xyz, sh.hi.xyz);
            if (hit.dist > lim || hit.s < EDGE || hit.s > 1.0 - EDGE || hit.dist > dEnds - DIP) { continue; }
            push(2u, NONE, hit.s, 0.0, hit.n, hit.q, r, mu, 0u, sid);
          }
        }
      }

      // --- against the segments after it, in the 27 cells around
      let mid = 0.5 * (p0 + p1);
      let cell = cellOf(mid);
      let aReach = info[a].rest * 0.5 + r + P.marginCap;
      let aNear = length(p1 - p0) * 0.5 + r;
      for (var ox = -1; ox <= 1; ox++) {
        for (var oy = -1; oy <= 1; oy++) {
          for (var oz = -1; oz <= 1; oz++) {
            let nc = cell + vec3i(ox, oy, oz);
            var e = atomicLoad(&hash[bucketOf(nc)]);
            loop {
              if (e == 0u) { break; }
              let b = e - 1u;
              e = atomicLoad(&hash[P.hashMask + 1u + b]);
              if (b <= a) { continue; }
              let bFixed = fixedNode(b) && fixedNode(b + 1u);
              if (aFixed && bFixed) { continue; }
              let b0 = pos[b].xyz; let b1 = pos[b + 1u].xyz;
              let bmid = 0.5 * (b0 + b1);
              let md = mid - bmid; let md2 = dot(md, md);
              let rb = pos[b].w;
              let reach = aReach + info[b].rest * 0.5 + rb;
              if (md2 > reach * reach) { continue; }
              if (any(cellOf(bmid) != nc)) { continue; }
              let cb = cableOf(b);
              if (ca == cb && b - a <= 2u) { continue; }
              let rs = r + rb;
              let margin = min(P.marginCap, P.margin + (va + segSpeed(b)) * P.frame * P.marginLead);
              let near = aNear + length(b1 - b0) * 0.5 + rb + margin;
              if (md2 > near * near) { continue; }
              let cl = segSeg(p0, p1, b0, b1);
              let lim = rs + margin;
              if (cl.d2 > lim * lim) { continue; }
              pairContacts(b, b0, b1, rs, lim, P.muCable, 0u, cl);
            }
          }
        }
      }

      // in the order of their keys, whatever the order of the buckets
      for (var i = 1u; i < nloc; i++) {
        let c = loc[i];
        var j = i;
        loop {
          if (j == 0u) { break; }
          let p = loc[j - 1u];
          if (!after(p, c)) { break; }
          loc[j] = p;
          j--;
        }
        loc[j] = c;
      }
      let rb = rbase() + a * REF;
      for (var i = 0u; i < nloc; i++) {
        let slot = MS + i;
        contacts[cbase() + a * M + slot] = loc[i];
        let b = loc[i].id.x;
        let h = atomicAdd(&refsA[rbase() + b * REF + 2u], 1u);
        if (h < MH) { atomicStore(&refsA[rbase() + b * REF + 4u + h], a * M + slot); } else { overflow[2]++; }
      }
      atomicStore(&refsA[rb], nstat);
      atomicStore(&refsA[rb + 1u], nloc);
      // and into the list of the contacts, for the kernels that run per contact
      if (overflow[0] + overflow[1] + overflow[2] > 0u) {
        atomicAdd(&foundA[1], overflow[0]); atomicAdd(&foundA[2], overflow[1]); atomicAdd(&foundA[3], overflow[2]);
      }
      let at = atomicAdd(&foundA[0], nstat + nloc);
      for (var i = 0u; i < nstat; i++) { atomicStore(&foundA[4u + at + i], a * M + i); }
      for (var i = 0u; i < nloc; i++) { atomicStore(&foundA[4u + at + nstat + i], a * M + MS + i); }
    `),
  },

  /**
   * The contacts of every node, in the order of its segments' (see
   * forContacts), and its weight in each: what the block solve reads, in one
   * place. After the count (one per node), MA (contact, weight) per node.
   */
  adjacency: {
    uses: ['P', 'info', 'contacts', 'refs', 'adj', 'foundA'],
    code: NODES + main(/* wgsl */ `
      let i = gid;
      if (i >= P.n) { return; }
      var n = 0u;
      ${forContacts('i', /* wgsl */ `
        if (n < MA) {
          adj[P.n + 2u * (i * MA + n)] = g;
          adj[P.n + 2u * (i * MA + n) + 1u] = bitcast<u32>(w);
          n++;
        } else { atomicAdd(&foundA[3], 1u); }`)}
      adj[i] = n;
    `),
  },

  /**
   * The number of workgroups of the kernels per contact (which also run one
   * invocation per segment), after the list's header; the counters to stats.
   */
  foundArgs: {
    uses: ['P', 'foundA', 'stats'],
    code: main(/* wgsl */ `
      if (gid != 0u) { return; }
      let n = atomicLoad(&foundA[0]);
      atomicStore(&stats[${STAT.contacts}], n);
      atomicStore(&stats[${STAT.overflowS}], atomicLoad(&foundA[1]));
      atomicStore(&stats[${STAT.overflowC}], atomicLoad(&foundA[2]));
      atomicStore(&stats[${STAT.overflowH}], atomicLoad(&foundA[3]));
      atomicStore(&foundA[4u + P.n * M], (max(n, P.n) + ${WG - 1}u) / ${WG}u);
      atomicStore(&foundA[5u + P.n * M], 1u);
      atomicStore(&foundA[6u + P.n * M], 1u);
    `),
  },

  /** the references of every segment, sorted (they were appended in any order) */
  sortRefs: {
    uses: ['P', 'refs'],
    code: NODES.replace(/fn flags[\s\S]*?fn cbase/, 'fn cbase') + main(/* wgsl */ `
      if (gid >= P.n) { return; }
      let rb = rbase() + gid * REF;
      let n = min(refs[rb + 2u], MH);
      refs[rb + 2u] = n;
      for (var i = 1u; i < n; i++) {
        let c = refs[rb + 4u + i];
        var j = i;
        loop {
          if (j == 0u || refs[rb + 4u + j - 1u] <= c) { break; }
          refs[rb + 4u + j] = refs[rb + 4u + j - 1u];
          j--;
        }
        refs[rb + 4u + j] = c;
      }
    `),
  },

  // ---------------------------------------------------------------- colouring

  colorInit: {
    uses: ['P', 'colors'],
    code: main(/* wgsl */ `
      if (gid >= P.nChunks) { return; }
      colors[${COL.node}u + gid] = NONE;
    `),
  },

  /**
   * Speculative colouring of the stretches of cord (Gebremedhin–Manne), in
   * rounds of two passes: every uncoloured stretch picks the smallest colour
   * none of its coloured neighbours has (\`colorPick\`), then keeps it unless a
   * neighbour of higher rank picked the same (\`colorKeep\`). Each pass reads
   * only what the one before wrote: the colouring does not depend on the
   * scheduling. Neighbours: the stretches its nodes are coupled to (within 2
   * nodes along the cord: stretch and bending; across a contact).
   */
  colorPick: {
    uses: ['P', 'info', 'contacts', 'refs', 'colors'],
    code: NODES + COLORING + main(/* wgsl */ `
      let k = gid;
      if (k >= P.nChunks) { return; }
      if (colors[${COL.node}u + k] != NONE) { colors[${COL.node}u + 4u * P.nChunks + k] = NONE; return; }
      me = k; phase = 0u; mask = 0u;
      neighbours(k);
      colors[${COL.node}u + 4u * P.nChunks + k] = min(firstTrailingBit(~mask), COLORS - 1u);
    `),
  },

  colorKeep: {
    uses: ['P', 'info', 'contacts', 'refs', 'colors'],
    code: NODES + COLORING + main(/* wgsl */ `
      let k = gid;
      if (k >= P.nChunks || colors[${COL.node}u + k] != NONE) { return; }
      me = k; phase = 1u; mine = colors[${COL.node}u + 4u * P.nChunks + k]; lose = false;
      neighbours(k);
      if (!lose) { colors[${COL.node}u + k] = mine; }
    `),
  },

  colorCount: {
    uses: ['P', 'colorsA', 'stats'],
    code: main(/* wgsl */ `
      if (gid >= P.nChunks) { return; }
      var c = atomicLoad(&colorsA[${COL.node}u + gid]);
      if (c == NONE) { atomicAdd(&stats[${STAT.uncoloured}], 1u); c = P.nColors - 1u; } else { atomicMax(&stats[${STAT.maxColor}], c); }
      // (beyond the colours dispatched: in the last one, with what that one holds)
      if (c >= P.nColors) { atomicAdd(&stats[${STAT.clamped}], 1u); c = P.nColors - 1u; }
      atomicStore(&colorsA[${COL.node}u + gid], c);
      atomicAdd(&colorsA[${COL.count}u + c], 1u);
    `),
  },

  colorScan: {
    uses: ['P', 'colors', 'stats'],
    code: main(/* wgsl */ `
      if (gid != 0u) { return; }
      var at = 0u;
      for (var c = 0u; c < COLORS; c++) {
        let n = colors[${COL.count}u + c];
        colors[${COL.start}u + c] = at;
        colors[${COL.fill}u + c] = at;
        colors[${COL.args}u + 3u * c] = (n + ${WG / LANES - 1}u) / ${WG / LANES}u;
        colors[${COL.args}u + 3u * c + 1u] = 1u;
        colors[${COL.args}u + 3u * c + 2u] = 1u;
        atomicStore(&stats[${STAT.colors}u + c], n);
        at += n;
      }
    `),
  },

  colorFill: {
    uses: ['P', 'colorsA'],
    code: main(/* wgsl */ `
      if (gid >= P.nChunks) { return; }
      let c = atomicLoad(&colorsA[${COL.node}u + gid]);
      let e = atomicAdd(&colorsA[${COL.fill}u + c], 1u);
      atomicStore(&colorsA[${COL.node}u + P.nChunks + e], gid);
    `),
  },

  // ---------------------------------------------------------------- substep

  /** start of a substep, per contact: tracking, separation (α stabilisation), warm start; per segment, the same for its stretch */
  contactStart: {
    uses: ['P', 'pos', 'info', 'statics', 'contacts', 'segs', 'found'],
    code: NODES + GEOM + CONTACTS + main(/* wgsl */ `
      // and per segment, the warm start of its stretch constraint, its error at the start of the substep
      if (segOK(gid)) {
        let m = max(info[gid].mass, info[gid + 1u].mass) * P.mhk;
        var sg = segs[gid];
        sg.x *= P.lamWarmS * P.gamma;
        sg.y = min(P.stretch * m, max(P.kMin * m, sg.y * P.gamma));
        sg.z = length(pos[gid].xyz - pos[gid + 1u].xyz) - info[gid].rest;
        segs[gid] = sg;
      }
      if (gid >= found[0]) { return; }
      let g = cbase() + found[4u + gid];
      let a = found[4u + gid] / M;
      var c = contacts[g];
      let b = c.id.x;
      if ((c.id.w >> 31u) == 1u && (b != NONE || (flags(a) & 24u) != 0u)) {
        // the closest point of the two segments follows them (see CableSim.track)
        var q0: vec3f; var q1: vec3f;
        if (b != NONE) { q0 = pos[b].xyz; q1 = pos[b + 1u].xyz; } else { let sh = statics[c.id.w & 0x7fffffffu]; q0 = sh.a.xyz; q1 = sh.b.xyz; }
        let p0 = pos[a].xyz; let p1 = pos[a + 1u].xyz;
        let cl = segSeg(p0, p1, q0, q1);
        let ds = cl.s - c.n.w;
        var dt = 0.0;
        if (b != NONE) { dt = cl.t - c.q.w; }
        let pa = mix(p0, p1, cl.s); let q = mix(q0, q1, cl.t);
        let nn = pa - q; let d = length(nn);
        if (d > 1e-9 && dot(nn, c.n.xyz) > 0.0) { c.n = vec4f(nn / d, c.n.w); }
        c.n.w = cl.s;
        if (b == NONE) { c.q = vec4f(q, cl.t); } else { c.q.w = cl.t; }
        var shift = (p1 - p0) * ds;
        if (b != NONE) { shift -= (q1 - q0) * dt; }
        c.anc = vec4f(c.anc.xyz + shift, c.anc.w);
      }
      c.k.w = contactC(c, a);
      c.lt = c.lt * P.lamWarm;
      contacts[g] = c;
    `),
  },

  /** inertial prediction */
  predict: {
    uses: ['P', 'pos', 'state', 'info'],
    code: NODES + main(/* wgsl */ `
      let i = gid;
      if (i >= P.n) { return; }
      let x = pos[i];
      var s = state[i];
      s.x0 = vec4f(x.xyz, 0.0);
      if (fixedNode(i)) { s.xt = s.x0; state[i] = s; return; }
      let xt = x.xyz + P.h * s.v.xyz + vec3f(0.0, P.gy, 0.0);
      s.xt = vec4f(xt, 0.0);
      state[i] = s;
      pos[i] = vec4f(xt, x.w);
    `),
  },

  /** the block solve of the nodes of one colour of stretches, each stretch in order (see CableSim.solveNode) */
  /**
   * The block solve of the stretches of one colour (see CableSim.solveNode):
   * LANES invocations per stretch, which solve its nodes one after the other
   * in the order of the CPU; the contacts of a node are shared out among
   * them, and their sums gathered in workgroup memory.
   */
  solve: {
    uses: ['P', 'pos', 'state', 'info', 'segs', 'tgt', 'contacts', 'adj', 'colorsR'],
    code: NODES + CONTACTS + /* wgsl */ `
${batch(WG)}
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  solveBatch(wg.x * SLOTS, ps.color, ps.forward == 1u, li);
}`,
  },

  /** dual update of the stretch constraints, of the contacts and of their friction (see CableSim.dualUpdate, frictionDual) */
  dualContact: {
    uses: ['P', 'pos', 'info', 'contacts', 'segs', 'found'],
    code: NODES + CONTACTS + DUAL + main(/* wgsl */ `
      // the stretch constraints first (one per invocation of the first N)
      dualSeg(gid);
      if (gid >= found[0]) { return; }
      dualOf(cbase() + found[4u + gid], found[4u + gid] / M);
    `),
  },

  /** one length pass along every cord, as on the CPU (see CableSim.inextensible) */
  pbd: {
    uses: ['P', 'pos', 'info', 'cables'],
    code: NODES + /* wgsl */ `
fn weight(i: u32) -> f32 {
  let f = flags(i);
  if ((f & 1u) != 0u) { return 0.0; }
  if ((f & 4u) != 0u) { return P.handWeight / info[i].mass; }
  return 1.0 / info[i].mass;
}
` + main(/* wgsl */ `
      if (gid >= P.nCables) { return; }
      let cb = cables[gid];
      if (cb.y < 2u) { return; }
      for (var k = 0u; k < cb.y - 1u; k++) {
        let s = select(cb.x + cb.y - 2u - k, cb.x + k, ps.forward == 1u);
        if (!segOK(s)) { continue; }
        let wa = weight(s); let wb = weight(s + 1u);
        if (wa + wb == 0.0) { continue; }
        let pa = pos[s]; let pb = pos[s + 1u];
        let d = pb.xyz - pa.xyz;
        let L = length(d);
        if (L < 1e-9) { continue; }
        let corr = (L - info[s].rest) / (L * (wa + wb));
        pos[s] = vec4f(pa.xyz + d * corr * wa, pa.w);
        pos[s + 1u] = vec4f(pb.xyz - d * corr * wb, pb.w);
      }
    `),
  },

  /** v = (x − x₀)/h, with drag */
  velocity: {
    uses: ['P', 'pos', 'state', 'info'],
    code: NODES + main(/* wgsl */ `
      let i = gid;
      if (i >= P.n) { return; }
      if (fixedNode(i)) { state[i].v = vec4f(0.0); return; }
      state[i].v = vec4f((pos[i].xyz - state[i].x0.xyz) * P.ih * P.damp, 0.0);
    `),
  },

  /** the impulse of every contact that pushed: no separating normal velocity, some sliding damped (see CableSim.contactVelocities) */
  cvImpulse: {
    uses: ['P', 'pos', 'state', 'info', 'contacts', 'found'],
    code: NODES + CONTACTS + main(/* wgsl */ `
      if (gid >= found[0]) { return; }
      let g = cbase() + found[4u + gid];
      let a = found[4u + gid] / M;
      let c = contacts[g];
      contacts[g].dv = vec4f(0.0);
      if (c.lt.w >= 0.0 && contactC(c, a) > 0.0) { return; }
      let s = c.n.w; let t = c.q.w; let b = c.id.x;
      var denom = 0.0;
      if (!fixedNode(a)) { denom += (1.0 - s) * (1.0 - s) / info[a].mass; }
      if (!fixedNode(a + 1u)) { denom += s * s / info[a + 1u].mass; }
      var rv = mix(state[a].v.xyz, state[a + 1u].v.xyz, s);
      if (b != NONE) {
        if (!fixedNode(b)) { denom += (1.0 - t) * (1.0 - t) / info[b].mass; }
        if (!fixedNode(b + 1u)) { denom += t * t / info[b + 1u].mass; }
        rv -= mix(state[b].v.xyz, state[b + 1u].v.xyz, t);
      }
      if (denom <= 0.0) { return; }
      let n = c.n.xyz;
      let vn = dot(rv, n);
      let imp = select(0.0, -vn / denom, vn > 0.0);
      let tv = rv - vn * n;
      let td = P.contactDamping / denom;
      if (!(vn > 0.0 || td > 0.0)) { return; }
      contacts[g].dv = vec4f(imp * n - td * tv, 1.0);
    `),
  },

  /** every node takes the mean of the impulses of its contacts (Jacobi with mass splitting) */
  cvApply: cvApply(false),
  /** the same, then rest friction: a node pressed against something and barely moving stops; faster, it loses some speed */
  cvApplyRest: cvApply(true),
};

function cvApply(rest: boolean): Kernel {
  return {
    uses: ['P', 'state', 'info', 'contacts', 'refs'],
    code: NODES + main(/* wgsl */ `
      let i = gid;
      if (i >= P.n || fixedNode(i)) { return; }
      var dv = vec3f(0.0); var cnt = 0.0; var pressed = false;
      ${forContacts('i', /* wgsl */ `
        let d = contacts[g].dv;
        if (d.w > 0.0) { dv += w * d.xyz; cnt += 1.0; }
        if (contacts[g].lt.w < 0.0) { pressed = true; }`)}
      var v = state[i].v.xyz;
      if (cnt > 0.0) { v += dv / (cnt * info[i].mass); }
      ${rest ? /* wgsl */ `
      if (pressed && P.restFriction > 0.0) {
        let sp = length(v);
        v *= select(0.0, 1.0 - P.restDecel / sp, sp > P.restFriction);
      }` : ''}
      state[i].v = vec4f(v, 0.0);
    `),
  };
}

/** the WGSL of a kernel: the shared declarations, its buffers, its code */
export function kernelSource(k: Kernel): string {
  const decls = k.uses.map((name, b) => `@group(0) @binding(${b}) ${RESOURCES[name].wgsl};`).join('\n');
  return COMMON + decls + k.code;
}
