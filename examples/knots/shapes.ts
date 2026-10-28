// The shapes the ropes of the knots example start from, as polylines (xyz
// interleaved): knots tied loose, ropes wound round a bar, nodes laid at an even
// distance along them. Nothing here knows of the solver nor of three.js.

export type Vec = [number, number, number];

/** nodes `seg` apart along the polyline (xyz interleaved), the first and last kept */
export function resample(path: number[], seg: number): number[] {
  const out = [path[0], path[1], path[2]];
  let carry = 0;
  for (let k = 3; k < path.length; k += 3) {
    const ax = path[k - 3], ay = path[k - 2], az = path[k - 1];
    const dx = path[k] - ax, dy = path[k + 1] - ay, dz = path[k + 2] - az;
    const len = Math.hypot(dx, dy, dz);
    let t = seg - carry;
    while (t <= len) {
      out.push(ax + (dx * t) / len, ay + (dy * t) / len, az + (dz * t) / len);
      t += seg;
    }
    carry = len - (t - seg);
  }
  return out;
}

/** the closed curve of a knot, at parameter t in [0, 2π), its outermost point at t = 0, going +y there */
export type KnotCurve = (t: number) => Vec;

/** a torus knot (2, q): q = 3 is the trefoil (an overhand knot once cut), 5 and 7 wind further */
export function torusKnot(q: number, depth = 1): KnotCurve {
  return (t) => {
    const r = 2 + Math.cos(q * t);
    return [r * Math.cos(2 * t), r * Math.sin(2 * t), depth * Math.sin(q * t)];
  };
}

/** the figure-eight knot (4₁) */
export function figureEight(depth = 1): KnotCurve {
  return (t) => {
    const r = 2 + Math.cos(2 * t);
    return [r * Math.cos(3 * t), r * Math.sin(3 * t), depth * Math.sin(4 * t)];
  };
}

/**
 * A rope tied in a loose knot: the closed curve cut open at its outermost point
 * (t = 0), its two ends led out along +x, away from the knot, `lead` metres
 * each and a little apart. The knot is `scale` metres a unit, centred on `at`;
 * standing, its ends point up and it hangs from them; lying, it lies flat
 * (its depth is world y), its ends pointing to −z.
 */
export function openKnot(curve: KnotCurve, scale: number, lead: number, at: Vec, seg: number, lying = false): number[] {
  const gap = 0.12, fine = 600;
  const local: Vec[] = [];
  // the first lead, coming in towards the knot
  const a = curve(gap), b = curve(2 * Math.PI - gap);
  const out = (p: Vec, side: number, s: number): Vec => [p[0] + s, p[1] + side * s * 0.35, p[2]];
  for (let k = 40; k > 0; k--) local.push(out(b, -1, ((k / 40) * lead) / scale));
  for (let k = fine; k >= 0; k--) local.push(curve(gap + ((2 * Math.PI - 2 * gap) * k) / fine));
  for (let k = 1; k <= 40; k++) local.push(out(a, 1, ((k / 40) * lead) / scale));
  const path: number[] = [];
  // standing: knot x → world y (up), y → x, z → −z; lying: x → −z, y → x, z → −y (both right-handed)
  for (const p of local) {
    if (lying) path.push(at[0] + p[1] * scale, at[1] - p[2] * scale, at[2] - p[0] * scale);
    else path.push(at[0] + p[1] * scale, at[1] + p[0] * scale, at[2] - p[2] * scale);
  }
  return resample(path, seg);
}

/**
 * A rope wound round a bar along z (centre `c`, radius R): it hangs `left`
 * metres on the −x side, goes over the top and round `turns` times more (a half
 * turn and `turns` full ones, `pitch` apart along the bar), and hangs `right`
 * metres on the +x side.
 */
export function wound(c: Vec, R: number, turns: number, pitch: number, left: number, right: number, seg: number): number[] {
  const path: number[] = [];
  const angle = Math.PI + 2 * Math.PI * turns;
  const steps = Math.max(24, Math.round(angle * 24));
  const z0 = c[2] - (pitch * turns) / 2;
  // up the −x side, then round from the left (θ = π) clockwise seen from +z, over the top
  path.push(c[0] - R, c[1] - left, z0);
  for (let k = 0; k <= steps; k++) {
    const u = k / steps, th = Math.PI - angle * u;
    path.push(c[0] + R * Math.cos(th), c[1] + R * Math.sin(th), z0 + pitch * turns * u);
  }
  path.push(c[0] + R, c[1] - right, z0 + pitch * turns);
  return resample(path, seg);
}
