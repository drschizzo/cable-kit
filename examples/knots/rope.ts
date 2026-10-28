// What the ropes are made of, drawn in the fragment shader from where each
// point lies on its tube: `tubeCoord` gives how far along the rope at rest
// (so the pattern stays put on the rope as it moves) and the angle round it.
// No texture: the strands are computed, with their relief as a bent normal,
// dark grooves between them, and a sheen for the fibres. Each rope picks its
// make with its tag (the last argument of `lay`):
//
//   LAID: three strands twisted round one another (a hemp or nylon rope);
//   BRAIDED: a sheath of strands braided over and under each other (a
//            climbing rope), with a stripe of a second colour;
//   JACKET: a smooth plastic jacket (a cable), with a printed band.
//
// The pattern is the rope's own, not its twist: the solver does not know how a
// rope turns about its axis, so the strands follow a frame carried along it
// (see three-cable-tubes).

import * as THREE from 'three';

export const LAID = 0, BRAIDED = 1, JACKET = 2;

const vertexDecl = /* glsl */ `
varying vec4 vRope;
varying vec3 vRopeT;
varying float vRopeR;
`;

// the tangent of the tube and its radius, from the sweep (tT, tEnd: see tube.ts)
const vertexMain = /* glsl */ `
  vRope = tubeCoord;
  vRopeT = normalize(normalMatrix * tT);
  vRopeR = tEnd.w;
#include <begin_vertex>`;

const fragmentDecl = /* glsl */ `
varying vec4 vRope;
varying vec3 vRopeT;
varying float vRopeR;
uniform vec3 ropeStripe;
float ropeH;
vec2 ropeSlope;
float ropeTone;

// a round strand across v in [0, 1]: its height and its slope, the slope kept
// finite at the edges (a groove, not a cliff)
vec2 strand(float v) {
  float w = clamp(2.0 * v - 1.0, -0.97, 0.97);
  float h = sqrt(1.0 - w * w);
  return vec2(h, -2.0 * w / h);
}

// height of the surface (in radii), its slope along the rope and round it (per metre),
// and which colour (0: the rope's, 1: the stripe)
void ropeSurface(float s, float a, float R, float tag) {
  float C = 6.2831853 * R;
  ropeH = 1.0; ropeSlope = vec2(0.0); ropeTone = 0.0;
  if (tag < 0.5) {
    // three strands, a full twist every 7 radii; fibres twisted the other way within each
    float L = 7.0 * R;
    float f = 3.0 * (a + s / L);
    vec2 st = strand(fract(f));
    float g = 18.0 * (a - s / (2.2 * R));
    float fib = 0.5 + 0.5 * sin(6.2831853 * g);
    ropeH = st.x * (0.94 + 0.06 * fib);
    float dg = 0.03 * st.x * 3.1415927 * cos(6.2831853 * g) * 18.0;
    ropeSlope = vec2(st.y * 3.0 / L - dg / (2.2 * R), st.y * 3.0 / C + dg / C) * 0.3 * R;
    ropeTone = mod(floor(f), 3.0) == 0.0 ? 0.08 : 0.0;
  } else if (tag < 1.5) {
    // a braided sheath: 12 bundles each way, two over two under
    float P = 9.0 * R;
    float u1 = 12.0 * (a + s / P), u2 = 12.0 * (a - s / P);
    float cell = mod(floor(u1) + floor(u2), 4.0);
    bool first = cell < 2.0;
    float u = first ? u1 : u2;
    vec2 st = strand(fract(u));
    // each bundle dips where it goes under the other
    float along = fract(first ? u2 : u1);
    float dip = 0.75 + 0.25 * sin(3.1415927 * along);
    ropeH = st.x * dip;
    float sgn = first ? 1.0 : -1.0;
    ropeSlope = vec2(st.y * dip * sgn * 12.0 / P, st.y * dip * 12.0 / C) * 0.12 * R;
    float bundle = mod(floor(u), 12.0);
    ropeTone = (first && (bundle == 0.0 || bundle == 6.0)) ? 1.0 : 0.0;
  } else {
    // a smooth jacket, a printed band every 30 cm
    ropeH = 1.0;
    ropeTone = fract(s / 0.3) < 0.04 ? 1.0 : 0.0;
  }
}
`;

/**
 * A material for ropes of every make (by the tag of each tube): give it to a
 * `TubeMesh` before anything else touches its shader. `stripe` is the second
 * colour of the braided ropes and the band of the jackets.
 */
export function ropeMaterial(stripe = new THREE.Color('#f2f2ea')): THREE.MeshPhysicalMaterial {
  const m = new THREE.MeshPhysicalMaterial({
    vertexColors: true, roughness: 0.75, sheen: 0.8, sheenRoughness: 0.45, sheenColor: new THREE.Color('#ffffff'),
  });
  m.onBeforeCompile = (s) => {
    s.uniforms.ropeStripe = { value: stripe };
    s.vertexShader = s.vertexShader
      .replace('void main() {', `${vertexDecl}\nvoid main() {`)
      .replace('#include <begin_vertex>', vertexMain);
    s.fragmentShader = s.fragmentShader
      .replace('void main() {', `${fragmentDecl}\nvoid main() {`)
      .replace('#include <color_fragment>', /* glsl */ `#include <color_fragment>
  float ropeA = atan(vRope.z, vRope.y) / 6.2831853;
  ropeSurface(vRope.x, ropeA, max(vRopeR, 1e-4), vRope.w);
  // fades the relief where it is finer than a pixel, rather than let it shimmer
  float ropeFine = clamp(1.5 - 6.0 * length(fwidth(vec2(vRope.x / vRopeR, ropeA * 6.2831853))), 0.0, 1.0);
  ropeH = mix(0.8, ropeH, ropeFine);
  ropeSlope *= ropeFine;
  vec3 ropeBase = vRope.w < 0.5 ? diffuseColor.rgb * (1.0 - ropeTone) : mix(diffuseColor.rgb, ropeStripe, ropeTone);
  // the grooves between strands are in shadow
  diffuseColor.rgb = ropeBase * mix(0.35, 1.0, smoothstep(0.0, 0.9, ropeH));`)
      .replace('#include <roughnessmap_fragment>', /* glsl */ `#include <roughnessmap_fragment>
  roughnessFactor = vRope.w > 1.5 ? 0.38 : mix(0.95, roughnessFactor, ropeH);`)
      .replace('#include <normal_fragment_maps>', /* glsl */ `#include <normal_fragment_maps>
  {
    vec3 T = normalize(vRopeT - dot(vRopeT, normal) * normal);
    vec3 B = cross(T, normal);
    normal = normalize(normal - ropeSlope.x * T + ropeSlope.y * B);
  }`)
      .replace('#include <lights_physical_fragment>', /* glsl */ `#include <lights_physical_fragment>
  // fibres catch the light at grazing angles; a plastic jacket does not
  material.sheenColor *= vRope.w > 1.5 ? vec3(0.0) : 0.6 * diffuseColor.rgb + 0.2;`);
  };
  m.customProgramCacheKey = () => 'rope';
  return m;
}
