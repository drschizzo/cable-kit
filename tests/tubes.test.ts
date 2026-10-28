import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { TubeMesh, TubeNodes, TubeSync, swept } from '@drschizzo/three-cable-tubes';
import * as GPU from '@drschizzo/three-cable-tubes/webgpu';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { float, vec3 } from 'three/tsl';

// The tubes without a GPU: what they send it (the node map) and the shader they make.

/** texel t of the node map (4 floats) */
function texel(nodes: TubeNodes, t: number): number[] {
  const d = nodes.texture.image.data as Float32Array;
  return [...d.slice(4 * t, 4 * t + 4)];
}

/** the vertex shader a material is compiled with, from a bare one */
function vertexOf(m: THREE.Material): string {
  const s = { uniforms: {}, vertexShader: 'void main() {\n}', fragmentShader: '' } as unknown as THREE.WebGLProgramParametersWithUniforms;
  m.onBeforeCompile(s, {} as THREE.WebGLRenderer);
  return s.vertexShader;
}

describe('tubes', () => {
  it('a node takes the radius of its tube, one for all or one per node', () => {
    const x = [0, 0, 0, 1, 0, 0, 2, 0, 0];
    const nodes = new TubeNodes(3);
    nodes.write(x, 0, 3, 0.01);
    expect(texel(nodes, 3)).toEqual([1, 0, 0, 0.01].map((v) => expect.closeTo(v, 6)));
    nodes.write(x, 0, 3, [0.01, 0.02, 0.03]);
    expect(texel(nodes, 3 * 2)[3]).toBeCloseTo(0.03);
    expect(nodes.radius).toBeCloseTo(0.03);
    // the frame's normal is square to the cord
    const nrm = texel(nodes, 3 * 1 + 1);
    expect(nrm[0]).toBeCloseTo(0);
    expect(Math.hypot(nrm[0], nrm[1], nrm[2])).toBeCloseTo(1);
  });

  it('only the cables that moved are written again', () => {
    const nodes = new TubeNodes(4);
    const src = { x: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]), cables: [{ start: 0, count: 2, asleep: false }, { start: 2, count: 2, asleep: true }] };
    const sync = new TubeSync(nodes, 0.01);
    expect(sync.cable(src, 0)).toBe(true);
    expect(sync.cable(src, 1)).toBe(true);
    // asleep and still: skipped; awake: written; asleep but moved, or forced: written
    expect(sync.cable(src, 1)).toBe(false);
    expect(sync.cable(src, 0)).toBe(true);
    src.x[3 * 3 + 2] = 0.5;
    expect(sync.cable(src, 1)).toBe(true);
    expect(sync.cable(src, 1)).toBe(false);
    expect(sync.cable(src, 1, true)).toBe(true);
  });

  it('the shader follows the options, and names its values as asked', () => {
    const nodes = new TubeNodes(2);
    const def = vertexOf(swept(new THREE.MeshBasicMaterial(), nodes));
    expect(def).toContain('vec4 tubeCoord');
    expect(def).toContain('vec3 tubeTint');
    expect(def).toContain('float(tK) / 8.0');
    // the coarsest rings face the camera
    expect(def).toMatch(/if \(tStep == 4\) \{[^}]*cameraPosition/);
    const own = vertexOf(swept(new THREE.MeshBasicMaterial(), nodes, { radial: 8, sub: 4, aliases: { glow: 'tubeTint' } }));
    expect(own).toContain('float(tK) / 4.0');
    expect(own).toContain(`float(tR) * ${(Math.PI / 4).toFixed(8)}`);
    expect(own).toContain('#define glow tubeTint');
    // a program of its own
    const a = swept(new THREE.MeshBasicMaterial(), nodes), b = swept(new THREE.MeshBasicMaterial(), nodes, { sub: 4 });
    expect(a.customProgramCacheKey()).not.toBe(b.customProgramCacheKey());
  });

  it('a mesh lays one instance per segment, with the rings its options ask for', () => {
    const nodes = new TubeNodes(5);
    const mesh = new TubeMesh(4, new THREE.MeshBasicMaterial(), nodes, { radial: 8, sub: 4 });
    expect(mesh.geometry.getAttribute('normal').count).toBe((4 + 1) * 8);
    mesh.lay(0, 0, 5, new THREE.Color(1, 0, 0), 0.5, 0.025, 2);
    const arc = mesh.geometry.getAttribute('tubeArc');
    expect([arc.getX(3), arc.getY(3), arc.getZ(3)]).toEqual([0.575, 0.025, 2].map((v) => expect.closeTo(v, 6)));
    mesh.dispose();
  });

  it('the WebGPU mesh sweeps its node material, with the same rings and segments', () => {
    const nodes = new GPU.TubeNodes(5);
    const material = new MeshStandardNodeMaterial();
    const mesh = new GPU.TubeMesh(4, material, nodes, { radial: 8, sub: 4 });
    expect(material.positionNode).not.toBeNull();
    expect(mesh.geometry.getAttribute('normal').count).toBe((4 + 1) * 8);
    expect(mesh.geometry.instanceCount).toBe(4);
    mesh.lay(0, 0, 5, new THREE.Color(1, 0, 0), 0, 0.025, 1);
    expect(mesh.geometry.getAttribute('tubeSeg').getZ(2)).toBe(4);
    // the coarser lists: a quarter, a sixteenth of the triangles
    const counts = [0, 1, 2].map((l) => { (mesh as unknown as { lod(l: number): void }).lod(l); return mesh.geometry.index!.count; });
    expect(counts).toEqual([counts[0], counts[0] / 4, counts[0] / 16]);
    mesh.dispose();
  });

  it('the WebGPU mesh also reads nodes the GPU writes, laid out as the CPU writes them', () => {
    const cpu = new GPU.TubeNodes(1000), gpu = new GPU.GPUTubeNodes(1000);
    const size = (t: THREE.Texture) => { const im = t.image as { width: number; height: number }; return [im.width, im.height]; };
    expect(size(gpu.texture)).toEqual(size(cpu.texture));
    expect(gpu.texture.type).toBe(THREE.FloatType);
    gpu.box.set(new THREE.Vector3(-1, 0, -1), new THREE.Vector3(1, 1, 1));
    gpu.radius = 0.001;
    const material = new MeshStandardNodeMaterial();
    const mesh = new GPU.TubeMesh(999, material, gpu);
    expect(material.positionNode).not.toBeNull();
    mesh.dispose();
  });

  it('GPU nodes widen their texture past 8192 rows', () => {
    const small = new GPU.GPUTubeNodes(1000), big = new GPU.GPUTubeNodes(5_000_000);
    expect(small.width).toBe(1024);
    const h = (big.texture.image as { height: number }).height;
    expect(h).toBeLessThanOrEqual(8192);
    expect(big.width * h).toBeGreaterThanOrEqual(3 * 5_000_000);
  });

  it('sweepStrands makes a compute shader per node, and one for the glow only when there is a tint', () => {
    const nodes = new GPU.GPUTubeNodes(10 * 8);
    const base = { count: 10, nodes: 8, segment: 0.02, root: () => vec3(0, 0, 0), direction: () => vec3(0, 1, 0), radius: () => float(0.001) };
    const plain = GPU.sweepStrands(nodes, base);
    expect(plain.update.count).toBe(80);
    expect(plain.glow).toBeNull();
    const lit = GPU.sweepStrands(nodes, { ...base, tint: () => vec3(1, 0, 0) });
    expect(lit.glow!.count).toBe(80);
    // laid strand by strand, each on its own segments
    const mesh = new GPU.TubeMesh(10 * 7, new MeshStandardNodeMaterial(), nodes);
    GPU.layStrands(mesh, base, () => new THREE.Color(1, 1, 1), (s) => s);
    const seg = mesh.geometry.getAttribute('tubeSeg'), arc = mesh.geometry.getAttribute('tubeArc');
    expect([seg.getX(7 * 3), seg.getY(7 * 3), seg.getZ(7 * 3), arc.getZ(7 * 3)]).toEqual([24, 24, 31, 3]);
    mesh.dispose();
  });

  it('its triangles face out of the tube', () => {
    // the stretch as the shader lays it along x: ring k at x = k, vertex r at the angle
    // 2πr/radial from side = y towards up = tangent × side = z
    const mesh = new TubeMesh(1, new THREE.MeshBasicMaterial(), new TubeNodes(2), { radial: 8, sub: 4 });
    const ring = mesh.geometry.getAttribute('normal'), index = mesh.geometry.index!;
    const at = (v: number) => {
      const a = (2 * Math.PI * ring.getY(v)) / 8;
      return new THREE.Vector3(ring.getX(v), Math.cos(a), Math.sin(a));
    };
    for (let t = 0; t < index.count; t += 3) {
      const [a, b, c] = [0, 1, 2].map((k) => at(index.getX(t + k)));
      const n = new THREE.Vector3().subVectors(b, a).cross(new THREE.Vector3().subVectors(c, a));
      const out = new THREE.Vector3(0, a.y + b.y + c.y, a.z + b.z + c.z);
      expect(n.dot(out)).toBeGreaterThan(0);
    }
    mesh.dispose();
  });

  it('a material draws the nodes of one mesh at a time, and is free again once it is disposed', () => {
    const uniformsOf = (m: THREE.Material) => {
      const sh = { uniforms: {}, vertexShader: 'void main() {\n}', fragmentShader: '' } as unknown as THREE.WebGLProgramParametersWithUniforms;
      m.onBeforeCompile(sh, {} as THREE.WebGLRenderer);
      return { nodes: (sh.uniforms.tubeNodes as THREE.IUniform).value, vertex: sh.vertexShader };
    };
    const material = new THREE.MeshStandardMaterial();
    material.onBeforeCompile = (sh) => { sh.vertexShader = sh.vertexShader.replace('void main() {', '// mine\nvoid main() {'); };
    const a = new TubeNodes(2), b = new TubeNodes(2), c = new TubeNodes(2);
    const first = new TubeMesh(1, material, a), second = new TubeMesh(1, material, b);
    // the second one draws a copy, with the material's own additions
    expect(first.material).toBe(material);
    expect(second.material).not.toBe(material);
    expect(uniformsOf(first.material as THREE.Material).nodes).toBe(a.texture);
    const copy = uniformsOf(second.material as THREE.Material);
    expect(copy.nodes).toBe(b.texture);
    expect(copy.vertex).toContain('// mine');
    // once the first is disposed, the next one draws the material itself, swept once
    first.dispose();
    const third = new TubeMesh(1, material, c);
    expect(third.material).toBe(material);
    const again = uniformsOf(material);
    expect(again.nodes).toBe(c.texture);
    expect(again.vertex.split('uniform highp sampler2D tubeNodes').length).toBe(2);
    second.dispose();
    third.dispose();
  });

  it('on WebGPU too, two meshes do not share a material', () => {
    const material = new MeshStandardNodeMaterial();
    const first = new GPU.TubeMesh(1, material, new GPU.TubeNodes(2));
    const second = new GPU.TubeMesh(1, material, new GPU.TubeNodes(2));
    expect(first.material).toBe(material);
    expect(second.material).not.toBe(material);
    const before = material.positionNode;
    first.dispose();
    const third = new GPU.TubeMesh(1, material, new GPU.TubeNodes(2));
    expect(third.material).toBe(material);
    expect(material.positionNode).not.toBe(before);
    second.dispose();
    third.dispose();
  });
});
