/**
 * Minimal Gaussian Splat loader supporting two formats:
 *
 *  - .splat  — antimatter15's compact binary format. Each splat is a fixed
 *              32-byte record: position (3x float32), scale (3x float32),
 *              color (4x uint8 RGBA), rotation quaternion (4x uint8, packed
 *              0-255 -> -1..1). This is the format most WebXR splat viewers
 *              expect, and what you'd convert a .ply into for faster loading.
 *
 *  - .ply    — the standard output format from gsplat/Nerfstudio training.
 *              Binary little-endian PLY with per-vertex properties: x,y,z
 *              position, f_dc_* (spherical harmonics DC term -> base color),
 *              opacity, scale_*, rot_* (quaternion). This is what your
 *              teammate's training pipeline will most likely produce
 *              directly — no conversion step needed, this loader reads it.
 *
 * Both loaders return a Float32Array-backed splat buffer compatible with
 * the renderer in main.js: interleaved [x,y,z, r,g,b,a, sx,sy,sz, qx,qy,qz,qw]
 * per splat, so the renderer doesn't need to know which format was loaded.
 */

const SPLAT_STRIDE_FLOATS = 14; // x,y,z, r,g,b,a, sx,sy,sz, qx,qy,qz,qw

export async function loadSplatFile(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${url}: ${resp.status} ${resp.statusText}`);
  }

  const contentLength = Number(resp.headers.get('content-length')) || 0;
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress && contentLength) {
      onProgress(received / contentLength);
    }
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  const isPly = url.toLowerCase().endsWith('.ply');
  return isPly ? parsePly(buffer.buffer) : parseSplatBinary(buffer.buffer);
}

function parseSplatBinary(arrayBuffer) {
  const RECORD_BYTES = 32;
  const count = Math.floor(arrayBuffer.byteLength / RECORD_BYTES);
  const out = new Float32Array(count * SPLAT_STRIDE_FLOATS);

  const f32 = new Float32Array(arrayBuffer);
  const u8 = new Uint8Array(arrayBuffer);

  for (let i = 0; i < count; i++) {
    const recordFloatOffset = (i * RECORD_BYTES) / 4;
    const recordByteOffset = i * RECORD_BYTES;
    const o = i * SPLAT_STRIDE_FLOATS;

    // position (3x float32)
    out[o + 0] = f32[recordFloatOffset + 0];
    out[o + 1] = f32[recordFloatOffset + 1];
    out[o + 2] = f32[recordFloatOffset + 2];

    // scale (3x float32)
    out[o + 7] = f32[recordFloatOffset + 3];
    out[o + 8] = f32[recordFloatOffset + 4];
    out[o + 9] = f32[recordFloatOffset + 5];

    // color (4x uint8 RGBA), normalized to 0-1
    out[o + 3] = u8[recordByteOffset + 24] / 255;
    out[o + 4] = u8[recordByteOffset + 25] / 255;
    out[o + 5] = u8[recordByteOffset + 26] / 255;
    out[o + 6] = u8[recordByteOffset + 27] / 255;

    // rotation quaternion (4x uint8, packed 0-255 -> -1..1)
    out[o + 10] = (u8[recordByteOffset + 28] - 128) / 128;
    out[o + 11] = (u8[recordByteOffset + 29] - 128) / 128;
    out[o + 12] = (u8[recordByteOffset + 30] - 128) / 128;
    out[o + 13] = (u8[recordByteOffset + 31] - 128) / 128;
  }

  return { count, data: out };
}

function parsePly(arrayBuffer) {
  // Parse the ASCII header first to find property offsets and where binary
  // data begins — gsplat/Nerfstudio output is binary_little_endian with a
  // fixed but not universally-identical property order, so we read the
  // header rather than assuming fixed offsets.
  const headerText = new TextDecoder('ascii').decode(arrayBuffer.slice(0, 4096));
  const headerEndMarker = 'end_header\n';
  const headerEndIdx = headerText.indexOf(headerEndMarker);
  if (headerEndIdx === -1) {
    throw new Error('Could not find PLY header end — file may not be binary_little_endian format.');
  }
  const headerEndByte = headerEndIdx + headerEndMarker.length;
  const header = headerText.slice(0, headerEndIdx);

  const vertexCountMatch = header.match(/element vertex (\d+)/);
  if (!vertexCountMatch) {
    throw new Error('Could not find vertex count in PLY header.');
  }
  const count = parseInt(vertexCountMatch[1], 10);
  console.log(`[parsePly] header parsed, vertex count: ${count.toLocaleString()}`);

  const propertyLines = header
    .split('\n')
    .filter((l) => l.trim().startsWith('property'))
    .map((l) => l.trim().split(/\s+/)); // ["property", "float", "x"]

  const propertyOffsets = {};
  let byteOffset = 0;
  const typeSizes = { float: 4, float32: 4, double: 8, uchar: 1, uint8: 1, int: 4, int32: 4, short: 2, ushort: 2 };
  for (const [, type, name] of propertyLines) {
    propertyOffsets[name] = { offset: byteOffset, type };
    byteOffset += typeSizes[type] ?? 4;
  }
  const recordBytes = byteOffset;

  const dv = new DataView(arrayBuffer, headerEndByte);
  const out = new Float32Array(count * SPLAT_STRIDE_FLOATS);

  const readProp = (name, recordStart) => {
    const prop = propertyOffsets[name];
    if (!prop) return 0;
    const at = recordStart + prop.offset;
    switch (prop.type) {
      case 'float':
      case 'float32':
        return dv.getFloat32(at, true);
      case 'double':
        return dv.getFloat64(at, true);
      case 'uchar':
      case 'uint8':
        return dv.getUint8(at);
      default:
        return dv.getFloat32(at, true);
    }
  };

  // Spherical harmonics DC term -> approximate base RGB color. This is the
  // standard gsplat/3DGS convention (SH degree-0 coefficient, scaled by a
  // fixed constant, offset to 0-1 range).
  const SH_C0 = 0.28209479177387814;

  for (let i = 0; i < count; i++) {
    const recordStart = i * recordBytes;
    const o = i * SPLAT_STRIDE_FLOATS;

    out[o + 0] = readProp('x', recordStart);
    out[o + 1] = readProp('y', recordStart);
    out[o + 2] = readProp('z', recordStart);

    const fdc0 = readProp('f_dc_0', recordStart);
    const fdc1 = readProp('f_dc_1', recordStart);
    const fdc2 = readProp('f_dc_2', recordStart);
    out[o + 3] = Math.max(0, Math.min(1, 0.5 + SH_C0 * fdc0));
    out[o + 4] = Math.max(0, Math.min(1, 0.5 + SH_C0 * fdc1));
    out[o + 5] = Math.max(0, Math.min(1, 0.5 + SH_C0 * fdc2));

    const rawOpacity = readProp('opacity', recordStart);
    // gsplat stores opacity pre-sigmoid; squash to 0-1.
    out[o + 6] = 1 / (1 + Math.exp(-rawOpacity));

    // gsplat stores log-scale; exponentiate to get actual scale.
    out[o + 7] = Math.exp(readProp('scale_0', recordStart));
    out[o + 8] = Math.exp(readProp('scale_1', recordStart));
    out[o + 9] = Math.exp(readProp('scale_2', recordStart));

    out[o + 10] = readProp('rot_1', recordStart); // qx
    out[o + 11] = readProp('rot_2', recordStart); // qy
    out[o + 12] = readProp('rot_3', recordStart); // qz
    out[o + 13] = readProp('rot_0', recordStart); // qw (gsplat stores w first)
  }

  return { count, data: out };
}

export { SPLAT_STRIDE_FLOATS };