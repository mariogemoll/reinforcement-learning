// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

const GEOM_TYPES = ['plane', 'sphere', 'capsule', 'ellipsoid', 'cylinder', 'box'] as const;

export const LANE_SPACING = 1.5;

export type GeomType = (typeof GEOM_TYPES)[number];

export interface Geom {
  type: GeomType;
  size: [number, number, number];
  rgba: [number, number, number, number];
}

export interface Rollout {
  fps: number;
  geoms: Geom[];
  nframes: number;
  nagents: number;
  forwardAxis: number;
  framePos: Float32Array;
  frameQuat: Float32Array;
}

export function parseRollout(buffer: ArrayBuffer): Rollout {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== 'MJRV') {
    throw new Error(`Unexpected rollout magic header: ${magic}`);
  }

  let offset = 4;
  const version = view.getUint32(offset, true);
  offset += 4;
  const fps = view.getUint32(offset, true);
  offset += 4;
  const ngeoms = view.getUint32(offset, true);
  offset += 4;
  const nframes = view.getUint32(offset, true);
  offset += 4;

  let nagents = 1;
  let forwardAxis = 0;
  let isPlanar = false;
  if (version >= 2) {
    nagents = view.getUint32(offset, true);
    offset += 4;
    forwardAxis = view.getUint32(offset, true);
    offset += 4;
    isPlanar = view.getUint32(offset, true) !== 0;
    offset += 4;
  }

  const geoms: Geom[] = [];
  for (let index = 0; index < ngeoms; index++) {
    const typeId = view.getUint8(offset);
    offset += 4;
    const size: [number, number, number] = [
      view.getFloat32(offset, true),
      view.getFloat32(offset + 4, true),
      view.getFloat32(offset + 8, true)
    ];
    offset += 12;
    const rgba: [number, number, number, number] = [
      view.getFloat32(offset, true),
      view.getFloat32(offset + 4, true),
      view.getFloat32(offset + 8, true),
      view.getFloat32(offset + 12, true)
    ];
    offset += 16;
    geoms.push({ type: GEOM_TYPES[typeId] ?? 'sphere', size, rgba });
  }

  const framePos = new Float32Array(nframes * ngeoms * 3);
  const frameQuat = new Float32Array(nframes * ngeoms * 4);

  if (isPlanar) {
    const floatsPerFrame = ngeoms * 3;
    const frameData = new Float32Array(buffer, offset, nframes * floatsPerFrame);
    const upAxis = 2;
    const lateralAxis = forwardAxis === 0 ? 1 : 0;
    const planeCount = geoms.filter(geom => geom.type === 'plane').length;
    const geomsPerAgent = nagents > 0 ? (ngeoms - planeCount) / nagents : 0;

    for (let frameIndex = 0; frameIndex < nframes; frameIndex++) {
      const base = frameIndex * floatsPerFrame;
      for (let geomIndex = 0; geomIndex < ngeoms; geomIndex++) {
        const forward = frameData[base + geomIndex * 2];
        const up = frameData[base + geomIndex * 2 + 1];
        const angle = frameData[base + ngeoms * 2 + geomIndex];

        let lateral = 0;
        if (geomIndex >= planeCount && nagents > 1 && geomsPerAgent > 0) {
          const agentIndex = Math.floor((geomIndex - planeCount) / geomsPerAgent);
          lateral = (agentIndex - (nagents - 1) / 2) * LANE_SPACING;
        }

        const positionOffset = (frameIndex * ngeoms + geomIndex) * 3;
        framePos[positionOffset + forwardAxis] = forward;
        framePos[positionOffset + upAxis] = up;
        framePos[positionOffset + lateralAxis] = lateral;

        const halfAngle = angle / 2;
        const quatOffset = (frameIndex * ngeoms + geomIndex) * 4;
        frameQuat[quatOffset] = Math.cos(halfAngle);
        frameQuat[quatOffset + 1] = 0;
        frameQuat[quatOffset + 2] = 0;
        frameQuat[quatOffset + 3] = 0;
        frameQuat[quatOffset + 1 + lateralAxis] = Math.sin(halfAngle);
      }
    }
  } else {
    const floatsPerFrame = ngeoms * 7;
    const frameData = new Float32Array(buffer, offset, nframes * floatsPerFrame);
    for (let frameIndex = 0; frameIndex < nframes; frameIndex++) {
      const base = frameIndex * floatsPerFrame;
      framePos.set(frameData.subarray(base, base + ngeoms * 3), frameIndex * ngeoms * 3);
      frameQuat.set(
        frameData.subarray(base + ngeoms * 3, base + floatsPerFrame),
        frameIndex * ngeoms * 4
      );
    }
  }

  return { fps, geoms, nframes, nagents, forwardAxis, framePos, frameQuat };
}
