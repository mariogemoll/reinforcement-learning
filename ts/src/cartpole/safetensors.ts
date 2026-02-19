// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

type SafetensorsHeader = Record<string, {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}>;

export async function loadSafetensors(url: string): Promise<Record<string, Float32Array>> {
  const buffer = await fetch(url).then(r => r.arrayBuffer());
  const view = new DataView(buffer);

  const headerLength = Number(view.getBigUint64(0, true));
  const headerBytes = new Uint8Array(buffer, 8, headerLength);
  const header = JSON.parse(new TextDecoder().decode(headerBytes)) as SafetensorsHeader;

  const dataStart = 8 + headerLength;
  const result: Record<string, Float32Array> = {};

  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') {continue;}
    const [start, end] = meta.data_offsets;
    result[name] = new Float32Array(buffer, dataStart + start, (end - start) / 4);
  }

  return result;
}
