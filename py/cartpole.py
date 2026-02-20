# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import json
import struct
from pathlib import Path

import anywidget
import jax
import numpy as np
import traitlets


def as_f32(value: object) -> np.ndarray:
    """Convert a JAX array or NNX Variable to a float32 numpy array."""
    if hasattr(value, "__getitem__"):
        try:
            value = value[...]
        except Exception:
            pass
    elif hasattr(value, "get_value"):
        value = value.get_value()
    return np.asarray(jax.device_get(value), dtype=np.float32)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write a minimal safetensors file containing float32 tensors."""
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized = {
        name: np.ascontiguousarray(arr, dtype=np.float32) for name, arr in tensors.items()
    }

    header: dict[str, object] = {}
    offset = 0
    for name, arr in normalized.items():
        size = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - (len(header_bytes) % 8)) % 8)

    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for arr in normalized.values():
            f.write(arr.tobytes(order="C"))


class CartPoleVisualization(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "dist" / "cartpole-visualization.js"
    _css = (Path(__file__).parent.parent / "ts" / "reinforcement-learning.css").read_text()

    weights_base64 = traitlets.Unicode("").tag(sync=True)
