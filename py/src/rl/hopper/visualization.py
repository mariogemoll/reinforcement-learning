# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import base64
from pathlib import Path

import anywidget
import traitlets

from rl.core.assets import dist_asset_path, shared_css_text


class HopperVisualization(anywidget.AnyWidget):
    _esm = dist_asset_path("hopper-visualization.js")
    _css = shared_css_text()

    rollout_base64 = traitlets.Unicode("").tag(sync=True)

    @classmethod
    def from_rollout_file(cls, rollout_path: str | Path) -> "HopperVisualization":
        payload = Path(rollout_path).read_bytes()
        return cls(rollout_base64=base64.b64encode(payload).decode("ascii"))
