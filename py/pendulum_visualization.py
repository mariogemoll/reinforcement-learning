# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

from pathlib import Path

import anywidget
import traitlets


class PendulumVisualization(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "dist" / "pendulum-visualization.js"
    _css = (Path(__file__).parent.parent / "ts" / "reinforcement-learning.css").read_text()
    weights_base64 = traitlets.Unicode("").tag(sync=True)
