# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

import anywidget
import traitlets

from rl.core.assets import dist_asset_path, shared_css_text


class PendulumVisualization(anywidget.AnyWidget):
    _esm = dist_asset_path("pendulum-visualization.js")
    _css = shared_css_text()
    weights_base64 = traitlets.Unicode("").tag(sync=True)
