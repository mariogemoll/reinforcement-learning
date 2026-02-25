// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import type { QBarEls } from '../pong-policy/types';

export type { QBarEls } from '../pong-policy/types';

export interface PixelPongPolicyVizDom {
  container: HTMLDivElement;
  canvas: HTMLCanvasElement;
  overlay: HTMLDivElement;
  overlayTitle: HTMLDivElement;
  playAgainBtn: HTMLButtonElement;
  qBars: { noop: QBarEls; up: QBarEls; down: QBarEls };
}
