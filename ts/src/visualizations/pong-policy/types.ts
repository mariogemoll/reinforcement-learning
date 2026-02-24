// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export interface QBarEls {
  bar: HTMLDivElement;
  value: HTMLSpanElement;
}

export interface PongPolicyVizDom {
  container: HTMLDivElement;
  canvas: HTMLCanvasElement;
  overlay: HTMLDivElement;
  overlayTitle: HTMLDivElement;
  playAgainBtn: HTMLButtonElement;
  qBars: { noop: QBarEls; up: QBarEls; down: QBarEls };
}
