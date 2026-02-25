// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export interface MinAtarBreakoutVizDom {
  container: HTMLDivElement;
  canvas: HTMLCanvasElement;
  userModeRadio: HTMLInputElement;
  policyModeRadio: HTMLInputElement;
  policyModeText: HTMLSpanElement;
  restartBtn: HTMLButtonElement;
  hint: HTMLSpanElement;
  qPanel: HTMLDivElement;
  overlay: HTMLDivElement;
  overlayTitle: HTMLDivElement;
  overlayRestartBtn: HTMLButtonElement;
  scoreValue: HTMLSpanElement;
  stepValue: HTMLSpanElement;
  qBars: {
    noop: { bar: HTMLDivElement; value: HTMLSpanElement };
    left: { bar: HTMLDivElement; value: HTMLSpanElement };
    right: { bar: HTMLDivElement; value: HTMLSpanElement };
  };
}
