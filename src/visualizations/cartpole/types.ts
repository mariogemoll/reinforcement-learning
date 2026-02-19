// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export interface CartpolePolicyVizDom {
  container: HTMLDivElement;
  canvas: HTMLCanvasElement;
  terminalOverlay: HTMLDivElement;
  terminalTitle: HTMLElement;
  terminalSummary: HTMLElement;
  episodesValue: HTMLElement;
  stepsValue: HTMLElement;
  positionValue: HTMLElement;
  velocityValue: HTMLElement;
  angleValue: HTMLElement;
  angularVelocityValue: HTMLElement;
  qLeftValue: HTMLElement;
  qRightValue: HTMLElement;
  qLeftBar: HTMLElement;
  qRightBar: HTMLElement;
  qLeftRow: HTMLElement;
  qRightRow: HTMLElement;
  actionValue: HTMLElement;
  pauseBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  speedSlider: HTMLInputElement;
  speedValueEl: HTMLElement;
}
