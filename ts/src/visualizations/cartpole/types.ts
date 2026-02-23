// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export interface CartPoleVizDom {
  container: HTMLDivElement;
  canvas: HTMLCanvasElement;
  actionChartCanvas: HTMLCanvasElement;
  timelineCanvas: HTMLCanvasElement;
  terminalOverlay: HTMLDivElement;
  terminalTitle: HTMLElement;
  terminalSummary: HTMLElement;
  episodesValue: HTMLElement;
  stepsValue: HTMLElement;
  positionValue: HTMLElement;
  velocityValue: HTMLElement;
  angleValue: HTMLElement;
  angularVelocityValue: HTMLElement;
  goToStartBtn: HTMLButtonElement;
  stepBackBtn: HTMLButtonElement;
  playBtn: HTMLButtonElement;
  stepForwardBtn: HTMLButtonElement;
  stepCounterEl: HTMLElement;
  resetBtn: HTMLButtonElement;
  speedSlider: HTMLInputElement;
  speedValueEl: HTMLElement;
}
