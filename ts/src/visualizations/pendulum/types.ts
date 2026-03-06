// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

export interface PendulumVizDom {
  container: HTMLDivElement;
  canvas: HTMLCanvasElement;
  torqueChartCanvas: HTMLCanvasElement;
  timelineCanvas: HTMLCanvasElement;
  terminalOverlay: HTMLDivElement;
  terminalTitle: HTMLElement;
  terminalSummary: HTMLElement;
  episodesValue: HTMLElement;
  stepsValue: HTMLElement;
  angleValue: HTMLElement;
  angVelValue: HTMLElement;
  torqueValue: HTMLElement;
  returnValue: HTMLElement;
  goToStartBtn: HTMLButtonElement;
  stepBackBtn: HTMLButtonElement;
  playBtn: HTMLButtonElement;
  stepForwardBtn: HTMLButtonElement;
  stepCounterEl: HTMLElement;
  resetBtn: HTMLButtonElement;
  speedSlider: HTMLInputElement;
  speedValueEl: HTMLElement;
}
