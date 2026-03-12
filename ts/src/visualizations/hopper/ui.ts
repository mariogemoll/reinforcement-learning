// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { createTimelineStepControls } from '../shared/ui';

const SPEED_SLIDER_MIN = 10;
const SPEED_SLIDER_MAX = 400;
const SPEED_SLIDER_DEFAULT = 100;

export interface HopperVisualizationUi {
  root: HTMLDivElement;
  viewport: HTMLDivElement;
  playButton: HTMLButtonElement;
  scrubber: HTMLInputElement;
  stepCounterEl: HTMLSpanElement;
  speedSlider: HTMLInputElement;
  speedValueEl: HTMLElement;
  cameraModeSelect: HTMLSelectElement;
}

export function sliderToSpeed(value: number): number {
  return value / 100;
}

export function formatSpeed(speed: number): string {
  if (speed >= 1) {
    return `${speed.toFixed(1).replace(/\.0$/, '')}x`;
  }
  return `${speed.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')}x`;
}

export function buildUi(parent: HTMLElement): HopperVisualizationUi {
  const root = document.createElement('div');
  root.className = 'visualization hopper-viz-root';

  const viewport = document.createElement('div');
  viewport.className = 'hopper-viz-viewport';

  const controls = document.createElement('div');
  controls.className = 'hopper-viz-controls';

  const leftControls = document.createElement('div');
  leftControls.className = 'hopper-viz-controls-left';
  const {
    stepRow,
    playBtn: playButton
  } = createTimelineStepControls();
  stepRow.classList.add('hopper-viz-step-row');
  for (const child of Array.from(stepRow.children)) {
    if (child instanceof HTMLButtonElement && child !== playButton) {
      child.style.display = 'none';
    }
  }
  playButton.textContent = '\u23F8';
  leftControls.append(stepRow);

  const timelineWrap = document.createElement('div');
  timelineWrap.className = 'hopper-viz-timeline-wrap';

  const scrubber = document.createElement('input');
  scrubber.type = 'range';
  scrubber.min = '0';
  scrubber.step = '1';
  scrubber.setAttribute('aria-label', 'Time');
  scrubber.className = 'hopper-viz-scrubber';
  const timelineInfo = document.createElement('span');
  timelineInfo.className = 'pi-viz-step-counter hopper-viz-timeline-counter';
  timelineInfo.textContent = '1/1';
  timelineWrap.append(scrubber, timelineInfo);

  const rightControls = document.createElement('div');
  rightControls.className = 'hopper-viz-controls-right';

  const speedWrap = document.createElement('div');
  speedWrap.className = 'slider-wrap hopper-viz-mini-control';
  const speedLabel = document.createElement('div');
  speedLabel.className = 'slider-label';
  const speedText = document.createElement('span');
  speedText.textContent = 'Speed';
  const speedValueEl = document.createElement('strong');
  speedValueEl.className = 'mono-value';
  speedValueEl.textContent = formatSpeed(sliderToSpeed(SPEED_SLIDER_DEFAULT));
  speedLabel.append(speedText, speedValueEl);

  const speedSlider = document.createElement('input');
  speedSlider.type = 'range';
  speedSlider.min = String(SPEED_SLIDER_MIN);
  speedSlider.max = String(SPEED_SLIDER_MAX);
  speedSlider.step = '1';
  speedSlider.value = String(SPEED_SLIDER_DEFAULT);
  speedSlider.setAttribute('aria-label', 'Playback speed');
  speedWrap.append(speedLabel, speedSlider);

  const cameraModeSelect = document.createElement('select');
  const cameraWrap = document.createElement('div');
  cameraWrap.className = 'hopper-viz-mini-control';
  const cameraLabel = document.createElement('div');
  cameraLabel.className = 'slider-label';
  const cameraText = document.createElement('span');
  cameraText.textContent = 'Camera';
  cameraLabel.append(cameraText);
  cameraWrap.append(cameraLabel, cameraModeSelect);

  rightControls.append(speedWrap, cameraWrap);
  controls.append(leftControls, timelineWrap, rightControls);
  root.append(viewport, controls);

  const placeholder = parent.querySelector('.placeholder');
  if (placeholder) {
    placeholder.replaceWith(root);
  } else {
    parent.append(root);
  }

  return {
    root,
    viewport,
    playButton,
    scrubber,
    stepCounterEl: timelineInfo,
    speedSlider,
    speedValueEl,
    cameraModeSelect
  };
}
