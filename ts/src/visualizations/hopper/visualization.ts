// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';

import { parseRollout } from './rollout';
import {
  createHopperScene,
  mujocoTrackingOffset,
  TRACK_AZIMUTH_DEG,
  TRACK_DISTANCE,
  TRACK_ELEVATION_DEG
} from './scene';
import { buildUi, formatSpeed, sliderToSpeed } from './ui';

type CameraMode = 'leader' | 'all' | number;

export interface HopperVisualization {
  destroy(): void;
}

export async function initializeHopperVisualization(
  parent: HTMLElement,
  rolloutUrl: string
): Promise<HopperVisualization> {
  const response = await fetch(rolloutUrl);
  if (!response.ok) {
    throw new Error(`Failed to load rollout: ${String(response.status)}`);
  }
  const rollout = parseRollout(await response.arrayBuffer());
  const ui = buildUi(parent);
  const hopperScene = createHopperScene(ui.viewport, rollout);
  const {
    scene,
    renderer,
    camera,
    orbitControls,
    directionalLight,
    lightOffset,
    meshes,
    planeCount,
    geomsPerAgent
  } = hopperScene;
  const ngeoms = rollout.geoms.length;

  const baseCameraOffset = mujocoTrackingOffset(
    TRACK_DISTANCE,
    TRACK_ELEVATION_DEG,
    TRACK_AZIMUTH_DEG
  );

  let currentFrame = 0;
  let playing = true;
  let speed = 1.0;
  let cameraMode: CameraMode = rollout.nagents > 1 ? 'all' : 0;
  let animationFrameId = 0;
  let destroyed = false;
  let accumulator = 0;
  let previousTime = performance.now();

  function resize(): void {
    const width = Math.max(ui.viewport.clientWidth, 1);
    const height = Math.max(ui.viewport.clientHeight, 1);
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function getAgentPosition(agentIndex: number, frameIndex: number): THREE.Vector3 {
    const geomIndex = planeCount + agentIndex * geomsPerAgent;
    const offset = frameIndex * ngeoms * 3 + geomIndex * 3;
    return new THREE.Vector3(
      rollout.framePos[offset],
      rollout.framePos[offset + 1],
      0.5
    );
  }

  function getLeaderIndex(frameIndex: number): number {
    let bestIndex = 0;
    let bestValue = -Infinity;
    for (let agentIndex = 0; agentIndex < rollout.nagents; agentIndex++) {
      const geomIndex = planeCount + agentIndex * geomsPerAgent;
      const value = rollout.framePos[frameIndex * ngeoms * 3 + geomIndex * 3 + rollout.forwardAxis];
      if (value > bestValue) {
        bestValue = value;
        bestIndex = agentIndex;
      }
    }
    return bestIndex;
  }

  function getCentroid(frameIndex: number): THREE.Vector3 {
    const centroid = new THREE.Vector3();
    for (let agentIndex = 0; agentIndex < rollout.nagents; agentIndex++) {
      centroid.add(getAgentPosition(agentIndex, frameIndex));
    }
    return centroid.divideScalar(rollout.nagents);
  }

  function getSpread(frameIndex: number): number {
    let maxDistance = 0;
    for (let leftIndex = 0; leftIndex < rollout.nagents; leftIndex++) {
      for (let rightIndex = leftIndex + 1; rightIndex < rollout.nagents; rightIndex++) {
        const distance = getAgentPosition(leftIndex, frameIndex).distanceTo(
          getAgentPosition(rightIndex, frameIndex)
        );
        maxDistance = Math.max(maxDistance, distance);
      }
    }
    return maxDistance;
  }

  function getTrackTarget(frameIndex: number): { distance: number; position: THREE.Vector3 } {
    if (rollout.nagents <= 1) {
      return { distance: TRACK_DISTANCE, position: getAgentPosition(0, frameIndex) };
    }
    if (cameraMode === 'all') {
      const spread = getSpread(frameIndex);
      const fovRad = THREE.MathUtils.degToRad(camera.fov);
      const requiredDistance = Math.max(TRACK_DISTANCE, (spread / 2) / Math.tan(fovRad / 2) + 2);
      return { distance: requiredDistance, position: getCentroid(frameIndex) };
    }
    const agentIndex = cameraMode === 'leader' ? getLeaderIndex(frameIndex) : cameraMode;
    return { distance: TRACK_DISTANCE, position: getAgentPosition(agentIndex, frameIndex) };
  }

  function applyFrame(frameIndex: number): void {
    const posOffset = frameIndex * ngeoms * 3;
    const quatOffset = frameIndex * ngeoms * 4;
    for (let geomIndex = 0; geomIndex < ngeoms; geomIndex++) {
      const posIndex = posOffset + geomIndex * 3;
      meshes[geomIndex].position.set(
        rollout.framePos[posIndex],
        rollout.framePos[posIndex + 1],
        rollout.framePos[posIndex + 2]
      );
      const quatIndex = quatOffset + geomIndex * 4;
      meshes[geomIndex].quaternion.set(
        rollout.frameQuat[quatIndex + 1],
        rollout.frameQuat[quatIndex + 2],
        rollout.frameQuat[quatIndex + 3],
        rollout.frameQuat[quatIndex]
      );
    }
    ui.scrubber.value = String(frameIndex);
    ui.stepCounterEl.textContent = `${String(frameIndex + 1)}/${String(rollout.nframes)}`;
  }

  function updateTrackedVisuals(target: THREE.Vector3): void {
    directionalLight.target.position.copy(target);
    directionalLight.position.copy(target).add(lightOffset);
    directionalLight.target.updateMatrixWorld();
  }

  function animate(): void {
    if (destroyed) {
      return;
    }
    animationFrameId = window.requestAnimationFrame(animate);
    const now = performance.now();
    const delta = now - previousTime;
    previousTime = now;

    if (playing) {
      accumulator += delta * speed;
      const frameDuration = 1000 / rollout.fps;
      while (accumulator >= frameDuration) {
        accumulator -= frameDuration;
        currentFrame = (currentFrame + 1) % rollout.nframes;
      }
      applyFrame(currentFrame);
    }

    const track = getTrackTarget(currentFrame);
    updateTrackedVisuals(track.position);
    const orbitOffset = camera.position.clone().sub(orbitControls.target);
    const currentDistance = orbitOffset.length();
    const targetDistance = track.distance;
    const lerpedDistance = currentDistance + (targetDistance - currentDistance) * 0.05;
    orbitOffset.normalize().multiplyScalar(lerpedDistance);
    orbitControls.target.copy(track.position);
    camera.position.copy(track.position).add(orbitOffset);
    orbitControls.update();
    renderer.render(scene, camera);
  }

  if (rollout.nagents > 1) {
    const options: { label: string; value: string }[] = [
      { label: 'All', value: 'all' },
      { label: 'Leader', value: 'leader' }
    ];
    for (let agentIndex = 0; agentIndex < rollout.nagents; agentIndex++) {
      options.push({ label: `Agent ${String(agentIndex + 1)}`, value: String(agentIndex) });
    }
    for (const optionDef of options) {
      const option = document.createElement('option');
      option.value = optionDef.value;
      option.textContent = optionDef.label;
      ui.cameraModeSelect.append(option);
    }
    ui.cameraModeSelect.value = 'all';
  } else {
    ui.cameraModeSelect.disabled = true;
    const option = document.createElement('option');
    option.value = '0';
    option.textContent = 'Agent 1';
    ui.cameraModeSelect.append(option);
  }

  ui.scrubber.max = String(Math.max(rollout.nframes - 1, 0));
  ui.playButton.addEventListener('click', () => {
    playing = !playing;
    ui.playButton.textContent = playing ? '\u23F8' : '\u25B6';
  });
  ui.scrubber.addEventListener('input', () => {
    currentFrame = Number(ui.scrubber.value);
    applyFrame(currentFrame);
  });
  ui.speedSlider.addEventListener('input', () => {
    speed = sliderToSpeed(Number(ui.speedSlider.value));
    ui.speedValueEl.textContent = formatSpeed(speed);
  });
  ui.cameraModeSelect.addEventListener('change', () => {
    const value = ui.cameraModeSelect.value;
    cameraMode = value === 'all' || value === 'leader' ? value : Number(value);
  });

  const resizeObserver = new ResizeObserver(() => {
    resize();
  });
  resizeObserver.observe(ui.viewport);

  resize();
  applyFrame(0);
  speed = sliderToSpeed(Number(ui.speedSlider.value));
  ui.speedValueEl.textContent = formatSpeed(speed);
  const initialTrack = getTrackTarget(0);
  updateTrackedVisuals(initialTrack.position);
  camera.position.copy(
    initialTrack.position.clone().add(
      baseCameraOffset.clone().normalize().multiplyScalar(initialTrack.distance)
    )
  );
  orbitControls.target.copy(initialTrack.position);
  orbitControls.update();
  animate();

  return {
    destroy(): void {
      destroyed = true;
      window.cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      hopperScene.destroy();
      ui.root.remove();
    }
  };
}
