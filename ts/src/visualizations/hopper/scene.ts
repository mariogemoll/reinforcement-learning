// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

import { type Geom, LANE_SPACING, type Rollout } from './rollout';

export const TRACK_DISTANCE = 4.0;
export const TRACK_ELEVATION_DEG = -20.0;
export const TRACK_AZIMUTH_DEG = -90.0;

export interface HopperScene {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  orbitControls: OrbitControls;
  directionalLight: THREE.DirectionalLight;
  lightOffset: THREE.Vector3;
  meshes: THREE.Mesh[];
  planeCount: number;
  geomsPerAgent: number;
  destroy(): void;
}

export function mujocoTrackingOffset(
  distance: number,
  elevationDeg: number,
  azimuthDeg: number
): THREE.Vector3 {
  const elevation = THREE.MathUtils.degToRad(elevationDeg);
  const azimuth = THREE.MathUtils.degToRad(azimuthDeg);
  const planar = distance * Math.cos(elevation);
  return new THREE.Vector3(
    planar * Math.cos(azimuth),
    planar * Math.sin(azimuth),
    -distance * Math.sin(elevation)
  );
}

function makeMesh(geom: Geom, floorHalfExtent: number): THREE.Mesh {
  let geometry: THREE.BufferGeometry;
  switch (geom.type) {
  case 'plane':
    geometry = new THREE.PlaneGeometry(floorHalfExtent * 2, floorHalfExtent * 2);
    break;
  case 'sphere':
    geometry = new THREE.SphereGeometry(geom.size[0], 24, 24);
    break;
  case 'capsule': {
    const capsule = new THREE.CapsuleGeometry(geom.size[0], geom.size[1] * 2, 8, 16);
    capsule.rotateX(Math.PI / 2);
    geometry = capsule;
    break;
  }
  case 'cylinder': {
    const cylinder = new THREE.CylinderGeometry(geom.size[0], geom.size[0], geom.size[1] * 2, 24);
    cylinder.rotateX(Math.PI / 2);
    geometry = cylinder;
    break;
  }
  case 'box':
    geometry = new THREE.BoxGeometry(geom.size[0] * 2, geom.size[1] * 2, geom.size[2] * 2);
    break;
  case 'ellipsoid': {
    const ellipsoid = new THREE.SphereGeometry(1, 24, 24);
    ellipsoid.scale(geom.size[0], geom.size[1], geom.size[2]);
    geometry = ellipsoid;
    break;
  }
  }

  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(geom.rgba[0], geom.rgba[1], geom.rgba[2]),
    opacity: geom.rgba[3],
    transparent: geom.rgba[3] < 1,
    roughness: 0.6,
    metalness: 0.1
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = geom.type !== 'plane';
  mesh.receiveShadow = true;
  mesh.visible = geom.type !== 'plane';
  return mesh;
}

export function createHopperScene(
  viewport: HTMLElement,
  rollout: Rollout
): HopperScene {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  viewport.append(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xb8cce8);
  scene.fog = new THREE.Fog(0xb8cce8, 40, 120);

  const planeGeom = rollout.geoms.find(geom => geom.type === 'plane');
  const planeHalfExtent = planeGeom ? Math.max(planeGeom.size[0], planeGeom.size[1]) : 5;
  const visualFloorHalfExtent = Math.max(planeHalfExtent, 100);
  const shadowHalfExtent = Math.max(planeHalfExtent, 20);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.up.set(0, 0, 1);

  const orbitControls = new OrbitControls(camera, renderer.domElement);
  orbitControls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  scene.add(new THREE.HemisphereLight(0x87ceeb, 0x2d8a4e, 0.4));
  const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
  const lightOffset = new THREE.Vector3(3, -3, 5);
  directionalLight.position.copy(lightOffset);
  directionalLight.castShadow = true;
  directionalLight.shadow.mapSize.set(2048, 2048);
  directionalLight.shadow.camera.near = 0.1;
  directionalLight.shadow.camera.far = Math.max(30, shadowHalfExtent * 4);
  directionalLight.shadow.camera.left = -shadowHalfExtent;
  directionalLight.shadow.camera.right = shadowHalfExtent;
  directionalLight.shadow.camera.top = shadowHalfExtent;
  directionalLight.shadow.camera.bottom = -shadowHalfExtent;
  scene.add(directionalLight);
  scene.add(directionalLight.target);

  const forwardAxis = rollout.forwardAxis;
  const trackLength = visualFloorHalfExtent * 2;
  const trackWidth = rollout.nagents * LANE_SPACING;
  const lineWidth = 0.05;

  const grassCanvas = document.createElement('canvas');
  const stripeCount = 20;
  grassCanvas.width = forwardAxis === 0 ? stripeCount : 1;
  grassCanvas.height = forwardAxis === 0 ? 1 : stripeCount;
  const grassContext = grassCanvas.getContext('2d');
  if (grassContext === null) {
    throw new Error('Could not create grass texture context');
  }
  for (let stripeIndex = 0; stripeIndex < stripeCount; stripeIndex++) {
    grassContext.fillStyle = stripeIndex % 2 === 0 ? '#2d8a4e' : '#267a42';
    if (forwardAxis === 0) {
      grassContext.fillRect(stripeIndex, 0, 1, 1);
    } else {
      grassContext.fillRect(0, stripeIndex, 1, 1);
    }
  }
  const grassTexture = new THREE.CanvasTexture(grassCanvas);
  grassTexture.minFilter = THREE.NearestFilter;
  grassTexture.magFilter = THREE.NearestFilter;
  grassTexture.generateMipmaps = false;

  const grass = new THREE.Mesh(
    new THREE.PlaneGeometry(trackLength, trackLength),
    new THREE.MeshStandardMaterial({
      map: grassTexture,
      roughness: 0.9,
      metalness: 0.0,
      polygonOffset: true,
      polygonOffsetFactor: 2,
      polygonOffsetUnits: 2
    })
  );
  grass.receiveShadow = true;
  grass.renderOrder = 0;
  scene.add(grass);

  const tartan = new THREE.Mesh(
    forwardAxis === 0
      ? new THREE.PlaneGeometry(trackLength, trackWidth)
      : new THREE.PlaneGeometry(trackWidth, trackLength),
    new THREE.MeshStandardMaterial({
      color: 0xb5651d,
      roughness: 0.8,
      metalness: 0.0,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1
    })
  );
  tartan.receiveShadow = true;
  tartan.renderOrder = 1;
  scene.add(tartan);

  for (let lineIndex = 0; lineIndex <= rollout.nagents; lineIndex++) {
    const lineOffset = (lineIndex - rollout.nagents / 2) * LANE_SPACING;
    const lane = new THREE.Mesh(
      forwardAxis === 0
        ? new THREE.PlaneGeometry(trackLength, lineWidth)
        : new THREE.PlaneGeometry(lineWidth, trackLength),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 })
    );
    if (forwardAxis === 0) {
      lane.position.set(0, lineOffset, 0.001);
    } else {
      lane.position.set(lineOffset, 0, 0.001);
    }
    lane.renderOrder = 2;
    scene.add(lane);
  }

  const meshes = rollout.geoms.map(geom => {
    const mesh = makeMesh(geom, visualFloorHalfExtent);
    scene.add(mesh);
    return mesh;
  });

  const planeCount = rollout.geoms.filter(geom => geom.type === 'plane').length;
  const geomsPerAgent =
    rollout.nagents > 0 ? (rollout.geoms.length - planeCount) / rollout.nagents : 0;

  let sum = 0;
  for (let agentIndex = 0; agentIndex < rollout.nagents; agentIndex++) {
    const geomIndex = planeCount + agentIndex * geomsPerAgent;
    sum += rollout.framePos[geomIndex * 3 + forwardAxis];
  }
  const startForwardPosition = sum / rollout.nagents + 0.4;
  const startLine = new THREE.Mesh(
    forwardAxis === 0
      ? new THREE.PlaneGeometry(lineWidth * 2, trackWidth)
      : new THREE.PlaneGeometry(trackWidth, lineWidth * 2),
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.5,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1
    })
  );
  if (forwardAxis === 0) {
    startLine.position.set(startForwardPosition, 0, 0.001);
  } else {
    startLine.position.set(0, startForwardPosition, 0.001);
  }
  startLine.renderOrder = 3;
  scene.add(startLine);

  return {
    scene,
    renderer,
    camera,
    orbitControls,
    directionalLight,
    lightOffset,
    meshes,
    planeCount,
    geomsPerAgent,
    destroy(): void {
      orbitControls.dispose();
      renderer.dispose();
    }
  };
}
