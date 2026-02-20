// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { initCartPoleVisualization } from '../visualizations/cartpole/index';

interface AnyModel {
  get(key: string): unknown;
  on(event: string, callback: () => void): void;
  off(event: string, callback: () => void): void;
}

async function render({
  model,
  el
}: {
  model: AnyModel;
  el: HTMLElement;
}): Promise<() => void> {
  let visualization: { destroy(): void } | null = null;
  let generation = 0;

  const parent = document.createElement('div');
  parent.id = 'cartpole-visualization';
  el.appendChild(parent);

  const mount = async(): Promise<void> => {
    const currentGeneration = ++generation;
    visualization?.destroy();
    visualization = null;
    parent.innerHTML = '';
    const base64 = model.get('weights_base64') as string;
    const weightsUrl = `data:application/octet-stream;base64,${base64}`;
    const created = await initCartPoleVisualization(parent, weightsUrl);
    if (currentGeneration !== generation) {
      created.destroy();
      return;
    }
    visualization = created;
  };

  const onWeightsBase64Change = (): void => {
    void mount();
  };
  model.on('change:weights_base64', onWeightsBase64Change);
  await mount();

  return () => {
    generation++;
    model.off('change:weights_base64', onWeightsBase64Change);
    visualization?.destroy();
    visualization = null;
    parent.remove();
  };
}

export default { render };
