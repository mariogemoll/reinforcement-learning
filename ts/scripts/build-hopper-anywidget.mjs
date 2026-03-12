// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import esbuild from 'esbuild';

const outfile = '../py/dist/hopper-visualization.js';
const hopperDir = './src/visualizations/hopper';
const threeCdnUrl = 'https://cdn.jsdelivr.net/npm/three@0.183.2/build/three.module.js';

const browserThreeShimPlugin = {
  name: 'hopper-browser-three-shim',
  setup(build) {
    build.onResolve({ filter: /^three$/ }, () => ({
      path: 'three-browser-shim',
      namespace: 'hopper-browser'
    }));

    build.onLoad({ filter: /^three-browser-shim$/, namespace: 'hopper-browser' }, () => ({
      contents: [
        `import * as THREE from '${threeCdnUrl}';`,
        `export * from '${threeCdnUrl}';`,
        'export default THREE;'
      ].join('\n'),
      loader: 'js'
    }));
  }
};

await esbuild.build({
  entryPoints: [`${hopperDir}/anywidget.ts`],
  bundle: true,
  format: 'esm',
  minify: true,
  outfile,
  plugins: [browserThreeShimPlugin],
  external: [threeCdnUrl]
});
