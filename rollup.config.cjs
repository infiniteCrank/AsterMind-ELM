// rollup.config.cjs
const typescript = require('rollup-plugin-typescript2');

module.exports = async () => {
  const { nodeResolve } = await import('@rollup/plugin-node-resolve');
  const commonjs = (await import('@rollup/plugin-commonjs')).default;

  return {
    input: 'src/index.ts',
    plugins: [
      nodeResolve({ browser: true }),
      commonjs(),
      typescript({ tsconfig: './tsconfig.json', useTsconfigDeclarationDir: true }),
    ],
    output: [
      { file: 'dist/index.esm.js', format: 'esm', sourcemap: true },
      { file: 'dist/index.cjs', format: 'cjs', sourcemap: true },
      { file: 'dist/index.umd.js', format: 'umd', name: 'astermind', sourcemap: true, exports: 'named' },
    ],
  };
};
