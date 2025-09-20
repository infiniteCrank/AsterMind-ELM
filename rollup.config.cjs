// rollup.config.cjs
const fs = require('fs');
const typescript = require('rollup-plugin-typescript2');
const { nodeResolve } = require('@rollup/plugin-node-resolve');
const commonjs = require('@rollup/plugin-commonjs');

const pkg = JSON.parse(fs.readFileSync('./package.json', 'utf-8'));

module.exports = {
    input: 'src/index.ts',
    output: [
        {
            file: pkg.main,      // dist/astermind.umd.js
            format: 'umd',
            name: 'astermind',   // window.astermind
            sourcemap: true
        },
        {
            file: pkg.module,    // dist/astermind.esm.js
            format: 'esm',
            sourcemap: true
        },
    ],
    plugins: [
        nodeResolve({ browser: true }),
        commonjs(),
        typescript({
            tsconfig: './tsconfig.json',
            useTsconfigDeclarationDir: true,
            clean: true,
        }),
    ],
    // If you want to keep your bundle “pure,” mark external libs here:
    // external: [],
};
