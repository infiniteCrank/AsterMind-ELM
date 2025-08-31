// rollup.config.js (CommonJS fallback)
const typescript = require('rollup-plugin-typescript2');
const fs = require('fs');

// Manually parse package.json instead of ESM import
const pkg = JSON.parse(fs.readFileSync('./package.json', 'utf-8'));

module.exports = {
    input: 'src/index.ts',
    output: [
        {
            file: pkg.main,
            format: 'umd',
            name: 'astermind',
            sourcemap: true,
        },
        {
            file: pkg.module,
            format: 'esm',
            sourcemap: true,
        },
    ],
    plugins: [
        typescript({
            tsconfig: './tsconfig.json',
            useTsconfigDeclarationDir: true,
        }),
    ],
};
