import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig(({ command }) => {
    const demoMatch = process.env.DEMO;
    const isDemo = command === 'serve' && !!demoMatch;
    const isVitest = !!process.env.VITEST;

    return {
        root: isDemo ? `examples/${demoMatch}` : '.',
        publicDir: path.resolve(__dirname, 'public'),
        server: {
            port: 5173,
        },
        test: {
            include: ['tests/**/*.test.ts'],
            environment: isVitest ? 'jsdom' : 'node',
        },
    };
});
