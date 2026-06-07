import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const environment = loadEnv(mode, '.', 'AEQA_');
  const target = environment.AEQA_API_PROXY_TARGET || 'http://127.0.0.1:58000';
  if (!/^https?:\/\/(?:127\.0\.0\.1|localhost|\[::1\])(?::\d{1,5})?\/?$/.test(target)) {
    throw new Error('AEQA_API_PROXY_TARGET 必须为 loopback HTTP API。');
  }
  const proxy = { '/v1': { target, changeOrigin: false } };
  return {
    plugins: [react()],
    server: { host: '127.0.0.1', port: 5173, proxy },
    preview: { host: '127.0.0.1', port: 4173, proxy },
  };
});
