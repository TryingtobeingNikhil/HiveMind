import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // CRITICAL: all /api requests are forwarded to FastAPI backend.
      // Without this proxy, the browser enforces CORS even if FastAPI
      // has CORS headers set, because the origin differs.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
