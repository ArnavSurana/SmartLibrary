import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // note the trailing slash: this makes Vite forward /books/<…> → http://127.0.0.1:8000/books/<…>
      '/books/': {
        target:      'http://127.0.0.1:8000',
        changeOrigin: true,
        secure:      false,
      },
      '/questions/': {
        target:      'http://127.0.0.1:8000',
        changeOrigin: true,
        secure:      false,
      },
    },
  },
});
