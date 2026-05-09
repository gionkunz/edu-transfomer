/// <reference types="vitest" />
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vitest/config';

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    conditions: ['import', 'node', 'default'],
  },
  test: {
    environment: 'node',
    include: ['apps/**/src/**/*.spec.ts', 'packages/**/src/**/*.spec.ts'],
  },
  root,
});
