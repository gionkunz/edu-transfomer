/// <reference types="vitest" />
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vitest/config';

import { createLibVitestConfig } from '../../vitest.config.base';

export default defineConfig(
  createLibVitestConfig(path.dirname(fileURLToPath(import.meta.url)), '@llm-transformer-diy/cli'),
);
