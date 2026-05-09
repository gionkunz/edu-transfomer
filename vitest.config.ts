/// <reference types="vitest" />
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vitest/config';

const workspaceRoot = path.dirname(fileURLToPath(import.meta.url));

/**
 * Workspace root: `vitest` / `vitest run` discovers each package/app
 * `vitest.config.ts` via `test.projects` (Vitest 3).
 *
 * Prefer Nx for CI: `nx run-many -t test --all` or `nx run <proj>:test`.
 */
export default defineConfig({
  root: workspaceRoot,
  test: {
    projects: ['packages/*', 'apps/*'],
  },
});
