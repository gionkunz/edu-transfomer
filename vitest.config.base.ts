import type { UserConfig } from 'vite';

/** Shared defaults for Node libraries/apps — one `vitest.config.ts` per Nx project. */
export function createLibVitestConfig(projectRootDir: string, testProjectName: string): UserConfig {
  return {
    root: projectRootDir,
    resolve: {
      conditions: ['import', 'node', 'default'],
    },
    test: {
      name: testProjectName,
      environment: 'node',
      include: ['src/*.spec.ts', 'src/**/*.spec.ts'],
      passWithNoTests: true,
      // Nx's @nx/vitest executor starts with an empty reporter list and only adds NxReporter;
      // without a built-in reporter, the console shows almost nothing from Vitest.
      reporters: ['default'],
    },
  };
}
