import eslint from '@eslint/js';
import eslintConfigPrettier from 'eslint-config-prettier';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: [
      'dist/**',
      '**/dist/**',
      'node_modules/**',
      '.nx/**',
      'coverage/**',
      'tmp/**',
      'package-lock.json',
      'eslint.config.mjs',
    ],
  },
  eslint.configs.recommended,
  {
    files: [
      'vitest.config.ts',
      'vitest.config.base.ts',
      'apps/**/vitest.config.ts',
      'packages/**/vitest.config.ts',
      'apps/**/src/**/*.ts',
      'apps/**/src/**/*.tsx',
      'packages/**/src/**/*.ts',
    ],
    extends: [...tseslint.configs.strictTypeChecked],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      '@typescript-eslint/consistent-type-imports': [
        'error',
        {
          disallowTypeAnnotations: false,
          fixStyle: 'separate-type-imports',
        },
      ],
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
        },
      ],
      // strictTypeChecked is valuable, but these rules misfire on defensive runtime checks
      // (JSON manifests, optional CLI flags) and loosely typed UI event payloads (wolf-tui).
      '@typescript-eslint/no-unnecessary-condition': 'off',
      '@typescript-eslint/no-unnecessary-boolean-literal-compare': 'off',
      '@typescript-eslint/no-unnecessary-type-conversion': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/restrict-template-expressions': ['error', { allowNumber: true }],
    },
  },
  {
    files: ['apps/**/vitest.config.ts', 'packages/**/vitest.config.ts'],
    rules: {
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-call': 'off',
    },
  },
  eslintConfigPrettier,
);
