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
      'pnpm-lock.yaml',
      'eslint.config.js',
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
