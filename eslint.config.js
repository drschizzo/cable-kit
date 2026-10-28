// Static analysis: likely mistakes only (unknown or unused variables, dead code…), not style.
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import globals from 'globals';

export default tseslint.config(
  { ignores: ['**/dist/', 'node_modules/'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    languageOptions: { ecmaVersion: 2024, sourceType: 'module' },
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { args: 'none', caughtErrors: 'none', ignoreRestSiblings: true }],
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-non-null-assertion': 'off',
    },
  },
  { files: ['packages/**/*.ts', 'examples/**/*.ts'], languageOptions: { globals: globals.browser } },
  // the solver has no dependency; a package imports nothing outside its folder
  { files: ['packages/**/*.ts'], rules: { 'no-restricted-imports': ['error', packageBounds()] } },
  {
    files: ['packages/avbd-cables/**/*.ts'],
    rules: { 'no-restricted-imports': ['error', { ...packageBounds(), paths: [{ name: 'three', message: 'avbd-cables has no dependency.' }] }] },
  },
  { files: ['tools/**/*.ts', 'tests/**/*.ts', '*.config.{js,ts}'], languageOptions: { globals: globals.node } },
);

function packageBounds() {
  return { patterns: [{ group: ['../../*'], message: 'A package imports nothing outside its folder.' }] };
}
