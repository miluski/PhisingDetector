const tseslint = require('typescript-eslint');
const angular = require('@angular-eslint/eslint-plugin');
const templateParser = require('@angular-eslint/template-parser');
const angularTemplate = require('@angular-eslint/eslint-plugin-template');

module.exports = tseslint.config(
  {
    files: ['src/**/*.ts'],
    plugins: { '@angular-eslint': angular },
    extends: [...tseslint.configs.recommended],
    rules: {
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': 'error',
      '@typescript-eslint/explicit-function-return-type': 'off',
    },
  },
  {
    files: ['src/**/*.html'],
    languageOptions: { parser: templateParser },
    plugins: { '@angular-eslint/template': angularTemplate },
    rules: angularTemplate.configs.recommended.rules,
  },
);
