#!/usr/bin/env node
/**
 * Make the bundled ESM entry at dist/apps/cli/main.js runnable as bin/llm.
 * esbuild keeps the `#!/usr/bin/env node` line of `apps/cli/src/main.ts`; this adds one
 * only if it is missing (without it the shell would run the file as a shell script).
 *
 * npm link also requires the bin target to be executable; esbuild output is often mode 0644.
 *
 * Prefer idempotent patching: preserve file contents if already has a shebang; always chmod +x.
 */
import fs from 'node:fs';

const SHEBANG = '#!/usr/bin/env node\n';
const MODE_SCRIPT = 0o755;

const [, , rel = 'dist/apps/cli/main.js'] = process.argv;

const resolved = fs.realpathSync(rel);

function chmodScript() {
  fs.chmodSync(resolved, MODE_SCRIPT);
}

let body = fs.readFileSync(resolved, 'utf8');

if (body.startsWith('#!')) {
  chmodScript();
  process.exit(0);
}

fs.writeFileSync(resolved, SHEBANG + body, 'utf8');
chmodScript();
