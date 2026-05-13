#!/usr/bin/env node
/**
 * The Nx non-bundling Node shim at dist/apps/cli/main.js lacks a Node shebang.
 * When npm links it as bin/llm, the shell executes it as bash and blows up on "use strict".
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

/** Nx bootstrap: CommonJS shim that patches Module._resolveFilename */
if (!body.includes('Module._resolveFilename')) {
  console.error(
    `cli-bootstrap-shebang: expected Nx resolver shim at ${resolved}, refusing to alter.`,
  );
  process.exit(1);
}

fs.writeFileSync(resolved, SHEBANG + body, 'utf8');
chmodScript();
