/**
 * After `tsc`, Node ESM needs `.js` on relative specifiers. With `moduleResolution: "bundler"`
 * we keep extensionless imports in `.ts` source; this patches emitted `.js` and `.d.ts`.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const workspaceRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function* walkPatchableFiles(dir) {
  if (!fs.existsSync(dir)) return;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) yield* walkPatchableFiles(p);
    else if (ent.isFile() && (ent.name.endsWith('.js') || ent.name.endsWith('.d.ts'))) yield p;
  }
}

function collectDistRoots() {
  const roots = [];
  for (const top of ['packages', 'apps']) {
    const base = path.join(workspaceRoot, top);
    if (!fs.existsSync(base)) continue;
    for (const ent of fs.readdirSync(base, { withFileTypes: true })) {
      if (!ent.isDirectory()) continue;
      const dist = path.join(base, ent.name, 'dist');
      if (fs.existsSync(dist)) roots.push(dist);
    }
  }
  const rootDist = path.join(workspaceRoot, 'dist');
  if (fs.existsSync(rootDist)) roots.push(rootDist);
  return roots;
}

function patchRelativeSpecifiers(content) {
  const needsJs = (spec) =>
    (spec.startsWith('./') || spec.startsWith('../')) && !/\.[a-zA-Z0-9]+$/.test(spec);

  return content.replace(/\b(from|import)\s+(['"])(\.\.?\/[^'"]+)\2/g, (full, kw, q, spec) => {
    if (!needsJs(spec)) return full;
    return `${kw} ${q}${spec}.js${q}`;
  });
}

for (const root of collectDistRoots()) {
  for (const file of walkPatchableFiles(root)) {
    const before = fs.readFileSync(file, 'utf8');
    const after = patchRelativeSpecifiers(before);
    if (after !== before) fs.writeFileSync(file, after);
  }
}
