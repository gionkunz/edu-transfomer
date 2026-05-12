#!/usr/bin/env node
import { runCli } from './cli-program.js';

void runCli(process.argv).catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
