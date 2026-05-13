#!/usr/bin/env node
import { runCli } from './cli-program';

void runCli(process.argv).catch((e: unknown) => {
  console.error(e);
  process.exit(1);
});
