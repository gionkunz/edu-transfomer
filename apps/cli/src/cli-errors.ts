/** Prints **`err`** to stderr and exits **`1`** (stack only when **`verbose`**). */
export function exitCliError(err: unknown, verbose: boolean): never {
  const msg =
    err instanceof Error
      ? verbose
        ? `${err.stack ?? err.message}\n`
        : `${err.message}\n`
      : `${String(err)}\n`;
  console.error(msg);
  process.exit(1);
}
