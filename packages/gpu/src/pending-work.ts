/**
 * Registry of GPU contexts that currently hold **recorded but unsubmitted**
 * compute work (see `GpuContext.encodeComputeDispatch`).
 *
 * It exists for code that must destroy a `GPUBuffer` without having a context
 * at hand (the buffer pool, foreign-buffer release). Destroying a buffer that a
 * not-yet-submitted command buffer still references makes WebGPU reject the
 * whole submit, silently dropping every dispatch in it — so those call sites
 * call {@link flushAllPendingGpuWork} first. Kept in its own module so
 * `buffer-pool.ts` does not import `device.ts` (which imports the pool).
 */
export interface PendingGpuWork {
  flush(): void;
}

const withPendingWork = new Set<PendingGpuWork>();

export function notePendingGpuWork(ctx: PendingGpuWork): void {
  withPendingWork.add(ctx);
}

export function clearPendingGpuWork(ctx: PendingGpuWork): void {
  withPendingWork.delete(ctx);
}

/** Submit every context's recorded dispatches. Cheap no-op when nothing is pending. */
export function flushAllPendingGpuWork(): void {
  if (withPendingWork.size === 0) return;
  for (const ctx of [...withPendingWork]) {
    ctx.flush();
  }
}
