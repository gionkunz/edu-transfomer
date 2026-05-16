/**
 * Doubly-linked list of token slots backing BPE training.
 *
 * The corpus is materialized once at construction time as a fixed-capacity slot pool
 * (one slot per initial UTF-8 byte, including any synthetic inter-file separators).
 * Each merge consumes the right slot of an `(L, R)` pair into the left slot in `O(1)`
 * by patching the four `prev` / `next` neighbor pointers — no slot is ever
 * reallocated, so the slot pool monotonically shrinks (`liveCount` decreases).
 *
 * Storage is **parallel typed arrays** (`Uint32Array` for ids and link pointers,
 * `Uint8Array` for the alive flag) so the hot path stays unboxed and cache-dense
 * — matches the data layout SentencePiece uses in C++ and `tiktoken` uses in Rust.
 */

/** End-of-list sentinel for `prev` / `next` pointers (slot indices are `0 .. capacity-1`). */
export const END = 0xff_ff_ff_ff;

/**
 * Maximum permitted slot capacity. We reserve `END = 0xFFFFFFFF` as the no-neighbor
 * sentinel, so the largest legal slot index is `0xFFFFFFFE` and capacity tops out at
 * `0xFFFFFFFE`. A 4 GiB-byte corpus already exceeds the project's reasonable single-corpus
 * ceiling, so this cap is plenty.
 */
export const MAX_SLOT_CAPACITY = 0xff_ff_ff_fe;

/**
 * Doubly-linked list of token slots. All four backing arrays are sized to `capacity`
 * at construction; mutation never reallocates.
 */
export interface BpeLinkedTokenStream {
  /** Token id at slot `i`. Only meaningful when `alive[i] === 1`. */
  readonly value: Uint32Array;
  /** Previous live slot index, or `END` when this is the head. */
  readonly prev: Uint32Array;
  /** Next live slot index, or `END` when this is the tail. */
  readonly next: Uint32Array;
  /** `1` while slot `i` carries a live token; `0` once a merge has consumed it. */
  readonly alive: Uint8Array;
  /** Index of the leftmost live slot, or `END` when `liveCount === 0`. */
  head: number;
  /** Index of the rightmost live slot, or `END` when `liveCount === 0`. Used to amortize append. */
  tail: number;
  /** Number of slots with `alive[i] === 1`; equals the logical token-stream length. */
  liveCount: number;
  /** Total slot capacity (length of `value` / `prev` / `next` / `alive`). */
  readonly capacity: number;
}

/**
 * Allocate a fresh slot pool of the given capacity. All slots start with `alive[i] === 0`
 * and `prev[i] === next[i] === END`; populate via {@link appendLiveSlot} during materialization.
 */
export function createLinkedTokenStream(capacity: number): BpeLinkedTokenStream {
  if (!Number.isInteger(capacity) || capacity < 0) {
    throw new Error(
      `BpeLinkedTokenStream: capacity must be a non-negative integer, got ${String(capacity)}`,
    );
  }
  if (capacity > MAX_SLOT_CAPACITY) {
    throw new Error(
      `BpeLinkedTokenStream: capacity ${String(capacity)} exceeds MAX_SLOT_CAPACITY ${String(MAX_SLOT_CAPACITY)} — split the corpus or use a chunked variant`,
    );
  }
  const value = new Uint32Array(capacity);
  const prev = new Uint32Array(capacity);
  const next = new Uint32Array(capacity);
  prev.fill(END);
  next.fill(END);
  return {
    value,
    prev,
    next,
    alive: new Uint8Array(capacity),
    head: END,
    tail: END,
    liveCount: 0,
    capacity,
  };
}

/**
 * Wire `slotIndex` onto the tail of the live list with token id `tokenId`.
 *
 * The slot must be unused (`alive[slotIndex] === 0`); during materialization callers
 * append slots in increasing index order so each call is `O(1)` amortized over the
 * whole corpus walk.
 */
export function appendLiveSlot(
  stream: BpeLinkedTokenStream,
  slotIndex: number,
  tokenId: number,
): void {
  if (slotIndex < 0 || slotIndex >= stream.capacity) {
    throw new Error(
      `BpeLinkedTokenStream.appendLiveSlot: slotIndex ${String(slotIndex)} out of range [0, ${String(stream.capacity)})`,
    );
  }
  if (stream.alive[slotIndex] === 1) {
    throw new Error(`BpeLinkedTokenStream.appendLiveSlot: slot ${String(slotIndex)} already alive`);
  }
  stream.value[slotIndex] = tokenId >>> 0;
  stream.alive[slotIndex] = 1;
  if (stream.head === END) {
    stream.head = slotIndex;
    stream.prev[slotIndex] = END;
  } else {
    stream.prev[slotIndex] = stream.tail;
    stream.next[stream.tail] = slotIndex;
  }
  stream.next[slotIndex] = END;
  stream.tail = slotIndex;
  stream.liveCount += 1;
}

/**
 * Merge the live slot at `leftSlot` with its successor `next[leftSlot]` into a single
 * slot carrying `mergedTokenId`. The right slot becomes dead (`alive = 0`); the left
 * slot's `value` is replaced with `mergedTokenId` and its `next` pointer is patched
 * to skip past the consumed slot.
 *
 * Pre-condition: `alive[leftSlot] === 1` and `alive[next[leftSlot]] === 1` (i.e. the
 * pair `(value[leftSlot], value[next[leftSlot]])` is a live adjacency). Returns the
 * slot index that was consumed (the former `next[leftSlot]`), so callers can use it
 * to look up neighboring pair entries before they're disturbed.
 */
export function unitMergeSlot(
  stream: BpeLinkedTokenStream,
  leftSlot: number,
  mergedTokenId: number,
): number {
  if (leftSlot < 0 || leftSlot >= stream.capacity) {
    throw new Error(
      `BpeLinkedTokenStream.unitMergeSlot: leftSlot ${String(leftSlot)} out of range [0, ${String(stream.capacity)})`,
    );
  }
  if (stream.alive[leftSlot] === 0) {
    throw new Error(
      `BpeLinkedTokenStream.unitMergeSlot: leftSlot ${String(leftSlot)} is not alive`,
    );
  }
  const rightSlot = stream.next[leftSlot]!;
  if (rightSlot === END) {
    throw new Error(
      `BpeLinkedTokenStream.unitMergeSlot: leftSlot ${String(leftSlot)} has no live successor`,
    );
  }
  if (stream.alive[rightSlot] === 0) {
    throw new Error(
      `BpeLinkedTokenStream.unitMergeSlot: successor slot ${String(rightSlot)} is not alive`,
    );
  }
  const afterRight = stream.next[rightSlot]!;
  stream.value[leftSlot] = mergedTokenId >>> 0;
  stream.next[leftSlot] = afterRight;
  if (afterRight !== END) {
    stream.prev[afterRight] = leftSlot;
  } else {
    // The pair sat at the tail; the merged slot is now the tail.
    stream.tail = leftSlot;
  }
  stream.alive[rightSlot] = 0;
  stream.liveCount -= 1;
  return rightSlot;
}

/**
 * Walk the live list head → tail once and return a freshly-allocated `Uint32Array`
 * sized to `liveCount` containing the live token ids in order. Allocation-light path
 * for tests and for any caller that wants a flat view of the current stream state.
 */
export function extractLiveTokenIds(stream: BpeLinkedTokenStream): Uint32Array {
  const out = new Uint32Array(stream.liveCount);
  let cur = stream.head;
  let i = 0;
  while (cur !== END) {
    out[i] = stream.value[cur]!;
    i += 1;
    cur = stream.next[cur]!;
  }
  if (i !== stream.liveCount) {
    throw new Error(
      `BpeLinkedTokenStream.extractLiveTokenIds: walked ${String(i)} slots but liveCount is ${String(stream.liveCount)} — corrupted list`,
    );
  }
  return out;
}
