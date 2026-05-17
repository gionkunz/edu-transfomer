import React from 'react';
import { Box, Text, render, useInput, useStdout } from '@wolf-tui/react';

import { createGptKvCache, destroyGptKvCache } from '@llm-transformer-diy/model';
import { generateAutoregressive } from '@llm-transformer-diy/inference';
import { loadGptModelFromCheckpoint } from '@llm-transformer-diy/training';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';
import type { FloatDType } from '@llm-transformer-diy/tensor';

import { resolvePackedCheckpointPaths } from '../paths';
import { assertRuntimeSupportsComputeDtype } from '../compute-dtype';
import { createTensorRuntime } from '../runtime';
import { primeTuiTerminalCanvas } from '../tui-prime';
import { PacingController } from './generate-pacing';

type GenViewProps = {
  readonly text: string;
  readonly newTokens: number;
  readonly tokensPerSec: number;
};

function GenView(props: GenViewProps): React.ReactElement {
  useInput((input, key) => {
    if (input === 'q') {
      process.exit(0);
    }
    if (key.ctrl && input === 'c') {
      process.exit(0);
    }
  });

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', padding: 1 }}>
      <Text style={{ fontWeight: 'bold' }}>Generate</Text>
      <Text
        style={{ color: 'gray' }}
      >{`new tokens ${String(props.newTokens)} · tok/s ${props.tokensPerSec.toFixed(1)} — q to quit`}</Text>
      <Text>{props.text}</Text>
    </Box>
  );
}

const INSPECTOR_PANEL_WIDTH = 36;
const INSPECTOR_MIN_COLUMNS = 60;
const INSPECTOR_MIN_ROWS = 8;
const INSPECTOR_CHROME_ROWS = 8;

const BAR_CHARS = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'] as const;

function barChar(p: number, scale: number): string {
  if (!(scale > 0)) return ' ';
  const ratio = Math.max(0, Math.min(1, p / scale));
  const idx = Math.min(BAR_CHARS.length - 1, Math.max(0, Math.floor(ratio * 8)));
  return BAR_CHARS[idx];
}

function fmtProb(p: number): string {
  if (p >= 0.0001 || p === 0) return p.toFixed(4);
  return p.toExponential(1);
}

/** Show whitespace / control runs visibly so the inspector is legible. */
function escapeForRow(s: string): string {
  return sanitizeControlChars(s)
    .replaceAll('\n', '↵')
    .replaceAll('\r', '␍')
    .replaceAll('\t', '→')
    .replaceAll(' ', '␣');
}

/**
 * Strip / replace control characters that would otherwise corrupt the TUI:
 * BEL (`\x07`) rings the terminal bell, ESC (`\x1b`) starts an ANSI escape
 * sequence and can move the cursor or repaint colors, and other C0/C1
 * control bytes (`\x00`–`\x1f` minus tab/newline/carriage-return, plus
 * `\x7f` and `\x80`–`\x9f`) cause similar layout damage. Streaming a model
 * that occasionally samples raw bytes through a byte-level BPE makes this a
 * routine occurrence, especially at low temperature where one bad logit can
 * stick.
 *
 * Visible whitespace (`\n`, `\r`, `\t`) is preserved so paragraphs and
 * indentation still render; everything else is replaced with U+FFFD so the
 * user sees that *something* was sampled without the terminal misbehaving.
 */
export function sanitizeControlChars(s: string): string {
  let out = '';
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    const isAllowedWs = code === 0x09 || code === 0x0a || code === 0x0d;
    const isC0 = code <= 0x1f;
    const isDel = code === 0x7f;
    const isC1 = code >= 0x80 && code <= 0x9f;
    if (!isAllowedWs && (isC0 || isDel || isC1)) {
      out += '\uFFFD';
    } else {
      out += s[i];
    }
  }
  return out;
}

function truncatePiece(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, Math.max(0, maxLen - 1)) + '…';
}

type InspectorRow = {
  readonly id: number;
  readonly probability: number;
  readonly piece: string;
};

type ExploreVm = {
  readonly promptText: string;
  readonly prefixText: string;
  readonly lastPiece: string;
  readonly newTokens: number;
  readonly tokensPerSec: number;
  readonly probabilities: Float32Array | null;
  readonly sortedIds: Int32Array | null;
  readonly maxProb: number;
  readonly displayFor: (id: number) => string;
  readonly sampledFooter: {
    readonly id: number;
    readonly probability: number;
    readonly rank: number;
  } | null;
  readonly modeLabel: 'NORMAL' | 'SLOW' | 'PAUSED';
};

type ExploreViewProps = {
  readonly vm: ExploreVm;
  readonly onToggleSpeed: () => void;
  readonly onTogglePause: () => void;
  readonly onQuit: () => void;
};

/** Slice prefix/lastPiece pair to keep at most **`maxChars`** trailing chars while preserving the highlight on the tail. */
export function tailSliceForHighlight(
  prefix: string,
  last: string,
  maxChars: number,
): { readonly prefix: string; readonly last: string } {
  if (maxChars <= 0) return { prefix: '', last: '' };
  const total = prefix.length + last.length;
  if (total <= maxChars) return { prefix, last };
  if (last.length >= maxChars) {
    return { prefix: '', last: last.slice(last.length - maxChars) };
  }
  const prefixKeep = maxChars - last.length;
  return { prefix: prefix.slice(prefix.length - prefixKeep), last };
}

function ExploreView(props: ExploreViewProps): React.ReactElement {
  const { vm } = props;
  const stdoutCtx = useStdout();
  const cols = stdoutCtx?.stdout.columns ?? 80;
  const rows = stdoutCtx?.stdout.rows ?? 24;

  const showInspector = cols >= INSPECTOR_MIN_COLUMNS && rows >= INSPECTOR_MIN_ROWS;
  const visibleRows = Math.max(0, rows - INSPECTOR_CHROME_ROWS);

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      props.onQuit();
      return;
    }
    if (input === 'q') {
      props.onQuit();
      return;
    }
    if (input === 's') {
      props.onToggleSpeed();
      return;
    }
    if (input === 'p') {
      props.onTogglePause();
      return;
    }
  });

  const inspectorRows: InspectorRow[] = (() => {
    if (!showInspector) return [];
    if (vm.sortedIds === null || vm.probabilities === null) return [];
    const out: InspectorRow[] = [];
    const limit = Math.min(visibleRows, vm.sortedIds.length);
    for (let i = 0; i < limit; i++) {
      const id = vm.sortedIds[i];
      out.push({ id, probability: vm.probabilities[id], piece: vm.displayFor(id) });
    }
    return out;
  })();

  const statusColor: string =
    vm.modeLabel === 'PAUSED' ? 'yellow' : vm.modeLabel === 'SLOW' ? 'magenta' : 'green';

  /**
   * Compute an explicit width for the streamed-text column so wolf-tui wraps to it instead of
   * letting long generations spill past the right-side inspector and break the row layout.
   * Buffer of `6` covers root padding (2) + left paddingRight (2) + inspector border/padding slop (2).
   */
  const inspectorBudget = showInspector ? INSPECTOR_PANEL_WIDTH + 4 : 0;
  const leftPaddingRight = showInspector ? 2 : 0;
  const leftWidth = Math.max(20, cols - inspectorBudget - leftPaddingRight - 2);
  const bodyHeight = Math.max(3, visibleRows + 1);
  /** Cap visible streamed text from the tail so vertical overflow can't push the inspector down. */
  const tailCharBudget = leftWidth * (bodyHeight + 4);
  const { prefix: prefixDisplay, last: lastDisplay } = tailSliceForHighlight(
    vm.prefixText,
    vm.lastPiece,
    tailCharBudget,
  );

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', padding: 1, flexGrow: 1 }}>
      <Text style={{ fontWeight: 'bold' }}>Generate · explore</Text>
      <Box style={{ display: 'flex', flexDirection: 'row' }}>
        <Text style={{ color: statusColor, fontWeight: 'bold' }}>{`mode: ${vm.modeLabel}`}</Text>
        <Text style={{ color: 'gray' }}>
          {` · tok/s ${vm.tokensPerSec.toFixed(1)} · new ${String(vm.newTokens)} · s = slow · p = pause · q = quit`}
        </Text>
      </Box>
      <Box
        style={{
          display: 'flex',
          flexDirection: 'row',
          height: bodyHeight,
          marginTop: 1,
          overflowX: 'hidden',
          overflowY: 'hidden',
        }}
      >
        <Box
          style={{
            display: 'flex',
            flexDirection: 'column',
            width: leftWidth,
            paddingRight: leftPaddingRight,
            overflowX: 'hidden',
            overflowY: 'hidden',
          }}
        >
          <Text style={{ textWrap: 'wrap' }}>
            <Text style={{ color: 'gray' }}>{vm.promptText}</Text>
            {prefixDisplay}
            <Text style={{ color: 'cyan', fontWeight: 'bold' }}>{lastDisplay}</Text>
          </Text>
        </Box>
        {showInspector ? (
          <Box
            style={{
              display: 'flex',
              flexDirection: 'column',
              width: INSPECTOR_PANEL_WIDTH,
              borderStyle: 'single',
              borderColor: 'gray',
              paddingLeft: 1,
              paddingRight: 1,
              overflowX: 'hidden',
              overflowY: 'hidden',
            }}
          >
            <Text style={{ fontWeight: 'bold' }}>next-token p</Text>
            {inspectorRows.length === 0 ? (
              <Text style={{ color: 'gray' }}>(waiting for first sample…)</Text>
            ) : (
              inspectorRows.map((r) => {
                const isSampled = vm.sampledFooter !== null && vm.sampledFooter.id === r.id;
                const idStr = String(r.id).padStart(5, ' ');
                const bar = barChar(r.probability, vm.maxProb);
                const piece = truncatePiece(escapeForRow(r.piece), 14).padEnd(14, ' ');
                const prob = fmtProb(r.probability).padStart(6, ' ');
                const line = `${idStr} ${bar} ${piece} ${prob}`;
                return (
                  <Text
                    key={r.id}
                    style={
                      isSampled
                        ? { color: 'cyan', fontWeight: 'bold', textWrap: 'truncate-end' }
                        : { textWrap: 'truncate-end' }
                    }
                  >
                    {line}
                  </Text>
                );
              })
            )}
          </Box>
        ) : null}
      </Box>
      {vm.sampledFooter !== null ? (
        <Text style={{ color: 'gray' }}>
          {`sampled: id=${String(vm.sampledFooter.id)} p=${fmtProb(vm.sampledFooter.probability)} rank=${String(
            vm.sampledFooter.rank + 1,
          )}`}
        </Text>
      ) : (
        <Text style={{ color: 'gray' }}>{'(no token sampled yet)'}</Text>
      )}
    </Box>
  );
}

export async function runGenerateCommand(opts: {
  readonly modelArg: string;
  readonly tokenizerPath: string;
  readonly prompt: string;
  readonly maxNewTokens: number;
  readonly temperature: number;
  readonly topK: number;
  /** When **`true`** (default in CLI) renders the educational explore view; when **`false`** uses the minimal **`GenView`**. */
  readonly explore: boolean;
  readonly repetitionPenalty?: number;
  readonly repetitionWindow?: number;
  readonly computeDtype?: FloatDType;
  readonly allowDtypeUpcast?: boolean;
}): Promise<void> {
  const rt = await createTensorRuntime();
  const dtype = opts.computeDtype ?? 'f32';
  assertRuntimeSupportsComputeDtype(rt.ctx, dtype, 'generate');
  if (rt.ctx.isLikelyCpuSoftwareAdapter) {
    console.error(
      'generate: WebGPU is using a software adapter (e.g. LLVMpipe). Use a discrete GPU runtime for speed; see train TUI note on Vulkan/ICD.',
    );
  }
  const { binPath, jsonPath } = resolvePackedCheckpointPaths(opts.modelArg);
  const { model, config } = await loadGptModelFromCheckpoint(rt, binPath, jsonPath, {
    computeDtype: dtype,
    ...(opts.allowDtypeUpcast === true ? { allowDtypeUpcast: true } : {}),
  });

  const tok = new BPETokenizer();
  await tok.load(opts.tokenizerPath);
  if (tok.vocabSize !== config.vocabSize) {
    throw new Error(
      `generate: tokenizer vocab ${String(tok.vocabSize)} ≠ checkpoint vocab ${String(config.vocabSize)}`,
    );
  }

  /** BPE merges are tracked in **bytes**; streaming must not call **`decode`** on single ids (often ill‑formed UTF‑8 fragments). */
  const promptIds = tok.encode(opts.prompt);
  const promptRendered = sanitizeControlChars(tok.decode(promptIds, { utf8: 'replace' }));
  const cache = createGptKvCache(rt, model);

  const generatedIds: number[] = [];
  let prevFullDecoded = promptRendered;
  let newTok = 0;
  const tStart = performance.now();

  primeTuiTerminalCanvas();

  if (opts.explore !== true) {
    let textOut = promptRendered;
    const inst = render(<GenView text={textOut} newTokens={0} tokensPerSec={0} />, {
      exitOnCtrlC: true,
      patchConsole: true,
    });
    try {
      await generateAutoregressive(rt, model, Uint32Array.from(promptIds), cache, {
        maxNewTokens: opts.maxNewTokens,
        sample: { temperature: opts.temperature, topK: opts.topK },
        ...(opts.repetitionPenalty !== undefined
          ? { repetitionPenalty: opts.repetitionPenalty }
          : {}),
        ...(opts.repetitionWindow !== undefined ? { repetitionWindow: opts.repetitionWindow } : {}),
        onSampledToken: (id: number) => {
          generatedIds.push(id);
          const fullDecoded = sanitizeControlChars(
            tok.decode([...promptIds, ...generatedIds], { utf8: 'replace' }),
          );
          textOut = fullDecoded;
          newTok += 1;
          const dt = (performance.now() - tStart) / 1000;
          inst.rerender(
            <GenView
              text={textOut}
              newTokens={newTok}
              tokensPerSec={newTok / Math.max(dt, 1e-6)}
            />,
          );
        },
      });
    } finally {
      destroyGptKvCache(cache);
      inst.unmount();
    }
    return;
  }

  const pacing = new PacingController();
  const decodeCache = new Map<number, string>();
  const displayFor = (id: number): string => {
    const cached = decodeCache.get(id);
    if (cached !== undefined) return cached;
    const piece = tok.decode([id], { utf8: 'replace' });
    const onlyReplacement = /^\uFFFD+$/.test(piece);
    const display = piece.length === 0 || onlyReplacement ? `‹${String(id)}›` : piece;
    decodeCache.set(id, display);
    return display;
  };

  const initialVm: ExploreVm = {
    promptText: promptRendered,
    prefixText: '',
    lastPiece: '',
    newTokens: 0,
    tokensPerSec: 0,
    probabilities: null,
    sortedIds: null,
    maxProb: 0,
    displayFor,
    sampledFooter: null,
    modeLabel: pacing.statusLabel,
  };
  let currentVm: ExploreVm = initialVm;

  let exiting = false;
  const requestExit = (): void => {
    if (exiting) return;
    exiting = true;
    pacing.cancel();
  };

  function paint(): void {
    inst.rerender(
      <ExploreView
        vm={currentVm}
        onToggleSpeed={() => {
          pacing.toggleSpeed();
          currentVm = { ...currentVm, modeLabel: pacing.statusLabel };
          paint();
        }}
        onTogglePause={() => {
          pacing.togglePause();
          currentVm = { ...currentVm, modeLabel: pacing.statusLabel };
          paint();
        }}
        onQuit={requestExit}
      />,
    );
  }

  const inst = render(
    <ExploreView
      vm={initialVm}
      onToggleSpeed={() => {
        pacing.toggleSpeed();
        currentVm = { ...currentVm, modeLabel: pacing.statusLabel };
        paint();
      }}
      onTogglePause={() => {
        pacing.togglePause();
        currentVm = { ...currentVm, modeLabel: pacing.statusLabel };
        paint();
      }}
      onQuit={requestExit}
    />,
    { exitOnCtrlC: false, patchConsole: true },
  );

  try {
    await generateAutoregressive(rt, model, Uint32Array.from(promptIds), cache, {
      maxNewTokens: opts.maxNewTokens,
      sample: { temperature: opts.temperature, topK: opts.topK },
      ...(opts.repetitionPenalty !== undefined
        ? { repetitionPenalty: opts.repetitionPenalty }
        : {}),
      ...(opts.repetitionWindow !== undefined ? { repetitionWindow: opts.repetitionWindow } : {}),
      onStep: async ({ tokenId, newTokenIndex, probabilities }) => {
        if (exiting) return;
        generatedIds.push(tokenId);
        const fullDecoded = sanitizeControlChars(
          tok.decode([...promptIds, ...generatedIds], { utf8: 'replace' }),
        );
        const visiblePrev = prevFullDecoded.slice(promptRendered.length);
        const visibleFull = fullDecoded.slice(promptRendered.length);
        const lastPiece = visibleFull.slice(visiblePrev.length);
        const prefixText = visibleFull.slice(0, visibleFull.length - lastPiece.length);
        prevFullDecoded = fullDecoded;
        newTok = newTokenIndex + 1;

        const V = probabilities.length;
        const sortedIds = new Int32Array(V);
        for (let i = 0; i < V; i++) sortedIds[i] = i;
        const probsArr = probabilities;
        sortedIds.sort((a, b) => probsArr[b] - probsArr[a]);
        const maxProb = V > 0 ? probabilities[sortedIds[0]] : 0;
        let rank = 0;
        for (let i = 0; i < sortedIds.length; i++) {
          if (sortedIds[i] === tokenId) {
            rank = i;
            break;
          }
        }

        const dt = (performance.now() - tStart) / 1000;
        currentVm = {
          promptText: promptRendered,
          prefixText,
          lastPiece,
          newTokens: newTok,
          tokensPerSec: newTok / Math.max(dt, 1e-6),
          probabilities,
          sortedIds,
          maxProb,
          displayFor,
          sampledFooter: { id: tokenId, probability: probabilities[tokenId], rank },
          modeLabel: pacing.statusLabel,
        };
        paint();

        await pacing.waitBetweenSteps();
        if (exiting) return;
      },
    });
  } finally {
    pacing.cancel();
    destroyGptKvCache(cache);
    inst.unmount();
    if (exiting) {
      process.exit(0);
    }
  }
}
