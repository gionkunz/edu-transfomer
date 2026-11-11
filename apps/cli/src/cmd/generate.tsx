import React from 'react';
import { Box, Text, Transform, render, useInput, useStdout } from '@wolf-tui/react';

import {
  createGptKvCache,
  destroyGptKvCache,
  disposeGptKvForwardGraph,
  forwardGptDecodeStep,
} from '@llm-transformer-diy/model';
import {
  bestRankByKeyIndex,
  generateAutoregressive,
  selectTop10GlobalConnections,
  buildFlowTrace,
  summarizeHeads,
  unstackAttentionWeights,
  type GlobalAttentionConnection,
  type FlowTrace,
  type HeadAttentionSummary,
} from '@llm-transformer-diy/inference';
import { loadGptModelFromCheckpoint } from '@llm-transformer-diy/training';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';
import type { FloatDType } from '@llm-transformer-diy/tensor';

import { resolvePackedCheckpointPaths } from '../paths';
import { assertRuntimeSupportsComputeDtype } from '../compute-dtype';
import { createTensorRuntime } from '../runtime';
import { primeTuiTerminalCanvas } from '../tui-prime';
import { PacingController } from './generate-pacing';
import {
  buildContextTokenSpans,
  tailSliceTokenSpans,
  type ContextTokenSpan,
} from './attention-context-spans';
import { rankHighlightTransform } from './attention-rank-palette';
import { exploreChromeRows, panelTier } from './generate-layout-tiers';
import { trailingMarkerTokens } from './generate-stop';
import {
  AttentionPanel,
  FlowPanel,
  flowPanelContentRows,
  type AttentionPanelData,
  type FlowPanelData,
  type FlowVariant,
} from './generate-panels';

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

/** Attention of one decode forward at one block, summarized for display. */
type AttentionView = {
  readonly layerIndex: number;
  readonly rows: readonly Float32Array[];
  readonly heads: readonly HeadAttentionSummary[];
  readonly connections: readonly GlobalAttentionConnection[];
};

function attentionViewFromWeights(
  layerIndex: number,
  weights: Float32Array,
  nHeads: number,
): AttentionView {
  const rows = unstackAttentionWeights(weights, nHeads, Math.floor(weights.length / nHeads));
  return {
    layerIndex,
    rows,
    heads: summarizeHeads(rows),
    connections: selectTop10GlobalConnections(rows),
  };
}

type BottomPanelMode = 'attention' | 'flow';

type ExploreVm = {
  readonly promptTokenCount: number;
  readonly contextTokenSpans: readonly ContextTokenSpan[];
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
  readonly nLayers: number;
  readonly nHeads: number;
  readonly dModel: number;
  readonly vocabSize: number;
  readonly temperature: number;
  readonly topK: number;
  readonly focusedLayerIndex: number;
  readonly panelMode: BottomPanelMode;
  /** Attention behind the current token; **`null`** on the first one or when not captured. */
  readonly attention: AttentionView | null;
  /** Per-block trace behind the current token; captured only while the flow panel is open. */
  readonly flow: FlowTrace | null;
  /** Set when generation ended because the model wrote the stop marker. */
  readonly stoppedAtMarker: string | null;
  /** Trailing tokens kept for the panels but not drawn in the text (the stop marker). */
  readonly hiddenTailTokens: number;
};

type ExploreViewProps = {
  readonly vm: ExploreVm;
  readonly onToggleSpeed: () => void;
  readonly onTogglePause: () => void;
  readonly onLayerUp: () => void;
  readonly onLayerDown: () => void;
  readonly onTogglePanel: () => void;
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

/** Short visible label for a token piece: whitespace made visible, length capped. */
function pieceLabel(piece: string): string {
  return truncatePiece(escapeForRow(piece), 10);
}

function renderContextText(
  spans: readonly ContextTokenSpan[],
  promptTokenCount: number,
  rankByKey: ReadonlyMap<number, number>,
  sampledTokenIndex: number | null,
  attentionQueryTokenIndex: number | null,
  tailCharBudget: number,
): React.ReactNode {
  const visible = tailSliceTokenSpans(spans, tailCharBudget);
  return visible.map((span) => {
    const key = span.tokenIndex;
    if (sampledTokenIndex !== null && span.tokenIndex === sampledTokenIndex) {
      return (
        <Text key={key} style={{ color: 'cyan', fontWeight: 'bold' }}>
          {span.text}
        </Text>
      );
    }
    const rank = rankByKey.get(span.tokenIndex);
    const highlightTransform = rank !== undefined ? rankHighlightTransform(rank) : undefined;
    if (highlightTransform !== undefined) {
      return (
        <Transform key={key} transform={highlightTransform}>
          {span.text}
        </Transform>
      );
    }
    if (attentionQueryTokenIndex !== null && span.tokenIndex === attentionQueryTokenIndex) {
      return (
        <Text key={key} style={{ fontWeight: 'bold', textDecoration: 'underline' }}>
          {span.text}
        </Text>
      );
    }
    if (span.tokenIndex < promptTokenCount) {
      return (
        <Text key={key} style={{ color: 'gray' }}>
          {span.text}
        </Text>
      );
    }
    return <Text key={key}>{span.text}</Text>;
  });
}

function ExploreView(props: ExploreViewProps): React.ReactElement {
  const { vm } = props;
  const stdoutCtx = useStdout();
  const cols = stdoutCtx?.stdout.columns ?? 80;
  const rows = stdoutCtx?.stdout.rows ?? 24;

  // Full panel height depends on the model: a row per head, or one or two rows
  // per block. The flow diagram prefers two rows per block and falls back to one.
  const hasSelector = vm.nLayers > 0;
  const boxedRows = flowPanelContentRows(vm.nLayers, 'boxed');
  const flowVariant: FlowVariant =
    panelTier(rows, cols, hasSelector, boxedRows) === 'full' ? 'boxed' : 'compact';
  const fullPanelContentRows =
    vm.panelMode === 'flow' ? flowPanelContentRows(vm.nLayers, flowVariant) : vm.nHeads + 1;
  const tier = panelTier(rows, cols, hasSelector, fullPanelContentRows);
  const chromeRows = exploreChromeRows(tier, vm.nLayers > 0, fullPanelContentRows);

  const showInspector = cols >= INSPECTOR_MIN_COLUMNS && rows >= INSPECTOR_MIN_ROWS;
  const visibleRows = Math.max(0, rows - chromeRows);

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
    if (input === 'f') {
      props.onTogglePanel();
      return;
    }
    // Blocks are drawn top to bottom, so ↓ moves to the next (deeper) block.
    // `+` / `-` stay as aliases; they are awkward to reach on many keyboards.
    if (key.downArrow || input === '+' || input === '=') {
      props.onLayerUp();
      return;
    }
    if (key.upArrow || input === '-') {
      props.onLayerDown();
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

  const inspectorBudget = showInspector ? INSPECTOR_PANEL_WIDTH + 4 : 0;
  const leftPaddingRight = showInspector ? 2 : 0;
  const leftWidth = Math.max(20, cols - inspectorBudget - leftPaddingRight - 2);
  const bodyHeight = Math.max(3, visibleRows);
  const tailCharBudget = leftWidth * (bodyHeight + 4);

  const spans = vm.contextTokenSpans;
  const sampledTokenIndex = spans.length > 0 ? (spans[spans.length - 1]?.tokenIndex ?? null) : null;
  // The token the model was reading when it produced the current one.
  const queryTokenIndex =
    sampledTokenIndex !== null && sampledTokenIndex > 0 ? sampledTokenIndex - 1 : null;
  const labelForIndex = (tokenIndex: number): string => {
    const span = spans[tokenIndex];
    return span !== undefined && span.tokenIndex === tokenIndex
      ? pieceLabel(span.text)
      : `#${String(tokenIndex)}`;
  };
  const labelForId = (id: number): string => pieceLabel(vm.displayFor(id));

  const hasStepLabels = queryTokenIndex !== null && sampledTokenIndex !== null;
  const attentionData: AttentionPanelData | null =
    vm.attention !== null && hasStepLabels
      ? {
          layerIndex: vm.attention.layerIndex,
          nLayers: vm.nLayers,
          rows: vm.attention.rows,
          heads: vm.attention.heads,
          connections: vm.attention.connections,
          queryLabel: labelForIndex(queryTokenIndex),
          sampledLabel: labelForIndex(sampledTokenIndex),
        }
      : null;

  const flowData: FlowPanelData | null =
    vm.flow !== null && hasStepLabels && vm.sampledFooter !== null
      ? {
          flow: vm.flow,
          focusedLayerIndex: vm.focusedLayerIndex,
          queryLabel: labelForIndex(queryTokenIndex),
          queryPosition: queryTokenIndex,
          sampledLabel: labelForIndex(sampledTokenIndex),
          sampledProbability: vm.sampledFooter.probability,
          temperature: vm.temperature,
          topK: vm.topK,
          dModel: vm.dModel,
          vocabSize: vm.vocabSize,
        }
      : null;

  // Highlights belong to the attention panel; the flow panel keeps the text plain.
  const showHighlights = vm.panelMode === 'attention' && tier !== 'hidden' && vm.attention !== null;
  const rankByKey = showHighlights
    ? bestRankByKeyIndex(vm.attention.connections)
    : new Map<number, number>();
  // Attention sparklines take what is left beside the labels, within sane bounds.
  const sparkWidth = Math.max(12, Math.min(48, cols - 64));
  const selectorNote =
    vm.attention !== null && vm.attention.layerIndex !== vm.focusedLayerIndex
      ? ` · showing block ${String(vm.attention.layerIndex + 1)} until the next token`
      : '';

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', padding: 1 }}>
      <Text style={{ fontWeight: 'bold' }}>Generate · explore</Text>
      <Text style={{ textWrap: 'truncate-end' }}>
        <Text style={{ color: statusColor, fontWeight: 'bold' }}>{`mode: ${vm.modeLabel}`}</Text>
        <Text style={{ color: 'gray' }}>
          {` · tok/s ${vm.tokensPerSec.toFixed(1)} · new ${String(vm.newTokens)}${selectorNote}`}
        </Text>
      </Text>
      <Text style={{ marginTop: 1, textWrap: 'truncate-end' }}>
        {vm.nLayers > 0 ? (
          <Text
            style={{ fontWeight: 'bold' }}
          >{`Block ${String(vm.focusedLayerIndex + 1)}/${String(vm.nLayers)}`}</Text>
        ) : null}
        <Text style={{ color: 'gray' }}>
          {`${vm.nLayers > 0 ? ' · ↑/↓ block · ' : ''}f ${vm.panelMode === 'flow' ? 'attention' : 'architecture'} view · s slow · p pause · q quit`}
        </Text>
      </Text>
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
            {renderContextText(
              vm.hiddenTailTokens > 0 ? spans.slice(0, -vm.hiddenTailTokens) : spans,
              vm.promptTokenCount,
              rankByKey,
              sampledTokenIndex,
              showHighlights ? queryTokenIndex : null,
              tailCharBudget,
            )}
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
      {vm.stoppedAtMarker !== null ? (
        <Text style={{ color: 'yellow', textWrap: 'truncate-end' }}>
          {`finished: the model wrote ${vm.stoppedAtMarker} (end of text) · --no-stop to keep going`}
        </Text>
      ) : vm.sampledFooter !== null ? (
        <Text style={{ color: 'gray' }}>
          {`sampled: id=${String(vm.sampledFooter.id)} p=${fmtProb(vm.sampledFooter.probability)} rank=${String(
            vm.sampledFooter.rank + 1,
          )}`}
        </Text>
      ) : (
        <Text style={{ color: 'gray' }}>{'(no token sampled yet)'}</Text>
      )}
      {vm.panelMode === 'flow' ? (
        <FlowPanel
          tier={tier}
          variant={flowVariant}
          data={flowData}
          labelForId={labelForId}
          labelForIndex={labelForIndex}
          cols={cols}
        />
      ) : (
        <AttentionPanel
          tier={tier}
          data={attentionData}
          labelForIndex={labelForIndex}
          sparkWidth={sparkWidth}
        />
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
  /**
   * End generation when the generated text contains this marker (the marker
   * itself is not shown). Omit to generate until **`maxNewTokens`**.
   */
  readonly stopText?: string;
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
  const decodeIds = (ids: readonly number[]): string => tok.decode([...ids], { utf8: 'replace' });
  /** Trailing tokens that spell the stop marker, or 0 (also when `--no-stop`). */
  const stopMarkerTokens = (): number =>
    opts.stopText !== undefined ? trailingMarkerTokens(generatedIds, decodeIds, opts.stopText) : 0;
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
        onStep: ({ tokenId }) => {
          generatedIds.push(tokenId);
          const markerTokens = stopMarkerTokens();
          const shownIds = markerTokens > 0 ? generatedIds.slice(0, -markerTokens) : generatedIds;
          textOut = sanitizeControlChars(decodeIds([...promptIds, ...shownIds]));
          newTok += 1;
          const dt = (performance.now() - tStart) / 1000;
          inst.rerender(
            <GenView
              text={textOut}
              newTokens={newTok}
              tokensPerSec={newTok / Math.max(dt, 1e-6)}
            />,
          );
          return markerTokens > 0 ? 'stop' : undefined;
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
    const piece = sanitizeControlChars(tok.decode([id], { utf8: 'replace' }));
    const onlyReplacement = /^�+$/.test(piece);
    const display = piece.length === 0 || onlyReplacement ? `‹${String(id)}›` : piece;
    decodeCache.set(id, display);
    return display;
  };

  let focusedLayerIndex = config.nLayers - 1;
  let panelMode: BottomPanelMode = 'attention';

  const initialVm: ExploreVm = {
    promptTokenCount: promptIds.length,
    contextTokenSpans: buildContextTokenSpans(promptIds, [], displayFor),
    newTokens: 0,
    tokensPerSec: 0,
    probabilities: null,
    sortedIds: null,
    maxProb: 0,
    displayFor,
    sampledFooter: null,
    modeLabel: pacing.statusLabel,
    nLayers: config.nLayers,
    nHeads: config.nHeads,
    dModel: config.dModel,
    vocabSize: config.vocabSize,
    temperature: opts.temperature,
    topK: opts.topK,
    focusedLayerIndex,
    panelMode,
    attention: null,
    flow: null,
    stoppedAtMarker: null,
    hiddenTailTokens: 0,
  };
  let currentVm: ExploreVm = initialVm;

  let exiting = false;
  const requestExit = (): void => {
    if (exiting) return;
    exiting = true;
    pacing.cancel();
  };

  let recaptureInFlight = false;

  /**
   * Re-run the last decode forward to fetch what the panel now needs — attention
   * for a newly focused block, or the lens after switching to the flow view.
   * Only useful while generation is waiting (paused / slow); otherwise the next
   * token brings fresh data anyway.
   *
   * The forward re-processes the token at the last cache position: it rewrites
   * that position's K/V rows with identical values and leaves `cache.seqLen`
   * where it was, so generation continues undisturbed.
   */
  async function recaptureCurrentStep(): Promise<void> {
    if (recaptureInFlight || generatedIds.length < 2) return;
    // Valid only between steps: every generated token except the newest is in
    // the cache. If the main loop has already forwarded the newest one, the
    // last cache row belongs to it and must not be rewritten.
    if (cache.seqLen !== promptIds.length + generatedIds.length - 1) return;
    const prevTokenId = generatedIds[generatedIds.length - 2];
    const sampledTokenId = generatedIds[generatedIds.length - 1];
    if (prevTokenId === undefined || sampledTokenId === undefined) return;
    recaptureInFlight = true;
    const stepAtStart = generatedIds.length;
    const layerAtStart = focusedLayerIndex;
    const wantFlow = panelMode === 'flow';
    try {
      cache.seqLen -= 1;
      const decodeResult = await forwardGptDecodeStep(rt, model, prevTokenId, cache, {
        attentionCapture: { layerIndex: layerAtStart },
        ...(wantFlow ? { flowTrace: true } : {}),
      });
      disposeGptKvForwardGraph(decodeResult.logits, model, cache);
      // A new token may have arrived while the readback was pending; its own
      // step already painted newer data, so drop this result.
      if (generatedIds.length !== stepAtStart) return;
      currentVm = {
        ...currentVm,
        ...(decodeResult.attention !== undefined
          ? {
              attention: attentionViewFromWeights(
                layerAtStart,
                decodeResult.attention,
                config.nHeads,
              ),
            }
          : {}),
        ...(decodeResult.layerLogits !== undefined &&
        decodeResult.allAttention !== undefined &&
        decodeResult.blockVectors !== undefined
          ? {
              flow: buildFlowTrace({
                layerLogits: decodeResult.layerLogits,
                allAttention: decodeResult.allAttention,
                blockVectors: decodeResult.blockVectors,
                nLayers: config.nLayers,
                nHeads: config.nHeads,
                sampledTokenId,
              }),
            }
          : {}),
      };
      paint();
    } finally {
      recaptureInFlight = false;
    }
  }

  const generationIsWaiting = (): boolean => pacing.paused || pacing.speed === 'slow';

  const cycleLayer = (delta: number): void => {
    const next = Math.max(0, Math.min(config.nLayers - 1, focusedLayerIndex + delta));
    if (next === focusedLayerIndex) return;
    focusedLayerIndex = next;
    currentVm = { ...currentVm, focusedLayerIndex };
    paint();
    if (generationIsWaiting()) {
      void recaptureCurrentStep();
    }
  };

  const togglePanel = (): void => {
    panelMode = panelMode === 'flow' ? 'attention' : 'flow';
    currentVm = { ...currentVm, panelMode };
    paint();
    if (panelMode === 'flow' && currentVm.flow === null && generationIsWaiting()) {
      void recaptureCurrentStep();
    }
  };

  const view = (vm: ExploreVm): React.ReactElement => (
    <ExploreView
      vm={vm}
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
      onLayerUp={() => {
        cycleLayer(1);
      }}
      onLayerDown={() => {
        cycleLayer(-1);
      }}
      onTogglePanel={togglePanel}
      onQuit={requestExit}
    />
  );

  function paint(): void {
    inst.rerender(view(currentVm));
  }

  const inst = render(view(initialVm), { exitOnCtrlC: false, patchConsole: true });

  try {
    await generateAutoregressive(rt, model, Uint32Array.from(promptIds), cache, {
      maxNewTokens: opts.maxNewTokens,
      sample: { temperature: opts.temperature, topK: opts.topK },
      ...(opts.repetitionPenalty !== undefined
        ? { repetitionPenalty: opts.repetitionPenalty }
        : {}),
      ...(opts.repetitionWindow !== undefined ? { repetitionWindow: opts.repetitionWindow } : {}),
      attentionLayerIndex: () => focusedLayerIndex,
      // The trace costs one vocabulary-sized matmul per block plus readbacks,
      // so it is only captured while the flow panel is the one on screen.
      captureFlow: () => panelMode === 'flow',
      onStep: async ({ tokenId, newTokenIndex, probabilities, attention, flow }) => {
        if (exiting) return;
        generatedIds.push(tokenId);
        newTok = newTokenIndex + 1;

        // When the model has just finished writing the stop marker, its tokens
        // are hidden from the text (the panels still describe that last step,
        // which really was "write the marker") and generation ends here.
        const markerTokens = stopMarkerTokens();

        const contextTokenSpans = buildContextTokenSpans(promptIds, generatedIds, displayFor);

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
          ...currentVm,
          contextTokenSpans,
          newTokens: newTok,
          tokensPerSec: newTok / Math.max(dt, 1e-6),
          probabilities,
          sortedIds,
          maxProb,
          sampledFooter: { id: tokenId, probability: probabilities[tokenId], rank },
          modeLabel: pacing.statusLabel,
          focusedLayerIndex,
          panelMode,
          attention:
            attention !== undefined
              ? attentionViewFromWeights(attention.layerIndex, attention.weights, attention.nHeads)
              : null,
          flow: flow ?? null,
          hiddenTailTokens: markerTokens,
          stoppedAtMarker: markerTokens > 0 ? (opts.stopText ?? null) : null,
        };
        paint();
        if (markerTokens > 0) {
          return 'stop';
        }

        await pacing.waitBetweenSteps();
        // Never start the next decode forward while a recapture is mid-readback:
        // both would move `cache.seqLen`.
        while (recaptureInFlight) {
          await new Promise<void>((resolve) => {
            setTimeout(resolve, 10);
          });
        }
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
