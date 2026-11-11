import React from 'react';
import { Box, Text } from '@wolf-tui/react';

import {
  bucketAttentionRow,
  groupConnectionsByHead,
  type GlobalAttentionConnection,
  type FlowLayer,
  type FlowTrace,
  type HeadAttentionSummary,
} from '@llm-transformer-diy/inference';

import { colorForGlobalRank } from './attention-rank-palette';
import type { PanelTier } from './generate-layout-tiers';

/**
 * Bottom panels of the **`llm generate`** explore view. Pure presentation: every
 * number shown here is computed in `@llm-transformer-diy/inference`.
 */

const SPARK_LEVELS = ['·', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'] as const;
const BAR_EIGHTHS = ['', '▏', '▎', '▍', '▌', '▋', '▊', '▉'] as const;

/** Relative weights below this draw as a dot: visually "this head ignores that position". */
const SPARK_FLOOR = 0.02;

/**
 * One character per column, oldest context on the left. Height is the weight
 * relative to **`scaleMax`** — the panel passes the largest weight of any head,
 * so bars stay comparable between heads while using the full height even when
 * a long context spreads every weight thin.
 */
export function attentionSparkline(row: Float32Array, width: number, scaleMax = 1): string {
  const cols = bucketAttentionRow(row, width);
  const scale = scaleMax > 0 ? scaleMax : 1;
  let out = '';
  for (const w of cols) {
    const rel = w / scale;
    const level = rel < SPARK_FLOOR ? 0 : Math.max(1, Math.min(8, Math.ceil(rel * 8)));
    out += SPARK_LEVELS[level] ?? '·';
  }
  return out;
}

/** Horizontal bar of **`width`** cells for a probability in **`[0, 1]`**, eighth-cell precision. */
export function probabilityBar(p: number, width: number): string {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(p) ? p : 0));
  const eighths = Math.round(clamped * width * 8);
  const full = Math.floor(eighths / 8);
  const rest = eighths % 8;
  return ('█'.repeat(full) + (BAR_EIGHTHS[rest] ?? '')).padEnd(width, ' ');
}

function fmtWeight(p: number): string {
  if (p >= 0.995) return '1.00';
  if (p >= 0.0001 || p === 0) return p.toFixed(2);
  return p.toExponential(0);
}

const panelBox = {
  display: 'flex',
  flexDirection: 'column',
  borderStyle: 'single',
  borderColor: 'gray',
  paddingLeft: 1,
  paddingRight: 1,
  marginTop: 1,
  overflow: 'hidden',
} as const;

export type AttentionPanelData = {
  /** Block the weights were captured from (may lag the selector by one token). */
  readonly layerIndex: number;
  readonly nLayers: number;
  /** One post-softmax row per head over the visible prefix. */
  readonly rows: readonly Float32Array[];
  readonly heads: readonly HeadAttentionSummary[];
  /** Global top connections; their rank colors match the highlights in the text. */
  readonly connections: readonly GlobalAttentionConnection[];
  readonly queryLabel: string;
  readonly sampledLabel: string;
};

function ConnectionEntry(props: {
  readonly connection: GlobalAttentionConnection;
  readonly labelForIndex: (tokenIndex: number) => string;
}): React.ReactElement {
  const c = props.connection;
  return (
    <Text>
      <Text style={{ color: colorForGlobalRank(c.globalRank) ?? 'white' }}>{' ■ '}</Text>
      <Text>{`«${props.labelForIndex(c.keyIndex)}»`}</Text>
      <Text style={{ color: 'gray' }}>{`@${String(c.keyIndex)} ${fmtWeight(c.strength)}`}</Text>
    </Text>
  );
}

/**
 * Where the focused block's heads look while predicting the next token.
 *
 * **Full**: one row per head — a sparkline of its weights over the whole
 * context (old → new), its entropy in bits, then its entries from the global
 * top connections (or, in gray, its own strongest link when it has none).
 * **Compressed**: the top connections grouped by head on up to two lines.
 */
export function AttentionPanel(props: {
  readonly tier: PanelTier;
  readonly data: AttentionPanelData | null;
  readonly labelForIndex: (tokenIndex: number) => string;
  readonly sparkWidth: number;
}): React.ReactElement | null {
  const { tier, data } = props;
  if (tier === 'hidden') return null;
  if (data === null) {
    return (
      <Box style={panelBox}>
        <Text style={{ color: 'gray', textWrap: 'truncate-end' }}>
          {
            'Attention · appears from the second generated token (the first comes from the prompt pass)'
          }
        </Text>
      </Box>
    );
  }

  const byHead = groupConnectionsByHead(data.connections);
  const caption = `Attention · block ${String(data.layerIndex + 1)}/${String(data.nLayers)} · reading «${data.queryLabel}» to predict «${data.sampledLabel}»`;

  if (tier === 'compressed') {
    const headOrder = [...byHead.keys()].sort((a, b) => a - b);
    return (
      <Box style={panelBox}>
        <Text style={{ color: 'gray', textWrap: 'truncate-end' }}>{caption}</Text>
        <Box style={{ height: 2, overflow: 'hidden' }}>
          <Text style={{ textWrap: 'wrap' }}>
            {headOrder.map((h) => (
              <Text key={h}>
                <Text style={{ fontWeight: 'bold' }}>{`H${String(h)}`}</Text>
                {(byHead.get(h) ?? []).map((c) => (
                  <ConnectionEntry
                    key={`${String(c.headIndex)}-${String(c.keyIndex)}`}
                    connection={c}
                    labelForIndex={props.labelForIndex}
                  />
                ))}
                <Text>{'  '}</Text>
              </Text>
            ))}
          </Text>
        </Box>
      </Box>
    );
  }

  const scaleMax = data.heads.reduce((m, h) => Math.max(m, h.peakStrength), 0);

  return (
    <Box style={panelBox}>
      <Text style={{ color: 'gray', textWrap: 'truncate-end' }}>
        {`${caption} · bars: weight over context, old → new · b: entropy in bits`}
      </Text>
      {data.rows.map((row, h) => {
        const head = data.heads[h];
        const entries = byHead.get(h) ?? [];
        return (
          <Text key={h} style={{ textWrap: 'truncate-end' }}>
            <Text style={{ fontWeight: 'bold' }}>{`H${String(h).padEnd(2, ' ')}`}</Text>
            <Text style={{ color: 'cyan' }}>
              {attentionSparkline(row, props.sparkWidth, scaleMax)}
            </Text>
            <Text style={{ color: 'gray' }}>
              {head !== undefined ? ` ${head.entropyBits.toFixed(1).padStart(4, ' ')}b` : ''}
            </Text>
            {entries.length > 0
              ? entries.map((c) => (
                  <ConnectionEntry
                    key={`${String(c.headIndex)}-${String(c.keyIndex)}`}
                    connection={c}
                    labelForIndex={props.labelForIndex}
                  />
                ))
              : head !== undefined && (
                  <Text style={{ color: 'gray' }}>
                    {` · «${props.labelForIndex(head.peakKeyIndex)}»@${String(head.peakKeyIndex)} ${fmtWeight(head.peakStrength)}`}
                  </Text>
                )}
          </Text>
        );
      })}
    </Box>
  );
}

export type FlowPanelData = {
  readonly flow: FlowTrace;
  readonly focusedLayerIndex: number;
  readonly queryLabel: string;
  readonly queryPosition: number;
  readonly sampledLabel: string;
  /** Probability the sampler actually drew from (after temperature / top-k). */
  readonly sampledProbability: number;
  readonly temperature: number;
  readonly topK: number;
  readonly dModel: number;
  readonly vocabSize: number;
};

/** How the full flow panel is drawn: two rows per block, or one when height is short. */
export type FlowVariant = 'boxed' | 'compact';

/** Content rows of the full flow panel: caption + input row + blocks + output row. */
export function flowPanelContentRows(nLayers: number, variant: FlowVariant): number {
  return (variant === 'boxed' ? 2 * nLayers : nLayers) + 3;
}

/**
 * One cell per attention head: tall = the head locks onto one position
 * (its largest weight is near 1), low = it spreads its attention thin.
 */
export function headActivityCells(heads: FlowLayer['heads']): string {
  let out = '';
  for (const h of heads) {
    const level = Math.max(1, Math.min(8, Math.ceil(h.peakStrength * 8)));
    out += SPARK_LEVELS[level] ?? '▁';
  }
  return out;
}

function fmtUpdate(ratio: number): string {
  const pct = ratio * 100;
  if (!Number.isFinite(pct)) return '+?%';
  return `+${pct >= 10 ? pct.toFixed(0) : pct.toFixed(1)}%`;
}

/** Color for an update size: the larger the step's change to the stream, the hotter. */
function updateColor(ratio: number): string {
  if (ratio >= 0.5) return 'red';
  if (ratio >= 0.15) return 'yellow';
  return 'green';
}

function strongestHead(layer: FlowLayer): FlowLayer['heads'][number] | undefined {
  let best: FlowLayer['heads'][number] | undefined;
  for (const h of layer.heads) {
    if (best === undefined || h.peakStrength > best.peakStrength) best = h;
  }
  return best;
}

/** `attention [▁█▃…] ⊕ +12%  ─▶  feed-forward ⊕ +8%` — the two steps of a block, with live activity. */
function BlockSteps(props: {
  readonly layer: FlowLayer;
  readonly short: boolean;
}): React.ReactElement {
  const { layer, short } = props;
  return (
    <Text>
      <Text style={{ color: 'gray' }}>{short ? 'att ' : 'attention '}</Text>
      <Text style={{ color: 'gray' }}>{'['}</Text>
      <Text style={{ color: 'cyan' }}>{headActivityCells(layer.heads)}</Text>
      <Text style={{ color: 'gray' }}>{'] ⊕ '}</Text>
      <Text style={{ color: updateColor(layer.attentionUpdate) }}>
        {fmtUpdate(layer.attentionUpdate).padEnd(6, ' ')}
      </Text>
      <Text style={{ color: 'gray' }}>{short ? ' ffn ⊕ ' : ' ─▶ feed-forward ⊕ '}</Text>
      <Text style={{ color: updateColor(layer.feedForwardUpdate) }}>
        {fmtUpdate(layer.feedForwardUpdate).padEnd(6, ' ')}
      </Text>
    </Text>
  );
}

/** `guess «tok» 0.02 · ████▏ 0.07` — the logit lens of a block. */
function BlockLens(props: {
  readonly layer: FlowLayer;
  readonly labelForId: (tokenId: number) => string;
  readonly short: boolean;
}): React.ReactElement {
  const { layer } = props;
  return (
    <Text>
      <Text style={{ color: 'gray' }}>{props.short ? ' ┆ ' : ' ┆ would say '}</Text>
      <Text>{`«${props.labelForId(layer.topTokenId)}»`.padEnd(13, ' ')}</Text>
      <Text style={{ color: 'gray' }}>{`${fmtWeight(layer.topProbability)} `}</Text>
      <Text style={{ color: 'cyan' }}>{probabilityBar(layer.sampledProbability, 8)}</Text>
      <Text style={{ color: 'gray' }}>{` ${fmtWeight(layer.sampledProbability)}`}</Text>
    </Text>
  );
}

/**
 * The model drawn as it is built, top to bottom, with what each part did for
 * the current token.
 *
 * The vertical line on the left is the **residual stream**: one vector that
 * enters at the top as "token + position" and leaves at the bottom as the
 * prediction. Every block reads it and **adds** two updates to it (`⊕`):
 *
 * - **attention** — the bracket holds one cell per head; a tall cell is a head
 *   locked onto a single earlier token, a low one a head averaging many. The
 *   percentage is the size of the update relative to the stream.
 * - **feed-forward** — the same percentage for the block's second step.
 *
 * Right of the divider is the **logit lens**: the token the model would predict
 * if it stopped after this block, and a bar for how much probability the token
 * that was finally chosen has at this depth.
 */
export function FlowPanel(props: {
  readonly tier: PanelTier;
  readonly variant: FlowVariant;
  readonly data: FlowPanelData | null;
  readonly labelForId: (tokenId: number) => string;
  readonly labelForIndex: (tokenIndex: number) => string;
  readonly cols: number;
}): React.ReactElement | null {
  const { tier, data } = props;
  if (tier === 'hidden') return null;
  if (data === null) {
    return (
      <Box style={panelBox}>
        <Text style={{ color: 'gray', textWrap: 'truncate-end' }}>
          {'Architecture · appears from the next generated token (press p to pause on one)'}
        </Text>
      </Box>
    );
  }

  const layers = data.flow.layers;
  const caption = `Architecture · «${data.queryLabel}» @${String(data.queryPosition)} → «${data.sampledLabel}»`;

  if (tier === 'compressed') {
    return (
      <Box style={panelBox}>
        <Text style={{ color: 'gray', textWrap: 'truncate-end' }}>{caption}</Text>
        <Box style={{ height: 2, overflow: 'hidden' }}>
          <Text style={{ textWrap: 'wrap' }}>
            {layers.map((l, i) => (
              <Text key={l.layerIndex}>
                {i > 0 ? <Text style={{ color: 'gray' }}>{' → '}</Text> : null}
                <Text style={{ color: 'gray' }}>{`b${String(l.layerIndex + 1)} `}</Text>
                <Text>{`«${props.labelForId(l.topTokenId)}»`}</Text>
                <Text style={{ color: 'gray' }}>{` ${fmtWeight(l.topProbability)}`}</Text>
              </Text>
            ))}
          </Text>
        </Box>
      </Box>
    );
  }

  const short = props.cols < 104;
  const sampler =
    data.temperature <= 1e-8
      ? 'take the top one'
      : `sample T=${String(data.temperature)}${data.topK > 0 ? ` top-${String(data.topK)}` : ''}`;
  const ruleWidth = Math.max(8, Math.min(props.cols - 24, 96));

  return (
    <Box style={panelBox}>
      <Text style={{ color: 'gray', textWrap: 'truncate-end' }}>
        {`${caption} · [head cells] tall = locked on one token · ⊕ +% = change to the stream · bar = p(«${data.sampledLabel}»)`}
      </Text>
      <Text style={{ textWrap: 'truncate-end' }}>
        <Text style={{ color: 'gray' }}>{'  ┌ '}</Text>
        <Text style={{ fontWeight: 'bold' }}>{`«${data.queryLabel}»`}</Text>
        <Text style={{ color: 'gray' }}>
          {` ─▶ token vector ⊕ position ${String(data.queryPosition)} ═▶ residual stream (${String(data.dModel)} numbers)`}
        </Text>
      </Text>
      {layers.map((l) => {
        const focused = l.layerIndex === data.focusedLayerIndex;
        const titleColor = focused ? 'yellow' : 'gray';
        const title = `block ${String(l.layerIndex + 1)}`;
        if (props.variant === 'compact') {
          return (
            <Text key={l.layerIndex} style={{ textWrap: 'truncate-end' }}>
              <Text style={{ color: titleColor }}>
                {`  ${focused ? '┝▶' : '├─'} ${title.padEnd(9, ' ')}`}
              </Text>
              <BlockSteps layer={l} short={true} />
              <BlockLens layer={l} labelForId={props.labelForId} short={true} />
            </Text>
          );
        }
        const top = strongestHead(l);
        const link =
          top !== undefined
            ? ` strongest: H${String(top.headIndex)} → «${props.labelForIndex(top.peakKeyIndex)}»@${String(top.peakKeyIndex)} ${fmtWeight(top.peakStrength)} `
            : ' ';
        const lead = `  ${focused ? '┝▶' : '├─'} ${title} ─`;
        const fill = '─'.repeat(Math.max(2, ruleWidth - lead.length - link.length));
        return (
          <React.Fragment key={l.layerIndex}>
            <Text style={{ textWrap: 'truncate-end' }}>
              <Text style={{ color: titleColor, fontWeight: focused ? 'bold' : 'normal' }}>
                {lead}
              </Text>
              <Text style={{ color: focused ? 'yellow' : 'gray' }}>{link}</Text>
              <Text style={{ color: 'gray' }}>{fill}</Text>
            </Text>
            <Text style={{ textWrap: 'truncate-end' }}>
              <Text style={{ color: 'gray' }}>{'  │   '}</Text>
              <BlockSteps layer={l} short={short} />
              <BlockLens layer={l} labelForId={props.labelForId} short={short} />
            </Text>
          </React.Fragment>
        );
      })}
      <Text style={{ textWrap: 'truncate-end' }}>
        <Text style={{ color: 'gray' }}>
          {`  └▶ final norm ─▶ ${String(data.vocabSize)} scores ─▶ softmax ─▶ ${sampler} ─▶ `}
        </Text>
        <Text style={{ color: 'cyan', fontWeight: 'bold' }}>{`«${data.sampledLabel}»`}</Text>
        <Text style={{ color: 'gray' }}>{` ${fmtWeight(data.sampledProbability)}`}</Text>
      </Text>
    </Box>
  );
}
