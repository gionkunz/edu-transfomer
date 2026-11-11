import React, { useMemo } from 'react';
import { detectColorMode } from '@chartist-js/terminal';
import { Badge, Box, ProgressBar, Spinner, Text, useInput } from '@wolf-tui/react';

import { formatDurationSec } from './format-duration';
import { renderGradNormChart, renderLossChart, renderLrChart } from './train-charts';
import {
  type TrainDashboardSnapshot,
  type TrainPipelinePhase,
  TRAIN_SETUP_PIPELINE_ORDER,
  setupMacroPct,
} from './train-dash-types';

const PHASE_LABEL: Record<TrainPipelinePhase, string> = {
  starting: 'start',
  gpu_init: 'GPU',
  tokenizer_load: 'tokenizer',
  resolve_config: 'config',
  discover_files: 'discover',
  read_corpus: 'read',
  encode_corpus: 'encode',
  assemble_corpus: 'assemble',
  verify_corpus: 'verify',
  build_model: 'model',
  trainer_init: 'trainer',
  training: 'train',
  save_checkpoint: 'save',
  finalize: 'finish',
  done: 'done',
};

function formatEncodeCountApprox(n: number): string {
  const x = Math.max(0, Math.floor(n));
  if (x >= 1_000_000) return `${(x / 1_000_000).toFixed(x >= 10_000_000 ? 0 : 2)}M`;
  if (x >= 1000) return `${(x / 1000).toFixed(x >= 10_000 ? 0 : 1)}k`;
  return String(x);
}

/** Regions encoded per second under the **`gpt2-v1`** pre-tokenizer. */
function formatRegionThroughput(regionsPerSec: number | null): string {
  if (regionsPerSec === null || !Number.isFinite(regionsPerSec)) return '—';
  if (regionsPerSec >= 1000) return `${(regionsPerSec / 1000).toFixed(1)}k region/s`;
  return `${regionsPerSec.toFixed(1)} region/s`;
}

function formatEtaSec(sec: number | null): string {
  if (sec === null || !Number.isFinite(sec)) return '—';
  return `~${formatDurationSec(sec)}`;
}

function LossSparkline({ values }: { readonly values: readonly number[] }): React.ReactElement {
  if (values.length === 0) {
    return <Text style={{ color: 'gray' }}>—</Text>;
  }
  const finite = values.filter((x) => Number.isFinite(x));
  if (finite.length === 0) {
    return <Text style={{ color: 'gray' }}>—</Text>;
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = Math.max(1e-9, max - min);
  const n = values.length;
  const cols = 28;
  const segments: number[] = [];
  for (let c = 0; c < cols; c++) {
    const idx = Math.min(n - 1, Math.floor(((c + 0.5) * n) / cols));
    const vAt = values[idx];
    segments.push(vAt !== undefined && Number.isFinite(vAt) ? vAt : min);
  }
  const glyphs = '▁▂▃▄▅▆▇█';
  const line = segments
    .map((avg) => {
      const t = max === min ? 0 : (avg - min) / span;
      const i = Math.min(7, Math.max(0, Math.round(t * 7)));
      return glyphs[i] ?? glyphs[0];
    })
    .join('');
  return <Text style={{ color: 'cyan' }}>{line}</Text>;
}

/**
 * Rows the training block needs besides the charts: header, paths, adapter,
 * phase, progress, three stat rows, log tail, footer and padding.
 */
const NON_CHART_ROWS = 32;
/** Extra rows of the software-adapter warning. */
const SOFTWARE_WARNING_ROWS = 4;
/** Title, legend and top margin around the loss chart. */
const LOSS_CHART_CHROME_ROWS = 3;
const LOSS_CHART_MIN_ROWS = 8;
const LOSS_CHART_MAX_ROWS = 16;
const SMALL_CHART_ROWS = 7;
/** Title and top margin of the LR / grad-norm row. */
const SMALL_CHART_CHROME_ROWS = 2;
const CHART_MAX_COLUMNS = 120;

/**
 * Chart sizes for the current terminal, so the dashboard never grows taller
 * than the screen (a taller frame scrolls and leaves stale lines behind).
 * Order of sacrifice as the terminal gets shorter: the LR / grad-norm pair,
 * then loss-chart height, then the loss chart itself (sparkline instead).
 */
function chartLayout(hasSoftwareWarning: boolean): {
  readonly columns: number;
  readonly lossRows: number;
  readonly showLoss: boolean;
  readonly showSmall: boolean;
} {
  const termColumns = process.stdout.columns ?? 100;
  const termRows = process.stdout.rows ?? 40;
  const columns = Math.max(40, Math.min(CHART_MAX_COLUMNS, termColumns - 4));
  const free = termRows - NON_CHART_ROWS - (hasSoftwareWarning ? SOFTWARE_WARNING_ROWS : 0);
  const smallBlock = SMALL_CHART_ROWS + SMALL_CHART_CHROME_ROWS;
  const lossMin = LOSS_CHART_MIN_ROWS + LOSS_CHART_CHROME_ROWS;
  const showLoss = free >= lossMin;
  const showSmall = free >= lossMin + smallBlock && columns >= 70;
  const forLoss = free - LOSS_CHART_CHROME_ROWS - (showSmall ? smallBlock : 0);
  const lossRows = Math.max(LOSS_CHART_MIN_ROWS, Math.min(LOSS_CHART_MAX_ROWS, forLoss));
  return { columns, lossRows, showLoss, showSmall };
}

function TrainCharts({ snap }: { readonly snap: TrainDashboardSnapshot }): React.ReactElement {
  const layout = chartLayout(snap.gpuSoftwareWarning !== null);
  const color = useMemo(() => detectColorMode(process.stdout, process.env), []);
  // Re-render charts only when their data changed, not on every clock tick.
  const lossChart = useMemo(
    () =>
      renderLossChart(snap.lossHistory, snap.valHistory, {
        columns: layout.columns,
        rows: layout.lossRows,
        color,
      }),
    [snap.lossHistory, snap.valHistory, layout.columns, layout.lossRows, color],
  );
  const half = Math.floor((layout.columns - 2) / 2);
  const lrChart = useMemo(
    () =>
      layout.showSmall && snap.lrSchedule !== null
        ? renderLrChart(snap.lrSchedule, snap.trainStepCompleted, {
            columns: half,
            rows: SMALL_CHART_ROWS,
            color,
          })
        : null,
    [layout.showSmall, snap.lrSchedule, snap.trainStepCompleted, half, color],
  );
  const gradChart = useMemo(
    () =>
      layout.showSmall
        ? renderGradNormChart(snap.gradNormHistory, snap.maxGradNorm, {
            columns: half,
            rows: SMALL_CHART_ROWS,
            color,
          })
        : null,
    [layout.showSmall, snap.gradNormHistory, snap.maxGradNorm, half, color],
  );

  if (lossChart === null || !layout.showLoss) {
    return (
      <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
        <Text style={{ color: 'gray' }}>{'Loss trend (recent steps)'}</Text>
        <LossSparkline values={snap.recentLosses} />
      </Box>
    );
  }
  return (
    <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
      <Text style={{ color: 'gray' }}>
        {`Loss over the run${snap.uniformLoss !== null ? ` · uniform guess = ln(V) = ${snap.uniformLoss.toFixed(2)}` : ''}`}
      </Text>
      <Text>{lossChart}</Text>
      {lrChart !== null || gradChart !== null ? (
        <Box style={{ display: 'flex', flexDirection: 'row', marginTop: 1 }}>
          {lrChart !== null ? (
            <Box style={{ display: 'flex', flexDirection: 'column', width: half + 2 }}>
              <Text style={{ color: 'gray' }}>{'Learning rate · done / remaining'}</Text>
              <Text>{lrChart}</Text>
            </Box>
          ) : null}
          {gradChart !== null ? (
            <Box style={{ display: 'flex', flexDirection: 'column', width: half }}>
              <Text style={{ color: 'gray' }}>
                {snap.maxGradNorm !== null
                  ? `Gradient norm before clipping · clip at ${String(snap.maxGradNorm)}`
                  : 'Gradient norm (clipping off)'}
              </Text>
              <Text>{gradChart}</Text>
            </Box>
          ) : null}
        </Box>
      ) : null}
    </Box>
  );
}

function SetupStepperRail(props: { readonly phase: TrainPipelinePhase }): React.ReactElement {
  const ix = TRAIN_SETUP_PIPELINE_ORDER.indexOf(props.phase);
  const segments = TRAIN_SETUP_PIPELINE_ORDER.map((p, i) => {
    let color: 'gray' | 'green' | 'cyan' = 'gray';
    if (ix >= 0) {
      if (i < ix) color = 'green';
      else if (i === ix) color = 'cyan';
    }
    return (
      <Text key={p} style={{ color }}>
        {(i === 0 ? '' : ' › ') + PHASE_LABEL[p]}
      </Text>
    );
  });
  return <Box style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap' }}>{segments}</Box>;
}

export type TrainDashboardViewProps = {
  readonly snapshot: TrainDashboardSnapshot;
  /** Toggle training pause; only honored while **`phase === 'training'`** (caller checks). */
  readonly onToggleTrainingPause?: () => void;
};

export function TrainDashboardView(props: TrainDashboardViewProps): React.ReactElement {
  const snap = props.snapshot;

  useInput((input, key) => {
    if (input === 'q') {
      process.exit(0);
    }
    if (key.ctrl && input === 'c') {
      process.exit(0);
    }
    if (input === 'p' && snap.phase === 'training' && props.onToggleTrainingPause !== undefined) {
      props.onToggleTrainingPause();
    }
  });

  const setupPct = setupMacroPct(snap.phase);
  const lossStr = useMemo(() => snap.trainLossLast.toFixed(4), [snap.trainLossLast]);
  const lossSmaStr = useMemo(
    () => (snap.trainLossSma20 !== null ? snap.trainLossSma20.toFixed(4) : '—'),
    [snap.trainLossSma20],
  );
  const bitsPerCharStr = useMemo(
    () => (snap.bitsPerCharSma20 !== null ? snap.bitsPerCharSma20.toFixed(3) : '—'),
    [snap.bitsPerCharSma20],
  );
  const ppxStr = useMemo(() => {
    if (snap.perplexitySma20 === null || !Number.isFinite(snap.perplexitySma20)) return '—';
    const v = snap.perplexitySma20;
    if (v >= 100_000) return v.toExponential(2);
    if (v >= 1000) return Math.round(v).toString();
    return v.toFixed(1);
  }, [snap.perplexitySma20]);
  const charsPerTokStr = useMemo(
    () =>
      snap.meanCharsPerToken !== null && Number.isFinite(snap.meanCharsPerToken)
        ? snap.meanCharsPerToken.toFixed(2)
        : '—',
    [snap.meanCharsPerToken],
  );
  const lrStr = useMemo(() => snap.trainLrDisplay.toExponential(3), [snap.trainLrDisplay]);

  const trainProgressPct =
    snap.totalTrainSteps > 0
      ? Math.min(100, Math.max(0, (snap.trainStepCompleted / snap.totalTrainSteps) * 100))
      : 0;

  const etaSec =
    snap.trainStepCompleted > 0 &&
    snap.trainStepCompleted < snap.totalTrainSteps &&
    snap.trainWallActiveSec > 0
      ? ((snap.totalTrainSteps - snap.trainStepCompleted) * snap.trainWallActiveSec) /
        snap.trainStepCompleted
      : null;
  const etaStr = etaSec !== null && Number.isFinite(etaSec) ? `~${formatDurationSec(etaSec)}` : '—';

  const inSetupFunnel = TRAIN_SETUP_PIPELINE_ORDER.includes(snap.phase);
  const showEncodeSpinner = snap.phase === 'encode_corpus';
  const showTrainingBlock =
    snap.phase === 'training' ||
    snap.phase === 'save_checkpoint' ||
    snap.phase === 'finalize' ||
    snap.phase === 'done';

  const filePct =
    snap.fileProgress !== null && snap.fileProgress.totalFiles > 0
      ? Math.min(100, (snap.fileProgress.currentIndex / snap.fileProgress.totalFiles) * 100)
      : 0;

  const savePct =
    snap.saveProgress !== null && snap.saveProgress.totalTensors > 0
      ? Math.min(100, ((snap.saveProgress.tensorIndex + 1) / snap.saveProgress.totalTensors) * 100)
      : 0;

  return (
    <Box style={{ display: 'flex', flexDirection: 'column', padding: 1 }}>
      <Box style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap' }}>
        <Text style={{ fontWeight: 'bold' }}>{'llm train  '}</Text>
        <Badge color="cyan">{snap.configTag}</Badge>
        {snap.archSummary !== null ? (
          <>
            <Text>{'  '}</Text>
            <Badge color="magenta">{snap.archSummary}</Badge>
          </>
        ) : null}
      </Box>

      <Text style={{ color: 'gray', marginTop: 1, textWrap: 'truncate-start' }}>
        {snap.outDirDisplay}
      </Text>

      {snap.gpuAdapterSummary !== null && snap.gpuAdapterSummary.length > 0 ? (
        <Text style={{ marginTop: 1, textWrap: 'truncate-end' }}>
          <Text style={{ color: 'gray' }}>{'WebGPU adapter  '}</Text>
          <Text style={{ color: 'cyan' }}>{snap.gpuAdapterSummary}</Text>
        </Text>
      ) : null}

      {snap.gpuSoftwareWarning !== null && snap.gpuSoftwareWarning.length > 0 ? (
        <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
          <Text style={{ color: 'yellow', fontWeight: 'bold' }}>{'Software WebGPU backend'}</Text>
          <Text style={{ color: 'gray' }}>{snap.gpuSoftwareWarning}</Text>
        </Box>
      ) : null}

      <Box style={{ marginTop: 1 }}>
        <Box style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap' }}>
          <Text style={{ color: 'gray' }}>{'Phase '}</Text>
          <Badge color="yellow">{PHASE_LABEL[snap.phase]}</Badge>
          {snap.phase === 'training' && snap.trainingPaused ? (
            <>
              <Text>{'  '}</Text>
              <Badge color="red">{'PAUSED'}</Badge>
            </>
          ) : null}
          {snap.detailLine.trim().length > 0 ? (
            <Text style={{ color: 'gray' }}>{` · ${snap.detailLine}`}</Text>
          ) : null}
        </Box>
      </Box>

      {inSetupFunnel ? (
        <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
          <Text style={{ color: 'gray' }}>{'Setup funnel'}</Text>
          <ProgressBar value={setupPct} />
          <Text
            style={{ color: 'gray' }}
          >{`${setupPct.toFixed(0)}% · ${formatDurationSec(snap.nowElapsedSec)} elapsed`}</Text>
          <Box style={{ marginTop: 1 }}>
            <SetupStepperRail phase={snap.phase} />
          </Box>
        </Box>
      ) : null}

      {snap.phase === 'done' ? (
        <Box style={{ marginTop: 1 }}>
          <Text style={{ color: 'green', fontWeight: 'bold' }}>{'Training run complete.'}</Text>
        </Box>
      ) : null}

      {snap.fileProgress !== null ? (
        <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
          <Text style={{ color: 'gray' }}>{'Corpus files'}</Text>
          <ProgressBar value={filePct} />
          <Text style={{ color: 'gray' }}>
            {`${String(snap.fileProgress.currentIndex)} / ${String(snap.fileProgress.totalFiles)} · ${snap.fileProgress.pathRel}`}
          </Text>
        </Box>
      ) : null}

      {snap.encodeProgress !== null ? (
        <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
          <Box style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap' }}>
            {showEncodeSpinner ? <Spinner type="dots" /> : null}
            <Box
              style={{
                display: 'flex',
                flexDirection: 'column',
                marginLeft: showEncodeSpinner ? 1 : 0,
              }}
            >
              <Text style={{ color: 'gray' }}>{'BPE encode'}</Text>
              <Text style={{ color: 'gray' }}>
                {`pre-tokenizer: ${snap.encodeProgress.pretokenizerPolicy}`}
              </Text>
              <Text>
                {snap.encodeProgress.mergeSubphase === 'expand'
                  ? `expand · ${String(snap.encodeProgress.textChars)} chars · ${String(snap.encodeProgress.byteLength)} bytes → ${String(snap.encodeProgress.tokenCount)} tok`
                  : `region ${formatEncodeCountApprox(snap.encodeProgress.regionIndex + 1)} / ${formatEncodeCountApprox(snap.encodeProgress.regionsTotal)} · pass #${formatEncodeCountApprox(snap.encodeProgress.mergeIteration)} · ${formatEncodeCountApprox(snap.encodeProgress.tokenCount)} tok`}
              </Text>
              <Text style={{ color: 'gray' }}>
                {`elapsed ${formatDurationSec(snap.encodeProgress.encodeElapsedSec)}${snap.encodeProgress.meanPassesPerRegion !== null ? ` · mean ${snap.encodeProgress.meanPassesPerRegion.toFixed(1)} passes/region` : ''}`}
              </Text>
              {snap.encodeProgress.mergeSubphase === 'merge' ? (
                <>
                  <Box style={{ marginTop: 1 }}>
                    <ProgressBar
                      value={
                        snap.encodeProgress.regionsTotal > 0
                          ? Math.min(
                              100,
                              (100 * snap.encodeProgress.regionIndex) /
                                snap.encodeProgress.regionsTotal,
                            )
                          : 0
                      }
                    />
                    <Text style={{ color: 'gray' }}>
                      {snap.encodeProgress.regionsTotal > 0
                        ? `${((100 * snap.encodeProgress.regionIndex) / snap.encodeProgress.regionsTotal).toFixed(1)}% of regions encoded`
                        : '—'}
                    </Text>
                  </Box>
                  <Text style={{ marginTop: 1 }}>
                    {`${formatRegionThroughput(snap.encodeProgress.regionsPerSec)} · ETA ${formatEtaSec(snap.encodeProgress.etaSec)}`}
                  </Text>
                </>
              ) : (
                <Text style={{ color: 'gray', marginTop: 1 }}>
                  ETA after region throughput is measured…
                </Text>
              )}
            </Box>
          </Box>
        </Box>
      ) : null}

      {showTrainingBlock ? (
        <>
          <Box style={{ display: 'flex', flexDirection: 'row', marginTop: 1 }}>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '50%' }}>
              <Text style={{ color: 'gray' }}>{'Corpus tokens'}</Text>
              <Text style={{ fontWeight: 'bold' }}>
                {snap.corpusTokens !== null
                  ? `${String(snap.corpusTokens)} · seq ${String(snap.seqLen)}`
                  : '—'}
              </Text>
            </Box>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '50%' }}>
              <Text style={{ color: 'gray' }}>{'Step'}</Text>
              <Text style={{ fontWeight: 'bold' }}>
                {`${String(snap.trainStepCompleted)} / ${String(snap.totalTrainSteps)}`}
              </Text>
            </Box>
          </Box>
          <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
            <Text style={{ color: 'gray' }}>{'Training progress'}</Text>
            <ProgressBar value={trainProgressPct} />
            <Text style={{ color: 'gray' }}>
              {`${trainProgressPct.toFixed(0)}% · train wall ${formatDurationSec(snap.trainWallActiveSec)} · ETA ${etaStr}`}
            </Text>
          </Box>
          <Box style={{ display: 'flex', flexDirection: 'row', marginTop: 1, flexWrap: 'wrap' }}>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '32%' }}>
              <Text style={{ color: 'yellow' }}>{'Loss · sma(20)'}</Text>
              <Text style={{ fontWeight: 'bold' }}>{`${lossStr} · ${lossSmaStr}`}</Text>
            </Box>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '32%' }}>
              <Text style={{ color: 'green' }}>{'LR'}</Text>
              <Text style={{ fontWeight: 'bold' }}>{lrStr}</Text>
            </Box>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '32%' }}>
              <Text style={{ color: 'blue' }}>{'tok/s ~'}</Text>
              <Text style={{ fontWeight: 'bold' }}>{snap.tokensPerSecApprox.toFixed(0)}</Text>
            </Box>
          </Box>
          <Box style={{ display: 'flex', flexDirection: 'row', marginTop: 1, flexWrap: 'wrap' }}>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '32%' }}>
              <Text style={{ color: 'magenta' }}>{'bits/char (sma)'}</Text>
              <Text style={{ fontWeight: 'bold' }}>{bitsPerCharStr}</Text>
            </Box>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '32%' }}>
              <Text style={{ color: 'magenta' }}>{'ppx/tok (sma)'}</Text>
              <Text style={{ fontWeight: 'bold' }}>{ppxStr}</Text>
            </Box>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '32%' }}>
              <Text style={{ color: 'gray' }}>{'chars/tok'}</Text>
              <Text style={{ fontWeight: 'bold' }}>{charsPerTokStr}</Text>
            </Box>
          </Box>
          <Box style={{ display: 'flex', flexDirection: 'row', marginTop: 1, flexWrap: 'wrap' }}>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '32%' }}>
              <Text style={{ color: 'yellow' }}>{'Validation loss'}</Text>
              <Text style={{ fontWeight: 'bold' }}>
                {snap.valLossLast !== null ? snap.valLossLast.toFixed(4) : '— (--val-data)'}
              </Text>
            </Box>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '32%' }}>
              <Text style={{ color: 'magenta' }}>{'Grad norm'}</Text>
              <Text style={{ fontWeight: 'bold' }}>
                {snap.gradNormLast !== null ? snap.gradNormLast.toFixed(3) : '—'}
              </Text>
            </Box>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '32%' }}>
              <Text style={{ color: 'gray' }}>{'Sequences / step'}</Text>
              <Text style={{ fontWeight: 'bold' }}>{String(snap.accum)}</Text>
            </Box>
          </Box>
          <TrainCharts snap={snap} />
        </>
      ) : null}

      {snap.saveProgress !== null ? (
        <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
          <Text style={{ color: 'gray' }}>{'Checkpoint (CPU readback)'}</Text>
          <ProgressBar value={savePct} />
          <Text style={{ color: 'gray' }}>
            {`${String(snap.saveProgress.tensorIndex + 1)} / ${String(snap.saveProgress.totalTensors)} · ${snap.saveProgress.tensorName}`}
          </Text>
        </Box>
      ) : null}

      {snap.logTail.length > 0 ? (
        <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
          {snap.logTail.map((line, i) => (
            <Text
              key={`${String(i)}-${line}`}
              style={{ color: 'gray', textWrap: 'truncate-middle' }}
            >
              {`· ${line}`}
            </Text>
          ))}
        </Box>
      ) : null}

      <Text style={{ color: 'gray', marginTop: 1, textWrap: 'truncate-end' }}>
        {snap.phase === 'training'
          ? 'p pause/resume GPU · Ctrl+C / q quit · full log: train.log in the output directory'
          : 'Ctrl+C / q quit · full log: train.log in the output directory'}
      </Text>
    </Box>
  );
}
