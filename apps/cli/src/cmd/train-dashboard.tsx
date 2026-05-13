import React, { useMemo } from 'react';
import { Badge, Box, ProgressBar, Spinner, Text, useInput } from '@wolf-tui/react';

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

function formatMergeThroughput(mergesPerSec: number | null): string {
  if (mergesPerSec === null || !Number.isFinite(mergesPerSec)) return '—';
  if (mergesPerSec >= 1000) return `${(mergesPerSec / 1000).toFixed(1)}k merge/s`;
  return `${mergesPerSec.toFixed(1)} merge/s`;
}

function formatPessimisticEtaSec(sec: number | null): string {
  if (sec === null || !Number.isFinite(sec)) return '—';
  const s = Math.max(0, sec);
  if (s >= 3600) return `~${(s / 3600).toFixed(1)}h`;
  if (s >= 60) return `~${Math.max(1, Math.round(s / 60))}m`;
  return `~${Math.max(1, Math.round(s))}s`;
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
  });

  const setupPct = setupMacroPct(snap.phase);
  const lossStr = useMemo(() => snap.trainLossLast.toFixed(4), [snap.trainLossLast]);
  const lrStr = useMemo(() => snap.trainLrDisplay.toExponential(3), [snap.trainLrDisplay]);

  const trainProgressPct =
    snap.totalTrainSteps > 0
      ? Math.min(100, Math.max(0, (snap.trainStepCompleted / snap.totalTrainSteps) * 100))
      : 0;

  const etaSec =
    snap.trainStepCompleted > 0 && snap.trainStepCompleted < snap.totalTrainSteps
      ? ((snap.totalTrainSteps - snap.trainStepCompleted) * snap.nowElapsedSec) /
        snap.trainStepCompleted
      : null;
  const etaStr =
    etaSec !== null && Number.isFinite(etaSec) ? `~${Math.max(0, Math.round(etaSec))}s` : '—';

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

      <Text style={{ color: 'gray', marginTop: 1 }}>{snap.outDirDisplay}</Text>

      {snap.gpuAdapterSummary !== null && snap.gpuAdapterSummary.length > 0 ? (
        <Box style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', marginTop: 1 }}>
          <Text style={{ color: 'gray' }}>{'WebGPU adapter  '}</Text>
          <Text style={{ color: 'cyan' }}>{snap.gpuAdapterSummary}</Text>
        </Box>
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
          {snap.detailLine.trim().length > 0 ? (
            <Text style={{ color: 'gray' }}>{` · ${snap.detailLine}`}</Text>
          ) : null}
        </Box>
      </Box>

      {inSetupFunnel ? (
        <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
          <Text style={{ color: 'gray' }}>{'Setup funnel'}</Text>
          <ProgressBar value={setupPct} />
          <Text style={{ color: 'gray' }}>{`${setupPct.toFixed(0)}% · ${snap.nowElapsedSec.toFixed(
            1,
          )}s elapsed`}</Text>
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
              <Text>
                {snap.encodeProgress.mergeSubphase === 'expand'
                  ? `expand · ${String(snap.encodeProgress.textChars)} chars · ${String(snap.encodeProgress.byteLength)} bytes → ${String(snap.encodeProgress.tokenCount)} tok`
                  : `merge #${formatEncodeCountApprox(snap.encodeProgress.mergeIteration)} · ${formatEncodeCountApprox(snap.encodeProgress.tokenCount)} tok`}
              </Text>
              <Text style={{ color: 'gray' }}>
                {`elapsed ${snap.encodeProgress.encodeElapsedSec.toFixed(1)}s · worst-case ≤${formatEncodeCountApprox(snap.encodeProgress.pessimisticMergeUpperBound)} merges · rem ${formatEncodeCountApprox(snap.encodeProgress.pessimisticMergeRemaining)}`}
              </Text>
              {snap.encodeProgress.mergeSubphase === 'merge' ? (
                <>
                  <Box style={{ marginTop: 1 }}>
                    <ProgressBar value={snap.encodeProgress.mergeProgressPct} />
                    <Text
                      style={{ color: 'gray' }}
                    >{`${snap.encodeProgress.mergeProgressPct.toFixed(1)}% (worst-case bound; may finish sooner)`}</Text>
                  </Box>
                  <Text style={{ marginTop: 1 }}>
                    {`${formatMergeThroughput(snap.encodeProgress.mergesPerSec)} · ETA ${formatPessimisticEtaSec(snap.encodeProgress.pessimisticEtaSec)} (pessimistic)`}
                  </Text>
                </>
              ) : (
                <Text style={{ color: 'gray', marginTop: 1 }}>
                  ETA after merge throughput is measured…
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
              {`${trainProgressPct.toFixed(0)}% · wall ${snap.nowElapsedSec.toFixed(0)}s · ETA ${etaStr}`}
            </Text>
          </Box>
          <Box style={{ display: 'flex', flexDirection: 'row', marginTop: 1, flexWrap: 'wrap' }}>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '33%' }}>
              <Text style={{ color: 'yellow' }}>{'Loss'}</Text>
              <Text style={{ fontWeight: 'bold' }}>{lossStr}</Text>
            </Box>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '33%' }}>
              <Text style={{ color: 'green' }}>{'LR'}</Text>
              <Text style={{ fontWeight: 'bold' }}>{lrStr}</Text>
            </Box>
            <Box style={{ display: 'flex', flexDirection: 'column', width: '34%' }}>
              <Text style={{ color: 'blue' }}>{'tok/s ~'}</Text>
              <Text style={{ fontWeight: 'bold' }}>{snap.tokensPerSecApprox.toFixed(0)}</Text>
            </Box>
          </Box>
          <Box style={{ display: 'flex', flexDirection: 'column', marginTop: 1 }}>
            <Text style={{ color: 'gray' }}>{'Loss trend (recent steps)'}</Text>
            <LossSparkline values={snap.recentLosses} />
          </Box>
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

      <Text style={{ color: 'gray', marginTop: 1 }}>
        {'Ctrl+C / q quit · stderr still shows GPU / encode diagnostics'}
      </Text>
    </Box>
  );
}
