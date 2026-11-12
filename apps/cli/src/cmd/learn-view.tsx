import React from 'react';
import { Box, Text, useInput, useStdout } from '@wolf-tui/react';

import { FlowPanel, flowPanelContentRows } from './generate-panels';
import { FLOW_PANEL_FRAME_ROWS } from './learn-chapter-stack';
import {
  EXPLANATION_ROWS,
  LESSON_CHROME_ROWS,
  LESSON_MIN_COLUMNS,
  LESSON_MIN_ROWS,
  LESSON_TEXT_WIDTH,
  footerLine,
  headerLines,
  wrapText,
} from './learn-draw';
import {
  isLastStep,
  overallStepIndex,
  type Chapter,
  type LessonContext,
  type LessonMeta,
  type Line,
} from './learn-lesson';
import type { LessonAction, LessonState } from './learn-state';

/**
 * The screen of **`llm learn`**: header, schematic, explanation, key legend.
 * Presentation and key mapping only; state lives in the command and every
 * value drawn comes from the lesson context.
 */

export type LearnVm = {
  readonly chapters: readonly Chapter[];
  readonly meta: LessonMeta;
  readonly state: LessonState;
  /** Everything a step needs except the screen size, which is known only here. */
  readonly context: Omit<LessonContext, 'rows' | 'cols'>;
  /** Whether another token still fits into the model's context. */
  readonly canContinue: boolean;
  /** Shown instead of the lesson while the model runs. */
  readonly busy: string | null;
};

export type LearnViewProps = {
  readonly vm: LearnVm;
  readonly onAction: (action: LessonAction) => void;
  readonly onRedraw: () => void;
  readonly onContinue: () => void;
  readonly onQuit: () => void;
};

function LineText(props: { readonly line: Line }): React.ReactElement {
  if (props.line.length === 0) {
    return <Text> </Text>;
  }
  return (
    <Text style={{ textWrap: 'truncate-end' }}>
      {props.line.map((s, i) => (
        <Text
          key={i}
          style={{
            ...(s.color !== undefined ? { color: s.color } : {}),
            ...(s.bold === true ? { fontWeight: 'bold' } : {}),
          }}
        >
          {s.text}
        </Text>
      ))}
    </Text>
  );
}

function blankRows(count: number, keyPrefix: string): React.ReactElement[] {
  return Array.from({ length: Math.max(0, count) }, (_, i) => (
    <Text key={`${keyPrefix}${String(i)}`}> </Text>
  ));
}

/** Shown until the first forward has finished; only quitting is possible. */
export function LearnBusyView(props: {
  readonly text: string;
  readonly onQuit: () => void;
}): React.ReactElement {
  useInput((input, key) => {
    if ((key.ctrl && input === 'c') || input === 'q') {
      props.onQuit();
    }
  });
  return (
    <Box style={{ display: 'flex', flexDirection: 'column' }}>
      <Text style={{ color: 'gray' }}>{'llm learn'}</Text>
      <Text>{props.text}</Text>
    </Box>
  );
}

export function LearnView(props: LearnViewProps): React.ReactElement {
  const { vm } = props;
  const stdoutCtx = useStdout();
  const cols = stdoutCtx?.stdout.columns ?? LESSON_MIN_COLUMNS;
  const rows = stdoutCtx?.stdout.rows ?? LESSON_MIN_ROWS;

  const { state, meta } = vm;
  const chapter = vm.chapters[state.chapterIndex];
  const step = chapter?.steps[state.stepIndex];
  const last = isLastStep(meta, state.chapterIndex, state.stepIndex);

  useInput((input, key) => {
    if ((key.ctrl && input === 'c') || input === 'q') {
      props.onQuit();
      return;
    }
    if (vm.busy !== null) return;
    if (key.rightArrow) props.onAction({ type: 'next' });
    else if (key.leftArrow) props.onAction({ type: 'prev' });
    else if (key.upArrow) props.onAction({ type: 'dialUp' });
    else if (key.downArrow) props.onAction({ type: 'dialDown' });
    else if (input === ' ') props.onAction({ type: 'togglePlay' });
    else if (input === 'h') props.onAction({ type: 'nextHead' });
    else if (input === 'H') props.onAction({ type: 'prevHead' });
    else if (input === 'k') props.onAction({ type: 'cycleTopK' });
    else if (input === 'r') {
      if (chapter?.samples === true) props.onRedraw();
    } else if (input === 'n') {
      if (last && vm.canContinue) props.onContinue();
    } else if (/^[1-9]$/.test(input)) {
      props.onAction({ type: 'jumpChapter', chapterIndex: Number(input) - 1 });
    }
  });

  if (cols < LESSON_MIN_COLUMNS || rows < LESSON_MIN_ROWS) {
    return (
      <Box style={{ display: 'flex', flexDirection: 'column' }}>
        <Text style={{ color: 'yellow' }}>
          {`llm learn needs a terminal of at least ${String(LESSON_MIN_COLUMNS)}×${String(LESSON_MIN_ROWS)}.`}
        </Text>
        <Text>{`This one is ${String(cols)}×${String(rows)}. Enlarge it to continue, or press q to quit.`}</Text>
      </Box>
    );
  }
  if (vm.busy !== null || chapter === undefined || step === undefined) {
    return (
      <Box style={{ display: 'flex', flexDirection: 'column' }}>
        <Text style={{ color: 'gray' }}>{'llm learn'}</Text>
        <Text>{vm.busy ?? 'nothing to show'}</Text>
      </Box>
    );
  }

  const schematicRows = rows - LESSON_CHROME_ROWS;
  const context: LessonContext = { ...vm.context, rows: schematicRows, cols };
  const view = step.render(context, state.frame);

  const [headerA, headerB] = headerLines(
    {
      chapterNumber: state.chapterIndex + 1,
      chapterCount: vm.chapters.length,
      chapterTitle: chapter.title,
      stepNumber: state.stepIndex + 1,
      stepCount: chapter.steps.length,
      stepTitle: step.title,
      stepsPerChapter: vm.chapters.map((c) => c.steps.length),
      overallIndex: overallStepIndex(meta, state.chapterIndex, state.stepIndex),
      selection: chapter.selectsHead
        ? `block ${String(state.layerIndex + 1)}/${String(meta.nLayers)} head ${String(state.headIndex + 1)}/${String(meta.nHeads)}`
        : step.dial === 'block'
          ? `block ${String(state.layerIndex + 1)}/${String(meta.nLayers)}`
          : '',
      tokenLabels: vm.context.tokenLabels,
    },
    cols,
  );

  let schematic: React.ReactElement[];
  if (view.kind === 'lines') {
    const shown = view.lines.slice(0, schematicRows);
    schematic = [
      ...shown.map((line, i) => <LineText key={`l${String(i)}`} line={line} />),
      ...blankRows(schematicRows - shown.length, 'pad'),
    ];
  } else {
    const used =
      flowPanelContentRows(view.data.flow.layers.length, view.variant) + FLOW_PANEL_FRAME_ROWS;
    schematic = [
      <FlowPanel
        key="flow"
        tier="full"
        variant={view.variant}
        data={view.data}
        labelForId={vm.context.labelForId}
        labelForIndex={(i) => vm.context.tokenLabels[i] ?? ''}
        cols={cols}
        reveal={view.reveal}
      />,
      ...blankRows(schematicRows - used, 'pad'),
    ];
  }

  const explanation = wrapText(view.explanation, LESSON_TEXT_WIDTH).slice(0, EXPLANATION_ROWS);
  const animated =
    !meta.noAnimation &&
    (meta.chapters[state.chapterIndex]?.steps[state.stepIndex]?.frames ?? 1) > 1;

  return (
    <Box style={{ display: 'flex', flexDirection: 'column' }}>
      <LineText line={headerA} />
      <LineText line={headerB} />
      {schematic}
      <Text> </Text>
      {explanation.map((text, i) => (
        <Text key={`e${String(i)}`}>{text}</Text>
      ))}
      {blankRows(EXPLANATION_ROWS - explanation.length, 'epad')}
      <Text style={{ color: 'gray' }}>
        {footerLine(
          {
            dial: step.dial,
            chapterCount: vm.chapters.length,
            selectsHead: chapter.selectsHead,
            samples: chapter.samples,
            animated,
            canContinue: last ? vm.canContinue : null,
          },
          cols,
        )}
      </Text>
    </Box>
  );
}
