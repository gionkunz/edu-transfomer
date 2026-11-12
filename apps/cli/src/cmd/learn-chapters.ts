import { attentionChapter } from './learn-chapter-attention';
import { samplingChapter } from './learn-chapter-sampling';
import { stackChapter } from './learn-chapter-stack';
import type { Chapter } from './learn-lesson';

/** The lesson, in teaching order. Chapter numbers are positions in this list. */
export const LESSON_CHAPTERS: readonly Chapter[] = [
  attentionChapter,
  stackChapter,
  samplingChapter,
];
