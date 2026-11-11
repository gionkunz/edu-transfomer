/**
 * Vertical budget for the **`llm generate`** explore view.
 *
 * The view is: header chrome, the streamed text + probability inspector (the
 * "body"), a footer line, and one bottom panel (attention or flow). The panel
 * has a full form whose height depends on the model (one row per head, or one
 * per block) and a short compressed form. This module decides which form fits.
 */

/** Title, mode line, sampled footer, and root padding — excludes body, block selector, panel. */
export const EXPLORE_BASE_CHROME_ROWS = 6;

/** Rows reserved for the transformer block selector row. */
export const BLOCK_SELECTOR_CHROME_ROWS = 1;

/** marginTop gaps between explore sections (block selector, body, panel). */
export const EXPLORE_SECTION_MARGIN_ROWS = 3;

/** Top and bottom border of the bottom panel. */
export const PANEL_BORDER_ROWS = 2;

/** Compressed panel content: one caption line and up to two wrapped lines of entries. */
export const PANEL_COMPRESSED_CONTENT_ROWS = 3;

/** Minimum terminal columns to show the bottom panel at all. */
export const PANEL_MIN_COLUMNS = 40;

/** Body rows that must remain for the full panel; below this the panel compresses. */
export const EXPLORE_MIN_BODY_ROWS_FULL = 5;

/** Body rows that must remain for the compressed panel; below this the panel is hidden. */
export const EXPLORE_MIN_BODY_ROWS = 3;

/** Documented minimum terminal size for the explore view with a compressed panel. */
export const EXPLORE_RECOMMENDED_COLUMNS = 60;
export const EXPLORE_RECOMMENDED_ROWS = 24;

export type PanelTier = 'full' | 'compressed' | 'hidden';

export function blockSelectorRows(showBlockSelector: boolean): number {
  return showBlockSelector ? BLOCK_SELECTOR_CHROME_ROWS : 0;
}

/** Rows the bottom panel occupies, borders included. */
export function panelRows(tier: PanelTier, fullContentRows: number): number {
  if (tier === 'full') return fullContentRows + PANEL_BORDER_ROWS;
  if (tier === 'compressed') return PANEL_COMPRESSED_CONTENT_ROWS + PANEL_BORDER_ROWS;
  return 0;
}

/** Total non-body rows for explore layout budgeting. */
export function exploreChromeRows(
  tier: PanelTier,
  showBlockSelector: boolean,
  fullContentRows: number,
): number {
  return (
    EXPLORE_BASE_CHROME_ROWS +
    EXPLORE_SECTION_MARGIN_ROWS +
    blockSelectorRows(showBlockSelector) +
    panelRows(tier, fullContentRows)
  );
}

/**
 * Largest panel form that still leaves a usable body.
 *
 * **`fullContentRows`** is the content height of the full panel (e.g. one
 * caption row + one row per attention head).
 */
export function panelTier(
  rows: number,
  cols: number,
  showBlockSelector: boolean,
  fullContentRows: number,
): PanelTier {
  if (cols < PANEL_MIN_COLUMNS) return 'hidden';
  const fits = (tier: PanelTier, minBody: number): boolean =>
    rows >= exploreChromeRows(tier, showBlockSelector, fullContentRows) + minBody;
  if (fits('full', EXPLORE_MIN_BODY_ROWS_FULL)) return 'full';
  if (fits('compressed', EXPLORE_MIN_BODY_ROWS)) return 'compressed';
  return 'hidden';
}

/** Terminal rows needed for the full panel of a model with this many heads / blocks. */
export function rowsForFullPanel(showBlockSelector: boolean, fullContentRows: number): number {
  return exploreChromeRows('full', showBlockSelector, fullContentRows) + EXPLORE_MIN_BODY_ROWS_FULL;
}
