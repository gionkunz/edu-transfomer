/**
 * Tiny hand-rolled parser + evaluator for an embedding-arithmetic expression
 * language used by **`llm embeddings expression`**:
 *
 * ```
 * expr   := term (('+' | '-') term)*
 * term   := factor
 * factor := atom | '(' expr ')' | '-' factor
 * atom   := identifier | quotedString
 * ```
 *
 * Lexer notes:
 * - Identifiers match **`[A-Za-z_][A-Za-z0-9_-]*`** (so `co-founder` is one identifier).
 * - **`-`** is an operator only when whitespace-separated, adjacent to **`(`** / **`)`**, or at start of input;
 *   otherwise it's part of an identifier.
 * - Double-quoted strings are atoms (`"new york"`); v1 has no escape sequences.
 *
 * No dependencies; output AST is then evaluated against a `ResolveAtom` callback that
 * turns each atom string into a **`Float32Array`** (typically tokenizer + mean-pooled
 * **`lookupEmbedding`**).
 */

import { EmbeddingDimensionError } from './embeddings';

export class EmbeddingExpressionParseError extends Error {
  readonly position: number;
  constructor(message: string, position: number) {
    super(`${message} (at position ${String(position)})`);
    this.name = 'EmbeddingExpressionParseError';
    this.position = position;
  }
}

export class EmbeddingExpressionEvalError extends Error {
  readonly atom?: string;
  constructor(message: string, atom?: string) {
    super(message);
    this.name = 'EmbeddingExpressionEvalError';
    if (atom !== undefined) this.atom = atom;
  }
}

export type ExprAtom = {
  readonly kind: 'atom';
  readonly value: string;
  readonly position: number;
  /** True when the atom came from a `"..."` literal; false for bare identifiers. */
  readonly quoted: boolean;
};
export type ExprAdd = { readonly kind: 'add'; readonly left: ExprNode; readonly right: ExprNode };
export type ExprSub = { readonly kind: 'sub'; readonly left: ExprNode; readonly right: ExprNode };
export type ExprNeg = { readonly kind: 'neg'; readonly operand: ExprNode };
export type ExprNode = ExprAtom | ExprAdd | ExprSub | ExprNeg;

type TokKind = 'ident' | 'string' | '+' | '-' | '(' | ')';
type Tok = { readonly kind: TokKind; readonly text: string; readonly pos: number };

function isIdentStart(ch: string): boolean {
  return /[A-Za-z_]/.test(ch);
}
function isIdentCont(ch: string): boolean {
  return /[A-Za-z0-9_-]/.test(ch);
}

/**
 * Tokenises `src` per the grammar above. The whitespace-aware **`-`** rule is implemented
 * by tracking whether the previous emitted token (or absence thereof) makes a unary/binary
 * operator the natural reading.
 */
export function tokenizeEmbeddingExpression(src: string): Tok[] {
  const out: Tok[] = [];
  const n = src.length;
  let i = 0;
  while (i < n) {
    const ch = src[i] as string;
    if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r') {
      i++;
      continue;
    }
    if (ch === '+') {
      out.push({ kind: '+', text: '+', pos: i });
      i++;
      continue;
    }
    if (ch === '(') {
      out.push({ kind: '(', text: '(', pos: i });
      i++;
      continue;
    }
    if (ch === ')') {
      out.push({ kind: ')', text: ')', pos: i });
      i++;
      continue;
    }
    if (ch === '"') {
      const start = i;
      i++;
      let acc = '';
      let closed = false;
      while (i < n) {
        const c = src[i] as string;
        if (c === '"') {
          closed = true;
          i++;
          break;
        }
        acc += c;
        i++;
      }
      if (!closed) {
        throw new EmbeddingExpressionParseError('unterminated quoted string', start);
      }
      out.push({ kind: 'string', text: acc, pos: start });
      continue;
    }
    if (ch === '-') {
      // `-` outside of an identifier is always an operator (binary or unary). The identifier
      // lexer below greedily consumes `-` only when the next char is identifier-cont, so when
      // we reach this branch the `-` is structurally an operator (e.g. `(king)-man`,
      // `king - man`, `man -(woman)`, leading `-foo`, etc.).
      out.push({ kind: '-', text: '-', pos: i });
      i++;
      continue;
    }
    if (isIdentStart(ch)) {
      const start = i;
      let acc = ch;
      i++;
      while (i < n) {
        const c = src[i] as string;
        if (c === '-') {
          // Treat `-` as part of the identifier only if the next char would continue the
          // identifier (e.g. `co-founder`); otherwise stop here so the outer lexer can
          // emit `-` as an operator (e.g. `king-(man)`, `king- woman`).
          const next = src[i + 1];
          if (next !== undefined && isIdentCont(next) && next !== '-') {
            acc += c;
            i++;
            continue;
          }
          break;
        }
        if (isIdentCont(c)) {
          acc += c;
          i++;
          continue;
        }
        break;
      }
      out.push({ kind: 'ident', text: acc, pos: start });
      continue;
    }
    throw new EmbeddingExpressionParseError(`unexpected character ${JSON.stringify(ch)}`, i);
  }
  return out;
}

class Parser {
  private readonly toks: Tok[];
  private p = 0;
  constructor(toks: Tok[]) {
    this.toks = toks;
  }
  private peek(): Tok | undefined {
    return this.toks[this.p];
  }
  private consume(): Tok {
    const t = this.toks[this.p];
    if (t === undefined) {
      const last = this.toks[this.toks.length - 1];
      const at = last !== undefined ? last.pos + last.text.length : 0;
      throw new EmbeddingExpressionParseError('unexpected end of input', at);
    }
    this.p++;
    return t;
  }
  private expect(kind: TokKind): Tok {
    const t = this.peek();
    if (t === undefined || t.kind !== kind) {
      const at = t !== undefined ? t.pos : this.endPos();
      throw new EmbeddingExpressionParseError(
        `expected '${kind}', got ${t !== undefined ? `'${t.text}'` : 'end of input'}`,
        at,
      );
    }
    return this.consume();
  }
  private endPos(): number {
    const last = this.toks[this.toks.length - 1];
    return last !== undefined ? last.pos + last.text.length : 0;
  }

  parseExpr(): ExprNode {
    let left = this.parseFactor();
    while (true) {
      const t = this.peek();
      if (t === undefined) break;
      if (t.kind === '+' || t.kind === '-') {
        this.consume();
        const right = this.parseFactor();
        left = t.kind === '+' ? { kind: 'add', left, right } : { kind: 'sub', left, right };
        continue;
      }
      break;
    }
    return left;
  }

  parseFactor(): ExprNode {
    const t = this.peek();
    if (t === undefined) {
      throw new EmbeddingExpressionParseError('unexpected end of input', this.endPos());
    }
    if (t.kind === '-') {
      this.consume();
      const operand = this.parseFactor();
      return { kind: 'neg', operand };
    }
    if (t.kind === '(') {
      this.consume();
      const inner = this.parseExpr();
      this.expect(')');
      return inner;
    }
    if (t.kind === 'ident' || t.kind === 'string') {
      this.consume();
      return { kind: 'atom', value: t.text, position: t.pos, quoted: t.kind === 'string' };
    }
    throw new EmbeddingExpressionParseError(`unexpected token '${t.text}'`, t.pos);
  }

  parseAll(): ExprNode {
    if (this.toks.length === 0) {
      throw new EmbeddingExpressionParseError('empty expression', 0);
    }
    const node = this.parseExpr();
    const t = this.peek();
    if (t !== undefined) {
      throw new EmbeddingExpressionParseError(`trailing token '${t.text}'`, t.pos);
    }
    return node;
  }
}

/** Parses the source string into an AST. Throws {@link EmbeddingExpressionParseError} on malformed input. */
export function parseEmbeddingExpression(src: string): ExprNode {
  const toks = tokenizeEmbeddingExpression(src);
  return new Parser(toks).parseAll();
}

/** Returns every distinct atom string referenced anywhere in `node` (preserves first-seen order). */
export function collectAtoms(node: ExprNode): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const n of collectAtomNodes(node)) {
    if (!seen.has(n.value)) {
      seen.add(n.value);
      out.push(n.value);
    }
  }
  return out;
}

/**
 * Returns every atom node (identifier or quoted phrase) in occurrence order.
 * Preserves duplicates, and exposes the **`quoted`** flag so callers can apply
 * different tokenisation rules to bare vs quoted atoms (e.g. auto-prefix space).
 */
export function collectAtomNodes(node: ExprNode): ExprAtom[] {
  const out: ExprAtom[] = [];
  const walk = (n: ExprNode): void => {
    switch (n.kind) {
      case 'atom':
        out.push(n);
        return;
      case 'neg':
        walk(n.operand);
        return;
      case 'add':
      case 'sub':
        walk(n.left);
        walk(n.right);
        return;
    }
  };
  walk(node);
  return out;
}

/**
 * Resolves an atom (identifier or quoted phrase) to a vector. The optional
 * **`opts.quoted`** mirrors {@link ExprAtom.quoted} so callers can apply
 * different tokenisation rules (e.g. prepending a space to bare identifiers
 * to match GPT-2-style "leading-space" tokens) without touching the parser.
 */
export type ResolveAtom = (atom: string, opts?: { readonly quoted: boolean }) => Float32Array;

function addInPlace(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length !== b.length) {
    throw new EmbeddingDimensionError(a.length, b.length, 'expression operand');
  }
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] as number) + (b[i] as number);
  return out;
}
function subVec(a: Float32Array, b: Float32Array): Float32Array {
  if (a.length !== b.length) {
    throw new EmbeddingDimensionError(a.length, b.length, 'expression operand');
  }
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = (a[i] as number) - (b[i] as number);
  return out;
}
function negVec(a: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = -(a[i] as number);
  return out;
}

/**
 * Evaluates an AST against `resolve`. Throws {@link EmbeddingExpressionEvalError}
 * if an atom resolves to an empty vector or `resolve` itself throws.
 */
export function evaluateEmbeddingExpression(node: ExprNode, resolve: ResolveAtom): Float32Array {
  switch (node.kind) {
    case 'atom': {
      let v: Float32Array;
      try {
        v = resolve(node.value, { quoted: node.quoted });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        throw new EmbeddingExpressionEvalError(
          `failed to resolve atom ${JSON.stringify(node.value)}: ${msg}`,
          node.value,
        );
      }
      if (v.length === 0) {
        throw new EmbeddingExpressionEvalError(
          `atom ${JSON.stringify(node.value)} resolved to an empty vector`,
          node.value,
        );
      }
      return v;
    }
    case 'neg':
      return negVec(evaluateEmbeddingExpression(node.operand, resolve));
    case 'add':
      return addInPlace(
        evaluateEmbeddingExpression(node.left, resolve),
        evaluateEmbeddingExpression(node.right, resolve),
      );
    case 'sub':
      return subVec(
        evaluateEmbeddingExpression(node.left, resolve),
        evaluateEmbeddingExpression(node.right, resolve),
      );
  }
}
