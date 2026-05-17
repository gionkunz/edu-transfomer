import { describe, expect, it } from 'vitest';

import {
  collectAtomNodes,
  collectAtoms,
  EmbeddingExpressionEvalError,
  EmbeddingExpressionParseError,
  evaluateEmbeddingExpression,
  parseEmbeddingExpression,
  tokenizeEmbeddingExpression,
  type ResolveAtom,
} from './embedding-expression';

function staticResolver(map: Record<string, number[]>): ResolveAtom {
  return (atom: string) => {
    const v = map[atom];
    if (v === undefined) {
      throw new Error(`unknown atom ${atom}`);
    }
    return Float32Array.from(v);
  };
}

describe('embedding-expression: lexer', () => {
  it('tokenises whitespace-separated +/-', () => {
    const t = tokenizeEmbeddingExpression('king - man + woman');
    expect(t.map((x) => `${x.kind}:${x.text}`)).toEqual([
      'ident:king',
      '-:-',
      'ident:man',
      '+:+',
      'ident:woman',
    ]);
  });

  it('keeps hyphenated identifier intact (co-founder)', () => {
    const t = tokenizeEmbeddingExpression('co-founder + ceo');
    expect(t[0]?.kind).toBe('ident');
    expect(t[0]?.text).toBe('co-founder');
    expect(t[1]?.kind).toBe('+');
    expect(t[2]?.text).toBe('ceo');
  });

  it('splits `king-(man)` into 4 tokens (ident, -, (, ident, ))', () => {
    const t = tokenizeEmbeddingExpression('king-(man)');
    expect(t.map((x) => x.kind)).toEqual(['ident', '-', '(', 'ident', ')']);
  });

  it('treats leading `-` as operator (unary)', () => {
    const t = tokenizeEmbeddingExpression('-king');
    expect(t.map((x) => x.kind)).toEqual(['-', 'ident']);
  });

  it('parses quoted multi-word atom as a single string token', () => {
    const t = tokenizeEmbeddingExpression('"new york" - usa');
    expect(t[0]?.kind).toBe('string');
    expect(t[0]?.text).toBe('new york');
    expect(t[1]?.kind).toBe('-');
    expect(t[2]?.text).toBe('usa');
  });

  it('errors on unterminated string', () => {
    expect(() => tokenizeEmbeddingExpression('"new york')).toThrow(EmbeddingExpressionParseError);
  });

  it('errors on unexpected character', () => {
    expect(() => tokenizeEmbeddingExpression('king $ man')).toThrow(EmbeddingExpressionParseError);
  });
});

describe('embedding-expression: parser', () => {
  it('parses left-to-right associatively', () => {
    const ast = parseEmbeddingExpression('a + b - c');
    expect(ast.kind).toBe('sub');
    if (ast.kind !== 'sub') throw new Error('unreachable');
    expect(ast.left.kind).toBe('add');
    expect(ast.right.kind).toBe('atom');
  });

  it('handles parentheses', () => {
    const ast = parseEmbeddingExpression('a - (b + c)');
    expect(ast.kind).toBe('sub');
    if (ast.kind !== 'sub') throw new Error('unreachable');
    expect(ast.right.kind).toBe('add');
  });

  it('handles unary minus', () => {
    const ast = parseEmbeddingExpression('-king');
    expect(ast.kind).toBe('neg');
  });

  it('rejects empty input', () => {
    expect(() => parseEmbeddingExpression('   ')).toThrow(EmbeddingExpressionParseError);
  });

  it('rejects trailing operator', () => {
    expect(() => parseEmbeddingExpression('king +')).toThrow(EmbeddingExpressionParseError);
  });

  it('rejects mismatched parens', () => {
    expect(() => parseEmbeddingExpression('(king + man')).toThrow(EmbeddingExpressionParseError);
    expect(() => parseEmbeddingExpression('king + man)')).toThrow(EmbeddingExpressionParseError);
  });

  it('parser error carries a character position', () => {
    try {
      parseEmbeddingExpression('king +');
      throw new Error('expected throw');
    } catch (e) {
      expect(e).toBeInstanceOf(EmbeddingExpressionParseError);
      expect((e as EmbeddingExpressionParseError).position).toBeGreaterThanOrEqual(5);
    }
  });
});

describe('embedding-expression: evaluator', () => {
  const resolver = staticResolver({
    king: [1, 0, 0],
    man: [0, 1, 0],
    woman: [0, 0, 1],
    queen: [1, -1, 1],
    'co-founder': [2, 2, 2],
    ceo: [3, 3, 3],
    'new york': [5, 5, 5],
    usa: [1, 1, 1],
  });

  it('evaluates basic addition and subtraction', () => {
    const ast = parseEmbeddingExpression('king - man + woman');
    const v = evaluateEmbeddingExpression(ast, resolver);
    expect(Array.from(v)).toEqual([1, -1, 1]);
  });

  it('handles unary minus and parens', () => {
    const ast = parseEmbeddingExpression('-(man) + woman + king');
    const v = evaluateEmbeddingExpression(ast, resolver);
    expect(Array.from(v)).toEqual([1, -1, 1]);
  });

  it('resolves quoted multi-word atom', () => {
    const ast = parseEmbeddingExpression('"new york" - usa');
    const v = evaluateEmbeddingExpression(ast, resolver);
    expect(Array.from(v)).toEqual([4, 4, 4]);
  });

  it('resolves hyphen identifier as a single atom', () => {
    const ast = parseEmbeddingExpression('co-founder + ceo');
    const v = evaluateEmbeddingExpression(ast, resolver);
    expect(Array.from(v)).toEqual([5, 5, 5]);
  });

  it('throws typed error for unknown atom', () => {
    const ast = parseEmbeddingExpression('frobnicator');
    expect(() => evaluateEmbeddingExpression(ast, resolver)).toThrow(EmbeddingExpressionEvalError);
  });

  it('throws on empty resolved vector', () => {
    const empty: ResolveAtom = () => new Float32Array(0);
    const ast = parseEmbeddingExpression('king');
    expect(() => evaluateEmbeddingExpression(ast, empty)).toThrow(EmbeddingExpressionEvalError);
  });
});

describe('embedding-expression: AST quoted flag', () => {
  it('marks bare identifiers as quoted=false and quoted strings as quoted=true', () => {
    const ast = parseEmbeddingExpression('king + "queen" - "new york"');
    const nodes = collectAtomNodes(ast);
    expect(nodes).toHaveLength(3);
    expect(nodes[0]?.value).toBe('king');
    expect(nodes[0]?.quoted).toBe(false);
    expect(nodes[1]?.value).toBe('queen');
    expect(nodes[1]?.quoted).toBe(true);
    expect(nodes[2]?.value).toBe('new york');
    expect(nodes[2]?.quoted).toBe(true);
  });

  it('passes the quoted flag to the resolver via opts.quoted', () => {
    const seen: { atom: string; quoted: boolean }[] = [];
    const resolver: ResolveAtom = (atom, atomOpts) => {
      seen.push({ atom, quoted: atomOpts?.quoted === true });
      return new Float32Array([atom.length]);
    };
    const ast = parseEmbeddingExpression('king + "queen"');
    evaluateEmbeddingExpression(ast, resolver);
    expect(seen).toEqual([
      { atom: 'king', quoted: false },
      { atom: 'queen', quoted: true },
    ]);
  });
});

describe('embedding-expression: collectAtoms', () => {
  it('returns distinct atoms in first-seen order', () => {
    const ast = parseEmbeddingExpression('king - man + woman + king');
    expect(collectAtoms(ast)).toEqual(['king', 'man', 'woman']);
  });
  it('includes quoted phrases verbatim', () => {
    const ast = parseEmbeddingExpression('"new york" - usa + "new york"');
    expect(collectAtoms(ast)).toEqual(['new york', 'usa']);
  });
});
