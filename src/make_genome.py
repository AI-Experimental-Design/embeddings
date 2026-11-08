#!/usr/bin/env python3
"""Generate unlabeled DNA sequences with planted elements.

Each sequence is random background with up to three kinds of element
planted at random positions.

    TATA box    TATAAA      functional, raises expression
    GC box      GGGCGG      functional, raises expression
    poly-A run  AAAAAAAAAA  common but has no function

Each element appears in a sequence with probability --p_element. The
sequences carry no labels. A model pretrained on them only ever sees the
sequence.

The file also records each sequence's true expression,

    expression = 1 + 2 * (TATA box present) + 1 * (GC box present) + noise

which pretraining never uses. Later lectures use it as a label.

Usage:
    python make_genome.py --n 20000 --out out/genome.tsv
"""
import argparse

import numpy as np

ELEMENTS = {'TATA': 'TATAAA', 'GC': 'GGGCGG', 'polyA': 'AAAAAAAAAA'}
WEIGHTS = {'TATA': 2.0, 'GC': 1.0, 'polyA': 0.0}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=20000)
    p.add_argument('--length', type=int, default=100)
    p.add_argument('--p_element', type=float, default=0.5)
    p.add_argument('--elements', default='TATA,GC,polyA')
    p.add_argument('--noise', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    return p.parse_args()


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)
    use = args.elements.split(',')
    counts = {e: 0 for e in use}
    with open(args.out, 'w') as f:
        f.write('id\tseq\texpression\telements\n')
        for n in range(args.n):
            seq = list(rng.choice(list('ACGT'), size=args.length))
            placed, taken, expr = [], np.zeros(args.length, bool), 1.0
            for e in use:
                if rng.random() < args.p_element:
                    m = ELEMENTS[e]
                    for _ in range(100):
                        pos = rng.integers(0, args.length - len(m) + 1)
                        if not taken[max(0, pos - 2):pos + len(m) + 2].any():
                            break
                    seq[pos:pos + len(m)] = list(m)
                    taken[pos:pos + len(m)] = True
                    placed.append(f'{e}:{pos}')
                    expr += WEIGHTS[e]
                    counts[e] += 1
            expr += rng.normal(0, args.noise)
            f.write(f'seq{n:05d}\t{"".join(seq)}\t{expr:.4f}\t{",".join(placed) or "none"}\n')
    print(f'wrote {args.out}: {args.n} sequences of {args.length} bp')
    for e in use:
        print(f'  {e:6s} {ELEMENTS[e]:12s} in {counts[e] / args.n:.1%} of sequences')


if __name__ == '__main__':
    main()
