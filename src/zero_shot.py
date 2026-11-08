#!/usr/bin/env python3
"""Score variants with a pretrained model and no labels.

For each position, hide the base (mask it) and ask the model for the
probability of each of the four bases given the rest of the sequence. A
variant's score is the log-likelihood ratio

    LLR = log P(alt) - log P(ref)

A negative LLR means the model finds the alternate base less expected than
the reference base in this context. ESM-1v and GPN score variants this way.

The script also computes each variant's true effect on expression, using
the rule the sequences were generated with,

    expression = 1 + 2 * (TATAAA present) + 1 * (GGGCGG present)

Usage:
    python zero_shot.py --model out/pretrained.pt --genome out/heldout.tsv --index 0 -o out/zs.seq0
    python zero_shot.py --model out/pretrained.pt --genome out/heldout.tsv --all -o out/zs.all
"""
import argparse

import numpy as np
import torch

import dna


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--genome', required=True)
    p.add_argument('--index', type=int, default=0, help='which sequence to score')
    p.add_argument('--all', action='store_true', help='score every sequence in the file')
    p.add_argument('-o', '--out_prefix', required=True)
    return p.parse_args()


def true_expression(s):
    return 1 + 2 * ('TATAAA' in s) + 1 * ('GGGCGG' in s)


def score_sequence(model, seq):
    """Returns a (length, 4) array of log P(base) with that position masked."""
    x = dna.encode([seq])[0]
    L = len(seq)
    batch = x.repeat(L, 1)
    batch[torch.arange(L), torch.arange(L)] = dna.MASK
    with torch.no_grad():
        logp = torch.log_softmax(model(batch), -1)
    return logp[torch.arange(L), torch.arange(L)].numpy()


def main():
    args = get_args()
    model = dna.load_model(args.model)
    rows = dna.read_genome(args.genome)
    if not args.all:
        rows = [rows[args.index]]
    lab = dna.element_mask(rows, len(rows[0][1]))
    with open(args.out_prefix + '.tsv', 'w') as out:
        out.write('seq\tpos\tref\talt\tregion\tllr\ttrue_effect\n')
        for n, r in enumerate(rows):
            seq = r[1]
            logp = score_sequence(model, seq)
            base = true_expression(seq)
            M = np.zeros((4, len(seq)))
            for i, ref in enumerate(seq):
                for j, alt in enumerate(dna.BASES):
                    if alt == ref:
                        continue
                    llr = logp[i, j] - logp[i, dna.BASES.index(ref)]
                    eff = true_expression(seq[:i] + alt + seq[i + 1:]) - base
                    M[j, i] = llr
                    out.write(f'{r[0]}\t{i}\t{ref}\t{alt}\t{lab[n, i]}\t{llr:.4f}\t{eff:.1f}\n')
    if not args.all:
        with open(args.out_prefix + '.matrix.tsv', 'w') as f:
            f.write('base\t' + '\t'.join(f'{i}{b}' for i, b in enumerate(seq)) + '\n')
            for j, a in enumerate(dna.BASES):
                f.write(a + '\t' + '\t'.join(f'{v:.4f}' for v in M[j]) + '\n')
        print(f'{rows[0][0]} elements: {rows[0][3]}')
    print(f'scored {3 * len(rows[0][1]) * len(rows)} variants in {len(rows)} sequences')


if __name__ == '__main__':
    main()
