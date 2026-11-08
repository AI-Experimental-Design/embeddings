#!/usr/bin/env python3
"""Summarize zero-shot scores by region and against the true effects.

Usage:
    python summarize_llr.py -i out/zs.all.tsv --threshold -3
"""
import argparse

import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('--threshold', type=float, default=-3.0,
                   help='call a variant damaging if LLR is below this')
    return p.parse_args()


def auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])


def main():
    args = get_args()
    d = np.genfromtxt(args.input, names=True, dtype=None, encoding=None, delimiter='\t')
    region, llr, eff = d['region'], d['llr'], d['true_effect']
    print(f'{"region":11s} {"n":>6s} {"mean LLR":>9s} {"mean true effect":>17s} '
          f'{"frac LLR < " + str(args.threshold):>16s}')
    for r in ['TATA', 'GC', 'polyA', 'background']:
        k = region == r
        print(f'{r:11s} {k.sum():6d} {llr[k].mean():9.2f} {eff[k].mean():17.2f} '
              f'{np.mean(llr[k] < args.threshold):16.3f}')
    functional = (eff < 0).astype(int)
    print(f'AUC of -LLR for finding variants that lower expression: '
          f'{auc(functional, -llr):.3f}')
    called = llr < args.threshold
    print(f'variants with LLR < {args.threshold}: {called.sum()}, of which '
          f'{functional[called].mean():.1%} lower expression '
          f'({np.mean(region[called] == "polyA"):.1%} are in poly-A runs)')


if __name__ == '__main__':
    main()
