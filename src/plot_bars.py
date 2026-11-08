#!/usr/bin/env python3
"""Grouped bar chart from a TSV file.

Rows are categories along the x axis (the --x column). Each --y column is one
bar within a category.

Usage:
    python plot_bars.py -i out/auc.tsv -x benchmark -y gene_model,model_a -o img/auc.png
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:gray']


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('-x', '--x_col', required=True)
    p.add_argument('-y', '--y_cols', required=True)
    p.add_argument('--labels')
    p.add_argument('--ylabel')
    p.add_argument('--title')
    p.add_argument('--y_min', type=float)
    p.add_argument('--y_max', type=float)
    p.add_argument('--hline', type=float, nargs='*', default=[])
    p.add_argument('--width', type=float, default=4)
    p.add_argument('--height', type=float, default=2.5)
    return p.parse_args()


def main():
    args = get_args()
    lines = [l.rstrip('\n').split('\t') for l in open(args.input) if not l.startswith('#')]
    h, rows = lines[0], lines[1:]
    cats = [r[h.index(args.x_col)].replace('\\n', '\n') for r in rows]
    ys = args.y_cols.split(',')
    labels = args.labels.split(',') if args.labels else ys
    width = 0.8 / len(ys)
    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=300)
    x = np.arange(len(cats))
    for i, (y, lab) in enumerate(zip(ys, labels)):
        vals = [float(r[h.index(y)]) for r in rows]
        bars = ax.bar(x + (i - (len(ys) - 1) / 2) * width, vals, width,
                      color=COLORS[i % len(COLORS)], label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}', ha='center',
                    va='bottom', fontsize=5)
    for hv in args.hline:
        ax.axhline(hv, lw=0.5, ls='--', color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=6)
    if args.y_min is not None or args.y_max is not None:
        ax.set_ylim(args.y_min, args.y_max)
    if args.ylabel:
        ax.set_ylabel(args.ylabel, fontsize=8)
    if args.title:
        ax.set_title(args.title, fontsize=8)
    if len(ys) > 1:
        ax.legend(frameon=False, fontsize=6)
    ax.tick_params(axis='y', labelsize=7, width=0.5, length=2)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()
