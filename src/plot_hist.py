
#!/usr/bin/env python3
"""Histogram of one column of a TSV file or stdin, with optional marker lines.

Usage:
    python src/plot_hist.py -i out/null.tsv -c shuffled_accuracy \
        --vline 0.533 -o img/null.png

    cat out/null.tsv | python src/plot_hist.py \
        -c shuffled_accuracy --vline 0.533 -o img/null.png

    cat out/null.tsv | python src/plot_hist.py \
        -c shuffled_accuracy --xmin 0 --xmax 1 -o img/null.png
"""

import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', default='-',
                   help='Input TSV file (default: stdin)')
    p.add_argument('-o', '--output', required=True)
    p.add_argument('-c', '--column', required=True)
    p.add_argument('--group', help='column that splits values into overlaid histograms')
    p.add_argument('--bins', type=int, default=30)
    p.add_argument('--vline', type=float, nargs='*', default=[])
    p.add_argument('--vline_label', nargs='*')
    p.add_argument('--xlabel')
    p.add_argument('--title')
    p.add_argument('--xmin', type=float, help='Minimum x-axis value')
    p.add_argument('--xmax', type=float, help='Maximum x-axis value')
    p.add_argument('--width', type=float, default=3.5)
    p.add_argument('--height', type=float, default=2.2)
    return p.parse_args()


def main():
    args = get_args()

    if args.input == '-':
        lines = [
            l.rstrip('\n').split('\t')
            for l in sys.stdin
            if not l.startswith('#')
        ]
    else:
        with open(args.input) as f:
            lines = [
                l.rstrip('\n').split('\t')
                for l in f
                if not l.startswith('#')
            ]

    h, rows = lines[0], lines[1:]
    v = np.array([float(r[h.index(args.column)]) for r in rows])

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=300)
    edges = np.histogram_bin_edges(v, bins=args.bins)

    if args.group:
        g = np.array([r[h.index(args.group)] for r in rows])
        for i, name in enumerate(sorted(set(g))):
            ax.hist(
                v[g == name],
                bins=edges,
                alpha=0.6,
                label=name,
                color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'][i % 4]
            )
        ax.legend(frameon=False, fontsize=6)
    else:
        ax.hist(v, bins=edges, color='tab:gray')

    labels = args.vline_label or [None] * len(args.vline)
    for x, lab in zip(args.vline, labels):
        ax.axvline(x, color='tab:red', lw=1)
        if lab:
            ax.text(
                x, ax.get_ylim()[1] * 0.95,
                ' ' + lab,
                color='tab:red',
                fontsize=6,
                va='top'
            )

    ax.set_xlabel(args.xlabel or args.column, fontsize=8)
    ax.set_ylabel('count', fontsize=8)

    if args.title:
        ax.set_title(args.title, fontsize=8)

    if args.xmin is not None or args.xmax is not None:
        ax.set_xlim(left=args.xmin, right=args.xmax)

    ax.tick_params(labelsize=7, width=0.5, length=2)

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    for s in ['bottom', 'left']:
        ax.spines[s].set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()

