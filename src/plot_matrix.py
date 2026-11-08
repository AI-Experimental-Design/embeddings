#!/usr/bin/env python3
"""Draw a matrix as a heatmap.

The input is a TSV file. With --row_labels the first column holds row
names, and with --col_labels the first line holds column names. Diverging
data (centered on zero) gets a red/blue color map by default, matching the
convolution lecture. Use --cmap viridis for data that are all positive,
like a spectrogram or attention weights.

Usage:
    python plot_matrix.py -i out/ism.tsv -o img/ism.png --row_labels --col_labels
    python plot_matrix.py -i out/spec.tsv -o img/spec.png --cmap viridis --origin lower
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--row_labels', action='store_true')
    p.add_argument('--col_labels', action='store_true')
    p.add_argument('--cmap', default='RdBu')
    p.add_argument('--vmin', type=float)
    p.add_argument('--vmax', type=float)
    p.add_argument('--origin', choices=['upper', 'lower'], default='upper')
    p.add_argument('--xlabel')
    p.add_argument('--ylabel')
    p.add_argument('--title')
    p.add_argument('--extent', type=float, nargs=4,
                   help='x_min x_max y_min y_max, for axes in real units')
    p.add_argument('--colorbar', action='store_true')
    p.add_argument('--every_col', type=int, default=1,
                   help='only label every Nth column')
    p.add_argument('--width', type=float, default=4)
    p.add_argument('--height', type=float, default=2)
    return p.parse_args()


def main():
    args = get_args()
    rows, row_names, col_names = [], [], None
    with open(args.input) as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            a = line.rstrip('\n').split('\t')
            if args.col_labels and col_names is None:
                col_names = a[1:] if args.row_labels else a
                continue
            if args.row_labels:
                row_names.append(a[0])
                a = a[1:]
            rows.append([float(v) for v in a])
    m = np.array(rows)

    vmin, vmax = args.vmin, args.vmax
    if args.cmap == 'RdBu' and vmin is None and vmax is None:
        lim = np.abs(m).max()
        vmin, vmax = -lim, lim

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=300)
    im = ax.imshow(m, cmap=args.cmap, vmin=vmin, vmax=vmax, aspect='auto',
                   origin=args.origin, extent=args.extent,
                   interpolation='nearest')
    if args.extent is None:
        if col_names:
            idx = list(range(0, len(col_names), args.every_col))
            ax.set_xticks(idx)
            ax.set_xticklabels([col_names[i] for i in idx], fontsize=5)
        else:
            ax.set_xticks([])
        if row_names:
            ax.set_yticks(range(len(row_names)))
            ax.set_yticklabels(row_names, fontsize=6)
        else:
            ax.set_yticks([])
    else:
        ax.tick_params(axis='both', labelsize=7, width=0.5, length=2)
    if args.xlabel:
        ax.set_xlabel(args.xlabel, fontsize=8)
    if args.ylabel:
        ax.set_ylabel(args.ylabel, fontsize=8)
    if args.title:
        ax.set_title(args.title, fontsize=8)
    if args.colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()
