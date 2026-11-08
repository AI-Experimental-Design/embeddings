#!/usr/bin/env python3
"""Plot one or more columns of a TSV file against another column.

Reads a tab-separated file with a header line. Lines starting with # are
skipped. Each --y column becomes one series. With --group, the rows are
split by the values in that column and each group gets its own color.

Usage:
    python plot_tsv.py -i out/curve.tsv -x epoch -y train_mse,val_mse -o img/curve.png --ylog
    python plot_tsv.py -i out/points.tsv -x pc1 -y pc2 --group label --kind scatter -o img/pca.png
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('-x', '--x_col', required=True)
    p.add_argument('-y', '--y_cols', required=True,
                   help='comma-separated column names')
    p.add_argument('--labels', help='comma-separated legend labels for the y columns')
    p.add_argument('--group', help='column whose values split rows into colored groups')
    p.add_argument('--kind', choices=['line', 'scatter', 'linepoint', 'bar'], default='line')
    p.add_argument('--xlabel')
    p.add_argument('--ylabel')
    p.add_argument('--title')
    p.add_argument('--xlog', action='store_true')
    p.add_argument('--ylog', action='store_true')
    p.add_argument('--y_min', type=float)
    p.add_argument('--y_max', type=float)
    p.add_argument('--x_min', type=float)
    p.add_argument('--x_max', type=float)
    p.add_argument('--hline', type=float, nargs='*', default=[],
                   help='draw horizontal reference lines at these y values')
    p.add_argument('--vline', type=float, nargs='*', default=[],
                   help='draw vertical reference lines at these x values')
    p.add_argument('--diag', action='store_true', help='draw the y = x line')
    p.add_argument('--point_size', type=float, default=6)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--width', type=float, default=3.5)
    p.add_argument('--height', type=float, default=2.5)
    p.add_argument('--no_legend', action='store_true')
    return p.parse_args()


def read_tsv(path):
    header = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            a = line.rstrip('\n').split('\t')
            if header is None:
                header = a
            else:
                rows.append(a)
    cols = {}
    for i, h in enumerate(header):
        vals = [r[i] for r in rows]
        try:
            cols[h] = np.array([float(v) for v in vals])
        except ValueError:
            cols[h] = np.array(vals)
    return cols


def draw(ax, x, y, kind, color, label, args):
    if kind == 'scatter':
        ax.scatter(x, y, s=args.point_size, color=color, label=label,
                   alpha=args.alpha, linewidths=0)
    elif kind == 'linepoint':
        ax.plot(x, y, '-o', color=color, label=label, lw=1, ms=args.point_size / 2,
                alpha=args.alpha)
    elif kind == 'bar':
        ax.bar(x, y, color=color, label=label, alpha=args.alpha)
    else:
        ax.plot(x, y, '-', color=color, label=label, lw=1, alpha=args.alpha)


def main():
    args = get_args()
    cols = read_tsv(args.input)
    y_cols = args.y_cols.split(',')
    labels = args.labels.split(',') if args.labels else y_cols

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=300)
    x = cols[args.x_col]
    n_series = 0
    if args.group:
        groups = cols[args.group]
        for gi, g in enumerate(sorted(set(groups.tolist()), key=str)):
            m = groups == g
            for y_col in y_cols:
                draw(ax, x[m], cols[y_col][m], args.kind, COLORS[gi % len(COLORS)],
                     f'{args.group}={g}' if len(y_cols) == 1 else f'{y_col} {g}', args)
                n_series += 1
    else:
        for i, (y_col, label) in enumerate(zip(y_cols, labels)):
            draw(ax, x, cols[y_col], args.kind, COLORS[i % len(COLORS)], label, args)
            n_series += 1

    for h in args.hline:
        ax.axhline(h, lw=0.5, ls='--', color='gray')
    for v in args.vline:
        ax.axvline(v, lw=0.5, ls='--', color='gray')
    if args.diag:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], '--', lw=0.5, color='gray')
    if args.xlog:
        ax.set_xscale('log')
    if args.ylog:
        ax.set_yscale('log')
    if args.y_min is not None or args.y_max is not None:
        ax.set_ylim(args.y_min, args.y_max)
    if args.x_min is not None or args.x_max is not None:
        ax.set_xlim(args.x_min, args.x_max)
    ax.set_xlabel(args.xlabel if args.xlabel is not None else args.x_col, fontsize=8)
    if args.ylabel is not None:
        ax.set_ylabel(args.ylabel, fontsize=8)
    if args.title:
        ax.set_title(args.title, fontsize=8)
    if n_series > 1 and not args.no_legend:
        ax.legend(frameon=False, fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7, width=0.5, length=2)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()
