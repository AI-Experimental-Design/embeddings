#!/usr/bin/env python3
"""Plot columns from a tab-separated training log.

Based on AI-Experimental-Design/neural-networks/src/plot_interval_training.py.

Example:
    python src/plot_training.py -i out/synthetic/training_loss.tsv \
        -o img/word2vec_loss.png --columns loss
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['legend.numpoints'] = 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Tab-separated training log')
    parser.add_argument('-o', '--output', required=True, help='Output image path')
    parser.add_argument('--columns', default='loss', help='Comma-separated columns to plot')
    parser.add_argument('--colors', default='blue,green,red,magenta,orange,purple')
    parser.add_argument('--plot_width', type=float, default=4)
    parser.add_argument('--plot_height', type=float, default=3)
    parser.add_argument('--title')
    parser.add_argument('--xlabel', default='epoch')
    parser.add_argument('--ylabel', default='loss')
    return parser.parse_args()


def parse_params(path):
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()
        if not line:
            raise SystemExit(f'no header found in {path}')
        header = line.strip().split('\t')
        rows = [dict(zip(header, line.strip().split('\t'))) for line in f
                if line.strip() and not line.startswith('#')]
    return header, rows


def main():
    args = get_args()
    header, rows = parse_params(args.input)
    if not rows:
        raise SystemExit(f'no data rows found in {args.input}')
    if 'epoch' not in header:
        raise SystemExit(f'epoch column not found in {args.input}')
    columns = [c.strip() for c in args.columns.split(',')]
    for col in columns:
        if col not in header:
            raise SystemExit(f'column {col!r} not in {args.input}; have: {", ".join(header)}')
    colors = [c.strip() for c in args.colors.split(',')]
    epochs = [float(row['epoch']) for row in rows]

    fig = plt.figure(figsize=(args.plot_width, args.plot_height), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2)

    lines = []
    for i, col in enumerate(columns):
        values = [float(row[col]) for row in rows]
        line, = ax.plot(epochs, values, '-', color=colors[i % len(colors)], linewidth=1)
        lines.append(line)
    if len(columns) > 1:
        ax.legend(lines, columns, frameon=False, fontsize=8)
    if args.title:
        ax.set_title(args.title, fontsize=8)
    ax.set_xlabel(args.xlabel, fontsize=8)
    ax.set_ylabel(args.ylabel, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
