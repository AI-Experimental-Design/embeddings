#!/usr/bin/env python3
"""Plot loss and comparable PCA snapshots from word2vec_train.py outputs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Compact, uncluttered figures matching the course's plotting conventions.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.facecolor': 'white',
})

GROUPS = {
    'animals': ('cat', 'dog', 'horse'),
    'vehicles': ('car', 'truck', 'bus'),
    'fruits': ('apple', 'banana', 'orange'),
    'temperature': ('hot', 'cold'),
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='out/synthetic', help='Training output directory')
    p.add_argument('--out', default='img', help='Figure output directory')
    p.add_argument('--epochs', default='0,1,3,12', help='Comma-separated snapshot epochs')
    p.add_argument('--words', default='', help='Optional comma-separated words to label (default: synthetic groups)')
    args = p.parse_args()
    src, dst = Path(args.input), Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(src / 'embedding_history.tsv', sep='\t')
    losses = pd.read_csv(src / 'training_loss.tsv', sep='\t')
    dimensions = [c for c in history.columns if c.startswith('dim')]
    if not dimensions:
        p.error('No embedding dimensions found')
    available = sorted(history.epoch.unique().tolist())
    requested = [int(x.strip()) for x in args.epochs.split(',') if x.strip()]
    epochs = list(dict.fromkeys(x for x in requested if x in available))
    if not epochs:
        p.error(f'No requested epochs available; found {available}')
    # Fit once on all snapshots, rather than rotating PCA independently at each epoch.
    pca = PCA(n_components=2)
    coords = pca.fit_transform(history[dimensions].to_numpy(dtype=float))
    history['pc1'], history['pc2'] = coords[:, 0], coords[:, 1]
    xpad = max(np.ptp(coords[:, 0]) * .07, .25)
    ypad = max(np.ptp(coords[:, 1]) * .07, .25)
    xlim = (coords[:, 0].min() - xpad, coords[:, 0].max() + xpad)
    ylim = (coords[:, 1].min() - ypad, coords[:, 1].max() + ypad)
    selected = set(x.strip() for x in args.words.split(',') if x.strip()) if args.words else set(sum((list(v) for v in GROUPS.values()), []))
    palette = {'animals':'tab:blue', 'vehicles':'tab:orange', 'fruits':'tab:green', 'temperature':'tab:red'}
    word_group = {w:g for g, ws in GROUPS.items() for w in ws}
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(losses.epoch, losses.loss, marker='o', markersize=2.5, linewidth=1.2)
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training loss')
    fig.tight_layout()
    fig.savefig(dst / 'word2vec_loss.png', dpi=200)
    plt.close(fig)
    for epoch in epochs:
        subset = history[history.epoch == epoch]
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        for group, group_words in GROUPS.items():
            part = subset[subset.word.isin(group_words)]
            if not part.empty:
                ax.scatter(part.pc1, part.pc2, label=group, s=22, color=palette[group], zorder=3)
        other = subset[~subset.word.isin(word_group)]
        if not other.empty:
            ax.scatter(other.pc1, other.pc2, color='0.78', s=9, alpha=.6, label='other words')
        for row in subset.itertuples():
            if row.word in selected:
                ax.annotate(row.word, (row.pc1, row.pc2), xytext=(4,4), textcoords='offset points', fontsize=7)
        ax.set(xlim=xlim, ylim=ylim, xlabel='PC1', ylabel='PC2', title=f'Epoch {epoch}')
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, markerscale=.8)
        fig.subplots_adjust(left=.18, right=.98, top=.89, bottom=.28)
        fig.savefig(dst / f'word2vec_epoch_{epoch:03d}.png', dpi=200)
        plt.close(fig)
    history[['epoch', 'word', 'pc1', 'pc2']].to_csv(dst / 'word2vec_pca.tsv', sep='\t', index=False)
    print(f'Wrote loss plot, {len(epochs)} PCA snapshots, and coordinates to {dst}')
    print('PCA variance explained:', ', '.join(f'{v:.1%}' for v in pca.explained_variance_ratio_))

if __name__ == '__main__':
    main()
