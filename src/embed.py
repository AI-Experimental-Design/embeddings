#!/usr/bin/env python3
"""Run sequences through the pretrained model and save their representations,
reduced to two dimensions with principal component analysis (PCA) so they
can be plotted.

--level position   one 64-number vector per base, labeled by the element
                   covering that base (or background)
--level sequence   one vector per sequence, the average over its positions,
                   labeled by which elements the sequence contains

PCA finds the directions in which the vectors vary the most. PC1 is the
direction of greatest variation, PC2 the next, at right angles to PC1.

Usage:
    python embed.py --model out/pretrained.pt --genome out/genome.tsv --level position -o out/pos.pca.tsv
    python embed.py --model out/pretrained.pt --genome out/genome.tsv --level sequence -o out/seq.pca.tsv
"""
import argparse

import numpy as np
import torch

import dna


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--genome', required=True)
    p.add_argument('--level', choices=['position', 'sequence'], required=True)
    p.add_argument('--n', type=int, default=500, help='sequences to use (from the start of the file)')
    p.add_argument('--per_class', type=int, default=300,
                   help='position level: sample this many positions of each label')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-o', '--output', required=True)
    return p.parse_args()


def pca2(H):
    H = H - H.mean(0)
    _, s, vt = np.linalg.svd(H, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    return H @ vt[:2].T, var[:2]


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)
    model = dna.load_model(args.model)
    rows = dna.read_genome(args.genome)[:args.n]
    x = dna.encode([r[1] for r in rows])
    with torch.no_grad():
        H = model.hidden(x).numpy()
    if args.level == 'position':
        lab = dna.element_mask(rows, x.shape[1])
        keep = []
        for e in ['background', 'TATA', 'GC', 'polyA']:
            idx = np.argwhere(lab == e)
            idx = idx[rng.choice(len(idx), size=min(args.per_class, len(idx)), replace=False)]
            keep += [(i, j, e) for i, j in idx]
        V = np.array([H[i, j] for i, j, _ in keep])
        labels = [e for _, _, e in keep]
    else:
        V = H.mean(axis=1)
        labels = []
        for r in rows:
            els = sorted(item.split(':')[0] for item in r[3].split(',')) if r[3] != 'none' else []
            labels.append('+'.join(els) or 'none')
    Z, var = pca2(V)
    with open(args.output, 'w') as f:
        f.write('pc1\tpc2\tlabel\n')
        for z, l in zip(Z, labels):
            f.write(f'{z[0]:.4f}\t{z[1]:.4f}\t{l}\n')
    print(f'{len(V)} {args.level} vectors of length {V.shape[1]}')
    print(f'PC1 explains {var[0]:.1%} of the variance, PC2 {var[1]:.1%}')


if __name__ == '__main__':
    main()
