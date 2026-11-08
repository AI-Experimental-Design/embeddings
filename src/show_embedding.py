#!/usr/bin/env python3
"""Print a model's embedding table, one row per token, and the distances
between rows.

Usage:
    python show_embedding.py --model out/pretrained.pt
"""
import argparse

import numpy as np

import dna


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    return p.parse_args()


def main():
    args = get_args()
    E = dna.load_model(args.model).embed.weight.detach().numpy()
    names = list(dna.BASES) + ['mask']
    print('token ' + ' '.join(f'{f"dim{j}":>6s}' for j in range(E.shape[1])))
    for n, row in zip(names, E):
        print(f'{n:5s} ' + ' '.join(f'{v:6.2f}' for v in row))
    print('\ndistance between token vectors')
    print('      ' + ' '.join(f'{n:>5s}' for n in names))
    for n, a in zip(names, E):
        print(f'{n:5s} ' + ' '.join(f'{np.linalg.norm(a - b):5.2f}' for b in E))


if __name__ == '__main__':
    main()
