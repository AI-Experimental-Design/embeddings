#!/usr/bin/env python3
"""Run PCA on a TSV: first column is an identifier, remaining columns numeric."""
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    with open(args.input, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        if len(header) < 3:
            parser.error('input must have an ID column and at least two numeric feature columns')
        rows = [row for row in reader if row]

    if len(rows) < 2:
        parser.error('PCA requires at least two rows')
    ids = [row[0] for row in rows]
    try:
        X = np.array([[float(v) for v in row[1:]] for row in rows], dtype=float)
    except ValueError as exc:
        parser.error(f'non-numeric feature value: {exc}')
    if not np.isfinite(X).all():
        parser.error('features must contain finite numbers')

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([header[0], 'pc1', 'pc2'])
        for identifier, (pc1, pc2) in zip(ids, coords):
            writer.writerow([identifier, f'{pc1:.8g}', f'{pc2:.8g}'])
    print(f'PC1: {pca.explained_variance_ratio_[0]:.1%} variance explained')
    print(f'PC2: {pca.explained_variance_ratio_[1]:.1%} variance explained')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
