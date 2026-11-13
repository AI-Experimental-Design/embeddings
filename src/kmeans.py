#!/usr/bin/env python3
"""Cluster numerical features in a TSV file using k-means.

The first column is a sample identifier. Optional non-feature columns such as
'group' and 'batch' are excluded from fitting and preserved in the output.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', required=True, help='Input TSV')
    parser.add_argument('-o', '--output', required=True, help='Output TSV')
    parser.add_argument('--k', type=int, default=3, help='Number of clusters')
    parser.add_argument('--exclude', default='group,batch',
                        help='Comma-separated non-feature columns to preserve but exclude')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.input, newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        columns = reader.fieldnames
        if not columns or len(columns) < 2:
            parser.error('input must contain an ID column and numerical features')
        rows = list(reader)

    if not rows:
        parser.error('input contains no samples')
    if args.k < 1 or args.k > len(rows):
        parser.error('--k must be between 1 and the number of samples')

    id_column = columns[0]
    excluded = {name.strip() for name in args.exclude.split(',') if name.strip()}
    excluded.add(id_column)
    feature_columns = [name for name in columns if name not in excluded]
    if not feature_columns:
        parser.error('no numerical feature columns found')

    try:
        X = np.array([[float(row[name]) for name in feature_columns]
                      for row in rows], dtype=float)
    except (ValueError, TypeError) as exc:
        parser.error(f'feature columns must be numeric: {exc}')
    if not np.isfinite(X).all():
        parser.error('feature values must be finite')

    model = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    assignments = model.fit_predict(X)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow([id_column, 'cluster'] +
                        [name for name in columns[1:] if name in excluded])
        for row, cluster in zip(rows, assignments):
            writer.writerow([row[id_column], int(cluster)] +
                            [row[name] for name in columns[1:] if name in excluded])

    print(f'{len(rows)} samples, {len(feature_columns)} features, {args.k} clusters')
    print(f'inertia={model.inertia_:.4f}')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
