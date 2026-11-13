
#!/usr/bin/env python3
"""Reduce numerical features to two dimensions using UMAP.

The first column is the sample ID. All remaining columns are
numerical features.

Examples:
    python src/umap.py -i out/clusters_features.tsv \
        -o out/clusters_umap.tsv

    cat out/clusters_features.tsv | python src/umap.py \
        --n_neighbors 15 --min_dist 0.1 \
        > out/clusters_umap.tsv
"""

import argparse
import csv
import sys

import numpy as np
import umap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", default="-",
        help="Input TSV file (default: stdin)"
    )
    parser.add_argument(
        "-o", "--output", default="-",
        help="Output TSV file (default: stdout)"
    )
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input == "-":
        rows = list(csv.reader(sys.stdin, delimiter="\t"))
    else:
        with open(args.input) as f:
            rows = list(csv.reader(f, delimiter="\t"))

    if len(rows) < 4:
        raise SystemExit("Need at least three samples")

    header = rows[0]
    ids = [row[0] for row in rows[1:]]
    X = np.array(
        [[float(v) for v in row[1:]] for row in rows[1:]],
        dtype=float
    )

    if not 2 <= args.n_neighbors < len(ids):
        parser.error("--n_neighbors must be >= 2 and less than sample count")
    if not 0 <= args.min_dist:
        parser.error("--min_dist must be nonnegative")

    model = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed
    )

    coords = model.fit_transform(X)

    if args.output == "-":
        out = sys.stdout
    else:
        out = open(args.output, "w")

    try:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow([header[0], "umap1", "umap2"])

        for sample_id, (x, y) in zip(ids, coords):
            writer.writerow([sample_id, x, y])
    finally:
        if args.output != "-":
            out.close()


if __name__ == "__main__":
    main()

