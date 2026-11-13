
#!/usr/bin/env python3
"""Plot two numerical columns from a TSV file or stdin.

The first column identifies each point. An optional groups file
maps identifiers to group names for coloring.

Examples:
    python src/plot_scatter.py \
        -i out/synthetic/epoch_12_pca.tsv \
        -o img/word2vec_epoch_012.png \
        --groups src/word2vec_groups.tsv \
        --labels

    cat out/clusters_pca.tsv | python src/plot_scatter.py \
        -o img/clusters_pca.png \
        --groups out/cluster_groups.tsv
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "Arial"


def read_tsv(path):
    if path == "-":
        return list(csv.DictReader(sys.stdin, delimiter="\t"))
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", default="-",
        help="Input TSV file (default: stdin)"
    )
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--x", default="pc1")
    parser.add_argument("--y", default="pc2")
    parser.add_argument("--groups", help="TSV mapping IDs to groups")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--title")
    parser.add_argument("--plot_width", type=float, default=4)
    parser.add_argument("--plot_height", type=float, default=3)
    parser.add_argument("--point_size", type=float, default=15)
    args = parser.parse_args()

    rows = read_tsv(args.input)
    if not rows:
        raise SystemExit("No data rows found")

    id_col = next(iter(rows[0]))

    groups = {}
    if args.groups:
        group_rows = read_tsv(args.groups)
        group_id_col, group_col = list(group_rows[0])[:2]
        groups = {
            r[group_id_col]: r[group_col]
            for r in group_rows
        }

    for row in rows:
        row["_group"] = groups.get(row[id_col], "other")

    group_names = sorted(set(r["_group"] for r in rows))
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(
        figsize=(args.plot_width, args.plot_height),
        dpi=300
    )

    for i, group in enumerate(group_names):
        subset = [r for r in rows if r["_group"] == group]

        x = [float(r[args.x]) for r in subset]
        y = [float(r[args.y]) for r in subset]

        ax.scatter(
            x, y,
            s=args.point_size,
            color=colors(i % 10),
            label=group,
            alpha=0.8
        )

        if args.labels:
            for r, px, py in zip(subset, x, y):
                ax.annotate(
                    r[id_col],
                    (px, py),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6
                )

    ax.set_xlabel(args.x, fontsize=8)
    ax.set_ylabel(args.y, fontsize=8)

    if args.title:
        ax.set_title(args.title, fontsize=8)

    ax.tick_params(
        axis="both", which="major",
        labelsize=8, width=0.5, length=2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)

    if args.groups:
        ax.legend(frameon=False, fontsize=7)

    Path(args.output).parent.mkdir(
        parents=True, exist_ok=True
    )

    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

