
#!/usr/bin/env python3
"""Generate synthetic groups in a four-dimensional feature space."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="out/clusters.tsv")
    args = parser.parse_args()

    if args.n <= 0 or args.n % 3 != 0:
        parser.error("--n must be a positive multiple of 3")
    if args.noise < 0:
        parser.error("--noise must be nonnegative")

    rng = np.random.default_rng(args.seed)

    centers = {
        "A": [0, 0, 0, 0],
        "B": [3, 3, 0, 0],
        "C": [0, 0, 3, 3],
    }

    rows = []
    sample_id = 0

    for group, center in centers.items():
        samples = rng.normal(
            loc=center,
            scale=args.noise,
            size=(args.n // 3, 4)
        )

        for sample in samples:
            rows.append(
                (f"sample{sample_id:04d}", *sample, group)
            )
            sample_id += 1

    rng.shuffle(rows)

    Path(args.out).parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(args.out, "w") as f:
        f.write("id\tx1\tx2\tx3\tx4\tgroup\n")
        for sample_id, x1, x2, x3, x4, group in rows:
            f.write(
                f"{sample_id}\t{x1:.4f}\t{x2:.4f}\t"
                f"{x3:.4f}\t{x4:.4f}\t{group}\n"
            )

    print(f"wrote {len(rows)} samples to {args.out}")


if __name__ == "__main__":
    main()

