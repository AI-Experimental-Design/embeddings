
#!/usr/bin/env python3
"""Generate DNA sequences containing different patterns.

Each sequence contains a pattern centered in random DNA.

Usage:
    python src/make_embedding_sequences.py \
        --n 200 \
        --length 17 \
        --out out/embedding_sequences.tsv
"""

import argparse
import random
from pathlib import Path


PATTERNS = {
    "TATA": "TATAAA",
    "GC": "GGGCGG",
    "polyA": "AAAAAAAAAA",
    "polyT": "TTTTTTTTTT",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--length", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []

    for group, pattern in PATTERNS.items():
        if len(pattern) > args.length:
            parser.error(
                f"Pattern {group} is longer than sequence length"
            )

        start = (args.length - len(pattern)) // 2

        for _ in range(args.n):
            seq = [
                rng.choice("ACGT")
                for _ in range(args.length)
            ]

            seq[start:start + len(pattern)] = pattern

            rows.append(("".join(seq), group))

    rng.shuffle(rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w") as f:
        f.write("id\tsequence\tgroup\n")
        for i, (seq, group) in enumerate(rows):
            f.write(f"seq{i}\t{seq}\t{group}\n")

    print(f"Wrote {len(rows)} sequences to {args.out}")


if __name__ == "__main__":
    main()

