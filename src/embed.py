
#!/usr/bin/env python3
"""Generate contextual embeddings for 17-base DNA sequences.

Reads one DNA sequence per line from stdin.
Outputs the 64-dimensional embedding at the center position.

Usage:
    cat sequences.txt | python src/embed.py --model out/pretrain.pt
"""

import argparse
import sys

import torch

import dna


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model = dna.load_model(args.model)

    print("id\t" + "\t".join(
        f"e{i+1}" for i in range(model.conv2.out_channels)
    ))

    with torch.no_grad():
        for i, line in enumerate(sys.stdin):
            seq = line.strip().upper()
            if not seq:
                continue

            if len(seq) != 17:
                raise SystemExit("Each sequence must contain exactly 17 bases")

            x = dna.encode([seq])
            embedding = model.hidden(x)[0, 8]

            print(
                f"seq{i}\t" +
                "\t".join(f"{v:.6f}" for v in embedding.tolist())
            )


if __name__ == "__main__":
    main()

