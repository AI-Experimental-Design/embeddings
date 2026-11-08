#!/usr/bin/env python3
"""Compare cosine similarities between word2vec embeddings.

Examples:
  python src/word2vec_similarity.py --model out/synthetic/model.pt --pairs cat,dog cat,car
  python src/word2vec_similarity.py --history out/synthetic/embedding_history.tsv --epochs 0,12 --pairs cat,dog cat,car
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def load_model(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    words = checkpoint['words']
    vectors = checkpoint['state_dict']['embedding.weight'].detach().cpu().numpy()
    return {word: vector for word, vector in zip(words, vectors)}


def load_history(path, epochs):
    result = {epoch: {} for epoch in epochs}
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        dims = [field for field in reader.fieldnames if field.startswith('dim')]
        if not dims or not {'epoch', 'word'}.issubset(reader.fieldnames):
            raise ValueError('Expected epoch, word, dim0, dim1, ... columns')
        dims.sort(key=lambda name: int(name[3:]))
        for row in reader:
            epoch = int(row['epoch'])
            if epoch in result:
                result[epoch][row['word']] = np.array(
                    [float(row[dim]) for dim in dims], dtype=float
                )
    for epoch in epochs:
        if not result[epoch]:
            raise ValueError(f'No embeddings found for epoch {epoch}')
    return result


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', type=Path, help='Trained model.pt checkpoint')
    source.add_argument('--history', type=Path, help='embedding_history.tsv')
    parser.add_argument('--epochs', default='0,12',
                        help='Comma-separated epochs to compare with --history (default: 0,12)')
    parser.add_argument('--pairs', nargs='+', required=True,
                        help='Word pairs as word1,word2 (e.g. cat,dog cat,car)')
    args = parser.parse_args()

    pairs = []
    for pair in args.pairs:
        parts = pair.split(',')
        if len(parts) != 2 or not all(parts):
            parser.error(f'Invalid pair {pair!r}; expected word1,word2')
        pairs.append(tuple(word.lower() for word in parts))

    if args.model:
        snapshots = {'final': load_model(args.model)}
    else:
        try:
            epochs = [int(e.strip()) for e in args.epochs.split(',')]
        except ValueError:
            parser.error('--epochs must contain integers, e.g. 0,12')
        snapshots = load_history(args.history, epochs)

    print('epoch\tword1\tword2\tcosine_similarity')
    for epoch, embeddings in snapshots.items():
        for word1, word2 in pairs:
            for word in (word1, word2):
                if word not in embeddings:
                    parser.error(f'{word!r} not in vocabulary for epoch {epoch}')
            score = cosine(embeddings[word1], embeddings[word2])
            print(f'{epoch}\t{word1}\t{word2}\t{score:.4f}')


if __name__ == '__main__':
    main()
