#!/usr/bin/env python3
"""Train a full-softmax Skip-gram word2vec model on one sentence per line."""
import argparse
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, centers):
        return self.output(self.embedding(centers))


def save_embeddings(path, model, words, epoch):
    values = model.embedding.weight.detach().cpu().numpy()
    with path.open('a', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        for word, vector in zip(words, values):
            writer.writerow([epoch, word, *[f'{x:.8f}' for x in vector]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Text file: one sentence per line')
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--window', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='out/synthetic')
    args = parser.parse_args()
    if min(args.dim, args.window, args.epochs, args.batch_size) < 1:
        parser.error('--dim, --window, --epochs, and --batch-size must be positive')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # CPU is fast for the small synthetic corpus and maximizes reproducibility.
    torch.set_num_threads(min(4, torch.get_num_threads()))

    sentences = [line.lower().split() for line in Path(args.data).read_text().splitlines()]
    sentences = [sentence for sentence in sentences if sentence]
    if not sentences:
        parser.error('Input contains no sentences')

    counts = Counter(word for sentence in sentences for word in sentence)
    words = sorted(counts)
    word_to_id = {word: i for i, word in enumerate(words)}
    pairs = []
    for sentence in sentences:
        ids = [word_to_id[word] for word in sentence]
        for i, center in enumerate(ids):
            for j in range(max(0, i - args.window), min(len(ids), i + args.window + 1)):
                if i != j:
                    pairs.append((center, ids[j]))
    if not pairs:
        parser.error('No center-context pairs could be generated')

    pair_tensor = torch.tensor(pairs, dtype=torch.long)
    model = SkipGram(len(words), args.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'vocab.tsv').open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['word', 'count'])
        writer.writerows((word, counts[word]) for word in words)

    embeddings_path = out / 'embedding_history.tsv'
    with embeddings_path.open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['epoch', 'word', *[f'dim{i}' for i in range(args.dim)]])
    save_embeddings(embeddings_path, model, words, 0)

    with (out / 'training_loss.tsv').open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['epoch', 'loss'])

    print(f'{len(sentences)} sentences, {len(words)} words, {len(pairs)} training pairs')
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(pair_tensor))
        total_loss = 0.0
        for batch_ids in order.split(args.batch_size):
            batch = pair_tensor[batch_ids]
            optimizer.zero_grad()
            logits = model(batch[:, 0])
            loss = loss_fn(logits, batch[:, 1])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        average_loss = total_loss / len(pair_tensor)
        with (out / 'training_loss.tsv').open('a', newline='') as handle:
            csv.writer(handle, delimiter='\t').writerow([epoch, f'{average_loss:.6f}'])
        save_embeddings(embeddings_path, model, words, epoch)
        print(f'epoch {epoch:02d} loss={average_loss:.4f}')

    torch.save({'state_dict': model.state_dict(), 'words': words, 'dim': args.dim}, out / 'model.pt')
    print(f'Wrote results to {out}')


if __name__ == '__main__':
    main()
