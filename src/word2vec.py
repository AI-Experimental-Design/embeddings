import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = {
    'animal': ['cat', 'dog', 'horse', 'rabbit'],
    'vehicle': ['car', 'truck', 'bus', 'train'],
    'fruit': ['apple', 'banana', 'orange', 'pear'],
    'temperature': ['hot', 'cold'],
}
TEMPLATES = {
    'animal': ['the {w} likes to sleep', 'the {w} likes to eat', 'we saw the {w} outside', 'the {w} moved quickly'],
    'vehicle': ['the {w} travels on roads', 'we saw the {w} outside', 'the {w} moved quickly', 'the {w} stopped nearby'],
    'fruit': ['the {w} tastes good', 'we bought the {w} today', 'the {w} is in the basket', 'we ate the {w} today'],
    'temperature': ['the water feels {w} today', 'the weather is {w} outside', 'the room feels {w} today', 'the air is {w} outside'],
}


def make_corpus(n, seed):
    rng = random.Random(seed)
    categories = list(GROUPS)
    sentences = []
    for _ in range(n):
        category = rng.choice(categories)
        word = rng.choice(GROUPS[category])
        sentence = rng.choice(TEMPLATES[category]).format(w=word)
        sentences.append(sentence.split())
    return sentences


class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.input = nn.Embedding(vocab_size, dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, centers):
        return self.output(self.input(centers))


def pairs_from_sentences(sentences, stoi, window):
    centers, contexts = [], []
    for sentence in sentences:
        ids = [stoi[w] for w in sentence]
        for i, center in enumerate(ids):
            for j in range(max(0, i-window), min(len(ids), i+window+1)):
                if i != j:
                    centers.append(center)
                    contexts.append(ids[j])
    return torch.tensor(centers, dtype=torch.long), torch.tensor(contexts, dtype=torch.long)


def pca_2d(x):
    centered = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--window', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', default='out')
    args = parser.parse_args()
    if args.n < 1 or args.dim < 2 or args.window < 1 or args.epochs < 1:
        parser.error('n, epochs and window must be positive; dim must be at least 2')
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sentences = make_corpus(args.n, args.seed)
    vocab = sorted({w for sentence in sentences for w in sentence})
    stoi = {w: i for i, w in enumerate(vocab)}
    with (out / 'corpus.txt').open('w') as f:
        for sentence in sentences:
            f.write(' '.join(sentence) + '\n')
    x, y = pairs_from_sentences(sentences, stoi, args.window)
    model = SkipGram(len(vocab), args.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    rng = torch.Generator().manual_seed(args.seed)
    print(f'{len(sentences)} sentences; {len(vocab)} tokens; {len(x)} center-context pairs')
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(x), generator=rng)
        total_loss = 0.0
        for batch in permutation.split(1024):
            logits = model(x[batch])
            loss = nn.functional.cross_entropy(logits, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f'epoch {epoch:02d} loss={total_loss / len(x):.4f}')
    vectors = model.input.weight.detach().numpy()
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    print('\nNearest neighbors (cosine similarity):')
    for word in ['cat', 'dog', 'car', 'apple', 'hot', 'cold']:
        if word not in stoi:
            continue
        similarities = normalized @ normalized[stoi[word]]
        similarities[stoi[word]] = -np.inf
        top = np.argsort(-similarities)[:4]
        print(f'{word:>7}: ' + ', '.join(f'{vocab[i]} ({similarities[i]:.2f})' for i in top))
    coords = pca_2d(vectors)
    categories = {w: category for category, words in GROUPS.items() for w in words}
    with (out / 'embeddings.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['word', 'category', 'pc1', 'pc2'] + [f'dim{i}' for i in range(args.dim)])
        for word, xy, vector in zip(vocab, coords, vectors):
            writer.writerow([word, categories.get(word, 'context'), *xy, *vector])
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {'animal': 'tab:blue', 'vehicle': 'tab:orange', 'fruit': 'tab:green', 'temperature': 'tab:red', 'context': '0.75'}
    for category in ['context', 'animal', 'vehicle', 'fruit', 'temperature']:
        indices = [i for i, w in enumerate(vocab) if categories.get(w, 'context') == category]
        ax.scatter(coords[indices, 0], coords[indices, 1], label=category, c=palette[category], alpha=0.8)
        for i in indices:
            if category != 'context':
                ax.annotate(vocab[i], coords[i], xytext=(4, 3), textcoords='offset points', fontsize=9)
    ax.set(xlabel='PC1', ylabel='PC2', title='Word2vec token embeddings (PCA)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'embeddings.png', dpi=170)
    torch.save({'model': model.state_dict(), 'vocab': vocab, 'dim': args.dim}, out / 'word2vec.pt')
    print(f'\nSaved results in {out}/')


if __name__ == '__main__':
    main()
