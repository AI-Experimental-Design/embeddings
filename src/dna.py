"""Shared code: reading sequences, encoding them, and the masked model.

Bases are stored as integer tokens, A=0 C=1 G=2 T=3, and 4 for a masked
position. The model turns each token into a learned vector (the embedding),
mixes information from neighboring positions with two convolution layers,
and predicts the four bases at every position.
"""
import numpy as np
import torch
import torch.nn as nn

BASES = 'ACGT'
MASK = 4


def read_genome(path):
    rows = [l.rstrip('\n').split('\t') for l in open(path)][1:]
    return rows


def encode(seqs):
    lut = {b: i for i, b in enumerate(BASES)}
    return torch.tensor([[lut[b] for b in s] for s in seqs], dtype=torch.long)


def element_mask(rows, length):
    """For every position, which element covers it (or 'background')."""
    from make_genome import ELEMENTS
    lab = np.full((len(rows), length), 'background', dtype=object)
    for i, r in enumerate(rows):
        if r[3] == 'none':
            continue
        for item in r[3].split(','):
            e, pos = item.split(':')
            pos = int(pos)
            lab[i, pos:pos + len(ELEMENTS[e])] = e
    return lab


class MaskedDNAModel(nn.Module):
    def __init__(self, emb_dim=8, hidden=64, kernel=9):
        super().__init__()
        self.embed = nn.Embedding(5, emb_dim)
        self.conv1 = nn.Conv1d(emb_dim, hidden, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.out = nn.Conv1d(hidden, 4, 1)

    def hidden(self, tokens):
        """The per-position representation, shape (batch, length, hidden)."""
        x = self.embed(tokens).transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x.transpose(1, 2)

    def forward(self, tokens):
        """Logits for the four bases at every position, shape (batch, length, 4)."""
        h = self.hidden(tokens).transpose(1, 2)
        return self.out(h).transpose(1, 2)


def load_model(path):
    ckpt = torch.load(path, weights_only=False)
    m = MaskedDNAModel(**ckpt['config'])
    m.load_state_dict(ckpt['state'])
    m.eval()
    return m
