#!/usr/bin/env python3
"""Pretrain a model by hiding bases and asking it to fill them in.

Each epoch, every sequence has a random 15% of its bases replaced by a mask
token. The model predicts the four bases at every position, and the loss is
cross-entropy on the masked positions only. No labels are used. The
sequence is its own answer key.

Cross-entropy for four classes is the four-class version of binary
cross-entropy. The model outputs four numbers per position, softmax turns
them into four probabilities that add to 1, and the loss is
-log(probability given to the true base).

Usage:
    python pretrain.py --genome out/genome.tsv --epochs 15 --out out/pretrained.pt
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

import dna


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--genome', required=True)
    p.add_argument('--n_train', type=int, help='use only this many sequences')
    p.add_argument('--mask_frac', type=float, default=0.15)
    p.add_argument('--emb_dim', type=int, default=8)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--kernel', type=int, default=9)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.003)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log', help='write per-epoch loss and accuracy here')
    p.add_argument('--out', required=True)
    return p.parse_args()


def mask_tokens(x, frac, gen):
    m = torch.rand(x.shape, generator=gen) < frac
    xm = x.clone()
    xm[m] = dna.MASK
    return xm, m


def evaluate(model, x, lab, frac, gen):
    """Masked-base accuracy, overall and by element type."""
    with torch.no_grad():
        xm, m = mask_tokens(x, frac, gen)
        pred = model(xm).argmax(-1)
        ok = (pred == x)[m].numpy()
        where = lab[m.numpy()]
    out = {'all': ok.mean()}
    for e in ['background', 'TATA', 'GC', 'polyA']:
        if (where == e).any():
            out[e] = ok[where == e].mean()
    return out


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    rows = dna.read_genome(args.genome)
    if args.n_train:
        rows = rows[:args.n_train]
    n_val = len(rows) // 10
    val_rows, tr_rows = rows[:n_val], rows[n_val:]
    x_tr = dna.encode([r[1] for r in tr_rows])
    x_val = dna.encode([r[1] for r in val_rows])
    lab_val = dna.element_mask(val_rows, x_val.shape[1])

    config = dict(emb_dim=args.emb_dim, hidden=args.hidden, kernel=args.kernel)
    model = dna.MaskedDNAModel(**config)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'{len(tr_rows)} training and {n_val} validation sequences, '
          f'{sum(p.numel() for p in model.parameters())} parameters')
    log = []
    t0 = time.time()
    for epoch in range(args.epochs + 1):
        if epoch > 0:
            model.train()
            perm = torch.randperm(len(x_tr), generator=gen)
            total = 0.0
            for i in range(0, len(x_tr), args.batch):
                xb = x_tr[perm[i:i + args.batch]]
                xm, m = mask_tokens(xb, args.mask_frac, gen)
                loss = F.cross_entropy(model(xm)[m], xb[m])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            train_loss = total / len(x_tr)
        else:
            train_loss = float('nan')
        model.eval()
        acc = evaluate(model, x_val, lab_val, args.mask_frac,
                       torch.Generator().manual_seed(123))
        log.append((epoch, train_loss, acc))
        print(f'epoch {epoch:02d} loss={train_loss:.4f} masked acc: ' +
              ' '.join(f'{k}={v:.3f}' for k, v in acc.items()) +
              f'  ({time.time() - t0:.0f}s)', flush=True)
    torch.save({'config': config, 'state': model.state_dict()}, args.out)
    print(f'wrote {args.out}')
    if args.log:
        keys = list(log[-1][2].keys())
        with open(args.log, 'w') as f:
            f.write('epoch\tloss\t' + '\t'.join(keys) + '\n')
            for e, l, a in log:
                f.write(f'{e}\t{l:.4f}\t' + '\t'.join(f'{a.get(k, float("nan")):.4f}' for k in keys) + '\n')


if __name__ == '__main__':
    main()
