#!/usr/bin/env python3
"""Stack TSV files with the same columns into one, adding a column that says
which file each row came from. Handy for plotting several runs together.

Usage:
    python stack_tsv.py -i out/a.tsv out/b.tsv --names a b --col model > out/ab.tsv
"""
import argparse
import sys


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--files', nargs='+', required=True)
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--col', default='source')
    return p.parse_args()


def main():
    args = get_args()
    header_done = False
    for name, path in zip(args.names, args.files):
        lines = [l.rstrip('\n') for l in open(path) if not l.startswith('#')]
        if not header_done:
            sys.stdout.write(lines[0] + '\t' + args.col + '\n')
            header_done = True
        for l in lines[1:]:
            sys.stdout.write(l + '\t' + name + '\n')


if __name__ == '__main__':
    main()
