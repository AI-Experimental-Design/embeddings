
import argparse
import random
from pathlib import Path

groups = {
    "animals": ["cat", "dog", "horse"],
    "vehicles": ["car", "truck", "bus"],
    "fruits": ["apple", "banana", "orange"],
    "temperature": ["hot", "cold"],
}

templates = {
    "animals": [
        "the {} likes to sleep",
        "the {} likes to eat",
        "the {} likes to run",
    ],
    "vehicles": [
        "the {} drives on the road",
        "the {} has an engine",
        "the {} moves very fast",
    ],
    "fruits": [
        "the {} grows on a tree",
        "the {} tastes very sweet",
        "the {} is good to eat",
    ],
    "temperature": [
        "the water is {} today",
        "the weather is very {}",
        "it feels {} outside",
    ],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--out", default="out/synthetic.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w") as f:
        for _ in range(args.n):
            group = random.choice(list(groups))
            word = random.choice(groups[group])
            template = random.choice(templates[group])
            f.write(template.format(word) + "\n")

    print(f"wrote {args.n} sentences to {args.out}")

if __name__ == "__main__":
    main()

