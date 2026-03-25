"""CLI: Generate contrastive pairs (all generators or specific ones)."""

from __future__ import annotations

import argparse

import pandas as pd

from latenet.generators.base import BaseGenerator
from latenet.generators.wordnet_gen import WordNetGenerator
from latenet.generators.geography import GeographyGenerator
from latenet.generators.chemistry import ChemistryGenerator

# Registry of all available generators
GENERATORS: dict[str, type[BaseGenerator]] = {
    "wordnet": WordNetGenerator,
    "geography": GeographyGenerator,
    "chemistry": ChemistryGenerator,
}


def main():
    available = list(GENERATORS.keys())
    parser = argparse.ArgumentParser(description="Generate contrastive statement pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-depth", type=int, default=8, help="Max depth for WordNet traversal")
    parser.add_argument("--output", type=str, default="candidates.parquet", help="Output parquet file")
    parser.add_argument(
        "--max-false-per-true", type=int, default=3, help="Max false statements per true statement"
    )
    parser.add_argument(
        "--generators", nargs="+", default=available,
        help=f"Which generators to run (default: all implemented). Available: {available}",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=None,
        help="Max pairs per generator (default: no limit)",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []

    for gen_name in args.generators:
        if gen_name not in GENERATORS:
            print(f"Unknown generator: {gen_name}. Available: {available}")
            continue

        gen_cls = GENERATORS[gen_name]
        kwargs: dict = {"seed": args.seed, "max_pairs": args.max_pairs}

        if gen_name == "wordnet":
            kwargs["max_depth"] = args.max_depth
            kwargs["max_false_per_true"] = args.max_false_per_true

        generator = gen_cls(**kwargs)
        print(f"Running {gen_name} generator (seed={args.seed})...")

        count = 0
        for pair in generator.generate():
            all_rows.extend(pair.to_rows())
            count += 1

        print(f"  {gen_name}: {count} pairs, {len(all_rows)} total rows")

    if not all_rows:
        print("No rows generated.")
        return

    df = pd.DataFrame(all_rows)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
