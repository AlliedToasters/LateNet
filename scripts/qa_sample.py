"""CLI: Sample true/false statements from each generator, stratified by difficulty.

Reads a generated parquet file and prints a human-readable sample for manual QA.
Stratifies by generator x difficulty so every combination is represented.
"""

from __future__ import annotations

import argparse

import pandas as pd


def _print_pair(pair_id: str, row_true: pd.Series, row_false: pd.Series) -> None:
    """Print a single true/false pair."""
    print(f"  pair_id:    {pair_id}")
    print(f"  relation:   {row_true['relation_type']}")
    print(f"  template:   {row_true['template_id']}")
    print(f"  difficulty:  {row_true['difficulty']}")
    neg = row_true.get("negation_strategy", "")
    if neg:
        print(f"  negation:   {neg}")
    print(f"  TRUE:  {row_true['statement']}")
    print(f"  FALSE: {row_false['statement']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Sample true/false pairs from each generator, stratified by difficulty"
    )
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument(
        "--n", type=int, default=3,
        help="Number of pairs to sample per generator x difficulty bucket (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--generators", nargs="+", default=None,
        help="Filter to specific generators (default: all)",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    # Work at the pair level: split into true/false rows keyed by pair_id
    true_df = df[df["label"] == True].set_index("pair_id")
    false_df = df[df["label"] == False].set_index("pair_id")

    # Only keep pair_ids present in both
    common_ids = true_df.index.intersection(false_df.index)
    true_df = true_df.loc[common_ids]
    false_df = false_df.loc[common_ids]

    # Use true_df as the base for grouping
    generators = sorted(true_df["generator"].unique())
    if args.generators:
        generators = [g for g in generators if g in args.generators]

    total_sampled = 0

    for gen in generators:
        gen_mask = true_df["generator"] == gen
        gen_df = true_df[gen_mask]

        difficulties = sorted(gen_df["difficulty"].unique(), key=lambda d: d or "")
        relation_types = sorted(gen_df["relation_type"].unique())

        print("=" * 70)
        print(f"Generator: {gen}")
        print(f"  Total pairs: {len(gen_df)}")
        print(f"  Relation types: {', '.join(relation_types)}")
        print(f"  Difficulties: {', '.join(d or '(none)' for d in difficulties)}")
        print("=" * 70)

        for diff in difficulties:
            diff_mask = gen_df["difficulty"] == diff
            bucket = gen_df[diff_mask]

            if bucket.empty:
                continue

            n_sample = min(args.n, len(bucket))
            sample = bucket.sample(n=n_sample, random_state=args.seed)

            print(f"\n--- {gen} / {diff or '(none)'} ({len(bucket)} pairs, showing {n_sample}) ---\n")

            for pair_id in sample.index:
                row_true = true_df.loc[pair_id]
                row_false = false_df.loc[pair_id]
                _print_pair(pair_id, row_true, row_false)
                total_sampled += 1

    print("=" * 70)
    print(f"Total pairs sampled: {total_sampled}")
    print("=" * 70)


if __name__ == "__main__":
    main()
