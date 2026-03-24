"""CLI: Dataset statistics (domain coverage, tier distribution, etc.)."""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Print dataset coverage statistics")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    print(f"Total rows: {len(df)}")
    print(f"  True: {(df['label'] == True).sum()}")
    print(f"  False: {(df['label'] == False).sum()}")
    print(f"  Unique pairs: {df['pair_id'].nunique()}")
    print()

    print(f"Generators ({df['generator'].nunique()}):")
    for gen, count in df['generator'].value_counts().items():
        print(f"  {gen}: {count}")
    print()

    print(f"Relation types ({df['relation_type'].nunique()}):")
    for rt, count in df['relation_type'].value_counts().items():
        print(f"  {rt}: {count}")
    print()

    print(f"Domains ({df['domain'].nunique()}):")
    for dom, count in df['domain'].value_counts().head(20).items():
        print(f"  {dom}: {count}")
    print()

    print("Difficulty distribution:")
    for diff, count in df['difficulty'].value_counts().items():
        print(f"  {diff or '(none)'}: {count}")
    print()

    if "tier" in df.columns:
        print("Tier distribution:")
        for tier, count in df['tier'].value_counts().sort_index().items():
            print(f"  Tier {tier}: {count}")


if __name__ == "__main__":
    main()
