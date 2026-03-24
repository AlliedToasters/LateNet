"""CLI: Generate contrastive pairs from WordNet."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive statement pairs from WordNet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-depth", type=int, default=8, help="Max depth for WordNet traversal")
    parser.add_argument("--output", type=str, default="candidates.parquet", help="Output parquet file")
    args = parser.parse_args()
    raise NotImplementedError("Generation pipeline not yet implemented")


if __name__ == "__main__":
    main()
