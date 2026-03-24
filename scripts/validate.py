"""CLI: Run LLM voting on generated pairs."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Validate statement pairs with LLM ensemble voting")
    parser.add_argument("--input", type=str, required=True, help="Input parquet of candidate pairs")
    parser.add_argument("--output", type=str, default="validated.parquet", help="Output parquet file")
    args = parser.parse_args()
    raise NotImplementedError("Validation pipeline not yet implemented")


if __name__ == "__main__":
    main()
