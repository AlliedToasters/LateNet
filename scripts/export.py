"""CLI: Export validated dataset."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Export validated dataset to final parquet")
    parser.add_argument("--input", type=str, required=True, help="Input parquet of validated pairs")
    parser.add_argument("--output", type=str, default="latenet_v1.parquet", help="Output parquet file")
    args = parser.parse_args()
    raise NotImplementedError("Export pipeline not yet implemented")


if __name__ == "__main__":
    main()
