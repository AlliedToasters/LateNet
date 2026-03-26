"""CLI: Build a stratified, validated dataset.

Generates contrastive pairs, validates them with an LLM ensemble
(Llama 405B via NDIF + Claude Sonnet), and keeps only rows where both
validators agree with the ground-truth label. Loops until per-stratum
quotas are met.

Usage:
    latenet-build --rows-per-stratum 50 --output latenet_v1.parquet
    latenet-build --rows-per-stratum 10 --generators biology temporal --max-rounds 5
    latenet-build --rows-per-stratum 20 --legs anthropic  # skip NDIF
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

AVAILABLE_GENERATORS = ["wordnet", "geography", "chemistry", "biology", "temporal"]
VALID_LEGS = {"ndif", "anthropic"}


def main():
    parser = argparse.ArgumentParser(
        description="Build a stratified, LLM-validated contrastive dataset",
    )
    parser.add_argument(
        "--rows-per-stratum", type=int, required=True,
        help="Target number of pairs per (generator, difficulty) cell",
    )
    parser.add_argument(
        "--generators", nargs="+", default=AVAILABLE_GENERATORS,
        help=f"Which generators to use (default: all). Available: {AVAILABLE_GENERATORS}",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max generate-validate rounds")
    parser.add_argument(
        "--oversample", type=float, default=3.0,
        help="Oversample factor per round (default: 3x needed)",
    )
    parser.add_argument(
        "--output", type=str, default="latenet_validated.parquet",
        help="Output parquet for the clean dataset",
    )
    parser.add_argument(
        "--disputes-output", type=str, default=None,
        help="Output parquet for disputed rows (default: <output>.disputes.parquet)",
    )
    parser.add_argument(
        "--legs", nargs="+", default=list(VALID_LEGS),
        help=f"Validation legs to run (default: all). Options: {sorted(VALID_LEGS)}",
    )
    args = parser.parse_args()

    legs = set(args.legs)
    unknown = legs - VALID_LEGS
    if unknown:
        parser.error(f"Unknown legs: {unknown}. Valid: {sorted(VALID_LEGS)}")

    output_path = Path(args.output)
    disputes_path = (
        Path(args.disputes_output)
        if args.disputes_output
        else output_path.with_suffix(".disputes.parquet")
    )
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    from latenet.validation.stratified import build_stratified_dataset

    clean_df, disputes_df = build_stratified_dataset(
        generators=args.generators,
        rows_per_stratum=args.rows_per_stratum,
        seed=args.seed,
        max_rounds=args.max_rounds,
        oversample_factor=args.oversample,
        output_dir=output_dir,
        legs=legs,
    )

    if len(clean_df) > 0:
        clean_df.to_parquet(output_path, index=False)
        logger.info("Wrote %d clean rows to %s", len(clean_df), output_path)
    else:
        logger.warning("No clean rows produced.")

    if len(disputes_df) > 0:
        disputes_df.to_parquet(disputes_path, index=False)
        logger.info("Wrote %d disputed rows to %s", len(disputes_df), disputes_path)


if __name__ == "__main__":
    main()
