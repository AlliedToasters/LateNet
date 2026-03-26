"""CLI: Run LLM voting on generated pairs.

Usage:
    latenet-validate --input candidates.parquet --output validated.parquet
    latenet-validate --input candidates.parquet --legs anthropic   # skip NDIF
    latenet-validate --input candidates.parquet --resume           # resume from checkpoint
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from latenet.validation.voting import run_anthropic_leg, run_ndif_leg
from latenet.validation.filters import filter_clean, merge_verdicts, validation_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

VALID_LEGS = {"ndif", "anthropic"}


def main():
    parser = argparse.ArgumentParser(description="Validate statement pairs with LLM ensemble voting")
    parser.add_argument("--input", type=str, required=True, help="Input parquet of candidate pairs")
    parser.add_argument("--output", type=str, default="validated.parquet", help="Output parquet (all rows + validation columns)")
    parser.add_argument("--clean-output", type=str, default=None, help="Output parquet for clean rows only (default: <output>.clean.parquet)")
    parser.add_argument(
        "--legs", nargs="+", default=list(VALID_LEGS),
        help=f"Which validation legs to run (default: all). Options: {sorted(VALID_LEGS)}",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints if available")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory (default: same dir as output)")
    args = parser.parse_args()

    legs = set(args.legs)
    unknown = legs - VALID_LEGS
    if unknown:
        parser.error(f"Unknown legs: {unknown}. Valid: {sorted(VALID_LEGS)}")

    input_path = Path(args.input)
    output_path = Path(args.output)
    clean_path = Path(args.clean_output) if args.clean_output else output_path.with_suffix(".clean.parquet")

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_path.parent / ".validation_checkpoints"
    if args.resume or True:  # always create checkpoint dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)

    ndif_verdicts = None
    anthropic_verdicts = None

    if "ndif" in legs:
        logger.info("=== NDIF / Llama 405B leg ===")
        ndif_verdicts = run_ndif_leg(df, checkpoint_dir=checkpoint_dir)

    if "anthropic" in legs:
        logger.info("=== Anthropic (Sonnet + Opus escalation) leg ===")
        anthropic_verdicts = run_anthropic_leg(df, checkpoint_dir=checkpoint_dir)

    # Merge and filter
    validated = merge_verdicts(df, ndif_verdicts, anthropic_verdicts)
    clean = filter_clean(validated)

    # Summary
    stats = validation_summary(validated)
    logger.info("=== Validation Summary ===")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    # Write outputs
    validated.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(validated), output_path)

    clean.to_parquet(clean_path, index=False)
    logger.info("Wrote %d clean rows to %s", len(clean), clean_path)


if __name__ == "__main__":
    main()
