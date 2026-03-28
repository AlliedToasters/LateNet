"""CLI: Curate a balanced dataset from the persistent ledger.

Reads the ledger, selects clean (uncontested) rows, and produces a
balanced dataset with even representation across generators and
relation types.

Usage:
    latenet-curate --rows-per-stratum 50 --output latenet_v1.parquet
    latenet-curate --rows-per-stratum 100 --cap chemistry 30 --output large.parquet
    latenet-curate --generators biology geography --rows-per-stratum 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Curate a balanced dataset from the ledger",
    )
    parser.add_argument(
        "--rows-per-stratum", type=int, required=True,
        help="Target number of pairs per (generator, relation_type) cell",
    )
    parser.add_argument(
        "--generators", nargs="+", default=None,
        help="Filter to specific generators (default: all in ledger)",
    )
    parser.add_argument(
        "--output", type=str, default="latenet_curated.parquet",
        help="Output parquet for the curated dataset",
    )
    parser.add_argument(
        "--ledger-dir", type=str, default="data",
        help="Directory containing the ledger (default: data/)",
    )
    parser.add_argument(
        "--cap", nargs=2, action="append", metavar=("GENERATOR", "MAX_PAIRS"),
        help="Per-generator cap, e.g. --cap chemistry 30. Can be repeated.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling when a stratum has more than needed",
    )
    args = parser.parse_args()

    ledger_dir = Path(args.ledger_dir)

    from latenet.io.ledger import load_ledger

    ledger = load_ledger(ledger_dir)
    if len(ledger) == 0:
        logger.error("Ledger is empty. Run latenet-batch first.")
        return

    # Filter to clean rows: must be validated AND not contested
    if "contested" in ledger.columns:
        clean = ledger[ledger["contested"] == False].copy()  # noqa: E712
    else:
        # No validation data at all — nothing is clean
        logger.error("Ledger has no 'contested' column — run validation first.")
        return

    unvalidated = len(ledger) - ledger["contested"].notna().sum() if "contested" in ledger.columns else len(ledger)
    logger.info(
        "Ledger: %d total rows, %d validated, %d clean, %d unvalidated",
        len(ledger), len(ledger) - unvalidated, len(clean), unvalidated,
    )

    # Filter to requested generators
    if args.generators:
        clean = clean[clean["generator"].isin(args.generators)]
        logger.info("Filtered to generators %s: %d rows", args.generators, len(clean))

    # Parse per-generator caps
    generator_caps: dict[str, int] = {}
    if args.cap:
        for gen_name, cap_str in args.cap:
            generator_caps[gen_name] = int(cap_str)

    # Group by pair_id to ensure we keep true+false together
    # Each pair_id has 2 rows (affirmative only) or 4 rows (with negated variants)
    pair_groups = clean.groupby("pair_id")
    valid_pairs = pair_groups.filter(lambda g: len(g) in (2, 4))

    # Get unique pairs with their stratum info
    pair_meta = valid_pairs.groupby("pair_id").first()[["generator", "relation_type"]].reset_index()

    # Sample per stratum: (generator, relation_type)
    sampled_pair_ids: list[str] = []

    for (gen, rel), group in pair_meta.groupby(["generator", "relation_type"]):
        target: int = generator_caps.get(gen, args.rows_per_stratum)
        available = len(group)
        n = min(target, available)
        selected = group.sample(n=n, random_state=args.seed)
        sampled_pair_ids.extend(selected["pair_id"].tolist())
        status = "FULL" if n >= target else f"SHORT ({available}/{target})"
        logger.info("  %s/%s: %d pairs [%s]", gen, rel, n, status)

    # Pull the full rows for selected pairs
    curated = valid_pairs[valid_pairs["pair_id"].isin(sampled_pair_ids)].copy()

    # Drop difficulty column from published view — it's kept in the ledger
    # for backward compatibility but is not a meaningful dataset axis
    if "difficulty" in curated.columns:
        curated = curated.drop(columns=["difficulty"])

    # Sort for clean output
    curated = curated.sort_values(["generator", "relation_type", "pair_id", "label"]).reset_index(drop=True)

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curated.to_parquet(output_path, index=False)

    n_pairs = curated["pair_id"].nunique()
    n_strata = curated.groupby(["generator", "relation_type"]).ngroups
    logger.info(
        "Curated dataset: %d pairs (%d rows) across %d strata -> %s",
        n_pairs, len(curated), n_strata, output_path,
    )


if __name__ == "__main__":
    main()
