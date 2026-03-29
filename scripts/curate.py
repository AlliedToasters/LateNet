"""CLI: Curate a balanced dataset from the persistent ledger.

Reads the ledger, selects clean (uncontested, not awkward) affirmative
rows, optionally expands with mechanically-negated variants, and produces
a balanced dataset with even representation across generators and
relation types.

Usage:
    latenet-curate --rows-per-stratum 50 --output latenet_v1.parquet
    latenet-curate --rows-per-stratum 100 --expand-negated --output latenet_v1.parquet
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

VALIDATION_COLS = [
    "llama_says_true", "llama_agrees", "llama_confidence", "llama_error",
    "haiku_says_true", "haiku_agrees", "haiku_error", "haiku_awkward",
    "sonnet_says_true", "sonnet_agrees", "sonnet_error", "sonnet_awkward",
    "opus_says_true", "opus_agrees", "opus_error", "contested",
]


def _expand_with_negated(curated: pd.DataFrame) -> pd.DataFrame:
    """Mechanically expand affirmative rows with negated variants.

    For each pair, produces two additional rows with negated statements,
    flipped labels, ``negated=True``, and null validation columns (since
    negated rows inherit curation status from the affirmative pair but
    are not independently validated).
    """
    from latenet.negation.strategies import negate_statement

    neg_rows = []
    for _, row in curated.iterrows():
        neg = row.copy()
        neg["statement"] = negate_statement(row["statement"])
        neg["label"] = not row["label"]
        neg["negated"] = True
        if row["id"].endswith("_true"):
            neg["id"] = row["id"].replace("_true", "_neg_true")
        elif row["id"].endswith("_false"):
            neg["id"] = row["id"].replace("_false", "_neg_false")
        for col in VALIDATION_COLS:
            if col in neg.index:
                neg[col] = None
        neg_rows.append(neg)

    neg_df = pd.DataFrame(neg_rows)
    return pd.concat([curated, neg_df], ignore_index=True)


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
        "--exclude-generators", nargs="+", default=None,
        help="Exclude specific generators from curation",
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
    parser.add_argument(
        "--expand-negated", action="store_true",
        help="Expand with mechanically-negated variants (4 rows per pair). "
             "Negated rows have null validation columns.",
    )
    args = parser.parse_args()

    ledger_dir = Path(args.ledger_dir)

    from latenet.io.ledger import load_ledger

    ledger = load_ledger(ledger_dir)
    if len(ledger) == 0:
        logger.error("Ledger is empty. Run latenet-batch first.")
        return

    if "contested" not in ledger.columns:
        logger.error("Ledger has no 'contested' column — run validation first.")
        return

    # Work from affirmative rows only — negated variants are produced
    # mechanically during curation via --expand-negated
    if "negated" in ledger.columns:
        aff = ledger[ledger["negated"] == False].copy()  # noqa: E712
    else:
        aff = ledger.copy()

    validated = aff["contested"].notna().sum()
    logger.info("Ledger: %d total, %d affirmative, %d validated", len(ledger), len(aff), validated)

    # Clean = pair-level filter: not contested AND not awkward
    pair_quality = aff.groupby("pair_id").agg(
        any_contested=("contested", "any"),
        any_awkward=("haiku_awkward", lambda x: x.fillna(False).any()),
    ).reset_index()
    good_pair_ids = set(pair_quality[~pair_quality.any_contested & ~pair_quality.any_awkward].pair_id)

    n_contested = pair_quality.any_contested.sum()
    n_awkward_only = (~pair_quality.any_contested & pair_quality.any_awkward).sum()
    clean = aff[aff["pair_id"].isin(good_pair_ids)].copy()
    logger.info(
        "After quality filter: %d clean rows (%d pairs), dropped %d contested + %d awkward",
        len(clean), len(good_pair_ids), n_contested, n_awkward_only,
    )

    # Filter to requested generators
    if args.generators:
        clean = clean[clean["generator"].isin(args.generators)]
        logger.info("Filtered to generators %s: %d rows", args.generators, len(clean))

    if args.exclude_generators:
        clean = clean[~clean["generator"].isin(args.exclude_generators)]
        logger.info("Excluded generators %s: %d rows remain", args.exclude_generators, len(clean))

    # Parse per-generator caps
    generator_caps: dict[str, int] = {}
    if args.cap:
        for gen_name, cap_str in args.cap:
            generator_caps[gen_name] = int(cap_str)

    # Each pair should have exactly 2 affirmative rows (true + false)
    pair_groups = clean.groupby("pair_id")
    valid_pairs = pair_groups.filter(lambda g: len(g) == 2)

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

    # Expand with negated variants if requested
    if args.expand_negated:
        pre_neg = len(curated)
        curated = _expand_with_negated(curated)
        logger.info("Expanded with negated: %d -> %d rows", pre_neg, len(curated))

    # Drop difficulty column from published view — it's kept in the ledger
    # for backward compatibility but is not a meaningful dataset axis
    if "difficulty" in curated.columns:
        curated = curated.drop(columns=["difficulty"])

    # Sort for clean output
    curated = curated.sort_values(
        ["generator", "relation_type", "pair_id", "negated", "label"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    # Write
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curated.to_parquet(output_path, index=False)

    n_pairs = curated["pair_id"].nunique()
    n_strata = curated.groupby(["generator", "relation_type"]).ngroups
    rows_per_pair = len(curated) / n_pairs if n_pairs > 0 else 0
    logger.info(
        "Curated dataset: %d pairs (%d rows, %.0f/pair) across %d strata -> %s",
        n_pairs, len(curated), rows_per_pair, n_strata, output_path,
    )


if __name__ == "__main__":
    main()
