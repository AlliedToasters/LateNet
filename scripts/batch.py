"""CLI: Generate a batch, validate it, and append to the persistent ledger.

This is the primary iterative workflow command. Run it repeatedly with
different generators, seeds, and sizes. Inspect results with latenet-stats.
Pull the final balanced dataset with latenet-curate.

Usage:
    latenet-batch --generators biology temporal --max-pairs 50 --seed 42
    latenet-batch --generators chemistry --max-pairs 20 --legs anthropic
    latenet-batch --generators wordnet --max-pairs 200 --ledger-dir data/
    latenet-batch --validate-from-ledger --validate-per-stratum 5
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

VALID_LEGS = {"ndif", "anthropic"}


def _validate_from_ledger(args: argparse.Namespace, ledger_dir: Path) -> None:
    """Sample unvalidated rows from the ledger and validate them."""
    from latenet.io.ledger import append_to_ledger

    ledger_path = ledger_dir / "ledger.parquet"
    if not ledger_path.exists():
        logger.error("No ledger found at %s", ledger_path)
        return

    ledger = pd.read_parquet(ledger_path)

    # Identify unvalidated rows: no llama_agrees column, or llama_agrees is NaN
    if "llama_agrees" in ledger.columns:
        unvalidated = ledger[ledger["llama_agrees"].isna()].copy()
    else:
        unvalidated = ledger.copy()

    logger.info(
        "Ledger has %d total rows, %d unvalidated",
        len(ledger), len(unvalidated),
    )

    if unvalidated.empty:
        logger.warning("No unvalidated rows in ledger.")
        return

    # Sample per stratum
    n = args.validate_per_stratum or 5
    pair_meta = unvalidated.groupby("pair_id").first()[["generator", "difficulty"]].reset_index()
    sampled_ids: list[str] = []
    for (gen, diff), group in pair_meta.groupby(["generator", "difficulty"]):
        selected = group.sample(n=min(n, len(group)), random_state=args.seed)
        sampled_ids.extend(selected["pair_id"].tolist())
        logger.info(
            "  validate sample %s/%s: %d of %d unvalidated pairs",
            gen, diff or "(none)", len(selected), len(group),
        )

    candidates_df = unvalidated[unvalidated["pair_id"].isin(sampled_ids)].copy()
    logger.info(
        "Selected %d rows (%d pairs) for validation from ledger",
        len(candidates_df), len(sampled_ids),
    )

    # Validate
    legs = set(args.legs)
    from latenet.validation.voting import run_parallel_validation
    from latenet.validation.filters import merge_verdicts

    checkpoint_dir = ledger_dir / ".validation_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for f in checkpoint_dir.glob("*_checkpoint.json"):
        f.unlink()

    ndif_verdicts, anthropic_verdicts = run_parallel_validation(
        candidates_df, legs=legs, checkpoint_dir=checkpoint_dir,
    )

    validated = merge_verdicts(candidates_df, ndif_verdicts, anthropic_verdicts)

    # Update ledger: remove the old unvalidated rows for these pair_ids, replace with validated
    remaining = ledger[~ledger["pair_id"].isin(sampled_ids)]
    updated = pd.concat([remaining, validated], ignore_index=True)
    updated.to_parquet(ledger_path)
    logger.info(
        "Ledger updated in-place: %d rows (%d validated this pass)",
        len(updated), len(validated),
    )

    if "contested" in validated.columns:
        clean = (validated["contested"] == False).sum()  # noqa: E712
        contested = (validated["contested"] == True).sum()  # noqa: E712
        logger.info(
            "Batch result: %d clean rows, %d contested, %d total",
            clean, contested, len(validated),
        )


def main():
    from scripts.generate import GENERATORS

    available = list(GENERATORS.keys())

    parser = argparse.ArgumentParser(
        description="Generate a batch of candidates, validate, and append to the ledger",
    )
    parser.add_argument(
        "--generators", nargs="+", default=available,
        help=f"Which generators to run (default: all). Available: {available}",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=50,
        help="Max pairs per generator in this batch (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--legs", nargs="+", default=list(VALID_LEGS),
        help=f"Validation legs to run (default: all). Options: {sorted(VALID_LEGS)}",
    )
    parser.add_argument(
        "--ledger-dir", type=str, default="data",
        help="Directory for the persistent ledger (default: data/)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Generate only, no validation. Useful for testing generators.",
    )
    parser.add_argument(
        "--validate-per-stratum", type=int, default=None,
        help="Only validate N pairs per (generator, difficulty) stratum. "
             "Generates all pairs but samples down before validation to save API costs.",
    )
    parser.add_argument(
        "--validate-from-ledger", action="store_true",
        help="Skip generation; sample unvalidated rows from the existing ledger "
             "for validation. Use with --validate-per-stratum to control sample size.",
    )
    args = parser.parse_args()

    legs = set(args.legs)
    unknown = legs - VALID_LEGS
    if unknown:
        parser.error(f"Unknown legs: {unknown}. Valid: {sorted(VALID_LEGS)}")

    ledger_dir = Path(args.ledger_dir)

    # --- 1. Load existing ledger for dedup ---
    from latenet.io.ledger import get_seen_statements, stamp_provenance, append_to_ledger

    # --- 1b. If --validate-from-ledger, skip generation entirely ---
    if args.validate_from_ledger:
        _validate_from_ledger(args, ledger_dir)
        return

    seen = get_seen_statements(ledger_dir)
    logger.info("Ledger has %d existing statements for dedup", len(seen))

    # --- 2. Generate candidates ---
    all_rows: list[dict] = []
    for gen_name in args.generators:
        if gen_name not in GENERATORS:
            logger.warning("Unknown generator: %s (skipping)", gen_name)
            continue

        gen_cls = GENERATORS[gen_name]
        kwargs: dict = {"seed": args.seed, "max_pairs": args.max_pairs}
        if gen_name == "wordnet":
            kwargs["max_depth"] = 5
            kwargs["max_false_per_true"] = 3

        gen = gen_cls(**kwargs)
        logger.info("Running %s generator (seed=%d, max_pairs=%d)...", gen_name, args.seed, args.max_pairs)

        count = 0
        dupes = 0
        for pair in gen.generate():
            true_text = pair.true_statement.strip().lower()
            false_text = pair.false_statement.strip().lower()
            if true_text in seen or false_text in seen:
                dupes += 1
                continue
            seen.add(true_text)
            seen.add(false_text)
            all_rows.extend(pair.to_rows())
            count += 1

        if dupes > 0:
            logger.info("  %s: %d unique pairs (%d duplicates skipped)", gen_name, count, dupes)
        else:
            logger.info("  %s: %d pairs", gen_name, count)

    if not all_rows:
        logger.warning("No new rows generated.")
        return

    candidates_df = pd.DataFrame(all_rows)
    logger.info("Generated %d candidate rows total", len(candidates_df))

    # --- 3. Stamp provenance ---
    candidates_df = stamp_provenance(candidates_df, seed=args.seed)

    # --- 4. Sample down to validate-per-stratum if requested ---
    unvalidated_rows: pd.DataFrame | None = None
    if args.validate_per_stratum is not None and not args.skip_validation:
        n = args.validate_per_stratum
        # Sample N pair_ids per (generator, difficulty) stratum
        pair_meta = candidates_df.groupby("pair_id").first()[["generator", "difficulty"]].reset_index()
        sampled_ids: list[str] = []
        for (gen, diff), group in pair_meta.groupby(["generator", "difficulty"]):
            selected = group.sample(n=min(n, len(group)), random_state=args.seed)
            sampled_ids.extend(selected["pair_id"].tolist())
            logger.info(
                "  validate sample %s/%s: %d of %d pairs",
                gen, diff or "(none)", len(selected), len(group),
            )
        # Keep the unsampled rows for the ledger (unvalidated)
        unvalidated_rows = candidates_df[~candidates_df["pair_id"].isin(sampled_ids)].copy()
        candidates_df = candidates_df[candidates_df["pair_id"].isin(sampled_ids)].copy()
        logger.info(
            "Sampled %d rows (%d pairs) for validation, %d rows deferred",
            len(candidates_df), len(sampled_ids), len(unvalidated_rows),
        )

    # --- 5. Validate (unless --skip-validation) ---
    if args.skip_validation:
        logger.info("Skipping validation (--skip-validation)")
        validated = candidates_df
    else:
        from latenet.validation.voting import run_parallel_validation
        from latenet.validation.filters import merge_verdicts

        checkpoint_dir = ledger_dir / ".validation_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Clear stale checkpoints
        for f in checkpoint_dir.glob("*_checkpoint.json"):
            f.unlink()

        ndif_verdicts, anthropic_verdicts = run_parallel_validation(
            candidates_df, legs=legs, checkpoint_dir=checkpoint_dir,
        )

        validated = merge_verdicts(candidates_df, ndif_verdicts, anthropic_verdicts)

    # --- 6. Append to ledger ---
    ledger = append_to_ledger(validated, ledger_dir=ledger_dir)
    # Also append unvalidated rows if we sampled down
    if unvalidated_rows is not None and len(unvalidated_rows) > 0:
        logger.info("Appending %d unvalidated rows to ledger", len(unvalidated_rows))
        ledger = append_to_ledger(unvalidated_rows, ledger_dir=ledger_dir)

    # --- 6. Summary ---
    if "contested" in validated.columns:
        clean = (validated["contested"] == False).sum()  # noqa: E712
        contested = (validated["contested"] == True).sum()  # noqa: E712
        logger.info(
            "Batch result: %d clean rows, %d contested, %d total",
            clean, contested, len(validated),
        )
    logger.info(
        "Ledger total: %d rows (%d unique pairs)",
        len(ledger),
        ledger["pair_id"].nunique() if "pair_id" in ledger.columns else 0,
    )


if __name__ == "__main__":
    main()
