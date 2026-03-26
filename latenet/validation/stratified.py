"""Stratified dataset builder with generate-validate-accept loop.

Produces a balanced dataset where every (generator, difficulty) stratum has
exactly N rows, and every row has been validated by both Llama 405B (NDIF)
and Claude Sonnet — both agreeing with the ground-truth label.

Process
-------
1. Define target quotas: ``rows_per_stratum`` per (generator, difficulty) cell.
2. For each under-filled stratum, generate a batch of candidate pairs
   (oversampled to account for the expected rejection rate).
3. Validate candidates with Llama 405B (logit-level) and Sonnet.
4. **Accept** rows where both validators agree with the ground-truth label.
5. **Dispute** rows where either validator disagrees. These go to a sidecar
   ``disputes.parquet`` for analysis — they are NOT included in the clean
   dataset. Sonnet disagreements are escalated to Opus for the dispute record.
6. Repeat from (2) until every stratum quota is met, or ``max_rounds`` is
   reached.

The clean output has *exactly* ``rows_per_stratum`` rows per stratum (pairs,
so 2x that in statement rows). Disputed rows are preserved separately with
full validation metadata.

Determinism
-----------
Each round increments the generator seed so new candidates are produced on
retry, while remaining reproducible: ``seed + round * 1000``.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

from latenet.generators.base import BaseGenerator
from latenet.types import ContrastivePair

from .filters import merge_verdicts, validation_summary
from .voting import run_anthropic_leg, run_ndif_leg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator registry (mirrors scripts/generate.py)
# ---------------------------------------------------------------------------

def _get_generators() -> dict[str, type[BaseGenerator]]:
    from latenet.generators.biology import BiologyGenerator
    from latenet.generators.chemistry import ChemistryGenerator
    from latenet.generators.geography import GeographyGenerator
    from latenet.generators.temporal import TemporalGenerator
    from latenet.generators.wordnet_gen import WordNetGenerator

    return {
        "wordnet": WordNetGenerator,
        "geography": GeographyGenerator,
        "chemistry": ChemistryGenerator,
        "biology": BiologyGenerator,
        "temporal": TemporalGenerator,
    }


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

StratumKey = tuple[str, str]  # (generator, difficulty)


def _discover_strata(generators: list[str]) -> list[StratumKey]:
    """Return the (generator, difficulty) strata we expect to fill.

    Determined by doing a small probe generation per generator.
    """
    registry = _get_generators()
    strata: set[StratumKey] = set()
    for gen_name in generators:
        gen_cls = registry[gen_name]
        kwargs: dict = {"seed": 0, "max_pairs": 50}
        if gen_name == "wordnet":
            kwargs["max_depth"] = 3
        gen = gen_cls(**kwargs)
        for pair in gen.generate():
            strata.add((pair.generator, pair.difficulty))
    return sorted(strata)


def build_stratified_dataset(
    generators: list[str],
    rows_per_stratum: int,
    seed: int = 42,
    max_rounds: int = 10,
    oversample_factor: float = 3.0,
    output_dir: Path = Path("."),
    legs: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the generate-validate-accept loop until strata are filled.

    Parameters
    ----------
    generators : list[str]
        Which generators to use (e.g. ["biology", "geography", "temporal"]).
    rows_per_stratum : int
        Target number of *pairs* per (generator, difficulty) cell.
        The output will have 2x this many statement rows per cell.
    seed : int
        Base random seed. Each round uses seed + round * 1000.
    max_rounds : int
        Maximum generation-validation rounds before giving up.
    oversample_factor : float
        Generate this many multiples of the remaining quota per round
        to account for expected rejections.
    output_dir : Path
        Directory for checkpoints and output files.
    legs : set[str] | None
        Which validation legs to run. Default: {"ndif", "anthropic"}.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (clean_df, disputes_df) — the clean dataset and disputed rows.
    """
    if legs is None:
        legs = {"ndif", "anthropic"}

    registry = _get_generators()
    checkpoint_dir = output_dir / ".validation_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Discover available strata
    logger.info("Discovering strata from generators: %s", generators)
    all_strata = _discover_strata(generators)
    logger.info("Found %d strata: %s", len(all_strata), all_strata)

    # Track accepted pairs per stratum
    accepted: dict[StratumKey, list[dict]] = defaultdict(list)
    all_disputes: list[dict] = []

    for round_num in range(max_rounds):
        # Check what's still needed
        needed: dict[StratumKey, int] = {}
        for stratum in all_strata:
            have = len(accepted[stratum])
            if have < rows_per_stratum:
                needed[stratum] = rows_per_stratum - have
        if not needed:
            logger.info("All strata filled after %d rounds.", round_num)
            break

        total_needed = sum(needed.values())
        logger.info(
            "=== Round %d/%d — need %d more pairs across %d strata ===",
            round_num + 1, max_rounds, total_needed, len(needed),
        )

        # Group needed strata by generator
        gen_needs: dict[str, int] = defaultdict(int)
        for (gen_name, _diff), count in needed.items():
            gen_needs[gen_name] += count

        # Generate candidates
        round_seed = seed + round_num * 1000
        candidate_rows: list[dict] = []

        for gen_name, need_count in gen_needs.items():
            gen_cls = registry[gen_name]
            target = int(need_count * oversample_factor)
            kwargs: dict = {"seed": round_seed, "max_pairs": target}
            if gen_name == "wordnet":
                kwargs["max_depth"] = 5
                kwargs["max_false_per_true"] = 3
            gen = gen_cls(**kwargs)

            count = 0
            for pair in gen.generate():
                stratum = (pair.generator, pair.difficulty)
                if stratum in needed:
                    candidate_rows.extend(pair.to_rows())
                    count += 1
            logger.info("  %s: generated %d candidate pairs", gen_name, count)

        if not candidate_rows:
            logger.warning("No candidates generated in round %d, stopping.", round_num + 1)
            break

        candidates_df = pd.DataFrame(candidate_rows)
        logger.info("Round %d: %d candidate rows to validate", round_num + 1, len(candidates_df))

        # Clear per-round checkpoints
        for f in checkpoint_dir.glob("*_checkpoint.json"):
            f.unlink()

        # Validate
        ndif_verdicts = None
        anthropic_verdicts = None

        if "ndif" in legs:
            ndif_verdicts = run_ndif_leg(candidates_df, checkpoint_dir=checkpoint_dir)

        if "anthropic" in legs:
            anthropic_verdicts = run_anthropic_leg(candidates_df, checkpoint_dir=checkpoint_dir)

        # Merge verdicts
        validated = merge_verdicts(candidates_df, ndif_verdicts, anthropic_verdicts)

        # Classify each row
        both_agree = pd.Series(True, index=validated.index)
        if "llama_agrees" in validated.columns:
            both_agree = both_agree & (validated["llama_agrees"] == True)  # noqa: E712
        if "sonnet_agrees" in validated.columns:
            both_agree = both_agree & (validated["sonnet_agrees"] == True)  # noqa: E712
        # Also exclude rows with errors (treat as not validated)
        if "llama_error" in validated.columns:
            both_agree = both_agree & validated["llama_error"].isna()
        if "sonnet_error" in validated.columns:
            both_agree = both_agree & validated["sonnet_error"].isna()

        clean_rows = validated[both_agree]
        dispute_rows = validated[~both_agree]

        # Collect disputes
        if len(dispute_rows) > 0:
            all_disputes.append(dispute_rows)
            logger.info(
                "Round %d: %d disputed rows (saved to sidecar)",
                round_num + 1, len(dispute_rows),
            )

        # Accept clean rows into strata (by pair_id to keep true+false together)
        accepted_this_round = 0
        for pair_id, pair_rows in clean_rows.groupby("pair_id"):
            if len(pair_rows) != 2:
                continue
            gen = pair_rows.iloc[0]["generator"]
            diff = pair_rows.iloc[0]["difficulty"]
            stratum = (gen, diff)
            if stratum not in needed:
                continue
            if len(accepted[stratum]) >= rows_per_stratum:
                continue
            accepted[stratum].append(pair_rows.to_dict("records"))
            accepted_this_round += 1

        logger.info(
            "Round %d: accepted %d pairs, disputed %d rows",
            round_num + 1, accepted_this_round, len(dispute_rows),
        )

        # Progress report
        for stratum in sorted(needed.keys()):
            have = len(accepted[stratum])
            logger.info(
                "  %s/%s: %d/%d",
                stratum[0], stratum[1] or "(none)", have, rows_per_stratum,
            )
    else:
        logger.warning(
            "Reached max_rounds=%d. Some strata may be under-filled.", max_rounds
        )

    # Assemble final datasets
    clean_records: list[dict] = []
    for stratum in all_strata:
        for pair_rows in accepted[stratum]:
            clean_records.extend(pair_rows)

    clean_df = pd.DataFrame(clean_records) if clean_records else pd.DataFrame()
    disputes_df = pd.concat(all_disputes, ignore_index=True) if all_disputes else pd.DataFrame()

    # Summary
    if len(clean_df) > 0:
        logger.info("=== Final Dataset ===")
        strata_counts = clean_df.groupby(["generator", "difficulty"])["pair_id"].nunique()
        for (gen, diff), count in strata_counts.items():
            logger.info("  %s/%s: %d pairs", gen, diff or "(none)", count)
        logger.info(
            "Total: %d pairs (%d rows), %d disputed rows in sidecar",
            clean_df["pair_id"].nunique(), len(clean_df), len(disputes_df),
        )

    return clean_df, disputes_df
