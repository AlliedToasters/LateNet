"""CLI: Audit high-confidence disputes and optionally flip mislabeled pairs.

Routes disputed pairs by NDIF confidence:
- Low confidence (<threshold): genuinely unresolvable, stay excluded
- High confidence + both validators agree against label: likely a label error,
  sent to Sonnet for structured audit

Usage:
    latenet-audit --threshold 5.0 --dry-run
    latenet-audit --threshold 5.0 --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

_AUDIT_PROMPT = """\
You are auditing a disputed factual statement from a research dataset.

This statement was labeled {label} by a structured data source ({generator}). \
During validation, both Llama 405B (confidence: {confidence:.1f}) and Claude Sonnet \
independently judged it as {opposite}. This suggests the label may be wrong.

Statement: "{statement}"
Relation type: {relation_type}

Respond with ONLY a JSON object:
{{
    "verdict": "flip" | "exclude" | "keep_original",
    "reasoning": "Brief explanation"
}}

- "flip": The label is wrong — the statement's real-world truth value is {opposite}.
- "exclude": Genuinely ambiguous, remove from dataset.
- "keep_original": Validators were wrong, original label is correct.\
"""


def _find_audit_candidates(
    ledger: pd.DataFrame, threshold: float,
) -> pd.DataFrame:
    """Find high-confidence disputes where both validators agree against the label."""
    if "contested" not in ledger.columns:
        return pd.DataFrame()

    disputed = ledger[ledger["contested"] == True].copy()  # noqa: E712
    if disputed.empty:
        return disputed

    # Both validators must disagree with label
    both_disagree = (
        (disputed["llama_agrees"] == False)  # noqa: E712
        & (disputed["sonnet_agrees"] == False)  # noqa: E712
    )

    # High confidence
    high_conf = disputed["llama_confidence"] >= threshold

    candidates = disputed[both_disagree & high_conf].copy()
    return candidates


def _call_sonnet_audit(
    client, statement: str, label: bool, generator: str,
    relation_type: str, confidence: float,
) -> dict:
    """Send one disputed row to Sonnet for structured audit."""
    prompt = _AUDIT_PROMPT.format(
        label="True" if label else "False",
        opposite="False" if label else "True",
        generator=generator,
        confidence=confidence,
        statement=statement,
        relation_type=relation_type or "unknown",
    )

    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.warning("Audit call failed: %s", e)
                return {"verdict": "exclude", "reasoning": f"API error: {e}"}


def main():
    parser = argparse.ArgumentParser(
        description="Audit high-confidence disputes from the ledger",
    )
    parser.add_argument(
        "--threshold", type=float, default=5.0,
        help="Min NDIF confidence to route to audit (default: 5.0)",
    )
    parser.add_argument(
        "--ledger-dir", type=str, default="data",
        help="Directory containing the ledger (default: data/)",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply verdicts to the ledger. Without this flag, dry-run only.",
    )
    args = parser.parse_args()

    ledger_path = Path(args.ledger_dir) / "ledger.parquet"
    if not ledger_path.exists():
        logger.error("No ledger at %s", ledger_path)
        return

    ledger = pd.read_parquet(ledger_path)
    logger.info("Ledger: %d rows", len(ledger))

    # Find candidates
    candidates = _find_audit_candidates(ledger, args.threshold)
    if candidates.empty:
        logger.info("No high-confidence disputes above threshold %.1f", args.threshold)
        return

    # Group by pair_id for pair-level reporting
    pair_ids = candidates["pair_id"].unique()
    logger.info(
        "Found %d rows (%d pairs) for audit (threshold=%.1f)",
        len(candidates), len(pair_ids), args.threshold,
    )

    # Show routing summary
    all_disputed = ledger[ledger["contested"] == True]  # noqa: E712
    excluded_count = len(all_disputed) - len(candidates)
    logger.info(
        "Routing: %d rows to audit, %d low-confidence excluded",
        len(candidates), excluded_count,
    )

    # Per-generator breakdown
    gen_counts = candidates.groupby("generator").size()
    for gen, count in gen_counts.items():
        logger.info("  %s: %d rows", gen, count)

    if not args.apply:
        # Dry run: just show what would be audited
        logger.info("=== DRY RUN — showing candidates ===")
        for pid in pair_ids:
            rows = candidates[candidates["pair_id"] == pid]
            for _, row in rows.iterrows():
                conf = row.get("llama_confidence", 0)
                logger.info(
                    "  [%s] %s/%s label=%s conf=%.1f: %s",
                    pid[:8], row["generator"], row["difficulty"],
                    row["label"], conf, row["statement"][:70],
                )
        logger.info("Re-run with --apply to send to Sonnet and update ledger.")
        return

    # Live run: call Sonnet for each candidate row
    from anthropic import Anthropic
    client = Anthropic()

    verdicts: dict[int, dict] = {}  # ledger index -> audit result
    flip_count = 0
    exclude_count = 0
    keep_count = 0

    for idx, row in candidates.iterrows():
        result = _call_sonnet_audit(
            client,
            statement=row["statement"],
            label=bool(row["label"]),
            generator=row.get("generator", "unknown"),
            relation_type=row.get("relation_type", "unknown"),
            confidence=row.get("llama_confidence", 0),
        )

        verdict = result.get("verdict", "exclude")
        reasoning = result.get("reasoning", "")
        verdicts[idx] = result

        if verdict == "flip":
            flip_count += 1
        elif verdict == "exclude":
            exclude_count += 1
        else:
            keep_count += 1

        logger.info(
            "  [%d] %s -> %s: %s",
            idx, row["statement"][:50], verdict, reasoning[:60],
        )
        time.sleep(0.2)

    logger.info(
        "Audit results: %d flip, %d exclude, %d keep_original",
        flip_count, exclude_count, keep_count,
    )

    # Apply verdicts to ledger
    for idx, result in verdicts.items():
        verdict = result.get("verdict", "exclude")
        ledger.loc[idx, "audit_verdict"] = verdict
        ledger.loc[idx, "audit_reasoning"] = result.get("reasoning", "")

        if verdict == "flip":
            original = ledger.loc[idx, "label"]
            ledger.loc[idx, "original_label"] = original
            ledger.loc[idx, "label"] = not original
            ledger.loc[idx, "contested"] = False
        elif verdict == "keep_original":
            ledger.loc[idx, "contested"] = False

    # Write back
    ledger.to_parquet(ledger_path, index=False)
    logger.info("Ledger updated: %s", ledger_path)

    # Summary by generator
    audited = ledger[ledger["audit_verdict"].notna()]
    for gen, group in audited.groupby("generator"):
        flips = (group["audit_verdict"] == "flip").sum()
        excludes = (group["audit_verdict"] == "exclude").sum()
        keeps = (group["audit_verdict"] == "keep_original").sum()
        logger.info("  %s: %d flip, %d exclude, %d keep", gen, flips, excludes, keeps)


if __name__ == "__main__":
    main()
