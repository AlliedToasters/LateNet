"""LLM ensemble voting pipeline.

Two independent validation legs:
1. NDIF / Llama 405B Instruct — logit-level True/False via lmprobe
2. Anthropic API — Sonnet first pass, Opus escalation on disagreements

Each leg produces per-row verdicts that are merged for consensus filtering.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RowVerdict:
    """Single validator's judgment on one row."""
    says_true: bool | None = None
    agrees_with_label: bool | None = None
    confidence: float | None = None
    raw: str | None = None
    error: str | None = None


@dataclass
class ValidationResult:
    """Aggregated validation results across all rows."""
    verdicts: dict[int, dict[str, RowVerdict]] = field(default_factory=dict)
    # verdicts[row_idx]["llama"] = RowVerdict(...)
    # verdicts[row_idx]["sonnet"] = RowVerdict(...)
    # verdicts[row_idx]["opus"] = RowVerdict(...)  (only if escalated)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path: Path, data: dict) -> None:
    """Atomic write of checkpoint JSON."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(path)


def _load_checkpoint(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# NDIF / Llama 405B leg (via lmprobe)
# ---------------------------------------------------------------------------

LLAMA_MODEL = "meta-llama/Llama-3.1-405B-Instruct"
LLAMA_TOKENIZER = "meta-llama/Llama-3.1-405B"


def _chat_format(statement: str) -> str:
    """Format statement as Llama 3.1 instruct chat prompt."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Is this statement true or false? Answer with just 'True' or 'False'.\n\n"
        f"{statement}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def run_ndif_leg(
    df: pd.DataFrame,
    checkpoint_dir: Path | None = None,
    max_retries: int = 5,
) -> dict[int, RowVerdict]:
    """Run Llama 405B logit-level validation via NDIF/lmprobe.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'statement' and 'label' columns.
    checkpoint_dir : Path | None
        Directory for checkpoint files. If None, no checkpointing.
    max_retries : int
        Per-row retry attempts (handled by lmprobe's retry_with_backoff).

    Returns
    -------
    dict[int, RowVerdict]
        Keyed by DataFrame index.
    """
    import os
    from lmprobe.extraction import ActivationExtractor
    from lmprobe.retry import retry_with_backoff
    from transformers import AutoTokenizer

    # nnsight expects NNSIGHT_API_KEY; bridge from NDIF_API_KEY if needed
    if "NNSIGHT_API_KEY" not in os.environ and "NDIF_API_KEY" in os.environ:
        os.environ["NNSIGHT_API_KEY"] = os.environ["NDIF_API_KEY"]

    checkpoint_path = checkpoint_dir / "ndif_checkpoint.json" if checkpoint_dir else None
    existing = {}
    if checkpoint_path:
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt:
            existing = {int(k): v for k, v in ckpt.get("results", {}).items()}
            logger.info("Resumed NDIF checkpoint with %d existing results", len(existing))

    # Tokenizer for True/False token IDs
    tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
    true_id = tok.encode("True", add_special_tokens=False)[0]
    false_id = tok.encode("False", add_special_tokens=False)[0]

    # Lightweight remote extractor — no local GPU needed
    ext = ActivationExtractor(
        model_name=LLAMA_MODEL,
        device="cpu",
        layers=[0],
        backend="nnsight",
        remote=True,
    )

    verdicts: dict[int, RowVerdict] = {}

    for idx, row in df.iterrows():
        if idx in existing and existing[idx].get("error") is None:
            v = existing[idx]
            verdicts[idx] = RowVerdict(
                says_true=v.get("says_true"),
                agrees_with_label=v.get("agrees_with_label"),
                confidence=v.get("confidence"),
                raw=v.get("raw"),
                error=v.get("error"),
            )
            continue

        prompt = _chat_format(row["statement"])
        verdict = RowVerdict()

        try:
            _acts, _mask, logits, logits_indices = retry_with_backoff(
                lambda p=prompt: ext.extract_batch_with_logits(
                    [p], [0], remote=True, logit_top_k=10,
                ),
                max_retries=max_retries,
                base_delay=3.0,
                max_delay=120.0,
                context=f"NDIF row {idx}",
            )
            # With top_k, logits are (batch, seq, k) and logits_indices
            # maps positions back to vocab IDs
            last_logits = logits[0, -1, :]         # shape (k,)
            last_indices = logits_indices[0, -1, :]  # shape (k,)
            t_logit = float("-inf")
            f_logit = float("-inf")
            for i, vid in enumerate(last_indices.tolist()):
                if vid == true_id:
                    t_logit = last_logits[i].item()
                elif vid == false_id:
                    f_logit = last_logits[i].item()
            verdict.says_true = t_logit > f_logit
            verdict.confidence = abs(t_logit - f_logit)
            verdict.raw = f"true_logit={t_logit:.4f}, false_logit={f_logit:.4f}"
            verdict.agrees_with_label = verdict.says_true == bool(row["label"])
        except Exception as e:
            logger.error("NDIF failed for row %d after retries: %s", idx, e)
            verdict.error = str(e)

        verdicts[idx] = verdict

        # Checkpoint after each row
        if checkpoint_path:
            serializable = {
                str(k): {
                    "says_true": v.says_true,
                    "agrees_with_label": v.agrees_with_label,
                    "confidence": v.confidence,
                    "raw": v.raw,
                    "error": v.error,
                }
                for k, v in verdicts.items()
            }
            _save_checkpoint(checkpoint_path, {
                "results": serializable,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        # Brief pause to avoid hammering NDIF
        time.sleep(0.5)

    logger.info(
        "NDIF leg complete: %d/%d rows validated",
        sum(1 for v in verdicts.values() if v.error is None),
        len(verdicts),
    )
    return verdicts


# ---------------------------------------------------------------------------
# Anthropic leg (Sonnet + Opus escalation)
# ---------------------------------------------------------------------------

SONNET_MODEL = "claude-sonnet-4-6"
OPUS_MODEL = "claude-opus-4-6"

JUDGE_PROMPT = (
    "Is this statement true or false? "
    "Answer with exactly one word: 'True' or 'False'.\n\n"
    "{statement}"
)


def _query_anthropic(
    client: Any,
    model: str,
    statement: str,
    max_retries: int = 3,
) -> RowVerdict:
    """Query a single Anthropic model for a True/False judgment."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=5,
                messages=[{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(statement=statement),
                }],
            )
            text = response.content[0].text.strip().lower()
            if "true" in text:
                return RowVerdict(says_true=True, raw=text)
            elif "false" in text:
                return RowVerdict(says_true=False, raw=text)
            else:
                return RowVerdict(raw=text, error="ambiguous_response")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return RowVerdict(error=str(e))
    return RowVerdict(error="unreachable")


def run_anthropic_leg(
    df: pd.DataFrame,
    checkpoint_dir: Path | None = None,
    max_retries: int = 3,
) -> dict[int, dict[str, RowVerdict]]:
    """Run Anthropic validation: Sonnet on all rows, Opus on disagreements.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'statement' and 'label' columns.
    checkpoint_dir : Path | None
        Directory for checkpoint files.
    max_retries : int
        Per-call retry attempts.

    Returns
    -------
    dict[int, dict[str, RowVerdict]]
        Keyed by row index, values are {"sonnet": ..., "opus": ...}.
    """
    from anthropic import Anthropic

    client = Anthropic()
    checkpoint_path = checkpoint_dir / "anthropic_checkpoint.json" if checkpoint_dir else None

    existing: dict[str, Any] = {}
    if checkpoint_path:
        ckpt = _load_checkpoint(checkpoint_path)
        if ckpt:
            existing = ckpt.get("results", {})
            logger.info("Resumed Anthropic checkpoint with %d existing results", len(existing))

    results: dict[int, dict[str, RowVerdict]] = {}

    # Phase 1: Sonnet on all rows
    logger.info("Phase 1: Running Sonnet on %d rows...", len(df))
    for idx, row in df.iterrows():
        str_idx = str(idx)
        if str_idx in existing and "sonnet" in existing[str_idx]:
            s = existing[str_idx]["sonnet"]
            verdict = RowVerdict(
                says_true=s.get("says_true"),
                agrees_with_label=s.get("agrees_with_label"),
                raw=s.get("raw"),
                error=s.get("error"),
            )
        else:
            verdict = _query_anthropic(client, SONNET_MODEL, row["statement"], max_retries)
            if verdict.says_true is not None:
                verdict.agrees_with_label = verdict.says_true == bool(row["label"])
            time.sleep(0.1)  # rate limit courtesy

        results[idx] = {"sonnet": verdict}

        # Checkpoint
        if checkpoint_path:
            _save_anthropic_checkpoint(checkpoint_path, results)

    sonnet_agrees = sum(
        1 for r in results.values()
        if r["sonnet"].agrees_with_label is True
    )
    logger.info("Sonnet pass: %d/%d agree with labels", sonnet_agrees, len(results))

    # Phase 2: Opus escalation on Sonnet disagreements
    disagree_idxs = [
        idx for idx, r in results.items()
        if r["sonnet"].agrees_with_label is False
    ]
    logger.info("Phase 2: Escalating %d disagreements to Opus...", len(disagree_idxs))

    for idx in disagree_idxs:
        str_idx = str(idx)
        if str_idx in existing and "opus" in existing.get(str_idx, {}):
            o = existing[str_idx]["opus"]
            verdict = RowVerdict(
                says_true=o.get("says_true"),
                agrees_with_label=o.get("agrees_with_label"),
                raw=o.get("raw"),
                error=o.get("error"),
            )
        else:
            row = df.loc[idx]
            verdict = _query_anthropic(client, OPUS_MODEL, row["statement"], max_retries)
            if verdict.says_true is not None:
                verdict.agrees_with_label = verdict.says_true == bool(row["label"])
            time.sleep(0.2)

        results[idx]["opus"] = verdict

        if checkpoint_path:
            _save_anthropic_checkpoint(checkpoint_path, results)

    if disagree_idxs:
        opus_agrees = sum(
            1 for idx in disagree_idxs
            if results[idx].get("opus", RowVerdict()).agrees_with_label is True
        )
        logger.info("Opus escalation: %d/%d agree with labels", opus_agrees, len(disagree_idxs))

    return results


def _save_anthropic_checkpoint(path: Path, results: dict[int, dict[str, RowVerdict]]) -> None:
    """Serialize anthropic results to checkpoint."""
    serializable: dict[str, dict[str, dict]] = {}
    for idx, models in results.items():
        serializable[str(idx)] = {}
        for model_name, verdict in models.items():
            serializable[str(idx)][model_name] = {
                "says_true": verdict.says_true,
                "agrees_with_label": verdict.agrees_with_label,
                "raw": verdict.raw,
                "error": verdict.error,
            }
    _save_checkpoint(path, {
        "results": serializable,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
