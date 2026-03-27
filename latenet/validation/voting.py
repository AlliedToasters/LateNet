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

    # Lightweight remote extractor — logits only, no activations downloaded
    ext = ActivationExtractor(
        model_name=LLAMA_MODEL,
        device="cpu",
        layers=[],
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
            logits, _mask, logits_indices = retry_with_backoff(
                lambda p=prompt: ext.extract_logits_only(
                    [p], remote=True, logit_top_k=10,
                ),
                max_retries=max_retries,
                base_delay=3.0,
                max_delay=120.0,
                context=f"NDIF row {idx}",
            )
            # logits: (batch, seq, k), logits_indices: (batch, seq, k) vocab IDs
            last_logits = logits[0, -1, :]         # shape (k,)
            last_indices = logits_indices[0, -1, :]  # shape (k,)
            t_logit = float("-inf")
            f_logit = float("-inf")
            for i, vid in enumerate(last_indices.tolist()):
                if vid == true_id:
                    t_logit = last_logits[i].item()
                elif vid == false_id:
                    f_logit = last_logits[i].item()

            # Guard: neither True nor False in top-k
            if t_logit == float("-inf") and f_logit == float("-inf"):
                top_ids = last_indices.tolist()
                logger.warning(
                    "NDIF row %d: neither True nor False in top-k. "
                    "Top token IDs: %s. Statement: %.80s",
                    idx, top_ids, row["statement"],
                )
                verdict.confidence = float("-inf")
                verdict.raw = f"true_logit=-inf, false_logit=-inf, top_ids={top_ids}"
                verdict.error = "true_false_not_in_topk"
            else:
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


def run_parallel_validation(
    df: pd.DataFrame,
    legs: set[str] | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[dict[int, RowVerdict] | None, dict[int, dict[str, RowVerdict]] | None]:
    """Run NDIF and Anthropic legs in parallel, per row.

    Every completed row has verdicts from both legs, enabling graceful
    early stopping — if the process is killed, all finished rows are
    atomically complete and usable for consensus.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'statement' and 'label' columns.
    legs : set[str] | None
        Which legs to run. Default: {"ndif", "anthropic"}.
    checkpoint_dir : Path | None
        Directory for checkpoint files.

    Returns
    -------
    (ndif_verdicts, anthropic_verdicts) — same format as the individual
    leg functions, suitable for passing to merge_verdicts().
    """
    from concurrent.futures import ThreadPoolExecutor

    if legs is None:
        legs = {"ndif", "anthropic"}

    run_ndif = "ndif" in legs
    run_anthropic = "anthropic" in legs

    # --- Set up NDIF resources (shared across rows) ---
    ndif_ext = None
    ndif_tok = None
    ndif_true_id = None
    ndif_false_id = None

    if run_ndif:
        import os
        from lmprobe.extraction import ActivationExtractor
        from transformers import AutoTokenizer

        if "NNSIGHT_API_KEY" not in os.environ and "NDIF_API_KEY" in os.environ:
            os.environ["NNSIGHT_API_KEY"] = os.environ["NDIF_API_KEY"]

        ndif_tok = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
        ndif_true_id = ndif_tok.encode("True", add_special_tokens=False)[0]
        ndif_false_id = ndif_tok.encode("False", add_special_tokens=False)[0]
        ndif_ext = ActivationExtractor(
            model_name=LLAMA_MODEL,
            device="cpu",
            layers=[],
            backend="nnsight",
            remote=True,
        )

    # --- Set up Anthropic client ---
    anthropic_client = None
    if run_anthropic:
        from anthropic import Anthropic
        anthropic_client = Anthropic()

    # --- Load checkpoints ---
    ndif_checkpoint_path = checkpoint_dir / "ndif_checkpoint.json" if checkpoint_dir else None
    anthropic_checkpoint_path = checkpoint_dir / "anthropic_checkpoint.json" if checkpoint_dir else None

    existing_ndif: dict[int, dict] = {}
    existing_anthropic: dict[str, dict] = {}

    if ndif_checkpoint_path:
        ckpt = _load_checkpoint(ndif_checkpoint_path)
        if ckpt:
            existing_ndif = {int(k): v for k, v in ckpt.get("results", {}).items()}
            logger.info("Resumed NDIF checkpoint with %d results", len(existing_ndif))

    if anthropic_checkpoint_path:
        ckpt = _load_checkpoint(anthropic_checkpoint_path)
        if ckpt:
            existing_anthropic = ckpt.get("results", {})
            logger.info("Resumed Anthropic checkpoint with %d results", len(existing_anthropic))

    # --- Result accumulators ---
    ndif_verdicts: dict[int, RowVerdict] = {}
    anthropic_verdicts: dict[int, dict[str, RowVerdict]] = {}

    def _ndif_one_row(idx: int, row: pd.Series) -> RowVerdict:
        """Validate a single row via NDIF."""
        from lmprobe.retry import retry_with_backoff

        if idx in existing_ndif and existing_ndif[idx].get("error") is None:
            v = existing_ndif[idx]
            return RowVerdict(
                says_true=v.get("says_true"),
                agrees_with_label=v.get("agrees_with_label"),
                confidence=v.get("confidence"),
                raw=v.get("raw"),
            )

        prompt = _chat_format(row["statement"])
        try:
            logits, _mask, logits_indices = retry_with_backoff(
                lambda p=prompt: ndif_ext.extract_logits_only(
                    [p], remote=True, logit_top_k=10,
                ),
                max_retries=5,
                base_delay=3.0,
                max_delay=120.0,
                context=f"NDIF row {idx}",
            )
            last_logits = logits[0, -1, :]
            last_indices = logits_indices[0, -1, :]
            t_logit = float("-inf")
            f_logit = float("-inf")
            for i, vid in enumerate(last_indices.tolist()):
                if vid == ndif_true_id:
                    t_logit = last_logits[i].item()
                elif vid == ndif_false_id:
                    f_logit = last_logits[i].item()

            # Guard: neither True nor False in top-k
            if t_logit == float("-inf") and f_logit == float("-inf"):
                top_ids = last_indices.tolist()
                logger.warning(
                    "NDIF row %d: neither True nor False in top-k. "
                    "Top token IDs: %s. Statement: %.80s",
                    idx, top_ids, row["statement"],
                )
                return RowVerdict(
                    confidence=float("-inf"),
                    raw=f"true_logit=-inf, false_logit=-inf, top_ids={top_ids}",
                    error="true_false_not_in_topk",
                )

            verdict = RowVerdict(
                says_true=t_logit > f_logit,
                confidence=abs(t_logit - f_logit),
                raw=f"true_logit={t_logit:.4f}, false_logit={f_logit:.4f}",
            )
            verdict.agrees_with_label = verdict.says_true == bool(row["label"])
            return verdict
        except Exception as e:
            logger.error("NDIF failed for row %d: %s", idx, e)
            return RowVerdict(error=str(e))

    def _anthropic_one_row(idx: int, row: pd.Series) -> dict[str, RowVerdict]:
        """Validate a single row via Sonnet (+ Opus escalation if needed)."""
        str_idx = str(idx)
        result: dict[str, RowVerdict] = {}

        # Sonnet
        if str_idx in existing_anthropic and "sonnet" in existing_anthropic[str_idx]:
            s = existing_anthropic[str_idx]["sonnet"]
            sonnet_v = RowVerdict(
                says_true=s.get("says_true"),
                agrees_with_label=s.get("agrees_with_label"),
                raw=s.get("raw"),
                error=s.get("error"),
            )
        else:
            sonnet_v = _query_anthropic(anthropic_client, SONNET_MODEL, row["statement"])
            if sonnet_v.says_true is not None:
                sonnet_v.agrees_with_label = sonnet_v.says_true == bool(row["label"])
        result["sonnet"] = sonnet_v

        # Opus escalation if Sonnet disagreed
        if sonnet_v.agrees_with_label is False:
            if str_idx in existing_anthropic and "opus" in existing_anthropic.get(str_idx, {}):
                o = existing_anthropic[str_idx]["opus"]
                opus_v = RowVerdict(
                    says_true=o.get("says_true"),
                    agrees_with_label=o.get("agrees_with_label"),
                    raw=o.get("raw"),
                    error=o.get("error"),
                )
            else:
                opus_v = _query_anthropic(anthropic_client, OPUS_MODEL, row["statement"])
                if opus_v.says_true is not None:
                    opus_v.agrees_with_label = opus_v.says_true == bool(row["label"])
            result["opus"] = opus_v

        return result

    # --- Per-row parallel loop ---
    logger.info("=== Per-row parallel validation: %d rows, legs=%s ===", len(df), legs)
    completed = 0
    validation_start = time.monotonic()
    row_times: list[float] = []
    ndif_times: list[float] = []
    anthropic_times: list[float] = []
    opus_escalations = 0

    with ThreadPoolExecutor(max_workers=2) as pool:
        for idx, row in df.iterrows():
            row_start = time.monotonic()
            futures = {}
            if run_ndif:
                futures["ndif"] = pool.submit(_ndif_one_row, idx, row)
            if run_anthropic:
                futures["anthropic"] = pool.submit(_anthropic_one_row, idx, row)

            if "ndif" in futures:
                t0 = time.monotonic()
                ndif_verdicts[idx] = futures["ndif"].result()
                ndif_times.append(time.monotonic() - t0)
            if "anthropic" in futures:
                t0 = time.monotonic()
                anthropic_verdicts[idx] = futures["anthropic"].result()
                anthropic_times.append(time.monotonic() - t0)
                if "opus" in anthropic_verdicts[idx]:
                    opus_escalations += 1

            row_elapsed = time.monotonic() - row_start
            row_times.append(row_elapsed)
            completed += 1

            # Per-row progress summary
            parts = []
            if run_ndif:
                nv = ndif_verdicts.get(idx)
                if nv and nv.error:
                    parts.append(f"ndif={nv.error}")
                elif nv:
                    parts.append(f"ndif={'T' if nv.says_true else 'F'} conf={nv.confidence:.1f}")
            if run_anthropic:
                av = anthropic_verdicts.get(idx, {})
                sv = av.get("sonnet")
                if sv and sv.error:
                    parts.append(f"sonnet={sv.error}")
                elif sv:
                    parts.append(f"sonnet={'T' if sv.says_true else 'F'}")
                if "opus" in av:
                    ov = av["opus"]
                    parts.append(f"opus={'T' if ov.says_true else 'F'}")
            logger.info(
                "  [%d/%d] %.1fs %s | %s",
                completed, len(df), row_elapsed,
                " ".join(parts),
                row["statement"][:60],
            )

            # Periodic throughput summary every 25 rows
            if completed % 25 == 0:
                elapsed_so_far = time.monotonic() - validation_start
                rate = completed / elapsed_so_far * 60
                remaining = len(df) - completed
                eta_min = remaining / (completed / elapsed_so_far) / 60
                logger.info(
                    "  --- %.1f rows/min | %d/%d done | ETA %.0f min ---",
                    rate, completed, len(df), eta_min,
                )

            # Checkpoint both legs after each row
            if ndif_checkpoint_path and run_ndif:
                serializable = {
                    str(k): {
                        "says_true": v.says_true,
                        "agrees_with_label": v.agrees_with_label,
                        "confidence": v.confidence,
                        "raw": v.raw,
                        "error": v.error,
                    }
                    for k, v in ndif_verdicts.items()
                }
                _save_checkpoint(ndif_checkpoint_path, {
                    "results": serializable,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

            if anthropic_checkpoint_path and run_anthropic:
                _save_anthropic_checkpoint(anthropic_checkpoint_path, anthropic_verdicts)

    total_elapsed = time.monotonic() - validation_start
    logger.info("Parallel validation complete: %d rows in %.1fs", completed, total_elapsed)

    # Timing summary
    if row_times:
        avg_row = sum(row_times) / len(row_times)
        logger.info(
            "  Timing: %.1fs/row avg, %.1fs/row median",
            avg_row, sorted(row_times)[len(row_times) // 2],
        )
    if ndif_times:
        logger.info(
            "  NDIF:   %.2fs avg, %.2fs median",
            sum(ndif_times) / len(ndif_times),
            sorted(ndif_times)[len(ndif_times) // 2],
        )
    if anthropic_times:
        logger.info(
            "  Anthropic: %.2fs avg, %.2fs median (%d Opus escalations)",
            sum(anthropic_times) / len(anthropic_times),
            sorted(anthropic_times)[len(anthropic_times) // 2],
            opus_escalations,
        )
    if total_elapsed > 0 and completed > 0:
        logger.info(
            "  Throughput: %.1f rows/min, ETA for 1000 rows: %.0f min",
            completed / total_elapsed * 60,
            1000 / (completed / total_elapsed) / 60,
        )

    return (
        ndif_verdicts if run_ndif else None,
        anthropic_verdicts if run_anthropic else None,
    )


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
