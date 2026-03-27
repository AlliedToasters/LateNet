"""Grammar scanning (Layer 2) and correction (Layer 3).

Layer 2: LanguageTool — cheap local detection. Flags grammar issues but does
         NOT correct them.  Degrades gracefully if Java is unavailable.

Layer 3: Sonnet — intelligent correction.  Only called on rows flagged by
         Layer 2.  Returns structured JSON with corrected/unchanged/flagged
         status per statement.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from latenet.types import ContrastivePair

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 2: LanguageTool scanner
# ---------------------------------------------------------------------------

@dataclass
class GrammarFlag:
    """A single grammar issue detected by LanguageTool."""

    statement: str
    rule_id: str
    category: str
    message: str
    context: str
    offset: int
    length: int


# Rules that produce false positives on our short declarative statements.
_DEFAULT_DISABLED_RULES = frozenset({
    "PASSIVE_VOICE",           # style, not grammar
    "TOO_LONG_SENTENCE",       # style
    "WORDINESS",               # style
    "MORFOLOGIK_RULE_EN_US",   # spelling — false positives on proper nouns / scientific terms
    "SENTENCE_FRAGMENT",       # our statements are short declaratives, not fragments
})


class GrammarScanner:
    """Cheap local grammar detection using LanguageTool.

    Flags issues but does NOT correct them — correction is delegated to
    Sonnet (Layer 3) for flagged rows only.

    Requires a Java Runtime Environment.  If Java is unavailable the scanner
    degrades gracefully: ``available`` is ``False`` and ``scan()`` returns
    an empty list.
    """

    def __init__(
        self,
        language: str = "en-US",
        disabled_rules: frozenset[str] | None = None,
    ) -> None:
        self._disabled_rules = disabled_rules or _DEFAULT_DISABLED_RULES
        self._tool = None
        self._available = False
        try:
            import language_tool_python

            self._tool = language_tool_python.LanguageTool(language)
            self._available = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("LanguageTool unavailable (%s) — grammar scanning disabled", exc)

    @property
    def available(self) -> bool:
        return self._available

    def scan(self, statement: str) -> list[GrammarFlag]:
        """Scan a statement for grammar issues. Returns list of flags (empty = clean)."""
        if not self._available or self._tool is None:
            return []
        matches = self._tool.check(statement)
        return [
            GrammarFlag(
                statement=statement,
                rule_id=m.rule_id,
                category=m.category,
                message=m.message,
                context=m.context,
                offset=m.offset,
                length=m.error_length,
            )
            for m in matches
            if m.rule_id not in self._disabled_rules
        ]

    def scan_pair(self, pair: ContrastivePair) -> dict:
        """Scan both statements in a pair.

        Returns ``{'true_flags': [...], 'false_flags': [...], 'needs_correction': bool}``.
        """
        true_flags = self.scan(pair.true_statement)
        false_flags = self.scan(pair.false_statement)
        return {
            "true_flags": true_flags,
            "false_flags": false_flags,
            "needs_correction": bool(true_flags or false_flags),
        }


# ---------------------------------------------------------------------------
# Layer 3: Sonnet grammar correction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a grammar corrector for a dataset of factual statements used in AI research.

For each statement, do one of three things:

1. CORRECT: Fix grammar errors and smooth awkward phrasing while strictly preserving the factual meaning. This includes:
   - Article agreement (a/an)
   - Subject-verb agreement
   - Pluralization errors
   - Repeated words
   - Awkward or stilted phrasing that can be made more natural
   - Missing or incorrect punctuation

2. UNCHANGED: If the statement has no grammar issues, return it as unchanged.

3. FLAG: If the statement is too malformed to correct without risking a change in factual meaning, flag it. Explain why in the changes field.

4. NONSENSICAL: If the statement is semantically incoherent, tautological, or meaningless regardless of grammar (e.g. "A food has a food", "blue is a organization"). These are generation errors, not grammar errors. Explain what makes it nonsensical in the changes field.

Rules:
- NEVER change entities, relationships, quantities, or truth values
- NEVER add information, qualifiers, or hedging
- NEVER change proper nouns or technical terms
- Preserve the period at the end of every statement
- When smoothing awkward phrasing, keep the same core sentence structure
- If in doubt about whether a correction changes meaning, FLAG instead of correcting

Respond with ONLY a JSON object matching this schema. No markdown, no backticks, no preamble.
{
    "statements": [
        {
            "index": <int>,
            "original": <string>,
            "corrected": <string or null>,
            "status": "corrected" | "unchanged" | "flagged" | "nonsensical",
            "changes": <string description of changes or null>
        }
    ]
}"""


@dataclass
class CorrectionResult:
    """Result for one statement from the Sonnet corrector."""

    original: str
    corrected: str | None
    status: str  # "corrected", "unchanged", "flagged", or "nonsensical"
    changes: str | None


@dataclass
class PairCorrectionResult:
    """Correction results for a ContrastivePair."""

    pair: ContrastivePair
    true_result: CorrectionResult
    false_result: CorrectionResult
    excluded: bool = False


class GrammarCorrector:
    """Send flagged statements to Sonnet for structured grammar correction.

    Requires ``ANTHROPIC_API_KEY`` to be set.  If the Anthropic SDK is not
    installed or API calls fail, degrades gracefully and keeps original text.
    """

    def __init__(
        self,
        batch_size: int = 50,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 3,
    ) -> None:
        self.batch_size = batch_size
        self.model = model
        self.max_retries = max_retries
        self._client = None
        self._available = False
        try:
            from anthropic import Anthropic

            self._client = Anthropic()
            self._available = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anthropic SDK unavailable (%s) — grammar correction disabled", exc)

    @property
    def available(self) -> bool:
        return self._available

    def correct_batch(self, statements: list[str]) -> list[CorrectionResult]:
        """Send a batch of flagged statements to Sonnet for correction.

        Returns one :class:`CorrectionResult` per input statement.  On
        failure, falls back to ``status='unchanged'`` for every statement.
        """
        if not self._available or self._client is None:
            return [
                CorrectionResult(original=s, corrected=None, status="unchanged", changes=None)
                for s in statements
            ]

        user_content = json.dumps({
            "statements": [
                {"index": i, "original": s}
                for i, s in enumerate(statements)
            ]
        })

        raw = self._call_api(user_content)
        if raw is None:
            return [
                CorrectionResult(original=s, corrected=None, status="unchanged", changes=None)
                for s in statements
            ]

        return self._parse_response(raw, statements)

    def correct_pairs(
        self,
        flagged_pairs: list[tuple[ContrastivePair, dict]],
    ) -> list[PairCorrectionResult]:
        """Correct grammar in flagged ContrastivePairs.

        ``flagged_pairs`` is a list of ``(pair, scan_result)`` where
        ``scan_result`` is the dict returned by ``GrammarScanner.scan_pair()``.

        Collects all flagged statements, batches them, sends to Sonnet,
        applies corrections back to pairs.

        Pairs where any statement gets ``status='flagged'`` are excluded.
        """
        if not flagged_pairs:
            return []

        # Flatten: collect (pair_index, "true"/"false", statement)
        entries: list[tuple[int, str, str]] = []
        for i, (pair, scan) in enumerate(flagged_pairs):
            if scan["true_flags"]:
                entries.append((i, "true", pair.true_statement))
            if scan["false_flags"]:
                entries.append((i, "false", pair.false_statement))

        # Batch and correct
        all_results: list[CorrectionResult] = []
        statements = [e[2] for e in entries]
        for batch_start in range(0, len(statements), self.batch_size):
            batch = statements[batch_start:batch_start + self.batch_size]
            results = self.correct_batch(batch)
            all_results.extend(results)

        # Map corrections back to pairs
        pair_true: dict[int, CorrectionResult] = {}
        pair_false: dict[int, CorrectionResult] = {}
        for (pair_idx, side, _stmt), result in zip(entries, all_results):
            if side == "true":
                pair_true[pair_idx] = result
            else:
                pair_false[pair_idx] = result

        # Build results
        output: list[PairCorrectionResult] = []
        for i, (pair, _scan) in enumerate(flagged_pairs):
            true_res = pair_true.get(i, CorrectionResult(
                original=pair.true_statement, corrected=None,
                status="unchanged", changes=None,
            ))
            false_res = pair_false.get(i, CorrectionResult(
                original=pair.false_statement, corrected=None,
                status="unchanged", changes=None,
            ))
            excluded = true_res.status in ("flagged", "nonsensical") or false_res.status in ("flagged", "nonsensical")

            if not excluded:
                # Apply corrections
                if true_res.status == "corrected" and true_res.corrected:
                    pair.true_statement = true_res.corrected
                if false_res.status == "corrected" and false_res.corrected:
                    pair.false_statement = false_res.corrected

            output.append(PairCorrectionResult(
                pair=pair,
                true_result=true_res,
                false_result=false_res,
                excluded=excluded,
            ))

        return output

    def _call_api(self, user_content: str) -> str | None:
        """Call Anthropic API. Returns raw response text or None on failure."""
        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                return response.content[0].text
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sonnet grammar call failed (attempt %d): %s", attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def _parse_response(
        self,
        raw_text: str,
        original_statements: list[str],
    ) -> list[CorrectionResult]:
        """Parse structured JSON response from Sonnet.

        Validates that every input index has a corresponding output and that
        corrected text isn't wildly different from the original.  Falls back
        to ``unchanged`` on any per-entry validation failure.
        """
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.warning("Sonnet returned non-JSON; falling back to unchanged")
            return [
                CorrectionResult(original=s, corrected=None, status="unchanged", changes=None)
                for s in original_statements
            ]

        entries = data.get("statements", [])
        by_index: dict[int, dict] = {e["index"]: e for e in entries if "index" in e}

        results: list[CorrectionResult] = []
        for i, orig in enumerate(original_statements):
            entry = by_index.get(i)
            if entry is None:
                results.append(CorrectionResult(
                    original=orig, corrected=None, status="unchanged", changes=None,
                ))
                continue

            status = entry.get("status", "unchanged")
            corrected = entry.get("corrected")
            changes = entry.get("changes")

            # Validate corrected text
            if status == "corrected" and corrected:
                if len(corrected) > 2 * len(orig) or len(corrected) < len(orig) // 3:
                    logger.warning(
                        "Sonnet correction too different in length for [%d]; keeping original", i,
                    )
                    status = "unchanged"
                    corrected = None
                    changes = None

            results.append(CorrectionResult(
                original=orig,
                corrected=corrected,
                status=status,
                changes=changes,
            ))

        return results
