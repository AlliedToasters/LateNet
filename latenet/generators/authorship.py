"""Author-work attribution: created_by, author_of, worked_in_domain.

Data source: Wikidata (authors/creators and their works across creative domains).
Relations: created_by, author_of, worked_in_domain.
Domains: authorship.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from latenet.datasources.wikidata import load_authors_and_works
from latenet.generators.base import BaseGenerator
from latenet.sanitize import render_template
from latenet.types import ContrastivePair, Difficulty, NegationStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AuthTemplate:
    id: str
    relation: str
    pattern: str


# Work attribution: "{work} was {verb} by {author}"
_CREATED_BY_TEMPLATES = [
    AuthTemplate("auth_work_01", "created_by",
                 "{work} was {verb} by {author}."),
    AuthTemplate("auth_work_02", "created_by",
                 "{author} {past_verb} {work}."),
    AuthTemplate("auth_work_03", "created_by",
                 "The {work_type} {work} was {verb} by {author}."),
    AuthTemplate("auth_work_04", "created_by",
                 "{work} is {a_work_type} by {author}."),
]

# Reverse attribution: "{author} is the {role} of {work}"
_AUTHOR_OF_TEMPLATES = [
    AuthTemplate("auth_reverse_01", "author_of",
                 "{author} is the {role} of {work}."),
    AuthTemplate("auth_reverse_02", "author_of",
                 "{work} is among {author}'s works."),
    AuthTemplate("auth_reverse_03", "author_of",
                 "{author} created {work}."),
]

# Domain attribution: "{author} was a {role}"
_DOMAIN_TEMPLATES = [
    AuthTemplate("auth_domain_01", "worked_in_domain",
                 "{author} was {a_role}."),
    AuthTemplate("auth_domain_02", "worked_in_domain",
                 "{author} is known as {a_role}."),
    AuthTemplate("auth_domain_03", "worked_in_domain",
                 "{author} worked as {a_role}."),
]

_ALL_TEMPLATES: dict[str, list[AuthTemplate]] = {
    "created_by": _CREATED_BY_TEMPLATES,
    "author_of": _AUTHOR_OF_TEMPLATES,
    "worked_in_domain": _DOMAIN_TEMPLATES,
}

# Domain -> past tense active verb (for "auth_work_02" template)
_DOMAIN_PAST_VERB: dict[str, str] = {
    "literature": "wrote",
    "music": "composed",
    "art": "painted",
    "film": "directed",
    "science": "proposed",
}

# Roles for false domain attribution swaps
_ALL_ROLES = ["author", "composer", "painter", "director", "discoverer"]


def _make_pair_id(parts: list[str]) -> str:
    key = ":".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _era_from_year(year: int | None) -> str | None:
    """Bucket a publication year into a century string for era comparison."""
    if year is None or pd.isna(year):
        return None
    year = int(year)
    if year > 0:
        return str((year - 1) // 100 + 1)
    return str(-((-year - 1) // 100 + 1))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class AuthorshipGenerator(BaseGenerator):
    """Generate contrastive pairs from author-work attribution data (Wikidata)."""

    def __init__(
        self,
        seed: int = 42,
        max_pairs: int | None = None,
        min_works_per_author: int = 2,
        per_domain_cap: int | None = None,
        force_refresh: bool = False,
        min_work_sitelinks: int = 5,
        min_author_sitelinks: int = 15,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)
        self.min_works_per_author = min_works_per_author
        self.per_domain_cap = per_domain_cap
        self.force_refresh = force_refresh
        self.min_work_sitelinks = min_work_sitelinks
        self.min_author_sitelinks = min_author_sitelinks

        # Loaded lazily
        self._data: pd.DataFrame | None = None
        self._domain_groups: dict[str, pd.DataFrame] | None = None
        self._author_works: dict[str, list[int]] | None = None
        self._row_weights: list[float] | None = None

    @property
    def name(self) -> str:
        return "authorship"

    def relation_types(self) -> list[str]:
        return ["created_by", "author_of", "worked_in_domain"]

    def domains(self) -> list[str]:
        return ["authorship"]

    # --- Data loading ---

    def _load_data(self) -> None:
        if self._data is not None:
            return

        logger.info("Loading authorship data from Wikidata...")
        df = load_authors_and_works(
            min_works_per_author=self.min_works_per_author,
            force_refresh=self.force_refresh,
            min_work_sitelinks=self.min_work_sitelinks,
            min_author_sitelinks=self.min_author_sitelinks,
        )

        if df.empty:
            self._data = df
            self._domain_groups = {}
            self._author_works = {}
            return

        # Add era column for difficulty computation
        df["era"] = df["publication_year"].apply(_era_from_year)

        self._data = df

        # Build author-sitelink-based sampling weights for each row
        if "author_sitelinks" in df.columns:
            sl = pd.to_numeric(df["author_sitelinks"], errors="coerce").fillna(0)
            self._row_weights = self.build_weights(sl.tolist())
        else:
            self._row_weights = self.build_weights([1.0] * len(df))

        # Group by creative domain
        self._domain_groups = {
            domain: group.reset_index(drop=True)
            for domain, group in df.groupby("creative_domain")
        }

        # Group work indices by author
        self._author_works = {}
        for idx, row in df.iterrows():
            aqid = row["author_qid"]
            self._author_works.setdefault(aqid, []).append(idx)

        logger.info(
            "Authorship data loaded: %d pairs, %d authors, %d domains",
            len(df),
            df["author_qid"].nunique(),
            len(self._domain_groups),
        )
        for domain, group in self._domain_groups.items():
            logger.info("  %s: %d pairs, %d authors",
                         domain, len(group), group["author_qid"].nunique())

    # --- Main generate ---

    def generate(self) -> Iterator[ContrastivePair]:
        self._load_data()
        if self._data is None or self._data.empty:
            return

        yield from self._round_robin_generate([
            self._generate_created_by,
            self._generate_author_of,
            self._generate_domain_attribution,
        ])

    # --- Helpers ---

    def _pick_template(self, relation: str) -> AuthTemplate:
        templates = _ALL_TEMPLATES[relation]
        return templates[self.rng.randint(0, len(templates) - 1)]

    def _difficulty_for_swap(
        self,
        true_domain: str,
        true_era: str | None,
        swap_domain: str,
        swap_era: str | None,
    ) -> str:
        """Determine difficulty based on domain and era similarity."""
        if true_domain != swap_domain:
            return Difficulty.EASY.value
        # Same domain
        if true_era is not None and swap_era is not None and true_era == swap_era:
            return Difficulty.HARD.value
        return Difficulty.MEDIUM.value

    def _pick_swap_author(
        self,
        exclude_author_qid: str,
        target_domain: str,
        target_era: str | None,
        difficulty: str,
    ) -> pd.Series | None:
        """Pick a swap author matching the desired difficulty constraints."""
        df = self._data
        candidates = df[df["author_qid"] != exclude_author_qid]

        if difficulty == Difficulty.HARD.value:
            # Same domain, same era
            candidates = candidates[candidates["creative_domain"] == target_domain]
            if target_era is not None:
                era_match = candidates[candidates["era"] == target_era]
                if not era_match.empty:
                    candidates = era_match
        elif difficulty == Difficulty.MEDIUM.value:
            # Same domain, different era (or same domain if era unavailable)
            candidates = candidates[candidates["creative_domain"] == target_domain]
            if target_era is not None:
                diff_era = candidates[candidates["era"] != target_era]
                if not diff_era.empty:
                    candidates = diff_era
        else:
            # Easy: different domain
            candidates = candidates[candidates["creative_domain"] != target_domain]

        if candidates.empty:
            return None

        # Pick a random unique author from candidates
        unique_authors = candidates.drop_duplicates(subset=["author_qid"])
        idx = self.rng.randint(0, len(unique_authors) - 1)
        return unique_authors.iloc[idx]

    def _ensure_weights(self) -> None:
        """Ensure row weights are initialized (for test fixtures that bypass _load_data)."""
        if self._row_weights is not None:
            return
        if self._data is not None and "author_sitelinks" in self._data.columns:
            sl = pd.to_numeric(self._data["author_sitelinks"], errors="coerce").fillna(0)
            self._row_weights = self.build_weights(sl.tolist())
        elif self._data is not None:
            n = len(self._data)
            self._row_weights = self.build_weights([1.0] * n)

    # --- created_by ---

    def _generate_created_by(self) -> Iterator[ContrastivePair]:
        """Generate '{work} was {verb} by {author}' pairs."""
        if self._data is None or self._data.empty:
            return

        df = self._data
        # Weighted sampling: prefer rows with more notable authors
        self._ensure_weights()
        indices = self.weighted_sample(list(df.index), self._row_weights, len(df))

        per_domain_counts: dict[str, int] = {}
        seen_pair_ids: set[str] = set()

        for idx in indices:
            row = df.loc[idx]
            domain = row["creative_domain"]

            # Enforce per-domain cap
            if self.per_domain_cap is not None:
                if per_domain_counts.get(domain, 0) >= self.per_domain_cap:
                    continue

            work_name = str(row["work_name"])
            author_name = str(row["author_name"])
            work_type = str(row["work_type"])
            verb = str(row["verb"])
            past_verb = _DOMAIN_PAST_VERB.get(domain, "created")
            era = row.get("era")

            # Generate at each difficulty tier
            for target_diff, strategy in [
                (Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value),
                (Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value),
                (Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value),
            ]:
                swap = self._pick_swap_author(
                    row["author_qid"], domain, era, target_diff,
                )
                if swap is None:
                    continue

                swap_author = str(swap["author_name"])
                swap_era = swap.get("era")
                actual_difficulty = self._difficulty_for_swap(
                    domain, era, swap["creative_domain"], swap_era,
                )

                template = self._pick_template("created_by")
                true_stmt = self._render_created_by(
                    template, work_name, author_name, work_type, verb, past_verb,
                )
                false_stmt = self._render_created_by(
                    template, work_name, swap_author, work_type, verb, past_verb,
                )

                pair_id = _make_pair_id([
                    "auth", "created_by", row["work_qid"],
                    row["author_qid"], swap["author_qid"],
                    actual_difficulty, template.id,
                ])

                if pair_id in seen_pair_ids:
                    continue
                seen_pair_ids.add(pair_id)

                per_domain_counts[domain] = per_domain_counts.get(domain, 0) + 1

                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="authorship",
                    relation_type="created_by",
                    difficulty="mixed",
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                    source_synset=row["work_qid"],
                    target_synset=row["author_qid"],
                    neg_synset=swap["author_qid"],
                    gen_params={
                        "work": work_name,
                        "work_idx": int(idx),
                        "author": author_name,
                        "false_author": swap_author,
                        "creative_domain": domain,
                        "swap_distance": actual_difficulty,
                    },
                )

    def _render_created_by(
        self,
        template: AuthTemplate,
        work: str, author: str, work_type: str, verb: str, past_verb: str,
    ) -> str:
        """Render a created_by template with domain-appropriate verb."""
        return render_template(template.pattern,
            work=work, author=author, work_type=work_type,
            verb=verb, past_verb=past_verb,
        )

    # --- author_of ---

    def _generate_author_of(self) -> Iterator[ContrastivePair]:
        """Generate '{author} is the {role} of {work}' pairs (swap the work)."""
        if self._data is None or self._data.empty:
            return

        df = self._data

        # For each author, pick one of their works, then swap with another author's work
        # Weight author selection by max sitelinks among their works
        authors = list(self._author_works.keys())
        if "author_sitelinks" in df.columns:
            author_sl = []
            for aqid in authors:
                widx = self._author_works[aqid]
                max_sl = df.loc[widx, "author_sitelinks"].max()
                author_sl.append(float(max_sl) if pd.notna(max_sl) else 1.0)
            author_weights = self.build_weights(author_sl)
            authors = self.weighted_sample(
                list(range(len(authors))), author_weights, len(authors)
            )
            authors = [list(self._author_works.keys())[i] for i in authors]
        else:
            self.rng.shuffle(authors)

        for author_qid in authors:
            work_indices = self._author_works[author_qid]
            if len(work_indices) < 1:
                continue

            # Pick a true work
            true_idx = self.rng.choice(work_indices)
            true_row = df.loc[true_idx]
            author_name = str(true_row["author_name"])
            true_work = str(true_row["work_name"])
            role = str(true_row["role"])
            domain = true_row["creative_domain"]
            era = true_row.get("era")

            # Find a work by a different author in the same domain for the false statement
            same_domain = df[
                (df["creative_domain"] == domain) &
                (df["author_qid"] != author_qid)
            ]
            if same_domain.empty:
                continue

            false_idx = self.rng.randint(0, len(same_domain) - 1)
            false_row = same_domain.iloc[false_idx]
            false_work = str(false_row["work_name"])
            false_era = false_row.get("era")

            difficulty = self._difficulty_for_swap(domain, era, domain, false_era)

            template = self._pick_template("author_of")
            true_stmt = render_template(template.pattern,
                author=author_name, work=true_work, role=role,
            )
            false_stmt = render_template(template.pattern,
                author=author_name, work=false_work, role=role,
            )

            pair_id = _make_pair_id([
                "auth", "author_of", author_qid,
                true_row["work_qid"], false_row["work_qid"], template.id,
            ])

            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="authorship",
                relation_type="author_of",
                difficulty="mixed",
                semantic_distance=None,
                generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                source_synset=author_qid,
                target_synset=true_row["work_qid"],
                neg_synset=false_row["work_qid"],
                gen_params={
                    "author": author_name,
                    "true_work": true_work,
                    "true_work_idx": int(true_idx),
                    "false_work": false_work,
                    "false_work_idx": int(false_idx),
                    "role": role,
                    "swap_distance": difficulty,
                },
            )

    # --- worked_in_domain ---

    def _generate_domain_attribution(self) -> Iterator[ContrastivePair]:
        """Generate '{author} was a {role}' pairs."""
        if self._data is None or self._data.empty:
            return

        df = self._data

        # Derive each author's primary role from their works' domain
        agg_dict = {
            "author_name": ("author_name", "first"),
            "role": ("role", "first"),
            "creative_domain": ("creative_domain", "first"),
        }
        if "author_sitelinks" in df.columns:
            agg_dict["author_sitelinks"] = ("author_sitelinks", "max")
        author_roles = df.groupby("author_qid").agg(**agg_dict).reset_index()

        if "author_sitelinks" in author_roles.columns:
            sl = pd.to_numeric(author_roles["author_sitelinks"], errors="coerce").fillna(0)
            ar_weights = self.build_weights(sl.tolist())
            indices = self.weighted_sample(list(author_roles.index), ar_weights, len(author_roles))
        else:
            indices = list(author_roles.index)
            self.rng.shuffle(indices)

        available_roles = sorted(set(author_roles["role"].unique()))
        if len(available_roles) < 2:
            return

        for idx in indices:
            row = author_roles.iloc[idx]
            author_name = str(row["author_name"])
            true_role = str(row["role"])

            # Pick a false role from a different domain
            false_roles = [r for r in available_roles if r != true_role]
            if not false_roles:
                continue
            false_role = self.rng.choice(false_roles)

            template = self._pick_template("worked_in_domain")
            true_stmt = render_template(template.pattern,author=author_name, role=true_role)
            false_stmt = render_template(template.pattern,author=author_name, role=false_role)

            pair_id = _make_pair_id([
                "auth", "domain", row["author_qid"],
                true_role, false_role, template.id,
            ])

            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="authorship",
                relation_type="worked_in_domain",
                difficulty="mixed",
                semantic_distance=None,
                generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.DISTANT_SWAP.value,
                source_synset=row["author_qid"],
                gen_params={
                    "author": author_name,
                    "author_idx": int(idx),
                    "true_role": true_role,
                    "false_role": false_role,
                    "swap_distance": Difficulty.EASY.value,
                },
            )
