"""Biology generator: taxonomic membership, sibling relations, rank ordering.

Data source: Wikidata (organisms with NCBI taxonomy links).
Relations: is_member_of, same_family, same_order, has_rank.
Domains: biology, taxonomy.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from latenet.datasources.wikidata import RANK_LEVEL, RANK_ORDER, load_organisms
from latenet.generators.base import BaseGenerator
from latenet.types import ContrastivePair, Difficulty, NegationStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BioTemplate:
    id: str
    relation: str
    pattern: str


# Taxonomic membership: "{organism} is a {taxon}"
_MEMBERSHIP_TEMPLATES = [
    BioTemplate("bio_membership_01", "is_member_of",
                 "A {organism} is a {taxon}."),
    BioTemplate("bio_membership_02", "is_member_of",
                 "A {organism} is a type of {taxon}."),
    BioTemplate("bio_membership_03", "is_member_of",
                 "The {organism} belongs to the group {taxon}."),
    BioTemplate("bio_membership_04", "is_member_of",
                 "{organism} is classified as a {taxon}."),
    BioTemplate("bio_membership_05", "is_member_of",
                 "A {organism} is a kind of {taxon}."),
]

# Taxonomic sibling: "{A} and {B} are in the same {rank}"
_SIBLING_TEMPLATES = [
    BioTemplate("bio_sibling_01", "same_taxon",
                 "{organismA} and {organismB} are in the same {rank}."),
    BioTemplate("bio_sibling_02", "same_taxon",
                 "{organismA} and {organismB} belong to the same {rank}."),
    BioTemplate("bio_sibling_03", "same_taxon",
                 "{organismA} is in the same {rank} as {organismB}."),
]

# Taxonomic rank ordering
_RANK_TEMPLATES = [
    BioTemplate("bio_rank_01", "has_rank",
                 "{rankA} is a more specific rank than {rankB}."),
    BioTemplate("bio_rank_02", "has_rank",
                 "{rankA} is a more general rank than {rankB}."),
    BioTemplate("bio_rank_03", "has_rank",
                 "In taxonomy, {rankA} is a higher level than {rankB}."),
]

_ALL_TEMPLATES: dict[str, list[BioTemplate]] = {
    "is_member_of": _MEMBERSHIP_TEMPLATES,
    "same_taxon": _SIBLING_TEMPLATES,
    "has_rank": _RANK_TEMPLATES,
}

# Rank display names (capitalized for templates)
_RANK_DISPLAY = {
    "species": "species",
    "genus": "genus",
    "family": "family",
    "order": "order",
    "class": "class",
    "phylum": "phylum",
    "kingdom": "kingdom",
}


def _make_pair_id(parts: list[str]) -> str:
    key = ":".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _taxonomic_distance(rank_a: str, rank_b: str) -> int:
    """Number of hops between two taxonomic ranks."""
    la = RANK_LEVEL.get(rank_a, -1)
    lb = RANK_LEVEL.get(rank_b, -1)
    if la < 0 or lb < 0:
        return 99
    return abs(la - lb)


def _lowest_common_rank(row_a: pd.Series, row_b: pd.Series) -> str | None:
    """Find the lowest (most specific) rank where two organisms share an ancestor.

    Walks up from genus to kingdom, returns the first rank where both organisms
    have the same non-null ancestor.
    """
    # Check ranks from most specific upward (skip species — organisms are at species level)
    for rank in RANK_ORDER[1:]:  # genus, family, order, class_, phylum, kingdom
        col = f"{rank}_qid" if rank != "class" else "class__qid"
        val_a = row_a.get(col)
        val_b = row_b.get(col)
        if pd.notna(val_a) and pd.notna(val_b) and val_a == val_b:
            return rank
    return None


def _lcr_distance(lcr: str | None) -> int:
    """Map lowest common rank to a numeric distance (higher = more distant)."""
    if lcr is None:
        return 99
    return RANK_LEVEL.get(lcr, 99)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class BiologyGenerator(BaseGenerator):
    """Generate contrastive pairs from biological taxonomy data (Wikidata)."""

    def __init__(
        self,
        seed: int = 42,
        max_pairs: int | None = None,
        require_common_name: bool = True,
        require_wikipedia: bool = True,
        min_per_family: int = 2,
        min_per_order: int = 2,
        force_refresh: bool = False,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)
        self.require_common_name = require_common_name
        self.require_wikipedia = require_wikipedia
        self.min_per_family = min_per_family
        self.min_per_order = min_per_order
        self.force_refresh = force_refresh

        # Loaded lazily
        self._organisms: pd.DataFrame | None = None
        self._family_groups: dict[str, list[int]] | None = None
        self._order_groups: dict[str, list[int]] | None = None
        self._class_groups: dict[str, list[int]] | None = None

    @property
    def name(self) -> str:
        return "biology"

    def relation_types(self) -> list[str]:
        return ["is_member_of", "same_taxon", "has_rank"]

    def domains(self) -> list[str]:
        return ["biology", "taxonomy"]

    # --- Data loading ---

    def _load_data(self) -> None:
        if self._organisms is not None:
            return

        logger.info("Loading biology data from Wikidata...")
        df = load_organisms(
            require_common_name=self.require_common_name,
            require_wikipedia=self.require_wikipedia,
            force_refresh=self.force_refresh,
        )

        if df.empty:
            self._organisms = df
            return

        # Filter to species only for pair generation
        species = df[df["taxon_rank"] == "species"].copy()

        # Require key lineage fields
        species = species.dropna(subset=["kingdom"]).copy()

        # Use common_name as the display name, lowercase for natural language
        species["display_name"] = species["common_name"].str.lower()

        # Build groupings for swap selection
        self._family_groups = self._build_groups(species, "family")
        self._order_groups = self._build_groups(species, "order")
        self._class_groups = self._build_groups(species, "class_")

        # Filter orphan taxa
        if self.min_per_family > 1:
            valid_families = {
                k for k, v in self._family_groups.items() if len(v) >= self.min_per_family
            }
            family_col = "family" if "family" in species.columns else "family"
            species = species[species[family_col].isin(valid_families)].copy()
            # Rebuild after filtering
            self._family_groups = self._build_groups(species, "family")
            self._order_groups = self._build_groups(species, "order")
            self._class_groups = self._build_groups(species, "class_")

        self._organisms = species.reset_index(drop=True)
        logger.info(
            "Biology data loaded: %d species, %d families, %d orders, %d classes",
            len(self._organisms),
            len(self._family_groups),
            len(self._order_groups),
            len(self._class_groups),
        )

        # Log order-level family density for medium-tier viability
        self._log_order_family_density(species)

    def _log_order_family_density(self, species: pd.DataFrame) -> None:
        """Log how many distinct families each order has, to assess medium-tier viability."""
        if "order" not in species.columns or "family" not in species.columns:
            return
        valid = species.dropna(subset=["order", "family"])
        order_family = valid.groupby("order")["family"].nunique()
        single_family = order_family[order_family == 1]
        multi_family = order_family[order_family >= 2]
        logger.info(
            "Order-family density: %d orders with 2+ families (medium-tier eligible), "
            "%d orders with only 1 family",
            len(multi_family), len(single_family),
        )
        if len(single_family) > 0:
            logger.debug(
                "Single-family orders (no medium-tier membership pairs): %s",
                sorted(single_family.index.tolist()),
            )

    def _build_groups(self, df: pd.DataFrame, col: str) -> dict[str, list[int]]:
        """Group DataFrame indices by a taxonomy column value."""
        groups: dict[str, list[int]] = {}
        for idx, row in df.iterrows():
            val = row.get(col)
            if pd.notna(val) and val:
                groups.setdefault(val, []).append(idx)
        return groups

    # --- Main generate ---

    def generate(self) -> Iterator[ContrastivePair]:
        self._load_data()
        if self._organisms is None or self._organisms.empty:
            return

        count = 0
        generators = [
            self._generate_membership,
            self._generate_sibling,
            self._generate_rank,
        ]

        for gen_fn in generators:
            for pair in gen_fn():
                yield pair
                count += 1
                if self.max_pairs is not None and count >= self.max_pairs:
                    return

    # --- Helpers ---

    def _pick_template(self, relation: str) -> BioTemplate:
        templates = _ALL_TEMPLATES[relation]
        return templates[self.rng.randint(0, len(templates) - 1)]

    def _organism_name(self, idx: int) -> str:
        return self._organisms.loc[idx, "display_name"]

    def _organism_row(self, idx: int) -> pd.Series:
        return self._organisms.loc[idx]

    # --- Taxonomic membership ---

    def _generate_membership(self) -> Iterator[ContrastivePair]:
        """Generate '{organism} is a {taxon}' pairs."""
        organisms = self._organisms
        indices = list(organisms.index)
        self.rng.shuffle(indices)

        # Ranks to generate membership statements for (skip species — it's the organism itself)
        membership_ranks = ["family", "order", "class_", "phylum", "kingdom"]

        for idx in indices:
            row = organisms.loc[idx]
            org_name = row["display_name"]

            for rank_col in membership_ranks:
                true_taxon = row.get(rank_col)
                if pd.isna(true_taxon) or not true_taxon:
                    continue

                # Pick a swap taxon at the same rank
                swap_info = self._pick_membership_swap(row, rank_col)
                if not swap_info:
                    continue

                for swap_taxon, difficulty, strategy, sem_dist in swap_info:
                    template = self._pick_template("is_member_of")
                    true_stmt = template.pattern.format(
                        organism=org_name, taxon=true_taxon.lower()
                    )
                    false_stmt = template.pattern.format(
                        organism=org_name, taxon=swap_taxon.lower()
                    )

                    # Rank display for pair_id
                    rank_label = rank_col.rstrip("_")
                    pair_id = _make_pair_id([
                        "bio", "member", org_name, rank_label,
                        true_taxon, swap_taxon, template.id,
                    ])

                    yield ContrastivePair(
                        true_statement=true_stmt,
                        false_statement=false_stmt,
                        pair_id=pair_id,
                        domain="biology",
                        relation_type="is_member_of",
                        difficulty=difficulty,
                        semantic_distance=sem_dist,
                        generator=self.name,
                        template_id=template.id,
                        negation_strategy=strategy,
                    )

    def _pick_membership_swap(
        self, row: pd.Series, rank_col: str,
    ) -> list[tuple[str, str, str, int]]:
        """Pick swap taxa for membership at different difficulties.

        Returns list of (swap_taxon_name, difficulty, strategy, semantic_distance).

        Difficulty tiers:
        - Hard: sibling taxon (same parent rank). E.g., different family in
          the same order.
        - Medium: cousin taxon (same grandparent rank, different parent).
          E.g., different family in the same class but different order.
        - Easy: distant taxon at the same rank.
        """
        true_taxon = row[rank_col]
        swaps = []

        rank_clean = rank_col.rstrip("_")
        rank_idx = RANK_LEVEL.get(rank_clean, -1)

        # Collect all distinct taxa at this rank
        all_taxa = set(
            self._organisms[rank_col].dropna().unique()
        ) - {true_taxon}

        if not all_taxa:
            return swaps

        sibling_picks: set[str] = set()

        # Hard: taxon in same parent rank (sibling)
        parent_rank_idx = rank_idx + 1
        if parent_rank_idx < len(RANK_ORDER):
            parent_rank = RANK_ORDER[parent_rank_idx]
            parent_col = f"{parent_rank}" if parent_rank != "class" else "class_"
            parent_val = row.get(parent_col)
            if pd.notna(parent_val) and parent_val:
                siblings = set(
                    self._organisms[
                        self._organisms[parent_col] == parent_val
                    ][rank_col].dropna().unique()
                ) - {true_taxon}
                if siblings:
                    pick = self.rng.choice(sorted(siblings))
                    swaps.append((pick, Difficulty.HARD.value,
                                  NegationStrategy.SIBLING_SWAP.value, 1))
                    sibling_picks = siblings

        # Medium: cousin taxon (same grandparent rank, different parent)
        grandparent_rank_idx = rank_idx + 2
        if grandparent_rank_idx < len(RANK_ORDER):
            gp_rank = RANK_ORDER[grandparent_rank_idx]
            gp_col = f"{gp_rank}" if gp_rank != "class" else "class_"
            gp_val = row.get(gp_col)
            if pd.notna(gp_val) and gp_val:
                # All taxa at this rank that share the grandparent
                cousins_all = set(
                    self._organisms[
                        self._organisms[gp_col] == gp_val
                    ][rank_col].dropna().unique()
                ) - {true_taxon}
                # Exclude siblings (same parent) to get true cousins
                cousins = sorted(cousins_all - sibling_picks)
                if cousins:
                    pick = self.rng.choice(cousins)
                    swaps.append((pick, Difficulty.MEDIUM.value,
                                  NegationStrategy.SIBLING_SWAP.value, 2))

        # Easy: distant taxon at same rank
        used = {s[0] for s in swaps}
        distant = sorted(all_taxa - used - sibling_picks)
        if distant:
            pick = self.rng.choice(distant)
            swaps.append((pick, Difficulty.EASY.value,
                          NegationStrategy.DISTANT_SWAP.value, 3))

        return swaps

    # --- Taxonomic sibling ---

    def _generate_sibling(self) -> Iterator[ContrastivePair]:
        """Generate '{A} and {B} are in the same {rank}' pairs."""
        organisms = self._organisms

        # Generate for family and order levels
        for group_dict, rank_label in [
            (self._family_groups, "family"),
            (self._order_groups, "order"),
        ]:
            groups = list(group_dict.items())
            self.rng.shuffle(groups)

            for taxon_name, member_indices in groups:
                if len(member_indices) < 2:
                    continue

                # Pick a true pair (same group)
                pair_indices = list(member_indices)
                self.rng.shuffle(pair_indices)
                idx_a, idx_b = pair_indices[0], pair_indices[1]

                name_a = self._organism_name(idx_a)
                name_b = self._organism_name(idx_b)

                template = self._pick_template("same_taxon")
                true_stmt = template.pattern.format(
                    organismA=name_a, organismB=name_b,
                    rank=_RANK_DISPLAY.get(rank_label, rank_label),
                )

                # False: pick an organism NOT in this group
                other_indices = [
                    i for i in organisms.index if i not in member_indices
                ]
                if not other_indices:
                    continue
                idx_c = self.rng.choice(other_indices)
                name_c = self._organism_name(idx_c)

                false_stmt = template.pattern.format(
                    organismA=name_a, organismB=name_c,
                    rank=_RANK_DISPLAY.get(rank_label, rank_label),
                )

                # Difficulty based on how close the false organism is
                row_a = self._organism_row(idx_a)
                row_c = self._organism_row(idx_c)
                lcr = _lowest_common_rank(row_a, row_c)
                lcr_dist = _lcr_distance(lcr)

                if lcr_dist <= 2:  # Same order or closer
                    difficulty = Difficulty.HARD.value
                elif lcr_dist <= 4:  # Same class or phylum
                    difficulty = Difficulty.MEDIUM.value
                else:
                    difficulty = Difficulty.EASY.value

                pair_id = _make_pair_id([
                    "bio", "sibling", rank_label, name_a, name_b, name_c, template.id,
                ])

                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="biology",
                    relation_type="same_taxon",
                    difficulty=difficulty,
                    semantic_distance=lcr_dist,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.SIBLING_SWAP.value,
                )

    # --- Rank ordering ---

    def _generate_rank(self) -> Iterator[ContrastivePair]:
        """Generate rank ordering pairs like 'species is more specific than genus'."""
        # This is a small, static set of pairs
        ranks = RANK_ORDER  # species, genus, family, order, class, phylum, kingdom

        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                specific = ranks[i]
                general = ranks[j]

                # "specific is a more specific rank than general" (true)
                template = _ALL_TEMPLATES["has_rank"][0]  # "more specific"
                true_stmt = template.pattern.format(
                    rankA=_RANK_DISPLAY[specific], rankB=_RANK_DISPLAY[general],
                )
                false_stmt = template.pattern.format(
                    rankA=_RANK_DISPLAY[general], rankB=_RANK_DISPLAY[specific],
                )

                # Difficulty: adjacent ranks = hard, distant = easy
                dist = j - i
                if dist == 1:
                    difficulty = Difficulty.HARD.value
                elif dist <= 3:
                    difficulty = Difficulty.MEDIUM.value
                else:
                    difficulty = Difficulty.EASY.value

                pair_id = _make_pair_id([
                    "bio", "rank", specific, general, template.id,
                ])

                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="taxonomy",
                    relation_type="has_rank",
                    difficulty=difficulty,
                    semantic_distance=dist,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                )

                # Also generate with "more general" template
                template2 = _ALL_TEMPLATES["has_rank"][1]  # "more general"
                true_stmt2 = template2.pattern.format(
                    rankA=_RANK_DISPLAY[general], rankB=_RANK_DISPLAY[specific],
                )
                false_stmt2 = template2.pattern.format(
                    rankA=_RANK_DISPLAY[specific], rankB=_RANK_DISPLAY[general],
                )

                pair_id2 = _make_pair_id([
                    "bio", "rank", general, specific, template2.id,
                ])

                yield ContrastivePair(
                    true_statement=true_stmt2,
                    false_statement=false_stmt2,
                    pair_id=pair_id2,
                    domain="taxonomy",
                    relation_type="has_rank",
                    difficulty=difficulty,
                    semantic_distance=dist,
                    generator=self.name,
                    template_id=template2.id,
                    negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                )
