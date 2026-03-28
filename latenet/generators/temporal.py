"""Temporal ordering: historical events, births, century attribution, contemporaneity.

Data source: Wikidata (historical events, notable people).
Relations: happened_before, born_before, occurred_in_century, lived_before_event, were_contemporaries.
Domains: temporal.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from latenet.datasources.wikidata import (
    DEFAULT_EXCLUDE_EVENT_TYPES,
    _century_label,
    _year_to_century,
    load_historical_events,
    load_notable_people,
)
from latenet.generators.base import BaseGenerator
from latenet.sanitize import render_template
from latenet.types import ContrastivePair, NegationStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TempTemplate:
    id: str
    relation: str
    pattern: str


_EVENT_ORDER_TEMPLATES = [
    TempTemplate("temp_event_order_01", "happened_before",
                 "{eventA} occurred before {eventB}."),
    TempTemplate("temp_event_order_02", "happened_before",
                 "{eventA} happened before {eventB}."),
    TempTemplate("temp_event_order_03", "happened_before",
                 "{eventA} took place before {eventB}."),
    TempTemplate("temp_event_order_04", "happened_before",
                 "{eventA} preceded {eventB}."),
]

_BIRTH_ORDER_TEMPLATES = [
    TempTemplate("temp_birth_order_01", "born_before",
                 "{personA} was born before {personB}."),
    TempTemplate("temp_birth_order_02", "born_before",
                 "{personA} was born earlier than {personB}."),
    TempTemplate("temp_birth_order_03", "born_before",
                 "{personA} predates {personB}."),
]

_CENTURY_TEMPLATES = [
    TempTemplate("temp_century_01", "occurred_in_century",
                 "{event} occurred in the {century}."),
    TempTemplate("temp_century_02", "occurred_in_century",
                 "{event} took place in the {century}."),
    TempTemplate("temp_century_03", "occurred_in_century",
                 "The {event} happened in the {century}."),
    TempTemplate("temp_century_person_01", "occurred_in_century",
                 "{person} was born in the {century}."),
]

_ERA_TEMPLATES = [
    TempTemplate("temp_era_01", "lived_before_event",
                 "{person} lived before {event}."),
    TempTemplate("temp_era_02", "lived_before_event",
                 "{person} died before {event} took place."),
    TempTemplate("temp_era_03", "lived_before_event",
                 "{person} predates {event}."),
]

_CONTEMPORARY_TEMPLATES = [
    TempTemplate("temp_contemporary_01", "were_contemporaries",
                 "{personA} and {personB} were contemporaries."),
    TempTemplate("temp_contemporary_02", "were_contemporaries",
                 "{personA} and {personB} lived at the same time."),
    TempTemplate("temp_contemporary_03", "were_contemporaries",
                 "{personA} was alive during {personB}'s lifetime."),
]

_ALL_TEMPLATES: dict[str, list[TempTemplate]] = {
    "happened_before": _EVENT_ORDER_TEMPLATES,
    "born_before": _BIRTH_ORDER_TEMPLATES,
    "occurred_in_century": _CENTURY_TEMPLATES,
    "lived_before_event": _ERA_TEMPLATES,
    "were_contemporaries": _CONTEMPORARY_TEMPLATES,
}


def _make_pair_id(parts: list[str]) -> str:
    key = ":".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _year_gap_difficulty(gap: int) -> str:
    """Map absolute year gap to a descriptive label (stashed in gen_params)."""
    if gap < 50:
        return "close"
    elif gap < 200:
        return "moderate"
    else:
        return "distant"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TemporalGenerator(BaseGenerator):
    """Generate contrastive pairs from temporal ordering data (Wikidata)."""

    def __init__(
        self,
        seed: int = 42,
        max_pairs: int | None = None,
        min_year_gap: int = 10,
        century_boundary_buffer: int = 5,
        contemporaneity_overlap_min: int = 20,
        contemporaneity_gap_min: int = 50,
        min_era_gap: int = 50,
        min_year_events: int = -3000,
        max_year_events: int = 2020,
        min_birth_year: int = -500,
        max_birth_year: int = 2000,
        force_refresh: bool = False,
        exclude_event_types: frozenset[str] = DEFAULT_EXCLUDE_EVENT_TYPES,
        min_sitelinks: int = 20,
        prefer_recent: bool = True,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)
        self.min_year_gap = min_year_gap
        self.century_boundary_buffer = century_boundary_buffer
        self.contemporaneity_overlap_min = contemporaneity_overlap_min
        self.contemporaneity_gap_min = contemporaneity_gap_min
        self.min_era_gap = min_era_gap
        self.min_year_events = min_year_events
        self.max_year_events = max_year_events
        self.min_birth_year = min_birth_year
        self.max_birth_year = max_birth_year
        self.force_refresh = force_refresh
        self.exclude_event_types = exclude_event_types
        self.min_sitelinks = min_sitelinks
        self.prefer_recent = prefer_recent

        # Loaded lazily
        self._events: pd.DataFrame | None = None
        self._people: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "temporal"

    def relation_types(self) -> list[str]:
        return [
            "happened_before", "born_before", "occurred_in_century",
            "lived_before_event", "were_contemporaries",
        ]

    def domains(self) -> list[str]:
        return ["temporal"]

    # --- Data loading ---

    def _load_data(self) -> None:
        if self._events is not None:
            return

        logger.info("Loading temporal data from Wikidata...")
        self._events = load_historical_events(
            min_year=self.min_year_events,
            max_year=self.max_year_events,
            force_refresh=self.force_refresh,
            exclude_event_types=self.exclude_event_types,
            min_sitelinks=self.min_sitelinks,
        )
        self._people = load_notable_people(
            min_birth_year=self.min_birth_year,
            max_birth_year=self.max_birth_year,
            force_refresh=self.force_refresh,
            min_sitelinks=self.min_sitelinks,
        )

        logger.info(
            "Temporal data loaded: %d events, %d people",
            len(self._events), len(self._people),
        )

    # --- Main generate ---

    def generate(self) -> Iterator[ContrastivePair]:
        self._load_data()

        yield from self._round_robin_generate([
            self._generate_event_ordering,
            self._generate_birth_ordering,
            self._generate_century_attribution,
            self._generate_era_ordering,
            self._generate_contemporaneity,
        ])

    # --- Helpers ---

    def _recency_weight(self, year: int) -> float:
        """Sampling weight favouring well-documented eras.

        Post-1800: 1.0, 1500-1800: 0.5, pre-1500: 0.25.
        Only applied when prefer_recent is True.
        """
        if not self.prefer_recent:
            return 1.0
        if year >= 1800:
            return 1.0
        if year >= 1500:
            return 0.5
        return 0.25

    def _notability_weights(self, df: pd.DataFrame, year_col: str) -> list[float]:
        """Build combined sitelink × recency weights for a DataFrame.

        Uses sitelinks as the primary notability signal (if available),
        multiplied by the recency weight to favour well-documented eras.
        Falls back to recency-only when sitelinks are absent.
        """
        if "sitelinks" in df.columns:
            sitelinks = pd.to_numeric(df["sitelinks"], errors="coerce").fillna(0).tolist()
        else:
            sitelinks = [1.0] * len(df)

        raw = [
            max(sl, 1.0) * self._recency_weight(int(df.loc[idx, year_col]))
            for idx, sl in zip(df.index, sitelinks)
        ]
        total = sum(raw)
        if total == 0:
            n = len(raw)
            return [1.0 / n] * n
        return [v / total for v in raw]

    def _pick_template(self, relation: str) -> TempTemplate:
        templates = _ALL_TEMPLATES[relation]
        return templates[self.rng.randint(0, len(templates) - 1)]

    # --- Event ordering ---

    def _generate_event_ordering(self) -> Iterator[ContrastivePair]:
        """Generate '{eventA} occurred before {eventB}' pairs."""
        if self._events is None or self._events.empty:
            return

        events = self._events
        indices = list(events.index)
        weights = self._notability_weights(events, "year")

        # Sample pairs rather than generating all O(n^2) combinations
        max_attempts = min(len(indices) * 5, 10000)
        seen = set()

        for _ in range(max_attempts):
            idx_a = self.rng.choices(indices, weights=weights, k=1)[0]
            idx_b = self.rng.choices(indices, weights=weights, k=1)[0]
            if idx_a == idx_b:
                continue

            pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            row_a = events.loc[idx_a]
            row_b = events.loc[idx_b]
            year_a = int(row_a["year"])
            year_b = int(row_b["year"])

            gap = abs(year_a - year_b)
            if gap < self.min_year_gap:
                continue

            # Ensure A is before B
            if year_a > year_b:
                row_a, row_b = row_b, row_a
                idx_a, idx_b = idx_b, idx_a
                year_a, year_b = year_b, year_a

            name_a = str(row_a["name"])
            name_b = str(row_b["name"])
            qid_a = str(row_a["qid"])
            qid_b = str(row_b["qid"])

            template = self._pick_template("happened_before")
            true_stmt = render_template(template.pattern,eventA=name_a, eventB=name_b)
            false_stmt = render_template(template.pattern,eventA=name_b, eventB=name_a)

            pair_id = _make_pair_id([
                "temp", "event_order", qid_a, qid_b, template.id,
            ])

            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="temporal",
                relation_type="happened_before",
                difficulty="mixed",
                semantic_distance=gap,
                generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                gen_params={
                    "event_a": name_a,
                    "event_a_idx": int(idx_a),
                    "event_b": name_b,
                    "event_b_idx": int(idx_b),
                    "year_a": year_a,
                    "year_b": year_b,
                    "gap_years": gap,
                    "gap_bucket": _year_gap_difficulty(gap),
                },
            )

    # --- Birth ordering ---

    def _generate_birth_ordering(self) -> Iterator[ContrastivePair]:
        """Generate '{personA} was born before {personB}' pairs."""
        if self._people is None or self._people.empty:
            return

        people = self._people
        indices = list(people.index)
        weights = self._notability_weights(people, "birth_year")

        max_attempts = min(len(indices) * 5, 10000)
        seen = set()

        for _ in range(max_attempts):
            idx_a = self.rng.choices(indices, weights=weights, k=1)[0]
            idx_b = self.rng.choices(indices, weights=weights, k=1)[0]
            if idx_a == idx_b:
                continue

            pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            row_a = people.loc[idx_a]
            row_b = people.loc[idx_b]
            year_a = int(row_a["birth_year"])
            year_b = int(row_b["birth_year"])

            gap = abs(year_a - year_b)
            if gap < self.min_year_gap:
                continue

            # Ensure A is born before B
            if year_a > year_b:
                row_a, row_b = row_b, row_a
                idx_a, idx_b = idx_b, idx_a
                year_a, year_b = year_b, year_a

            name_a = str(row_a["name"])
            name_b = str(row_b["name"])
            qid_a = str(row_a["qid"])
            qid_b = str(row_b["qid"])

            template = self._pick_template("born_before")
            true_stmt = render_template(template.pattern,personA=name_a, personB=name_b)
            false_stmt = render_template(template.pattern,personA=name_b, personB=name_a)

            pair_id = _make_pair_id([
                "temp", "birth_order", qid_a, qid_b, template.id,
            ])

            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="temporal",
                relation_type="born_before",
                difficulty="mixed",
                semantic_distance=gap,
                generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                gen_params={
                    "person_a": name_a,
                    "person_a_idx": int(idx_a),
                    "person_b": name_b,
                    "person_b_idx": int(idx_b),
                    "year_a": year_a,
                    "year_b": year_b,
                    "gap_years": gap,
                    "gap_bucket": _year_gap_difficulty(gap),
                },
            )

    # --- Century attribution ---

    def _generate_century_attribution(self) -> Iterator[ContrastivePair]:
        """Generate '{event} occurred in the {century} century' pairs."""
        if self._events is None or self._people is None:
            return

        # Generate for events
        yield from self._century_for_entities(
            self._events, "year", "name", "qid", "event"
        )

        # Generate for people (birth century)
        yield from self._century_for_entities(
            self._people, "birth_year", "name", "qid", "person"
        )

    def _century_for_entities(
        self, df: pd.DataFrame, year_col: str, name_col: str,
        qid_col: str, entity_type: str,
    ) -> Iterator[ContrastivePair]:
        """Generate century attribution pairs for a set of entities."""
        if df.empty:
            return

        weights = self._notability_weights(df, year_col)
        indices = self.weighted_sample(list(df.index), weights, len(df))

        # Collect distinct centuries for swap selection
        all_centuries = sorted(df[year_col].apply(_year_to_century).unique())

        for idx in indices:
            row = df.loc[idx]
            year = int(row[year_col])
            century = _year_to_century(year)

            # Skip entities near century boundaries
            year_in_century = year % 100
            if year > 0 and (year_in_century < self.century_boundary_buffer
                             or year_in_century > 100 - self.century_boundary_buffer):
                if year_in_century != 0:  # year 1800 is clearly 18th century
                    continue

            name = str(row[name_col])
            qid = str(row[qid_col])
            true_century = _century_label(century)

            # Pick a false century (different from true)
            false_centuries = [c for c in all_centuries if c != century]
            if not false_centuries:
                continue
            false_century_num = self.rng.choice(false_centuries)
            false_century = _century_label(false_century_num)

            # Pick template based on entity type
            if entity_type == "person":
                template = _ALL_TEMPLATES["occurred_in_century"][3]  # "was born in"
            else:
                template = self._pick_template("occurred_in_century")
                # Avoid using the person template for events
                while template.id == "temp_century_person_01":
                    template = self._pick_template("occurred_in_century")

            if entity_type == "person":
                true_stmt = render_template(template.pattern,person=name, century=true_century)
                false_stmt = render_template(template.pattern,person=name, century=false_century)
            else:
                true_stmt = render_template(template.pattern,event=name, century=true_century)
                false_stmt = render_template(template.pattern,event=name, century=false_century)

            century_gap = abs(century - false_century_num)

            pair_id = _make_pair_id([
                "temp", "century", entity_type, qid,
                str(century), str(false_century_num), template.id,
            ])

            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="temporal",
                relation_type="occurred_in_century",
                difficulty="mixed",
                semantic_distance=century_gap,
                generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.DISTANT_SWAP.value,
                gen_params={
                    "entity": name,
                    "entity_idx": int(idx),
                    "entity_type": entity_type,
                    "year": year,
                    "true_century": century,
                    "false_century": false_century_num,
                    "century_gap": century_gap,
                },
            )

    # --- Era ordering (person lived before event) ---

    def _generate_era_ordering(self) -> Iterator[ContrastivePair]:
        """Generate '{person} lived before {event}' pairs."""
        if (self._people is None or self._people.empty
                or self._events is None or self._events.empty):
            return

        people = self._people
        events = self._events

        # Only use people with death dates for "lived before" claims
        people_with_death = people.dropna(subset=["death_year"])
        if people_with_death.empty:
            return

        p_indices = list(people_with_death.index)
        e_indices = list(events.index)
        p_weights = self._notability_weights(people_with_death, "birth_year")
        e_weights = self._notability_weights(events, "year")

        max_attempts = min(len(p_indices) * 5, 10000)
        seen = set()

        for _ in range(max_attempts):
            p_idx = self.rng.choices(p_indices, weights=p_weights, k=1)[0]
            e_idx = self.rng.choices(e_indices, weights=e_weights, k=1)[0]

            pair_key = (p_idx, e_idx)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            person = people_with_death.loc[p_idx]
            event = events.loc[e_idx]
            death_year = int(person["death_year"])
            event_year = int(event["year"])

            gap = event_year - death_year
            if gap < self.min_era_gap:
                continue

            person_name = str(person["name"])
            event_name = str(event["name"])
            p_qid = str(person["qid"])
            e_qid = str(event["qid"])

            template = self._pick_template("lived_before_event")
            true_stmt = render_template(template.pattern,person=person_name, event=event_name)
            # False: reverse — claim person lived before event when they actually lived after
            false_stmt = render_template(template.pattern,person=event_name, event=person_name)

            # For a cleaner false statement, swap to claim the event happened before the person
            # But templates are "person lived before event" so we need a person who lived AFTER
            # Find a person born after the event for the false claim
            later_people = people[people["birth_year"] > event_year + self.min_era_gap]
            if later_people.empty:
                # Fall back to direct negation style: flip person and event in claim
                false_stmt = render_template(template.pattern,
                    person=person_name, event=event_name
                ).replace(" lived before ", " lived after ").replace(
                    " died before ", " died after "
                ).replace(" predates ", " postdates ")
                strategy = NegationStrategy.DIRECT_NEGATION.value
            else:
                later_person = later_people.iloc[self.rng.randint(0, len(later_people) - 1)]
                false_stmt = render_template(template.pattern,
                    person=str(later_person["name"]), event=event_name,
                )
                strategy = NegationStrategy.DISTANT_SWAP.value

            pair_id = _make_pair_id([
                "temp", "era", p_qid, e_qid, template.id,
            ])

            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="temporal",
                relation_type="lived_before_event",
                difficulty="mixed",
                semantic_distance=gap,
                generator=self.name,
                template_id=template.id,
                negation_strategy=strategy,
                gen_params={
                    "person": person_name,
                    "person_idx": int(p_idx),
                    "event": event_name,
                    "event_idx": int(e_idx),
                    "death_year": death_year,
                    "event_year": event_year,
                    "gap_years": gap,
                },
            )

    # --- Contemporaneity ---

    def _generate_contemporaneity(self) -> Iterator[ContrastivePair]:
        """Generate '{personA} and {personB} were contemporaries' pairs."""
        if self._people is None or self._people.empty:
            return

        # Need people with both birth and death dates
        people = self._people.dropna(subset=["birth_year", "death_year"]).copy()
        if len(people) < 2:
            return

        indices = list(people.index)
        weights = self._notability_weights(people, "birth_year")

        max_attempts = min(len(indices) * 5, 10000)
        seen = set()

        for _ in range(max_attempts):
            idx_a = self.rng.choices(indices, weights=weights, k=1)[0]
            idx_b = self.rng.choices(indices, weights=weights, k=1)[0]
            if idx_a == idx_b:
                continue

            pair_key = (min(idx_a, idx_b), max(idx_a, idx_b))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            row_a = people.loc[idx_a]
            row_b = people.loc[idx_b]

            birth_a = int(row_a["birth_year"])
            death_a = int(row_a["death_year"])
            birth_b = int(row_b["birth_year"])
            death_b = int(row_b["death_year"])

            overlap = min(death_a, death_b) - max(birth_a, birth_b)

            if overlap >= self.contemporaneity_overlap_min:
                # True contemporaries — generate true pair
                name_a = str(row_a["name"])
                name_b = str(row_b["name"])
                qid_a = str(row_a["qid"])
                qid_b = str(row_b["qid"])

                template = self._pick_template("were_contemporaries")
                true_stmt = render_template(template.pattern,personA=name_a, personB=name_b)

                # Find a non-contemporary for false statement
                non_contemp = self._find_non_contemporary(people, row_a, indices, weights)
                if non_contemp is None:
                    continue

                false_stmt = render_template(template.pattern,
                    personA=name_a, personB=str(non_contemp["name"]),
                )

                pair_id = _make_pair_id([
                    "temp", "contemporary", qid_a, qid_b, template.id,
                ])

                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="temporal",
                    relation_type="were_contemporaries",
                    difficulty="mixed",
                    semantic_distance=overlap,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.DISTANT_SWAP.value,
                    gen_params={
                        "person_a": name_a,
                        "person_a_idx": int(idx_a),
                        "person_b": name_b,
                        "person_b_idx": int(idx_b),
                        "birth_a": birth_a,
                        "death_a": death_a,
                        "birth_b": birth_b,
                        "death_b": death_b,
                        "overlap_years": overlap,
                    },
                )

    def _find_non_contemporary(
        self, people: pd.DataFrame, person: pd.Series, indices: list[int],
        weights: list[float] | None = None,
    ) -> pd.Series | None:
        """Find a person whose lifespan doesn't overlap with the given person."""
        death = int(person["death_year"])
        birth = int(person["birth_year"])

        # Try random candidates (weighted if available)
        for _ in range(50):
            if weights:
                idx = self.rng.choices(indices, weights=weights, k=1)[0]
            else:
                idx = self.rng.choice(indices)
            candidate = people.loc[idx]
            c_birth = int(candidate["birth_year"])
            c_death = int(candidate["death_year"])

            # No overlap, with sufficient gap
            gap = max(c_birth - death, birth - c_death)
            if gap >= self.contemporaneity_gap_min:
                return candidate

        return None
