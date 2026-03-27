"""Chemistry generator: periodic table relations, element properties, group membership.

Data source: mendeleev (MIT license) — bundled SQLite database of all 118 elements.
Relations: symbol_of, member_of_group, state_at_room_temp, in_block,
           atomic_number_greater, property_greater.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from mendeleev import element as get_element
from mendeleev.fetch import fetch_table

from latenet.generators.base import BaseGenerator
from latenet.sanitize import render_template
from latenet.types import ContrastivePair, Difficulty, NegationStrategy

logger = logging.getLogger(__name__)

ROOM_TEMP_K = 298.15
MAX_ATOMIC_NUMBER_DEFAULT = 103  # Skip synthetic/short-lived elements


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChemTemplate:
    id: str
    relation: str
    pattern: str


# Symbol equivalence
_SYMBOL_TEMPLATES = [
    ChemTemplate("chem_symbol_01", "symbol_of",
                 "The chemical symbol for {element} is {symbol}."),
    ChemTemplate("chem_symbol_02", "symbol_of",
                 "{symbol} is the chemical symbol for {element}."),
    ChemTemplate("chem_symbol_03", "symbol_of",
                 "The element {element} is represented by the symbol {symbol}."),
    ChemTemplate("chem_symbol_04", "symbol_of",
                 "{element} has the chemical symbol {symbol}."),
]

# Group/series membership
_GROUP_TEMPLATES = [
    ChemTemplate("chem_group_01", "member_of_group",
                 "{element} is {a_series_name}."),
    ChemTemplate("chem_group_02", "member_of_group",
                 "{element} belongs to the {series_name} group."),
    ChemTemplate("chem_group_03", "member_of_group",
                 "{element} is classified as {a_series_name}."),
    ChemTemplate("chem_group_04", "member_of_group",
                 "The element {element} is {a_series_name}."),
]

# State of matter
_STATE_TEMPLATES = [
    ChemTemplate("chem_state_01", "state_at_room_temp",
                 "{element} is {a_state} at room temperature."),
    ChemTemplate("chem_state_02", "state_at_room_temp",
                 "At room temperature, {element} is {a_state}."),
    ChemTemplate("chem_state_03", "state_at_room_temp",
                 "{element} exists as {a_state} under standard conditions."),
]

# Block membership
_BLOCK_TEMPLATES = [
    ChemTemplate("chem_block_01", "in_block",
                 "{element} is a {block}-block element."),
    ChemTemplate("chem_block_02", "in_block",
                 "{element} is in the {block} block of the periodic table."),
    ChemTemplate("chem_block_03", "in_block",
                 "The element {element} belongs to the {block} block."),
]

# Atomic number magnitude
_ATOMIC_NUM_TEMPLATES = [
    ChemTemplate("chem_atomnum_01", "atomic_number_greater",
                 "{big} has a higher atomic number than {small}."),
    ChemTemplate("chem_atomnum_02", "atomic_number_greater",
                 "{big}'s atomic number is greater than {small}'s."),
    ChemTemplate("chem_atomnum_03", "atomic_number_greater",
                 "In terms of atomic number, {big} exceeds {small}."),
]

# Property comparisons
_PROP_TEMPLATES = [
    ChemTemplate("chem_prop_01", "property_greater",
                 "{big} has a higher {property} than {small}."),
    ChemTemplate("chem_prop_02", "property_greater",
                 "{big}'s {property} is greater than {small}'s."),
    ChemTemplate("chem_prop_03", "property_greater",
                 "In terms of {property}, {big} exceeds {small}."),
]

_ALL_TEMPLATES: dict[str, list[ChemTemplate]] = {
    "symbol_of": _SYMBOL_TEMPLATES,
    "member_of_group": _GROUP_TEMPLATES,
    "state_at_room_temp": _STATE_TEMPLATES,
    "in_block": _BLOCK_TEMPLATES,
    "atomic_number_greater": _ATOMIC_NUM_TEMPLATES,
    "property_greater": _PROP_TEMPLATES,
}

# States for swapping
_ALL_STATES = ["solid", "liquid", "gas"]
_ALL_BLOCKS = ["s", "p", "d", "f"]

# Property display names
_PROPERTY_NAMES = {
    "en_pauling": "electronegativity",
    "atomic_weight": "atomic weight",
    "density": "density",
}


def _make_pair_id(parts: list[str]) -> str:
    key = ":".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _state_at_room_temp(mp: float | None, bp: float | None) -> str | None:
    """Derive state of matter at 298.15 K from melting/boiling points."""
    if mp is None or bp is None:
        return None
    if ROOM_TEMP_K < mp:
        return "solid"
    elif ROOM_TEMP_K < bp:
        return "liquid"
    else:
        return "gas"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class ChemistryGenerator(BaseGenerator):
    """Generate contrastive pairs from periodic table data."""

    def __init__(
        self,
        seed: int = 42,
        max_pairs: int | None = None,
        max_atomic_number: int = MAX_ATOMIC_NUMBER_DEFAULT,
        magnitude_ratio: float = 3.0,
        max_false_per_true: int = 2,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)
        self.max_atomic_number = max_atomic_number
        self.magnitude_ratio = magnitude_ratio
        self.max_false_per_true = max_false_per_true

        # Loaded lazily
        self._elements: pd.DataFrame | None = None
        self._series_map: dict[int, str] | None = None

    @property
    def name(self) -> str:
        return "chemistry"

    def relation_types(self) -> list[str]:
        return [
            "symbol_of", "member_of_group", "state_at_room_temp",
            "in_block", "atomic_number_greater", "property_greater",
        ]

    def domains(self) -> list[str]:
        return ["chemistry"]

    # --- Data loading ---

    def _load_data(self) -> None:
        if self._elements is not None:
            return

        logger.info("Loading chemistry data from mendeleev...")

        # Load series lookup
        series_df = fetch_table("series")
        self._series_map = dict(zip(series_df["id"], series_df["name"]))

        # Load elements table
        elements_df = fetch_table("elements")
        elements_df = elements_df[
            elements_df["atomic_number"] <= self.max_atomic_number
        ].copy()

        # Fetch melting/boiling points per element (not in table)
        mps, bps = [], []
        for _, row in elements_df.iterrows():
            try:
                el = get_element(int(row["atomic_number"]))
                mps.append(el.melting_point)
                bps.append(el.boiling_point)
            except Exception:
                mps.append(None)
                bps.append(None)

        elements_df["melting_point"] = mps
        elements_df["boiling_point"] = bps

        # Derive state at room temperature
        elements_df["state_rt"] = [
            _state_at_room_temp(mp, bp)
            for mp, bp in zip(elements_df["melting_point"], elements_df["boiling_point"])
        ]

        # Map series_id to series name
        elements_df["_series_name"] = elements_df["series_id"].map(self._series_map)

        # Make series name lowercase for natural-sounding templates
        # e.g. "Alkali metals" -> "alkali metal" (singular for templates like "X is a ...")
        elements_df["series_label"] = elements_df["_series_name"].apply(
            lambda s: s.lower().rstrip("s") if pd.notna(s) and s.lower().endswith("s") else (
                s.lower() if pd.notna(s) else None
            )
        )

        self._elements = elements_df
        logger.info(
            "Chemistry data loaded: %d elements (max Z=%d)",
            len(self._elements), self.max_atomic_number,
        )

    # --- Main generate ---

    def generate(self) -> Iterator[ContrastivePair]:
        self._load_data()
        count = 0

        generators = [
            self._generate_symbol,
            self._generate_group,
            self._generate_state,
            self._generate_block,
            self._generate_atomic_number,
            self._generate_property,
        ]

        # Round-robin across sub-generators so max_pairs doesn't starve later relations
        iterators = [gen_fn() for gen_fn in generators]
        counts_by_relation: dict[str, int] = {fn.__name__: 0 for fn in generators}
        active = list(range(len(iterators)))

        while active:
            next_active = []
            for idx in active:
                try:
                    pair = next(iterators[idx])
                except StopIteration:
                    continue
                yield pair
                count += 1
                counts_by_relation[generators[idx].__name__] += 1
                next_active.append(idx)
                if self.max_pairs is not None and count >= self.max_pairs:
                    logger.info("Chemistry generator pair counts: %s", counts_by_relation)
                    return
            active = next_active

        logger.info(
            "Chemistry generator produced %d total pairs. Per relation: %s",
            count, counts_by_relation,
        )

    # --- Helpers ---

    def _pick_template(self, relation: str) -> ChemTemplate:
        templates = _ALL_TEMPLATES[relation]
        return templates[self.rng.randint(0, len(templates) - 1)]

    def _elements_in_same_group(self, el_row: pd.Series) -> list[pd.Series]:
        """Elements in the same group_id (column of periodic table)."""
        gid = el_row["group_id"]
        if pd.isna(gid):
            return []
        return [
            r for _, r in self._elements.iterrows()
            if r["group_id"] == gid and r["atomic_number"] != el_row["atomic_number"]
        ]

    def _elements_in_same_period(self, el_row: pd.Series) -> list[pd.Series]:
        """Elements in the same period (row of periodic table)."""
        period = el_row["period"]
        return [
            r for _, r in self._elements.iterrows()
            if r["period"] == period and r["atomic_number"] != el_row["atomic_number"]
        ]

    def _elements_in_same_block(self, el_row: pd.Series) -> list[pd.Series]:
        """Elements in the same block."""
        block = el_row["block"]
        if pd.isna(block):
            return []
        return [
            r for _, r in self._elements.iterrows()
            if r["block"] == block and r["atomic_number"] != el_row["atomic_number"]
        ]

    def _elements_in_different_block(self, el_row: pd.Series) -> list[pd.Series]:
        """Elements in a different block."""
        block = el_row["block"]
        return [
            r for _, r in self._elements.iterrows()
            if r["block"] != block and pd.notna(r["block"])
        ]

    def _pick_swap_element(
        self, el_row: pd.Series, exclude_names: set[str] | None = None,
    ) -> list[tuple[pd.Series, str, str]]:
        """Pick swap elements at hard/medium/easy difficulty levels.

        Returns list of (swap_element_row, difficulty, negation_strategy).
        """
        exclude = exclude_names or set()
        swaps = []

        # Hard: same group or period
        same_gp = self._elements_in_same_group(el_row) + self._elements_in_same_period(el_row)
        # Deduplicate by atomic number
        seen_z = set()
        hard_candidates = []
        for r in same_gp:
            z = r["atomic_number"]
            if z not in seen_z and r["name"] not in exclude:
                seen_z.add(z)
                hard_candidates.append(r)
        if hard_candidates:
            pick = hard_candidates[self.rng.randint(0, len(hard_candidates) - 1)]
            swaps.append((pick, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value))

        # Medium: same block, different group/period
        same_block = [
            r for r in self._elements_in_same_block(el_row)
            if r["atomic_number"] not in seen_z and r["name"] not in exclude
        ]
        if same_block:
            pick = same_block[self.rng.randint(0, len(same_block) - 1)]
            swaps.append((pick, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value))

        # Easy: different block
        diff_block = [
            r for r in self._elements_in_different_block(el_row)
            if r["name"] not in exclude
        ]
        if diff_block:
            pick = diff_block[self.rng.randint(0, len(diff_block) - 1)]
            swaps.append((pick, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value))

        return swaps

    # --- Symbol equivalence ---

    def _generate_symbol(self) -> Iterator[ContrastivePair]:
        """Element name <-> symbol pairs."""
        indices = list(self._elements.index)
        self.rng.shuffle(indices)

        for idx in indices:
            el_row = self._elements.loc[idx]
            name = el_row["name"]
            symbol = el_row["symbol"]

            swaps = self._pick_swap_element(el_row, exclude_names={name})
            if not swaps:
                continue

            template = self._pick_template("symbol_of")

            for swap_el, difficulty, strategy in swaps[:self.max_false_per_true]:
                # Swap the symbol: "The chemical symbol for gold is Ag"
                false_stmt = render_template(template.pattern,
                    element=name, symbol=swap_el["symbol"]
                )
                true_stmt = render_template(template.pattern,element=name, symbol=symbol)
                pair_id = _make_pair_id([
                    "chem", "symbol", name, swap_el["symbol"], template.id
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="chemistry",
                    relation_type="symbol_of",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                    gen_params={
                        "element": name,
                        "true_symbol": symbol,
                        "false_symbol": swap_el["symbol"],
                        "atomic_number": int(el_row["atomic_number"]),
                    },
                )

    # --- Group/series membership ---

    def _generate_group(self) -> Iterator[ContrastivePair]:
        """Element is a {series_name} pairs."""
        indices = list(self._elements.index)
        self.rng.shuffle(indices)

        all_series = [
            s for s in self._elements["series_label"].dropna().unique()
        ]
        if not all_series:
            return

        for idx in indices:
            el_row = self._elements.loc[idx]
            series_label = el_row["series_label"]
            if pd.isna(series_label) or series_label is None:
                continue

            el_name = el_row["name"]
            template = self._pick_template("member_of_group")
            true_stmt = render_template(template.pattern,
                element=el_name, series_name=series_label
            )

            # Pick wrong series at different difficulty levels
            other_series = [s for s in all_series if s != series_label]
            if not other_series:
                continue

            # Hard: adjacent series (nearby in series_id)
            sid = el_row["series_id"]

            # Sort other series by series_id distance
            series_to_id = {}
            for _, er in self._elements.iterrows():
                sl = er["series_label"]
                if pd.notna(sl) and sl in other_series:
                    series_to_id[sl] = er["series_id"]

            nearby = sorted(series_to_id.keys(), key=lambda s: abs(series_to_id[s] - sid))

            swaps_emitted = 0
            if nearby:
                # Hard: closest series
                hard_series = nearby[0]
                false_stmt = render_template(template.pattern,
                    element=el_name, series_name=hard_series
                )
                pair_id = _make_pair_id([
                    "chem", "group", el_name, hard_series, template.id
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="chemistry",
                    relation_type="member_of_group",
                    difficulty=Difficulty.HARD.value,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.SIBLING_SWAP.value,
                    gen_params={
                        "element": el_name,
                        "true_series": series_label,
                        "false_series": hard_series,
                        "atomic_number": int(el_row["atomic_number"]),
                    },
                )
                swaps_emitted += 1

            if len(nearby) > 1 and swaps_emitted < self.max_false_per_true:
                # Easy: farthest series
                easy_series = nearby[-1]
                false_stmt = render_template(template.pattern,
                    element=el_name, series_name=easy_series
                )
                pair_id = _make_pair_id([
                    "chem", "group", el_name, easy_series, template.id
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="chemistry",
                    relation_type="member_of_group",
                    difficulty=Difficulty.EASY.value,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.DISTANT_SWAP.value,
                    gen_params={
                        "element": el_name,
                        "true_series": series_label,
                        "false_series": easy_series,
                        "atomic_number": int(el_row["atomic_number"]),
                    },
                )

    # --- State of matter ---

    def _generate_state(self) -> Iterator[ContrastivePair]:
        """Element is a {state} at room temperature."""
        valid = self._elements[self._elements["state_rt"].notna()].copy()
        indices = list(valid.index)
        self.rng.shuffle(indices)

        for idx in indices:
            el_row = valid.loc[idx]
            el_name = el_row["name"]
            true_state = el_row["state_rt"]
            wrong_states = [s for s in _ALL_STATES if s != true_state]
            if not wrong_states:
                continue

            template = self._pick_template("state_at_room_temp")
            true_stmt = render_template(template.pattern,element=el_name, state=true_state)

            # Hard: adjacent state (solid<->liquid, liquid<->gas)
            if true_state == "liquid":
                hard_state = self.rng.choice(wrong_states)
            elif true_state == "solid":
                hard_state = "liquid"
            else:  # gas
                hard_state = "liquid"

            false_stmt = render_template(template.pattern,element=el_name, state=hard_state)
            pair_id = _make_pair_id([
                "chem", "state", el_name, hard_state, template.id
            ])
            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="chemistry",
                relation_type="state_at_room_temp",
                difficulty=Difficulty.HARD.value,
                semantic_distance=None,
                generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.SIBLING_SWAP.value,
                gen_params={
                    "element": el_name,
                    "true_state": true_state,
                    "false_state": hard_state,
                    "atomic_number": int(el_row["atomic_number"]),
                },
            )

            # Easy: maximally different state
            if true_state == "solid":
                easy_state = "gas"
            elif true_state == "gas":
                easy_state = "solid"
            else:
                # Liquid — already emitted one, pick the other wrong state
                easy_state = [s for s in wrong_states if s != hard_state][0]

            false_stmt = render_template(template.pattern,element=el_name, state=easy_state)
            pair_id = _make_pair_id([
                "chem", "state", el_name, easy_state, template.id
            ])
            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="chemistry",
                relation_type="state_at_room_temp",
                difficulty=Difficulty.EASY.value,
                semantic_distance=None,
                generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.DISTANT_SWAP.value,
                gen_params={
                    "element": el_name,
                    "true_state": true_state,
                    "false_state": easy_state,
                    "atomic_number": int(el_row["atomic_number"]),
                },
            )

    # --- Block membership ---

    def _generate_block(self) -> Iterator[ContrastivePair]:
        """Element is a {block}-block element."""
        valid = self._elements[self._elements["block"].notna()].copy()
        indices = list(valid.index)
        self.rng.shuffle(indices)

        for idx in indices:
            el_row = valid.loc[idx]
            el_name = el_row["name"]
            true_block = el_row["block"]
            wrong_blocks = [b for b in _ALL_BLOCKS if b != true_block]
            if not wrong_blocks:
                continue

            template = self._pick_template("in_block")
            true_stmt = render_template(template.pattern,element=el_name, block=true_block)

            # Pick a wrong block
            wrong = self.rng.choice(wrong_blocks)
            false_stmt = render_template(template.pattern,element=el_name, block=wrong)

            # Difficulty: adjacent blocks are harder
            # s<->p is adjacent, d<->p is adjacent, d<->f is adjacent
            _adjacent = {
                "s": {"p"}, "p": {"s", "d"}, "d": {"p", "f"}, "f": {"d"},
            }
            if wrong in _adjacent.get(true_block, set()):
                difficulty = Difficulty.HARD.value
                strategy = NegationStrategy.SIBLING_SWAP.value
            else:
                difficulty = Difficulty.EASY.value
                strategy = NegationStrategy.DISTANT_SWAP.value

            pair_id = _make_pair_id([
                "chem", "block", el_name, wrong, template.id
            ])
            yield ContrastivePair(
                true_statement=true_stmt,
                false_statement=false_stmt,
                pair_id=pair_id,
                domain="chemistry",
                relation_type="in_block",
                difficulty=difficulty,
                semantic_distance=None,
                generator=self.name,
                template_id=template.id,
                negation_strategy=strategy,
                gen_params={
                    "element": el_name,
                    "true_block": true_block,
                    "false_block": wrong,
                    "atomic_number": int(el_row["atomic_number"]),
                },
            )

    # --- Atomic number magnitude ---

    def _generate_atomic_number(self) -> Iterator[ContrastivePair]:
        """Element A has a higher atomic number than Element B."""
        elements = self._elements
        names = list(elements["name"])
        z_map = dict(zip(elements["name"], elements["atomic_number"]))
        block_map = dict(zip(elements["name"], elements["block"]))

        self.rng.shuffle(names)
        templates = _ALL_TEMPLATES["atomic_number_greater"]

        for i, a in enumerate(names):
            for j in range(i + 1, len(names)):
                b = names[j]
                z_a, z_b = z_map[a], z_map[b]
                big, small = (a, b) if z_a > z_b else (b, a)
                z_big, z_small = max(z_a, z_b), min(z_a, z_b)

                if z_small <= 0 or z_big / z_small < self.magnitude_ratio:
                    continue

                template = templates[self.rng.randint(0, len(templates) - 1)]
                true_stmt = render_template(template.pattern,big=big, small=small)
                false_stmt = render_template(template.pattern,big=small, small=big)
                pair_id = _make_pair_id([
                    "chem", "atomnum", big, small, template.id
                ])

                # Difficulty: same block = hard, else easy
                if block_map.get(big) == block_map.get(small) and pd.notna(block_map.get(big)):
                    diff = Difficulty.HARD.value
                else:
                    ratio = z_big / z_small
                    if ratio < 5:
                        diff = Difficulty.MEDIUM.value
                    else:
                        diff = Difficulty.EASY.value

                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="chemistry",
                    relation_type="atomic_number_greater",
                    difficulty=diff,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                    gen_params={
                        "big_element": big,
                        "small_element": small,
                        "big_z": int(z_big),
                        "small_z": int(z_small),
                    },
                )

    # --- Property comparisons ---

    def _generate_property(self) -> Iterator[ContrastivePair]:
        """Property comparison pairs: electronegativity, atomic weight, density."""
        elements = self._elements
        templates = _ALL_TEMPLATES["property_greater"]

        for prop_col, prop_label in _PROPERTY_NAMES.items():
            valid = elements[elements[prop_col].notna() & (elements[prop_col] > 0)].copy()
            names = list(valid["name"])
            val_map = dict(zip(valid["name"], valid[prop_col]))
            block_map = dict(zip(valid["name"], valid["block"]))

            self.rng.shuffle(names)

            for i, a in enumerate(names):
                for j in range(i + 1, len(names)):
                    b = names[j]
                    va, vb = val_map[a], val_map[b]
                    big, small = (a, b) if va > vb else (b, a)
                    v_big, v_small = max(va, vb), min(va, vb)

                    if v_small <= 0 or v_big / v_small < self.magnitude_ratio:
                        continue

                    template = templates[self.rng.randint(0, len(templates) - 1)]
                    true_stmt = render_template(template.pattern,
                        big=big, small=small, property=prop_label
                    )
                    false_stmt = render_template(template.pattern,
                        big=small, small=big, property=prop_label
                    )
                    pair_id = _make_pair_id([
                        "chem", "prop", prop_col, big, small, template.id
                    ])

                    # Difficulty: same block = hard
                    if (block_map.get(big) == block_map.get(small)
                            and pd.notna(block_map.get(big))):
                        diff = Difficulty.HARD.value
                    else:
                        ratio = v_big / v_small
                        if ratio < 5:
                            diff = Difficulty.MEDIUM.value
                        else:
                            diff = Difficulty.EASY.value

                    yield ContrastivePair(
                        true_statement=true_stmt,
                        false_statement=false_stmt,
                        pair_id=pair_id,
                        domain="chemistry",
                        relation_type="property_greater",
                        difficulty=diff,
                        semantic_distance=None,
                        generator=self.name,
                        template_id=template.id,
                        negation_strategy=NegationStrategy.REVERSE_RELATION.value,
                        gen_params={
                            "big_element": big,
                            "small_element": small,
                            "property": prop_col,
                            "big_value": float(v_big),
                            "small_value": float(v_small),
                        },
                    )
