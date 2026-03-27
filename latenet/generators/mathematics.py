"""Mathematics generator: number theory, arithmetic, and abstract mathematical properties.

Data source: pure computation — no external dependencies.
Relations: has_property, greater_than, is_divisible_by, arithmetic_result,
           more_factors, shares_factor.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from latenet.generators.base import BaseGenerator
from latenet.generators.math_utils import (
    fibonacci_up_to,
    gcd,
    is_fibonacci,
    is_perfect_cube,
    is_perfect_number,
    is_perfect_square,
    is_power_of_two,
    is_prime,
    num_factors,
    primes_up_to,
)
from latenet.sanitize import render_template
from latenet.types import ContrastivePair, Difficulty, NegationStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Number ranges
# ---------------------------------------------------------------------------

PROPERTY_RANGE = (2, 1000)
COMPARISON_RANGE = (2, 10_000)
ARITHMETIC_OPERAND_RANGE = (2, 100)
ARITHMETIC_RESULT_MAX = 10_000
DIVISIBILITY_DIVIDEND_RANGE = (4, 500)
DIVISIBILITY_DIVISOR_RANGE = (2, 50)


# ---------------------------------------------------------------------------
# Curated number pools
# ---------------------------------------------------------------------------

# Numbers that look prime but aren't — for hard-tier property membership
_PSEUDOPRIME_TRAPS = [
    9, 15, 21, 25, 27, 33, 35, 39, 49, 51, 57, 63, 69, 77, 87, 91, 93, 95,
    111, 119, 121, 133, 143, 161, 169, 187, 203, 209, 221, 247, 253, 259,
    289, 299, 301, 319, 323, 329, 341, 361, 377, 391, 403, 407, 437, 451,
    481, 493, 511, 517, 527, 529, 533, 551, 559, 583, 589, 611, 629, 667,
    689, 697, 703, 713, 721, 731, 737, 749, 763, 779, 793, 799, 803, 817,
    841, 851, 869, 871, 893, 899, 901, 913, 921, 943, 949, 961, 979, 989,
]

# Near perfect squares — for hard-tier "is a perfect square" false statements
_NEAR_SQUARES = [
    3, 5, 8, 10, 15, 17, 24, 26, 35, 37, 48, 50, 63, 65, 80, 82,
    99, 101, 120, 122, 143, 145, 168, 170, 195, 197, 224, 226,
    255, 257, 288, 290, 323, 325, 360, 362, 399, 401,
]

# Highly composite numbers (many factors) — useful for factor comparison
_HIGHLY_COMPOSITE = [
    12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680, 2520,
]

# Perfect numbers below 10000
_PERFECT_NUMBERS = [6, 28, 496, 8128]


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MathTemplate:
    id: str
    relation: str
    pattern: str


# Property membership
_PROPERTY_TEMPLATES = [
    MathTemplate("math_property_01", "has_property",
                 "{number} is {a_property}."),
    MathTemplate("math_property_02", "has_property",
                 "The value {number} is {a_property}."),
    MathTemplate("math_property_03", "has_property",
                 "The number {number} is {a_property}."),
    MathTemplate("math_property_04", "has_property",
                 "{number} can be classified as {a_property}."),
]

# Magnitude comparison
_COMPARE_TEMPLATES = [
    MathTemplate("math_compare_01", "greater_than",
                 "{a} is greater than {b}."),
    MathTemplate("math_compare_02", "greater_than",
                 "{a} is larger than {b}."),
    MathTemplate("math_compare_03", "greater_than",
                 "{a} exceeds {b}."),
]

# Divisibility
_DIVISIBLE_TEMPLATES = [
    MathTemplate("math_divisible_01", "is_divisible_by",
                 "{a} is divisible by {b}."),
    MathTemplate("math_divisible_02", "is_divisible_by",
                 "{b} divides {a}."),
    MathTemplate("math_divisible_03", "is_divisible_by",
                 "{b} is a factor of {a}."),
    MathTemplate("math_divisible_04", "is_divisible_by",
                 "{a} is a multiple of {b}."),
]

# Arithmetic
_ARITHMETIC_TEMPLATES = {
    "multiply": [
        MathTemplate("math_arithmetic_mul_01", "arithmetic_result",
                     "{a} times {b} equals {result}."),
        MathTemplate("math_arithmetic_mul_02", "arithmetic_result",
                     "{a} multiplied by {b} is {result}."),
        MathTemplate("math_arithmetic_mul_03", "arithmetic_result",
                     "The product of {a} and {b} is {result}."),
    ],
    "add": [
        MathTemplate("math_arithmetic_add_01", "arithmetic_result",
                     "{a} plus {b} equals {result}."),
        MathTemplate("math_arithmetic_add_02", "arithmetic_result",
                     "The sum of {a} and {b} is {result}."),
    ],
    "subtract": [
        MathTemplate("math_arithmetic_sub_01", "arithmetic_result",
                     "{a} minus {b} equals {result}."),
        MathTemplate("math_arithmetic_sub_02", "arithmetic_result",
                     "The difference of {a} and {b} is {result}."),
    ],
    "divide": [
        MathTemplate("math_arithmetic_div_01", "arithmetic_result",
                     "{a} divided by {b} equals {result}."),
        MathTemplate("math_arithmetic_div_02", "arithmetic_result",
                     "The quotient of {a} and {b} is {result}."),
    ],
}

# Factor comparison
_FACTORS_TEMPLATES = [
    MathTemplate("math_factors_01", "more_factors",
                 "{a} has more factors than {b}."),
    MathTemplate("math_factors_02", "more_factors",
                 "{a} has more divisors than {b}."),
]

# Coprimality / shared factors
_COPRIME_TEMPLATES = [
    MathTemplate("math_coprime_01", "shares_factor",
                 "{a} and {b} share a common factor."),
    MathTemplate("math_coprime_02", "shares_factor",
                 "{a} and {b} have the greatest common divisor greater than 1."),
    MathTemplate("math_coprime_03", "shares_factor",
                 "{a} and {b} are coprime."),
]

_ALL_TEMPLATES: dict[str, list[MathTemplate]] = {
    "has_property": _PROPERTY_TEMPLATES,
    "greater_than": _COMPARE_TEMPLATES,
    "is_divisible_by": _DIVISIBLE_TEMPLATES,
    "more_factors": _FACTORS_TEMPLATES,
    "shares_factor": _COPRIME_TEMPLATES,
}


# ---------------------------------------------------------------------------
# Property definitions
# ---------------------------------------------------------------------------

# (label_for_true, label_for_article, checker)
# label_for_article: the form used in templates, e.g. "a prime" or "an even"
_PROPERTIES: dict[str, tuple[str, Callable[[int], bool]]] = {
    "prime": ("prime number", is_prime),
    "even": ("even number", lambda n: n % 2 == 0),
    "odd": ("odd number", lambda n: n % 2 != 0),
    "perfect_square": ("perfect square", is_perfect_square),
    "perfect_cube": ("perfect cube", is_perfect_cube),
    "power_of_two": ("power of two", is_power_of_two),
    "fibonacci": ("Fibonacci number", is_fibonacci),
    "perfect_number": ("perfect number", is_perfect_number),
}

# Groups of properties for difficulty-based swapping
# Same-category properties (number-theoretic) for medium tier
_SAME_CATEGORY_PROPS = ["prime", "even", "odd", "perfect_square", "perfect_cube",
                        "power_of_two", "fibonacci", "perfect_number"]
# Different-category (cross-category) for easy tier — structural vs divisibility
_STRUCTURAL_PROPS = ["perfect_square", "perfect_cube", "power_of_two", "fibonacci",
                     "perfect_number"]
_DIVISIBILITY_PROPS = ["prime", "even", "odd"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair_id(parts: list[str]) -> str:
    key = ":".join(str(p) for p in parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class MathematicsGenerator(BaseGenerator):
    """Generate contrastive pairs from number theory and arithmetic."""

    def __init__(
        self,
        seed: int = 42,
        max_pairs: int | None = None,
        property_range: tuple[int, int] = PROPERTY_RANGE,
        comparison_range: tuple[int, int] = COMPARISON_RANGE,
        arithmetic_operand_range: tuple[int, int] = ARITHMETIC_OPERAND_RANGE,
        arithmetic_result_max: int = ARITHMETIC_RESULT_MAX,
    ):
        super().__init__(seed=seed, max_pairs=max_pairs)
        self.property_range = property_range
        self.comparison_range = comparison_range
        self.arithmetic_operand_range = arithmetic_operand_range
        self.arithmetic_result_max = arithmetic_result_max

        # Precomputed pools (built by _build_pools before use)
        self._primes: list[int] = []
        self._fibs: list[int] = []
        self._squares: list[int] = []
        self._cubes: list[int] = []
        self._powers_of_two: list[int] = []

    @property
    def name(self) -> str:
        return "mathematics"

    def relation_types(self) -> list[str]:
        return [
            "has_property", "greater_than", "is_divisible_by",
            "arithmetic_result", "more_factors", "shares_factor",
        ]

    def domains(self) -> list[str]:
        return ["mathematics"]

    # --- Precomputation ---

    def _build_pools(self) -> None:
        lo, hi = self.property_range
        self._primes = primes_up_to(hi)
        self._fibs = fibonacci_up_to(hi)
        self._squares = [i * i for i in range(2, hi + 1) if lo <= i * i <= hi]
        self._cubes = [i ** 3 for i in range(2, hi + 1) if lo <= i ** 3 <= hi]
        self._powers_of_two = [2 ** i for i in range(1, 20) if lo <= 2 ** i <= hi]

    # --- Main generate ---

    def generate(self) -> Iterator[ContrastivePair]:
        self._build_pools()
        count = 0

        sub_generators = [
            self._generate_property,
            self._generate_greater_than,
            self._generate_divisibility,
            self._generate_arithmetic,
            self._generate_more_factors,
            self._generate_shares_factor,
        ]

        # Round-robin across sub-generators so max_pairs doesn't starve later relations
        iterators = [gen_fn() for gen_fn in sub_generators]
        counts_by_relation: dict[str, int] = {fn.__name__: 0 for fn in sub_generators}
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
                counts_by_relation[sub_generators[idx].__name__] += 1
                next_active.append(idx)
                if self.max_pairs is not None and count >= self.max_pairs:
                    logger.info("Math generator pair counts: %s", counts_by_relation)
                    return
            active = next_active

        logger.info(
            "Math generator produced %d total pairs. Per relation: %s",
            count, counts_by_relation,
        )

    # --- Template picking ---

    def _pick_template(self, relation: str, operation: str | None = None) -> MathTemplate:
        if relation == "arithmetic_result" and operation:
            templates = _ARITHMETIC_TEMPLATES[operation]
        else:
            templates = _ALL_TEMPLATES[relation]
        return templates[self.rng.randint(0, len(templates) - 1)]

    # --- 1. Property membership ---

    def _generate_property(self) -> Iterator[ContrastivePair]:
        """Generate has_property pairs: 'N is prime', 'N is even', etc."""
        lo, hi = self.property_range

        # Build pool of (number, true_properties) tuples
        # Use curated numbers plus a random sample
        curated = set(self._primes[:46])  # primes below ~200
        curated.update(self._squares)
        curated.update(self._cubes)
        curated.update(self._fibs)
        curated.update(self._powers_of_two)
        curated.update(n for n in _PERFECT_NUMBERS if lo <= n <= hi)
        curated.update(n for n in _HIGHLY_COMPOSITE if lo <= n <= hi)
        curated.update(n for n in _PSEUDOPRIME_TRAPS if lo <= n <= hi)
        curated.update(n for n in _NEAR_SQUARES if lo <= n <= hi)

        # Add some random numbers for diversity
        extra_count = max(0, 100 - len(curated))
        for _ in range(extra_count):
            curated.add(self.rng.randint(lo, hi))

        numbers = sorted(curated)
        self.rng.shuffle(numbers)

        for n in numbers:
            # Find all true properties
            true_props = [
                prop_name for prop_name, (_, checker) in _PROPERTIES.items()
                if checker(n)
            ]
            false_props = [
                prop_name for prop_name, (_, checker) in _PROPERTIES.items()
                if not checker(n)
            ]
            if not true_props or not false_props:
                continue

            true_prop = self.rng.choice(true_props)
            true_label = _PROPERTIES[true_prop][0]

            # Hard tier: pick a false property that's "close" or tricky
            hard_false = self._pick_hard_false_property(n, true_prop, false_props)
            # Medium tier: same category false property
            medium_false = self._pick_medium_false_property(n, true_prop, false_props, hard_false)
            # Easy tier: different category
            easy_false = self._pick_easy_false_property(n, true_prop, false_props,
                                                        {hard_false, medium_false})

            for false_prop, difficulty, strategy in [
                (hard_false, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value),
                (medium_false, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value),
                (easy_false, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value),
            ]:
                if false_prop is None:
                    continue

                false_label = _PROPERTIES[false_prop][0]
                template = self._pick_template("has_property")
                true_stmt = render_template(template.pattern,number=n, property=true_label)
                false_stmt = render_template(template.pattern,number=n, property=false_label)

                pair_id = _make_pair_id([
                    "math", "property", str(n), true_prop, false_prop, template.id,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="mathematics",
                    relation_type="has_property",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                )

    def _pick_hard_false_property(
        self, n: int, true_prop: str, false_props: list[str],
    ) -> str | None:
        """Pick a false property that's close/tricky for this number."""
        # Specific traps
        if true_prop == "odd" and "prime" in false_props and n in _PSEUDOPRIME_TRAPS:
            return "prime"
        if true_prop == "prime" and "even" in false_props:
            return "even"  # Primes > 2 are odd, so "even" is a sibling-swap
        if true_prop in ("perfect_square", "perfect_cube") and "prime" in false_props:
            return "prime"
        # Default: pick from same divisibility/structural category
        if true_prop in _DIVISIBILITY_PROPS:
            candidates = [p for p in false_props if p in _DIVISIBILITY_PROPS]
        else:
            candidates = [p for p in false_props if p in _STRUCTURAL_PROPS]
        if candidates:
            return self.rng.choice(candidates)
        return self.rng.choice(false_props) if false_props else None

    def _pick_medium_false_property(
        self, n: int, true_prop: str, false_props: list[str],
        exclude: str | None,
    ) -> str | None:
        candidates = [p for p in false_props if p != exclude]
        if not candidates:
            return None
        # Prefer same-category
        same_cat = [p for p in candidates if p in _SAME_CATEGORY_PROPS]
        return self.rng.choice(same_cat) if same_cat else self.rng.choice(candidates)

    def _pick_easy_false_property(
        self, n: int, true_prop: str, false_props: list[str],
        exclude: set[str | None],
    ) -> str | None:
        candidates = [p for p in false_props if p not in exclude]
        if not candidates:
            return None
        # Prefer cross-category
        if true_prop in _DIVISIBILITY_PROPS:
            cross = [p for p in candidates if p in _STRUCTURAL_PROPS]
        else:
            cross = [p for p in candidates if p in _DIVISIBILITY_PROPS]
        return self.rng.choice(cross) if cross else self.rng.choice(candidates)

    # --- 2. Magnitude comparison ---

    def _generate_greater_than(self) -> Iterator[ContrastivePair]:
        lo, hi = self.comparison_range
        pairs_per_tier = 60

        # Curate number pools by magnitude
        small = list(range(2, 10))
        medium = list(range(10, 100))
        large = list(range(100, 1000))
        very_large = list(range(1000, min(hi + 1, 10001)))

        def _make_pair(big: int, small_n: int, difficulty: str) -> ContrastivePair:
            template = self._pick_template("greater_than")
            true_stmt = render_template(template.pattern,a=big, b=small_n)
            false_stmt = render_template(template.pattern,a=small_n, b=big)
            pair_id = _make_pair_id(["math", "compare", str(big), str(small_n), template.id])
            return ContrastivePair(
                true_statement=true_stmt, false_statement=false_stmt,
                pair_id=pair_id, domain="mathematics",
                relation_type="greater_than", difficulty=difficulty,
                semantic_distance=None, generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.REVERSE_RELATION.value,
            )

        # Hard tier: close numbers (ratio 1.1–2x), same magnitude
        hard_pool = medium + large
        self.rng.shuffle(hard_pool)
        hard_count = 0
        for base in hard_pool:
            if hard_count >= pairs_per_tier:
                break
            # Pick a number 10-90% larger
            offset = self.rng.randint(max(1, base // 10), max(2, base // 2))
            other = base + offset
            if other > hi or other == base:
                continue
            yield _make_pair(other, base, Difficulty.HARD.value)
            hard_count += 1

        # Medium tier: moderate ratio (2x–10x)
        medium_bases = small + medium
        self.rng.shuffle(medium_bases)
        med_count = 0
        for base in medium_bases:
            if med_count >= pairs_per_tier:
                break
            multiplier = self.rng.randint(2, 9)
            other = base * multiplier
            if other > hi or other < lo:
                continue
            yield _make_pair(other, base, Difficulty.MEDIUM.value)
            med_count += 1

        # Easy tier: very different magnitudes (ratio > 10x)
        easy_smalls = list(small)
        easy_larges = list(large + very_large)
        self.rng.shuffle(easy_smalls)
        self.rng.shuffle(easy_larges)
        easy_count = 0
        for s in easy_smalls:
            for lg in easy_larges:
                if easy_count >= pairs_per_tier:
                    break
                if lg > 10 * s:
                    yield _make_pair(lg, s, Difficulty.EASY.value)
                    easy_count += 1
            if easy_count >= pairs_per_tier:
                break

    # --- 3. Divisibility ---

    def _generate_divisibility(self) -> Iterator[ContrastivePair]:
        lo_div, hi_div = DIVISIBILITY_DIVIDEND_RANGE
        lo_d, hi_d = DIVISIBILITY_DIVISOR_RANGE

        # Generate dividends with known divisors
        dividends = list(range(lo_div, hi_div + 1))
        self.rng.shuffle(dividends)

        for a in dividends[:200]:
            # Find true and false divisors
            true_divisors = [d for d in range(lo_d, min(hi_d + 1, a)) if a % d == 0]
            false_divisors = [d for d in range(lo_d, min(hi_d + 1, a)) if a % d != 0]

            if not true_divisors or not false_divisors:
                continue

            true_d = self.rng.choice(true_divisors)
            template = self._pick_template("is_divisible_by")
            true_stmt = render_template(template.pattern,a=a, b=true_d)

            # Hard: false divisor close to a true divisor (off by 1)
            hard_candidates = [
                d for d in false_divisors
                if any(abs(d - td) == 1 for td in true_divisors)
            ]
            # Medium: small prime that doesn't divide
            medium_candidates = [
                d for d in false_divisors
                if is_prime(d) and d <= 13 and d not in (set(hard_candidates) if hard_candidates else set())
            ]
            # Easy: obviously wrong (large relative to dividend)
            easy_candidates = [
                d for d in false_divisors
                if d > a // 2 and d not in (set(hard_candidates) | set(medium_candidates))
            ]

            for candidates, difficulty, strategy in [
                (hard_candidates, Difficulty.HARD.value, NegationStrategy.SIBLING_SWAP.value),
                (medium_candidates, Difficulty.MEDIUM.value, NegationStrategy.SIBLING_SWAP.value),
                (easy_candidates, Difficulty.EASY.value, NegationStrategy.DISTANT_SWAP.value),
            ]:
                if not candidates:
                    continue

                false_d = self.rng.choice(candidates)
                false_stmt = render_template(template.pattern,a=a, b=false_d)

                pair_id = _make_pair_id([
                    "math", "divisible", str(a), str(true_d), str(false_d), template.id,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="mathematics",
                    relation_type="is_divisible_by",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=strategy,
                )

    # --- 4. Arithmetic ---

    def _generate_arithmetic(self) -> Iterator[ContrastivePair]:
        lo, hi = self.arithmetic_operand_range
        result_max = self.arithmetic_result_max

        operands = list(range(lo, hi + 1))

        # Generate pairs for each operation
        for operation in ["multiply", "add", "subtract", "divide"]:
            self.rng.shuffle(operands)
            pair_count = 0

            for _ in range(300):
                a = self.rng.choice(operands)
                b = self.rng.choice(operands)

                if operation == "multiply":
                    result = a * b
                    if result > result_max:
                        continue
                elif operation == "add":
                    result = a + b
                elif operation == "subtract":
                    if a <= b:
                        continue  # Keep result positive
                    result = a - b
                elif operation == "divide":
                    if b == 0 or a % b != 0:
                        continue  # Only integer division
                    result = a // b
                    if result < 2:
                        continue  # Non-trivial results only
                else:
                    continue

                # Generate false results at different difficulty levels
                for diff_label, perturbation_range in [
                    (Difficulty.HARD.value, (1, 2)),
                    (Difficulty.MEDIUM.value, (3, 10)),
                    (Difficulty.EASY.value, (11, 100)),
                ]:
                    lo_p, hi_p = perturbation_range
                    offset = self.rng.randint(lo_p, min(hi_p, max(lo_p, result // 2 or 1)))
                    if self.rng.random() < 0.5:
                        offset = -offset
                    false_result = result + offset
                    if false_result <= 0 or false_result == result:
                        continue

                    template = self._pick_template("arithmetic_result", operation)
                    true_stmt = render_template(template.pattern,a=a, b=b, result=result)
                    false_stmt = render_template(template.pattern,a=a, b=b, result=false_result)

                    pair_id = _make_pair_id([
                        "math", "arith", operation, str(a), str(b),
                        str(false_result), template.id,
                    ])
                    yield ContrastivePair(
                        true_statement=true_stmt,
                        false_statement=false_stmt,
                        pair_id=pair_id,
                        domain="mathematics",
                        relation_type="arithmetic_result",
                        difficulty=diff_label,
                        semantic_distance=None,
                        generator=self.name,
                        template_id=template.id,
                        negation_strategy=NegationStrategy.SIBLING_SWAP.value,
                    )

                pair_count += 1
                if pair_count >= 50:
                    break

    # --- 5. Factor comparison ---

    def _generate_more_factors(self) -> Iterator[ContrastivePair]:
        lo, hi = self.property_range

        # Build factor-count buckets for all numbers in range
        factor_counts: dict[int, int] = {}
        for n in range(lo, min(hi + 1, 500)):
            factor_counts[n] = num_factors(n)

        # Group numbers by factor count for hard-tier (similar counts) pairing
        by_count: dict[int, list[int]] = {}
        for n, nf in factor_counts.items():
            by_count.setdefault(nf, []).append(n)

        # --- Hard tier: pair numbers with similar but different factor counts ---
        # e.g. 4 factors vs 6 factors — the difference is subtle
        hard_pairs: list[tuple[int, int]] = []
        sorted_counts = sorted(by_count.keys())
        for i in range(len(sorted_counts)):
            for j in range(i + 1, len(sorted_counts)):
                nf_a, nf_b = sorted_counts[i], sorted_counts[j]
                if nf_b - nf_a > 2:
                    break  # Only pair counts within 1-2 of each other
                if nf_a < 3:
                    continue  # Skip trivial (primes, etc.)
                for a in by_count[nf_b][:5]:
                    for b in by_count[nf_a][:5]:
                        if a != b:
                            hard_pairs.append((a, b))
        self.rng.shuffle(hard_pairs)

        for a, b in hard_pairs[:60]:
            template = self._pick_template("more_factors")
            true_stmt = render_template(template.pattern,a=a, b=b)
            false_stmt = render_template(template.pattern,a=b, b=a)
            pair_id = _make_pair_id(["math", "factors", str(a), str(b), template.id])
            yield ContrastivePair(
                true_statement=true_stmt, false_statement=false_stmt,
                pair_id=pair_id, domain="mathematics",
                relation_type="more_factors", difficulty=Difficulty.HARD.value,
                semantic_distance=None, generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.REVERSE_RELATION.value,
            )

        # --- Medium tier: moderate factor ratio (2x-4x) ---
        medium_pairs: list[tuple[int, int]] = []
        mid_factor_nums = [n for n, nf in factor_counts.items() if 4 <= nf <= 8]
        low_factor_nums = [n for n, nf in factor_counts.items() if nf == 2 or nf == 3]
        self.rng.shuffle(mid_factor_nums)
        self.rng.shuffle(low_factor_nums)
        for a in mid_factor_nums[:40]:
            for b in low_factor_nums[:20]:
                if a != b:
                    ratio = factor_counts[a] / factor_counts[b]
                    if 2 <= ratio <= 4:
                        medium_pairs.append((a, b))
        self.rng.shuffle(medium_pairs)

        for a, b in medium_pairs[:60]:
            template = self._pick_template("more_factors")
            true_stmt = render_template(template.pattern,a=a, b=b)
            false_stmt = render_template(template.pattern,a=b, b=a)
            pair_id = _make_pair_id(["math", "factors", str(a), str(b), template.id])
            yield ContrastivePair(
                true_statement=true_stmt, false_statement=false_stmt,
                pair_id=pair_id, domain="mathematics",
                relation_type="more_factors", difficulty=Difficulty.MEDIUM.value,
                semantic_distance=None, generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.REVERSE_RELATION.value,
            )

        # --- Easy tier: large factor ratio (highly composite vs primes) ---
        high_factor_nums = [n for n in _HIGHLY_COMPOSITE if lo <= n <= hi]
        for n in range(lo, min(hi + 1, 500)):
            if num_factors(n) >= 8:
                high_factor_nums.append(n)
        high_factor_nums = sorted(set(high_factor_nums))
        prime_pool = self._primes[:50]
        self.rng.shuffle(high_factor_nums)
        self.rng.shuffle(prime_pool)

        easy_pairs: list[tuple[int, int]] = []
        for a in high_factor_nums[:40]:
            for b in prime_pool[:20]:
                if a != b and factor_counts.get(a, num_factors(a)) / factor_counts.get(b, num_factors(b)) > 4:
                    easy_pairs.append((a, b))
        self.rng.shuffle(easy_pairs)

        for a, b in easy_pairs[:60]:
            template = self._pick_template("more_factors")
            true_stmt = render_template(template.pattern,a=a, b=b)
            false_stmt = render_template(template.pattern,a=b, b=a)
            pair_id = _make_pair_id(["math", "factors", str(a), str(b), template.id])
            yield ContrastivePair(
                true_statement=true_stmt, false_statement=false_stmt,
                pair_id=pair_id, domain="mathematics",
                relation_type="more_factors", difficulty=Difficulty.EASY.value,
                semantic_distance=None, generator=self.name,
                template_id=template.id,
                negation_strategy=NegationStrategy.REVERSE_RELATION.value,
            )

    # --- 6. Shared factor / coprime ---

    def _generate_shares_factor(self) -> Iterator[ContrastivePair]:
        lo, hi = self.property_range

        # Pool of numbers
        pool = list(range(lo, min(hi + 1, 200)))
        self.rng.shuffle(pool)

        # Bucket candidates by difficulty tier, then yield in balanced order
        hard_candidates: list[tuple[int, int, bool, int]] = []   # (a, b, share_factor, gcd)
        medium_candidates: list[tuple[int, int, bool, int]] = []
        easy_candidates: list[tuple[int, int, bool, int]] = []

        for i in range(len(pool)):
            for j in range(i + 1, min(i + 20, len(pool))):
                a, b = pool[i], pool[j]
                g = gcd(a, b)
                share_factor = g > 1

                # Difficulty based on how obvious the shared-factor relationship is:
                # Hard: GCD is a small prime (2 or 3) — subtle shared factor
                #       or coprime pair where numbers are both even-looking composites
                # Medium: GCD is a moderate composite or mid-range prime
                # Easy: GCD is large / obviously share a factor (e.g. both multiples of 10)
                if share_factor:
                    if g <= 3:
                        hard_candidates.append((a, b, share_factor, g))
                    elif g <= 10:
                        medium_candidates.append((a, b, share_factor, g))
                    else:
                        easy_candidates.append((a, b, share_factor, g))
                else:
                    # Coprime pairs: hard when numbers are close together (looks like they share)
                    gap = abs(a - b)
                    if gap <= 3:
                        hard_candidates.append((a, b, share_factor, g))
                    elif gap <= 15:
                        medium_candidates.append((a, b, share_factor, g))
                    else:
                        easy_candidates.append((a, b, share_factor, g))

        self.rng.shuffle(hard_candidates)
        self.rng.shuffle(medium_candidates)
        self.rng.shuffle(easy_candidates)

        pairs_per_tier = 60
        for candidates, difficulty in [
            (hard_candidates[:pairs_per_tier], Difficulty.HARD.value),
            (medium_candidates[:pairs_per_tier], Difficulty.MEDIUM.value),
            (easy_candidates[:pairs_per_tier], Difficulty.EASY.value),
        ]:
            for a, b, share_factor, _g in candidates:
                if share_factor:
                    template_pool = _COPRIME_TEMPLATES[:2]
                    template = template_pool[self.rng.randint(0, len(template_pool) - 1)]
                    true_stmt = render_template(template.pattern, a=a, b=b)
                    false_template = _COPRIME_TEMPLATES[2]
                    false_stmt = render_template(false_template.pattern, a=a, b=b)
                else:
                    template = _COPRIME_TEMPLATES[2]
                    true_stmt = render_template(template.pattern, a=a, b=b)
                    false_template = _COPRIME_TEMPLATES[0]
                    false_stmt = render_template(false_template.pattern, a=a, b=b)

                pair_id = _make_pair_id([
                    "math", "coprime", str(a), str(b),
                    str(share_factor), template.id,
                ])
                yield ContrastivePair(
                    true_statement=true_stmt,
                    false_statement=false_stmt,
                    pair_id=pair_id,
                    domain="mathematics",
                    relation_type="shares_factor",
                    difficulty=difficulty,
                    semantic_distance=None,
                    generator=self.name,
                    template_id=template.id,
                    negation_strategy=NegationStrategy.DIRECT_NEGATION.value,
                )
