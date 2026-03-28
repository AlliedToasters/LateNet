"""Tests for the mathematics generator and math utilities."""

from __future__ import annotations

from latenet.generators.math_utils import (
    digit_sum,
    fibonacci_up_to,
    gcd,
    is_divisible,
    is_fibonacci,
    is_perfect_cube,
    is_perfect_number,
    is_perfect_square,
    is_power_of_two,
    is_prime,
    num_factors,
    prime_factors,
    primes_up_to,
)
from latenet.generators.mathematics import MathematicsGenerator
from latenet.types import ContrastivePair, Difficulty


# ===========================================================================
# Math utilities tests
# ===========================================================================


class TestIsPrime:
    def test_small_primes(self):
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
            assert is_prime(p), f"{p} should be prime"

    def test_not_prime(self):
        for n in [0, 1, 4, 6, 8, 9, 10, 15, 21, 25, 100]:
            assert not is_prime(n), f"{n} should not be prime"

    def test_pseudoprime_traps(self):
        """Numbers that look prime-ish but aren't."""
        for n in [51, 57, 91, 119, 121, 169, 289, 323, 341, 529, 841, 961]:
            assert not is_prime(n), f"{n} should not be prime"

    def test_larger_primes(self):
        for p in [101, 127, 131, 137, 149, 997]:
            assert is_prime(p), f"{p} should be prime"

    def test_negative(self):
        assert not is_prime(-7)


class TestPrimeFactors:
    def test_prime_input(self):
        assert prime_factors(17) == [17]

    def test_composite(self):
        assert prime_factors(12) == [2, 2, 3]

    def test_power_of_two(self):
        assert prime_factors(64) == [2, 2, 2, 2, 2, 2]

    def test_one(self):
        assert prime_factors(1) == []

    def test_zero(self):
        assert prime_factors(0) == []

    def test_large(self):
        assert prime_factors(91) == [7, 13]


class TestPerfectSquare:
    def test_squares(self):
        for n in [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 225, 400, 625, 900]:
            assert is_perfect_square(n), f"{n} should be a perfect square"

    def test_not_squares(self):
        for n in [2, 3, 5, 7, 8, 10, 15, 24, 26, 48, 50, 99, 101]:
            assert not is_perfect_square(n), f"{n} should not be a perfect square"

    def test_negative(self):
        assert not is_perfect_square(-4)


class TestPerfectCube:
    def test_cubes(self):
        for n in [0, 1, 8, 27, 64, 125, 216, 343, 512, 729]:
            assert is_perfect_cube(n), f"{n} should be a perfect cube"

    def test_not_cubes(self):
        for n in [2, 3, 7, 9, 26, 28, 63, 65, 124, 126]:
            assert not is_perfect_cube(n), f"{n} should not be a perfect cube"

    def test_negative(self):
        assert not is_perfect_cube(-8)


class TestIsFibonacci:
    def test_fibonacci_numbers(self):
        fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        for n in fibs:
            assert is_fibonacci(n), f"{n} should be Fibonacci"

    def test_not_fibonacci(self):
        for n in [4, 6, 7, 9, 10, 14, 22, 35, 56, 90, 145]:
            assert not is_fibonacci(n), f"{n} should not be Fibonacci"


class TestPowerOfTwo:
    def test_powers(self):
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            assert is_power_of_two(n), f"{n} should be a power of two"

    def test_not_powers(self):
        for n in [0, 3, 5, 6, 7, 9, 10, 12, 15, 17, 100]:
            assert not is_power_of_two(n), f"{n} should not be a power of two"


class TestGcd:
    def test_coprime(self):
        assert gcd(7, 11) == 1

    def test_common_factor(self):
        assert gcd(12, 18) == 6

    def test_one_is_multiple(self):
        assert gcd(15, 5) == 5

    def test_same(self):
        assert gcd(7, 7) == 7

    def test_with_zero(self):
        assert gcd(5, 0) == 5


class TestIsDivisible:
    def test_divisible(self):
        assert is_divisible(12, 3)
        assert is_divisible(100, 25)

    def test_not_divisible(self):
        assert not is_divisible(12, 5)
        assert not is_divisible(100, 7)

    def test_divide_by_zero(self):
        assert not is_divisible(12, 0)


class TestDigitSum:
    def test_single_digit(self):
        assert digit_sum(5) == 5

    def test_multi_digit(self):
        assert digit_sum(123) == 6
        assert digit_sum(999) == 27

    def test_negative(self):
        assert digit_sum(-123) == 6


class TestNumFactors:
    def test_prime(self):
        assert num_factors(7) == 2  # 1 and 7

    def test_composite(self):
        assert num_factors(12) == 6  # 1, 2, 3, 4, 6, 12

    def test_perfect_square(self):
        assert num_factors(36) == 9  # 1, 2, 3, 4, 6, 9, 12, 18, 36

    def test_one(self):
        assert num_factors(1) == 1

    def test_zero(self):
        assert num_factors(0) == 0


class TestIsPerfectNumber:
    def test_perfect_numbers(self):
        for n in [6, 28, 496, 8128]:
            assert is_perfect_number(n), f"{n} should be a perfect number"

    def test_not_perfect(self):
        for n in [1, 2, 3, 4, 5, 7, 10, 12, 100]:
            assert not is_perfect_number(n), f"{n} should not be a perfect number"


class TestFibonacciUpTo:
    def test_small(self):
        assert fibonacci_up_to(10) == [1, 1, 2, 3, 5, 8]

    def test_hundred(self):
        result = fibonacci_up_to(100)
        assert result[-1] == 89
        assert all(is_fibonacci(n) for n in result)


class TestPrimesUpTo:
    def test_small(self):
        assert primes_up_to(10) == [2, 3, 5, 7]

    def test_count(self):
        # There are 25 primes below 100
        assert len(primes_up_to(100)) == 25

    def test_below_two(self):
        assert primes_up_to(1) == []


# ===========================================================================
# Generator tests
# ===========================================================================


def _make_generator(**kwargs) -> MathematicsGenerator:
    defaults = {"seed": 42, "max_pairs": 200}
    defaults.update(kwargs)
    return MathematicsGenerator(**defaults)


class TestGeneratorContract:
    def test_name(self):
        gen = _make_generator()
        assert gen.name == "mathematics"

    def test_relation_types(self):
        gen = _make_generator()
        assert set(gen.relation_types()) == {
            "has_property", "greater_than", "is_divisible_by",
            "arithmetic_result", "more_factors", "shares_factor",
        }

    def test_domains(self):
        gen = _make_generator()
        assert gen.domains() == ["mathematics"]

    def test_yields_contrastive_pairs(self):
        gen = _make_generator(max_pairs=20)
        pairs = list(gen.generate())
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, ContrastivePair)
            assert p.generator == "mathematics"
            assert p.domain == "mathematics"


class TestPropertyMembership:
    def test_generates_pairs(self):
        gen = _make_generator()
        gen._build_pools()
        pairs = list(gen._generate_property())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "has_property"

    def test_true_and_false_differ(self):
        gen = _make_generator()
        gen._build_pools()
        for p in gen._generate_property():
            assert p.true_statement != p.false_statement

    def test_template_ids_start_with_math(self):
        gen = _make_generator()
        gen._build_pools()
        for p in gen._generate_property():
            assert p.template_id.startswith("math_property_")


class TestGreaterThan:
    def test_generates_pairs(self):
        gen = _make_generator()
        gen._build_pools()
        pairs = list(gen._generate_greater_than())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "greater_than"
            assert p.negation_strategy == "reverse_relation"

    def test_true_statement_correct(self):
        gen = _make_generator()
        gen._build_pools()
        for p in gen._generate_greater_than():
            # The true statement should have the bigger number first
            # Parse "X is greater than Y." or "X is larger than Y." or "X exceeds Y."
            stmt = p.true_statement
            if "greater than" in stmt:
                parts = stmt.replace(".", "").split(" is greater than ")
            elif "larger than" in stmt:
                parts = stmt.replace(".", "").split(" is larger than ")
            elif "exceeds" in stmt:
                parts = stmt.replace(".", "").split(" exceeds ")
            else:
                continue
            a, b = int(parts[0]), int(parts[1])
            assert a > b, f"True statement '{stmt}' is wrong: {a} <= {b}"


class TestDivisibility:
    def test_generates_pairs(self):
        gen = _make_generator()
        gen._build_pools()
        pairs = list(gen._generate_divisibility())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "is_divisible_by"

    def test_true_divisibility_correct(self):
        gen = _make_generator()
        gen._build_pools()
        for p in gen._generate_divisibility():
            # Check the "X is divisible by Y" template
            stmt = p.true_statement
            if "is divisible by" in stmt:
                parts = stmt.replace(".", "").split(" is divisible by ")
                a, b = int(parts[0]), int(parts[1])
                assert a % b == 0, f"True: '{stmt}' but {a} % {b} = {a % b}"
            elif "is a multiple of" in stmt:
                parts = stmt.replace(".", "").split(" is a multiple of ")
                a, b = int(parts[0]), int(parts[1])
                assert a % b == 0, f"True: '{stmt}' but {a} % {b} = {a % b}"


class TestArithmetic:
    def test_generates_pairs(self):
        gen = _make_generator()
        gen._build_pools()
        pairs = list(gen._generate_arithmetic())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "arithmetic_result"

    def test_true_arithmetic_correct(self):
        gen = _make_generator()
        gen._build_pools()
        for p in gen._generate_arithmetic():
            stmt = p.true_statement
            if " times " in stmt and " equals " in stmt:
                left, result = stmt.replace(".", "").split(" equals ")
                parts = left.split(" times ")
                a, b = int(parts[0]), int(parts[1])
                assert a * b == int(result), f"Wrong: {stmt}"
            elif " plus " in stmt and " equals " in stmt:
                left, result = stmt.replace(".", "").split(" equals ")
                parts = left.split(" plus ")
                a, b = int(parts[0]), int(parts[1])
                assert a + b == int(result), f"Wrong: {stmt}"
            elif " minus " in stmt and " equals " in stmt:
                left, result = stmt.replace(".", "").split(" equals ")
                parts = left.split(" minus ")
                a, b = int(parts[0]), int(parts[1])
                assert a - b == int(result), f"Wrong: {stmt}"
            elif " divided by " in stmt and " equals " in stmt:
                left, result = stmt.replace(".", "").split(" equals ")
                parts = left.split(" divided by ")
                a, b = int(parts[0]), int(parts[1])
                assert a // b == int(result), f"Wrong: {stmt}"

    def test_false_arithmetic_wrong(self):
        gen = _make_generator()
        gen._build_pools()
        for p in gen._generate_arithmetic():
            assert p.true_statement != p.false_statement


class TestMoreFactors:
    def test_generates_pairs(self):
        gen = _make_generator()
        gen._build_pools()
        pairs = list(gen._generate_more_factors())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "more_factors"

    def test_true_has_more_factors(self):
        gen = _make_generator()
        gen._build_pools()
        for p in gen._generate_more_factors():
            if "has more factors than" in p.true_statement:
                parts = p.true_statement.replace(".", "").split(" has more factors than ")
                a, b = int(parts[0]), int(parts[1])
                assert num_factors(a) > num_factors(b), (
                    f"{a} has {num_factors(a)} factors, {b} has {num_factors(b)}"
                )
            elif "has more divisors than" in p.true_statement:
                parts = p.true_statement.replace(".", "").split(" has more divisors than ")
                a, b = int(parts[0]), int(parts[1])
                assert num_factors(a) > num_factors(b)


class TestSharesFactor:
    def test_generates_pairs(self):
        gen = _make_generator()
        gen._build_pools()
        pairs = list(gen._generate_shares_factor())
        assert len(pairs) > 0
        for p in pairs:
            assert p.relation_type == "shares_factor"

    def test_coprime_correctness(self):
        gen = _make_generator()
        gen._build_pools()
        for p in gen._generate_shares_factor():
            stmt = p.true_statement
            if "are coprime" in stmt:
                # Extract numbers
                parts = stmt.replace(".", "").split(" and ")
                a = int(parts[0])
                b = int(parts[1].replace(" are coprime", ""))
                assert gcd(a, b) == 1, f"{a} and {b} are not coprime (gcd={gcd(a, b)})"
            elif "share a common factor" in stmt:
                parts = stmt.replace(".", "").split(" and ")
                a = int(parts[0])
                b = int(parts[1].replace(" share a common factor", ""))
                assert gcd(a, b) > 1, f"{a} and {b} don't share a factor (gcd={gcd(a, b)})"


class TestDeterminism:
    def test_same_seed_same_output(self):
        gen1 = _make_generator(max_pairs=50)
        gen2 = _make_generator(max_pairs=50)
        pairs1 = list(gen1.generate())
        pairs2 = list(gen2.generate())
        assert len(pairs1) == len(pairs2)
        for p1, p2 in zip(pairs1, pairs2):
            assert p1.pair_id == p2.pair_id
            assert p1.true_statement == p2.true_statement
            assert p1.false_statement == p2.false_statement

    def test_different_seed_different_output(self):
        gen1 = _make_generator(max_pairs=50)
        gen2 = MathematicsGenerator(seed=99, max_pairs=50)
        pairs1 = list(gen1.generate())
        pairs2 = list(gen2.generate())
        ids1 = {p.pair_id for p in pairs1}
        ids2 = {p.pair_id for p in pairs2}
        assert ids1 != ids2


class TestTemplateDiversity:
    def test_multiple_templates_used(self):
        gen = _make_generator(max_pairs=2000)
        pairs = list(gen.generate())
        template_ids = {p.template_id for p in pairs}
        assert len(template_ids) >= 5

    def test_template_ids_are_stable(self):
        gen = _make_generator()
        pairs = list(gen.generate())
        for p in pairs:
            assert p.template_id.startswith("math_")


class TestDifficultyTiers:
    def test_all_difficulties_present(self):
        gen = _make_generator(max_pairs=500)
        pairs = list(gen.generate())
        for p in pairs:
            assert p.difficulty == "mixed"
        swap_dists = {p.gen_params.get("swap_distance") for p in pairs if p.gen_params}
        valid = {Difficulty.HARD.value, Difficulty.MEDIUM.value, Difficulty.EASY.value}
        assert swap_dists.issubset(valid)
        # Should have at least 2 swap distance levels
        assert len(swap_dists) >= 2


class TestMaxPairs:
    def test_max_pairs_respected(self):
        gen = MathematicsGenerator(seed=42, max_pairs=10)
        pairs = list(gen.generate())
        assert len(pairs) <= 10

    def test_max_pairs_none(self):
        gen = MathematicsGenerator(seed=42, max_pairs=None,
                                   property_range=(2, 50),
                                   comparison_range=(2, 50))
        pairs = list(gen.generate())
        assert len(pairs) > 10  # Should produce many pairs


class TestPairIds:
    def test_unique(self):
        gen = _make_generator(max_pairs=200)
        pairs = list(gen.generate())
        ids = [p.pair_id for p in pairs]
        assert len(ids) == len(set(ids)), "Duplicate pair IDs found"


class TestToRows:
    def test_to_rows_schema(self):
        gen = _make_generator(max_pairs=5)
        pairs = list(gen.generate())
        assert len(pairs) > 0
        rows = pairs[0].to_rows()
        assert len(rows) == 2
        true_row, false_row = rows
        assert true_row["label"] is True
        assert false_row["label"] is False
        assert true_row["pair_id"] == false_row["pair_id"]
        assert true_row["generator"] == "mathematics"
