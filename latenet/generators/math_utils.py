"""Pure-computation math utilities for the mathematics generator.

No external dependencies — standard library only.
"""

from __future__ import annotations

import math


def is_prime(n: int) -> bool:
    """Test primality by trial division."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def prime_factors(n: int) -> list[int]:
    """Return the prime factorization of n as a sorted list (with repeats)."""
    if n < 2:
        return []
    factors: list[int] = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def is_perfect_square(n: int) -> bool:
    """Check if n is a perfect square."""
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def is_perfect_cube(n: int) -> bool:
    """Check if n is a perfect cube."""
    if n < 0:
        return False
    root = round(n ** (1 / 3))
    # Check root and neighbors to handle float imprecision
    for candidate in (root - 1, root, root + 1):
        if candidate >= 0 and candidate ** 3 == n:
            return True
    return False


def is_fibonacci(n: int) -> bool:
    """Check if n is a Fibonacci number.

    A number is Fibonacci iff one of (5*n^2 + 4) or (5*n^2 - 4)
    is a perfect square.
    """
    if n < 0:
        return False
    x = 5 * n * n
    return is_perfect_square(x + 4) or is_perfect_square(x - 4)


def is_power_of_two(n: int) -> bool:
    """Check if n is a power of two (n >= 1)."""
    return n >= 1 and (n & (n - 1)) == 0


def gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def is_divisible(a: int, b: int) -> bool:
    """Check if a is divisible by b."""
    if b == 0:
        return False
    return a % b == 0


def digit_sum(n: int) -> int:
    """Sum of the digits of |n|."""
    return sum(int(d) for d in str(abs(n)))


def num_factors(n: int) -> int:
    """Count the number of positive divisors of n."""
    if n < 1:
        return 0
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count


def is_perfect_number(n: int) -> bool:
    """Check if n is a perfect number (sum of proper divisors equals n)."""
    if n < 2:
        return False
    total = 1
    i = 2
    while i * i <= n:
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
        i += 1
    return total == n


def fibonacci_up_to(limit: int) -> list[int]:
    """Return all Fibonacci numbers up to limit."""
    fibs = []
    a, b = 0, 1
    while a <= limit:
        if a > 0:
            fibs.append(a)
        a, b = b, a + b
    return fibs


def primes_up_to(limit: int) -> list[int]:
    """Sieve of Eratosthenes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, math.isqrt(limit) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]
