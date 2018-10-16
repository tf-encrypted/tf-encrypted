from functools import reduce
from math import log
from typing import Tuple


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def gcd(a: int, b: int) -> int:
    g, _, _ = egcd(a, b)
    return g


def inverse(a: int, m: int) -> int:
    g, b, _ = egcd(a, m)
    assert g == 1
    return b % m


log2 = lambda x: log(x) / log(2)
prod = lambda xs: reduce(lambda x, y: x * y, xs)
