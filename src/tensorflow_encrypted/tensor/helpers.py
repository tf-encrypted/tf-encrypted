from functools import reduce
from math import log

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def gcd(a, b):
    g, _, _ = egcd(a, b)
    return g

def inverse(a, m):
    _, b, _ = egcd(a, m)
    return b % m

log2 = lambda x: log(x)/log(2)

prod = lambda xs: reduce(lambda x,y: x*y, xs)
