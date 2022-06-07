"""
The following Python implementation of Shamir's Secret Sharing is
released into the Public Domain under the terms of CC0 and OWFa:
https://creativecommons.org/publicdomain/zero/1.0/
http://www.openwebfoundation.org/legal/the-owf-1-0-agreements/owfa-1-0
"""

from __future__ import division
from __future__ import print_function

import random
import functools

_R_INT = functools.partial(random.SystemRandom().randint, 0)


def _extended_gcd(a, b):
    """
    Division in integers modulus p means finding the inverse of the
    denominator modulo p and then multiplying the numerator by this
    inverse (Note: inverse of A is B such that A*B % p == 1) this can
    be computed via extended Euclidean algorithm
    http://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Computation
    """
    x = 0
    last_x = 1
    y = 1
    last_y = 0
    while b != 0:
        quot = a // b
        a, b = b, a % b
        x, last_x = last_x - quot * x, x
        y, last_y = last_y - quot * y, y
    return last_x, last_y


def _div_mod(num, den, p):
    """Compute num / den modulo prime p

    To explain what this means, the return value will be such that
    the following is true: den * _divmod(num, den, p) % p == num
    """
    inv, _ = _extended_gcd(den, p)
    return num * inv


def _lagrange_interpolate(x, x_s, y_s, p):
    """
    Find the y-value for the given x, given n (x, y) points;
    k points will define a polynomial of up to kth order.
    """
    k = len(x_s)
    assert k == len(set(x_s)), "points must be distinct"

    def PI(vals):  # upper-case PI -- product of inputs
        accum = 1
        for v in vals:
            accum *= v
        return accum

    nums = []  # avoid inexact division
    dens = []
    for i in range(k):
        others = list(x_s)
        cur = others.pop(i)
        nums.append(PI(x - o for o in others))
        dens.append(PI(cur - o for o in others))
    den = PI(dens)
    num = sum([_div_mod(nums[i] * den * y_s[i] % p, dens[i], p)
               for i in range(k)])
    return (_div_mod(num, den, p) + p) % p


class ShamirSecretSharing:

    def __init__(self, m, n, p=2 ** 521 - 1):
        """

        Args:
            m: minimum of shares to recover the secure
            n: total number of shares
            p: a large prime number. Default to be 2**127-1
                other choices: 2**127-1, 2**521-1
        """
        self._m = m
        self._n = n
        self._p = p

    def _eval_at(self, poly, x):
        """Evaluates polynomial (coefficient tuple) at x, used to generate a
        shamir pool in make_random_shares below.
        """
        accum = 0
        for coefficient in reversed(poly):
            accum *= x
            accum += coefficient
            accum %= self._p
        return accum

    def share(self, secret):
        """
        Generates a random shamir pool for a given secret, returns share points.
        """
        poly = [secret] + [_R_INT(self._p - 1) for i in range(self._m - 1)]
        points = [(i, self._eval_at(poly, i))for i in range(1, self._n + 1)]
        return points

    def recon(self, shares):
        """
        Recover the secret from share points
        (x, y points on the polynomial).
        """
        if len(shares) < self._m:
            raise ValueError(f"need at least {self._m} shares")
        x_s, y_s = zip(*shares)
        return _lagrange_interpolate(0, x_s, y_s, self._p)
