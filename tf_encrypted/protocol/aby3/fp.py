# This file include the  floating-point operations
from math import ceil
from math import log2

import numpy as np
import tensorflow as tf

from ..protocol import TFEPrivateTensor
from ..protocol import TFEPublicTensor


def _fp_div(prot, a: "TFEPrivateTensor", b: "TFEPrivateTensor", nonsigned: bool):
    with tf.name_scope("fp_div"):
        return a * _fp_recip_private(prot, b, nonsigned)


def _fp_div_private_private(
    prot, a: "TFEPrivateTensor", b: "TFEPrivateTensor", nonsigned: bool
):
    return _fp_div(prot, a, b, nonsigned)


def _fp_div_public_private(
    prot, a: "TFEPublicTensor", b: "TFEPrivateTensor", nonsigned: bool
):
    return _fp_div(prot, a, b, nonsigned)


def _fp_div_private_public(
    prot, a: "TFEPrivateTensor", b: "TFEPublicTensor", nonsigned: bool
):
    return a / b


def _fp_div_public_public(
    prot, a: "TFEPublicTensor", b: "TFEPublicTensor", nonsigned: bool
):
    return a / b


def _fp_sqrt2(prot, a: "TFEPrivateTensor"):
    c15 = prot.define_constant(1.5)
    c05 = prot.define_constant(0.5)

    y = approx_sqrt_inv(prot, a)  # y approixmates 1 / sqrt(a)
    g = a * y
    h = y * c05
    for _ in range(1):
        """
        Over iterations,
            g -> sqrt(b)
            h -> sqrt(1 / b) * 0.5
        1 iteration should give less than 10^{-4} relative error
        """
        r = c15 - g * h  # r = 1 + error
        g = g * r
        h = h * r
    return g, h


def _fp_recip_private(prot, x: "TFEPrivateTensor", nonsigned):
    """
    Approxiamtedly compute 1/x from x.

    Apply the quintic iteration from
    http://numbers.computation.free.fr/Constants/Algorithms/inverse.html
    """
    with tf.name_scope("fp_reciprocal"):
        sgf, exp = __fp_normalize(
            prot, x, nonsigned
        )  # x = sgf / exp, then 1 / x = 1/sgf * exp
        one = prot.define_constant(1.0)
        two = prot.define_constant(2, apply_scaling=False)
        # 2.9281928 - 2 * s approxiates 1/s with small relative errors
        # in the interval s \in [0.5, 1.)
        # By using integer factor '2', we can save one truncation.
        inv_sgf = 2.9281928 - two * sgf
        appr_recip = inv_sgf * exp  # ~ 1/x
        for _ in range(1):
            """One iteration should give us very good approximation
            (10^{-5} relative error ratio).
            More iterations, more precision."""
            res = one - x * appr_recip
            res2 = res * res
            appr_recip = appr_recip + appr_recip * (one + res2) * (
                res + res2
            )  # quintic iteration
        return appr_recip


def _fp_inv_sqrt_private(prot, a: "TFEPrivateTensor"):
    # low precision
    return approx_sqrt_inv(prot, a)
    # high precision with extra 3 rounds of communication
    # two = prot.define_constant(2, apply_scaling=False)
    # _, h = _fp_sqrt2(prot, a)
    # return h * two


def _fp_sqrt_private(prot, a: "TFEPrivateTensor"):
    g, _ = _fp_sqrt2(prot, a)
    return g


def prefix_ORs(b: "TFEPrivateTensor", k: int):
    r"""
    b := (..., b3, b2, b1, b0) where b0 is the least significant bit
    compute y := (y_{k-1}, y_{k-1}, ..., y0) where y_i \in {0, 1}
         y_i = bit_or_{i <= j < k} bj
         The first yi = 1 is the first significant bit that bi = 1.

    """
    # running ORs from MSB to LSB.
    n, e = int(ceil(log2(k))), 1
    with tf.name_scope("prefix_ORs"):
        for i in range(0, n):
            b = b | (b >> e)
            e = e << 1
    return b


def _do_fp_log_private(prot, x: "TFEPrivateTensor", base: "float"):
    k = prot.fixedpoint_config.precision_fractional
    m = k + prot.fixedpoint_config.precision_integral
    n = prot.int_factory.nbits
    logn = (
        int(log2(n) + 1) * 2
    )  # enough bit length to represent the exponent \in [0, 128)
    assert x.is_scaled, "Er.. tricky here."
    assert (
        2 * k > m
    ), "We assume 2^{-j} can be represent with 2k-bit precisions for all j in [0, m)"
    assert base >= 2.0, "log(x, base) shoule with base >= 2."
    adjust = 1.0 / np.log2(base)
    with tf.name_scope("fp_log"):
        # bit-decomposition. Make sure the higher bits are all 0 via shifting.
        x_bits = (prot.a2b(x, m) << (n - m)) >> (n - m)
        y_bits = prefix_ORs(x_bits, m)
        z_bits = y_bits ^ (y_bits >> 1)
        rev_z_bits = prot.bit_reverse(z_bits)

        exponent = prot.b2a(rev_z_bits >> (n - 2 * k - 1), 2 * k)  # NOTE: shift 1-less.
        exponent.is_scaled = True

        log_exponent = (
            prot.b2a(prot.xor_indices(z_bits), logn) * 2**k
        )  # j + k with k-bit precision
        log_exponent.is_scaled = True
        log_exponent = log_exponent - k
        if base != 2.0:
            log_exponent = log_exponent * adjust

        frac = x * exponent  # frac is in the interval [1., 2.]
        # The approximation coefficients are for log2(x), we need to
        # adjust them via multiplying 1./log2(base)
        """ """
        log_frac = ((-0.4326728 * adjust) * frac + (2.276597 * adjust)) * frac + (
            -1.843924 * adjust
        )
        return log_frac + log_exponent


def _fp_log2_private(prot, x: "TFEPrivateTensor"):
    return _do_fp_log_private(prot, x, 2.0)


def _fp_log10_private(prot, x: "TFEPrivateTensor"):
    return _do_fp_log_private(prot, x, 10.0)


def _fp_ln_private(prot, x: "TFEPrivateTensor"):
    return _do_fp_log_private(prot, x, np.e)


def __fp_normalize(prot, b: "TFEPrivateTensor", nonsigned=False):
    r"""
    Given [b], to compute [sgf], and [exp] such that b = sgf / exp
    where sgf \in [0.5, 1)
    """
    k = prot.fixedpoint_config.precision_fractional
    m = k + prot.fixedpoint_config.precision_integral
    n = b.backing_dtype.nbits
    assert b.is_scaled, "Er.. tricky here."
    assert (
        2 * k > m
    ), "We assume 2^{-j} can be represent with 2k-bit precisions for all j in [0, m)"
    with tf.name_scope("fp_normalize"):
        # juhou: sign.is_scaled is False
        if not nonsigned:
            msb = prot.bit_extract(b, m)
            two = np.ones(shape=msb.shape, dtype=int) * 2
            sign = 1 - prot.mul_ab(prot.define_constant(two), msb)
            x = sign * b  # abs(b)
        else:
            sign, x = 1, b

        # bit-decomposition. Make sure the higher bits are all 0 via shifting.
        x_bits = (prot.a2b(x, m) << (n - m)) >> (n - m)
        """
           y_j = 1 <-> for all i <= j, y_i = 1.
           z_j = 1 <-> 2^j < x = 2^j + x0 < 2^{j+1}.
           And there is only one z_j = 1, and others are all 0.
        """
        y_bits = prefix_ORs(x_bits, m)
        z = y_bits ^ (y_bits >> 1)

        r"""
            There exists one z_j = 1 and other z_i = 0.
            As as result, \sum_i 2^i * z_i = 2^j (i.e., b2a(z))
            If we reverse the bits {z_i} to obtain {z'_i} for z'_i = z_{128 - i}.
            Then \sum_i 2^i * z'_i = 2^{128 - j}.
            Also, if we right shift {z'_i} by (128 - m)-steps.
            Then we obtain 2^{m - j}, which equals to the fixed-point
            representation of 2^{-j} within m-bits precision.
        """
        # NOTE: We couldn't obtain 2^{-j} with only k-bits precision
        # because j > k might be possible (i.e., b = 2^k * b' for
        # real value b' > 1).
        # Instead, we preserve 2k-bit precision, i.e., c = 2^{-j} with 2k bits precision
        _exp = prot.b2a(prot.bit_reverse(z) >> (n - 2 * k), 2 * k)
        _exp.is_scaled = True
        sgf = x * _exp  # significant should in [0.5, 1). Thus, sgf should be positive.
        exp = _exp if nonsigned else sign * _exp
    return sgf, exp


def approx_sqrt_inv(prot, x: "TFEPrivateTensor"):
    """
    From x, to compute an approximation of 1/sqrt(x).
    """

    def select(x, y, bit):
        """
        return x if bit = 0 else y.
        """
        c = np.ones(shape=bit.shape) * (x - y)
        return prot.mul_ab(prot.define_constant(c, apply_scaling=False), bit) + y

    k = prot.fixedpoint_config.precision_fractional
    f = prot.fixedpoint_config.precision_integral
    m = k + f
    n = x.backing_dtype.nbits
    assert x.is_scaled, "Er.. tricky here."
    assert (
        2 * k > m
    ), "We assume 2^{-j} can be represent with 2k-bit precisions for all j in [0, m)"
    with tf.name_scope("normalize_for_sqrt"):
        # bit-decomposition. Make sure the higher bits are all 0 via shifting.
        x_bits = (prot.a2b(x, m) << (n - m)) >> (n - m)
        y_bits = prefix_ORs(x_bits, m)
        z_bits = (
            y_bits ^ (y_bits >> 1)
        ) << 1  # note: x = c * 2^m where c \in [0.25, 0.5)
        rev_z_bits = prot.bit_reverse(z_bits) >> (n - 2 * k)

        frac = prot.b2a(rev_z_bits, 2 * k)
        frac.is_scaled = True
        normalized = frac * x  # normalized \in [0.25, 0.5)
        """
        f(b) = 4.7979 * b^2 - 5.9417 * b + 3.1855 approixmates 1/sqrt(b) in [0.25, 0.5)
        with less than 0.7% relative error
        """
        sqrt_inv = ((4.7979 * normalized) - 5.9417) * normalized + 3.1855
        """
            Indeed, the exponetent part is 2^{j+k} where k is the scaling factor.
            We want to compute sqrt(2^{-j}) with k-bit precision,
            i.e., sqrt(2^{-j}) * 2^k.
            In other words, we compute sqrt(2^{-j}) * 2^k from 2^{j+k}.

            1. We first obtain 2^{-(j+k)} from 2^{j + k}.
            2. Then we compute 2^{floor(-(j+k)/2)}. Rewrite it as
               2^{floor(-(j+k)/2)} = c * 2^{floor(-j/2)} * 2^{floor(-k/2)}
               where c depends on the parity of j, and k.
            3. We compute the parity of j + k, i.e., check the LSB of j + k.
            4. Suppose k is even, 2^{floor(-k/2)} = 2^{-k/2}.
               Then we can cancel this term via 2^{k/2}.
               If lsb(j + k) = 0 <-> j is even. In this case,
               2^{floor(-j/2)} = 2^{-j/2} = sqrt(2^{-j}).
               If lsb(j + k) = 1 <-> j is odd. Then
               2^{floor(-j/2)} * 2^{-1} = sqrt(2^{-j}).

               Suppose k is odd: We need 2^{k//2} * 2 to cancel 2^{floor(-k/2)}.
               If lsb(j + k) = 0 <-> j is odd. In this case,
               2^{floor(-j/2)} * 2^{-1} = sqrt(2^{-j}).
               If lsb(j + k) = 1 <-> j is odd. Then
               2^{floor(-j/2)} = 2^{-j/2} = sqrt(2^{-j}).
        """
        j_add_k = prot.xor_indices(z_bits)  # j + k
        lsb = prot.bit_extract(j_add_k, 0)  # lsb = 0 <-> j + k is even
        exponet = prot.b2a(
            prot.bit_gather(rev_z_bits | rev_z_bits >> 1, 0, 2), k
        )  # 2^{floor(-(j+k)/2)}
        exponet.is_scaled = False  # Stop truncation
        if k & 1 == 0:  # k is even which means lsb = 1 <=> j is odd
            exponet = exponet * select(2 ** (k // 2), 2 ** (k // 2) * np.sqrt(2.0), lsb)
        else:  # k is odd which means lsb = 1 <=> j is even
            exponet = exponet * select(
                2 ** (k // 2) * np.sqrt(2.0), 2 ** (k // 2 + 1), lsb
            )
        exponet.is_scaled = True  # 2^{-j/2} with k-bit precision

    return sqrt_inv * exponet
