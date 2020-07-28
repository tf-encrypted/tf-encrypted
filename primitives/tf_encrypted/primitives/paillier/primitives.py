from typing import Optional

import tensorflow as tf
import tf_big

tf_big.set_secure_default(True)


class EncryptionKey:
    """Paillier encryption key.

    Note that the generator `g` has been fixed to `1 + n`.
    """

    def __init__(self, n, bitlength=None):
        limb_format = bitlength is not None
        n = tf_big.convert_to_tensor(n, limb_format=limb_format, bitlength=bitlength)

        self.n = n
        self.nn = n * n

    def export(self, dtype: tf.DType = tf.string, bitlength: Optional[int] = None):
        limb_format = bitlength is not None
        return tf_big.convert_from_tensor(
            self.n, dtype=dtype, limb_format=limb_format, bitlength=bitlength
        )


class DecryptionKey:
    def __init__(self, p, q, bitlength=None):
        limb_format = bitlength is not None
        self.p = tf_big.convert_to_tensor(
            p, limb_format=limb_format, bitlength=bitlength
        )
        self.q = tf_big.convert_to_tensor(
            q, limb_format=limb_format, bitlength=bitlength
        )

        self.n = self.p * self.q
        self.nn = self.n * self.n

        order_of_n = (self.p - 1) * (self.q - 1)
        self.d1 = order_of_n
        self.d2 = tf_big.inv(order_of_n, self.n)
        self.e = tf_big.inv(self.n, order_of_n)

    def export(self, dtype: tf.DType = tf.string, bitlength: Optional[int] = None):
        limb_format = bitlength is not None
        return (
            tf_big.convert_from_tensor(
                self.p, dtype=dtype, limb_format=limb_format, bitlength=bitlength
            ),
            tf_big.convert_from_tensor(
                self.q, dtype=dtype, limb_format=limb_format, bitlength=bitlength
            ),
        )


def gen_keypair(bitlength=2048):
    p, q, n = tf_big.random_rsa_modulus(bitlength=bitlength)
    ek = EncryptionKey(n)
    dk = DecryptionKey(p, q)
    return ek, dk


class Randomness:
    def __init__(self, raw_randomness, bitlength=None):
        limb_format = bitlength is not None
        self.raw = tf_big.convert_to_tensor(
            raw_randomness, limb_format=limb_format, bitlength=bitlength
        )

    def export(self, dtype: tf.DType = tf.string, bitlength: Optional[int] = None):
        limb_format = bitlength is not None
        return tf_big.convert_from_tensor(
            self.raw, dtype=dtype, limb_format=limb_format, bitlength=bitlength
        )


def gen_randomness(ek, shape):
    return Randomness(tf_big.random_uniform(shape=shape, maxval=ek.n))


class Ciphertext:
    def __init__(self, ek: EncryptionKey, raw_ciphertext, bitlength=None):
        limb_format = bitlength is not None
        self.ek = ek
        self.raw = tf_big.convert_to_tensor(
            raw_ciphertext, limb_format=limb_format, bitlength=bitlength
        )

    def export(self, dtype: tf.DType = tf.string, bitlength: Optional[int] = None):
        limb_format = bitlength is not None
        return tf_big.convert_from_tensor(
            self.raw, dtype=dtype, limb_format=limb_format, bitlength=bitlength
        )

    def __add__(self, other):
        assert self.ek == other.ek
        return add(self.ek, self, other)


def encrypt(
    ek: EncryptionKey, plaintext: tf.Tensor, randomness: Optional[Randomness] = None,
):
    x = tf_big.convert_to_tensor(plaintext)

    randomness = randomness or gen_randomness(ek=ek, shape=x.shape)
    r = randomness.raw
    assert r.shape == x.shape

    gx = 1 + (ek.n * x) % ek.nn
    rn = tf_big.pow(r, ek.n, ek.nn)
    c = gx * rn % ek.nn
    return Ciphertext(ek, c)


def decrypt(
    dk: DecryptionKey,
    ciphertext: Ciphertext,
    dtype: tf.DType = tf.int32,
    bitlength: Optional[int] = None,
):
    limb_format = bitlength is not None
    c = ciphertext.raw

    gxd = tf_big.pow(c, dk.d1, dk.nn)
    xd = (gxd - 1) // dk.n
    x = (xd * dk.d2) % dk.n

    if dtype == tf.variant:
        return x

    return tf_big.convert_from_tensor(
        x, dtype=dtype, limb_format=limb_format, bitlength=bitlength,
    )


def refresh(ek: EncryptionKey, ciphertext: Ciphertext):
    c = ciphertext.raw
    s = gen_randomness(ek=ek, shape=c.shape).raw
    sn = tf_big.pow(s, ek.n, ek.nn)
    d = (c * sn) % ek.nn
    return Ciphertext(ek, d)


def add(
    ek: EncryptionKey, lhs: Ciphertext, rhs: Ciphertext, do_refresh: bool = True,
):
    c0 = tf_big.convert_to_tensor(lhs.raw)
    c1 = tf_big.convert_to_tensor(rhs.raw)
    c = (c0 * c1) % ek.nn
    res = Ciphertext(ek, c)

    if not do_refresh:
        return res
    return refresh(ek, res)


def mul(
    ek: EncryptionKey, lhs: Ciphertext, rhs: tf.Tensor, do_refresh: bool = True,
):
    c = lhs.raw
    k = tf_big.convert_to_tensor(rhs)
    d = tf_big.pow(c, k) % ek.nn
    res = Ciphertext(ek, d)

    if not do_refresh:
        return res
    return refresh(ek, res)
