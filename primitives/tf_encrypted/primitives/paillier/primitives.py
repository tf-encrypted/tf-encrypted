from typing import Optional

import numpy as np
import tensorflow as tf

import tf_big

tf_big.set_secure_default(True)


class EncryptionKey:
    def __init__(self, n):
        n = tf_big.convert_to_tensor(n)

        self.n = n
        self.nn = n * n
        self.g = 1 + n

    def convert_to_tensor(self):
        if not isinstance(self.n, tf_big.Tensor):
            self.n = tf_big.convert_to_tensor(self.n)
        if not isinstance(self.nn, tf_big.Tensor):
            self.nn = tf_big.convert_to_tensor(self.nn)
        if not isinstance(self.g, tf_big.Tensor):
            self.g = tf_big.convert_to_tensor(self.g)

    def convert_from_tensor(self, dtype=tf.variant):
        if dtype != tf.variant:
            self.n = tf_big.convert_from_tensor(self.n, dtype)
            self.nn = tf_big.convert_from_tensor(self.nn, dtype)
            self.g = tf_big.convert_from_tensor(self.g, dtype)

class DecryptionKey:
    def __init__(self, p, q):
        p = tf_big.convert_to_tensor(p)
        q = tf_big.convert_to_tensor(q)

        n = p * q
        self.n = n
        self.nn = n * n
        self.g = 1 + n

        order_of_n = (p - 1) * (q - 1)
        self.d1 = order_of_n
        self.d2 = tf_big.inv(order_of_n, n)
        self.e = tf_big.inv(n, order_of_n)

    def convert_to_tensor(self):
        if not isinstance(self.n, tf_big.Tensor):
            self.n = tf_big.convert_to_tensor(self.n)
        if not isinstance(self.nn, tf_big.Tensor):
            self.nn = tf_big.convert_to_tensor(self.nn)
        if not isinstance(self.g, tf_big.Tensor):
            self.g = tf_big.convert_to_tensor(self.g)
        if not isinstance(self.n, tf_big.Tensor):
            self.d1 = tf_big.convert_to_tensor(self.d1)
        if not isinstance(self.d2, tf_big.Tensor):
            self.d2 = tf_big.convert_to_tensor(self.d2)
        if not isinstance(self.e, tf_big.Tensor):
            self.e = tf_big.convert_to_tensor(self.e)

    def convert_from_tensor(self, dtype=tf.variant):
        if dtype != tf.variant:
            self.n = tf_big.convert_from_tensor(self.n, dtype)
            self.nn = tf_big.convert_from_tensor(self.nn, dtype)
            self.g = tf_big.convert_from_tensor(self.g, dtype)
            self.d1 = tf_big.convert_from_tensor(self.d1, dtype)
            self.d2 = tf_big.convert_from_tensor(self.d2, dtype)
            self.e = tf_big.convert_from_tensor(self.e, dtype)


def gen_keypair(bitlength=2048, dtype=tf.variant):
    p, q, n = tf_big.random_rsa_modulus(bitlength=bitlength)
    ek = EncryptionKey(n)
    dk = DecryptionKey(p, q)
    ek.convert_from_tensor(dtype=dtype)
    dk.convert_from_tensor(dtype=dtype)
    return ek, dk


class Randomness:
    def __init__(self, raw_randomness):
        self.raw = tf_big.convert_to_tensor(raw_randomness)


def gen_randomness(ek, shape, dtype: tf.DType = tf.uint8):
    return Randomness(tf_big.random_uniform(shape=shape, maxval=ek.n))


class Ciphertext:
    def __init__(self, ek: EncryptionKey, raw_ciphertext):
        self.ek = ek
        self.raw = raw_ciphertext

    def __add__(self, other):
        assert self.ek == other.ek
        return add(self.ek, self, other)


def encrypt(
    ek,
    plaintext: tf.Tensor,
    randomness: Optional[Randomness] = None,
    dtype: tf.DType = tf.variant,
):
    x = tf_big.convert_to_tensor(plaintext)
    ek.convert_to_tensor()

    randomness = randomness or gen_randomness(ek=ek, shape=x.shape, dtype=tf.variant)
    r = randomness.raw
    assert r.shape == x.shape

    gx = tf_big.pow(ek.g, x, ek.nn)
    rn = tf_big.pow(r, ek.n, ek.nn)
    c = gx * rn % ek.nn

    if dtype != tf.variant:
        c = tf_big.convert_from_tensor(c, dtype=dtype)

    return Ciphertext(ek, c)


def decrypt(dk: DecryptionKey, ciphertext: Ciphertext, dtype: tf.variant):
    c = tf_big.convert_to_tensor(ciphertext.raw)
    dk.convert_to_tensor()

    gxd = tf_big.pow(c, dk.d1, dk.nn)
    xd = (gxd - 1) // dk.n
    x = (xd * dk.d2) % dk.n

    assert x._raw.dtype == tf.variant, (c.dtype, x.dtype)
    if dtype == tf.variant:
        return x
    return tf_big.convert_from_tensor(x, dtype=dtype)


def refresh(
    ek: EncryptionKey, ciphertext: Ciphertext, dtype: Optional[tf.DType] = None
):
    dtype = dtype or ciphertext.raw.dtype
    c = ciphertext.raw
    s = gen_randomness(ek=ek, shape=c.shape, dtype=tf.variant).raw
    sn = tf_big.pow(s, ek.n, ek.nn)
    d = (c * sn) % ek.nn
    return Ciphertext(ek, d)


def add(
    ek: EncryptionKey,
    lhs: Ciphertext,
    rhs: Ciphertext,
    do_refresh: bool = True,
    dtype: Optional[tf.DType] = None,
):
    dtype = dtype or lhs.raw.dtype or rhs.raw.dtype
    c0 = tf_big.convert_to_tensor(lhs.raw)
    c1 = tf_big.convert_to_tensor(rhs.raw)
    c = (c0 * c1) % ek.nn
    c = Ciphertext(ek, c)
    return refresh(ek, c, dtype=dtype) if do_refresh else c


def mul(
    ek: EncryptionKey,
    lhs: Ciphertext,
    rhs: tf.Tensor,
    do_refresh: bool = True,
    dtype: Optional[tf.DType] = None,
):
    dtype = dtype or lhs.raw.dtype
    c = lhs.raw
    k = tf_big.convert_to_tensor(rhs)
    d = tf_big.pow(c, k) % ek.nn
    d = Ciphertext(ek, d)
    return refresh(ek, d, dtype=dtype) if do_refresh else d
