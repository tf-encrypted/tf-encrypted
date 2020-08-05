from typing import Optional
from typing import Tuple

import tensorflow as tf
import tf_big

tf_big.set_secure_default(True)


def _import_maybe_limbs(tensor):
    if isinstance(tensor, tf_big.Tensor):
        return tensor
    if isinstance(tensor, tf.Tensor):
        if tensor.dtype == tf.string:
            return tf_big.import_tensor(tensor)
        return tf_big.import_limbs_tensor(tensor)
    raise ValueError("Don't know how to import tensors of type {}".format(type(tensor)))


def _export_maybe_limbs(tensor, dtype):
    assert isinstance(tensor, tf_big.Tensor), type(tensor)
    if dtype == tf.string:
        return tf_big.export_tensor(tensor, dtype=dtype)
    return tf_big.export_limbs_tensor(tensor, dtype=dtype)


class EncryptionKey:
    """Paillier encryption key.

    Note that the generator `g` has been fixed to `1 + n`.
    """

    def __init__(self, n: tf.Tensor):
        n = _import_maybe_limbs(n)

        self.n = n
        self.nn = n * n

    def export(self, dtype: tf.DType = tf.uint8) -> tf.Tensor:
        return _export_maybe_limbs(self.n, dtype)


class DecryptionKey:
    def __init__(self, p: tf.Tensor, q: tf.Tensor):
        self.p = _import_maybe_limbs(p)
        self.q = _import_maybe_limbs(q)

        self.n = self.p * self.q
        self.nn = self.n * self.n

        order_of_n = (self.p - 1) * (self.q - 1)
        self.d1 = order_of_n
        self.d2 = tf_big.inv(order_of_n, self.n)
        self.e = tf_big.inv(self.n, order_of_n)

    def export(self, dtype: tf.DType = tf.uint8) -> Tuple[tf.Tensor, tf.Tensor]:
        return (
            _export_maybe_limbs(self.p, dtype),
            _export_maybe_limbs(self.q, dtype),
        )


def gen_keypair(bitlength=2048) -> Tuple[EncryptionKey, DecryptionKey]:
    p, q, n = tf_big.random_rsa_modulus(bitlength=bitlength)
    ek = EncryptionKey(n)
    dk = DecryptionKey(p, q)
    return ek, dk


class Randomness:
    def __init__(self, raw_randomness: tf.Tensor):
        self.raw = _import_maybe_limbs(raw_randomness)

    def export(self, dtype: tf.DType = tf.uint8) -> tf.Tensor:
        return _export_maybe_limbs(self.raw, dtype=dtype)


def gen_randomness(ek: EncryptionKey, shape) -> Randomness:
    return Randomness(tf_big.random_uniform(shape=shape, maxval=ek.n))


class Ciphertext:
    def __init__(self, ek: EncryptionKey, raw_ciphertext: tf.Tensor):
        self.ek = ek
        self.raw = _import_maybe_limbs(raw_ciphertext)

    def export(self, dtype: tf.DType = tf.uint8) -> tf.Tensor:
        return _export_maybe_limbs(self.raw, dtype=dtype)

    def __add__(self, other):
        assert self.ek == other.ek
        return add(self.ek, self, other)

    def __mul__(self, other):
        return mul(self.ek, self, other)


def encrypt(
    ek: EncryptionKey, plaintext: tf.Tensor, randomness: Optional[Randomness] = None,
) -> Ciphertext:
    x = tf_big.import_tensor(plaintext)

    randomness = randomness or gen_randomness(ek=ek, shape=x.shape)
    r = randomness.raw
    assert r.shape == x.shape

    gx = 1 + (ek.n * x) % ek.nn
    rn = tf_big.pow(r, ek.n, ek.nn)
    c = gx * rn % ek.nn
    return Ciphertext(ek, c)


def decrypt(
    dk: DecryptionKey, ciphertext: Ciphertext, dtype: tf.DType = tf.int32
) -> tf.Tensor:
    c = ciphertext.raw

    gxd = tf_big.pow(c, dk.d1, dk.nn)
    xd = (gxd - 1) // dk.n
    x = (xd * dk.d2) % dk.n

    if dtype == tf.variant:
        return x

    return tf_big.export_tensor(x, dtype=dtype)


def refresh(ek: EncryptionKey, ciphertext: Ciphertext) -> Ciphertext:
    c = ciphertext.raw
    s = gen_randomness(ek=ek, shape=c.shape).raw
    sn = tf_big.pow(s, ek.n, ek.nn)
    d = (c * sn) % ek.nn
    return Ciphertext(ek, d)


def add(
    ek: EncryptionKey, lhs: Ciphertext, rhs: Ciphertext, do_refresh: bool = True,
) -> Ciphertext:
    c0 = lhs.raw
    c1 = rhs.raw
    c = (c0 * c1) % ek.nn
    res = Ciphertext(ek, c)

    if not do_refresh:
        return res
    return refresh(ek, res)


def mul(
    ek: EncryptionKey, lhs: Ciphertext, rhs: tf.Tensor, do_refresh: bool = True,
) -> Ciphertext:
    c = lhs.raw
    k = tf_big.import_tensor(rhs)
    d = tf_big.pow(c, k) % ek.nn
    res = Ciphertext(ek, d)

    if not do_refresh:
        return res
    return refresh(ek, res)
