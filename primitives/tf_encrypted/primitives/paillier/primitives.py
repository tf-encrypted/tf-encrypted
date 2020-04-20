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


def gen_keypair():
    # TODO
    p = tf_big.convert_to_tensor(np.array([[200000005627]]))
    q = tf_big.convert_to_tensor(np.array([[200000005339]]))
    n = p * q
    return EncryptionKey(n), DecryptionKey(p, q)


class Nonce:
    def __init__(self, raw_nonce):
        self.raw = tf_big.convert_to_tensor(raw_nonce)


def gen_nonce(ek, shape):
    return Nonce(tf_big.random_uniform(shape=shape, maxval=ek.n))


class Ciphertext:
    def __init__(self, raw_ciphertext):
        self.raw = tf_big.convert_to_tensor(raw_ciphertext)


def encrypt(ek, plaintext, nonce):
    x = tf_big.convert_to_tensor(plaintext)
    r = nonce.raw
    assert r.shape == x.shape
    gx = tf_big.pow(ek.g, x, ek.nn)
    rn = tf_big.pow(r, ek.n, ek.nn)
    c = gx * rn % ek.nn
    return Ciphertext(c)


def decrypt(dk, ciphertext, dtype=tf.int32):
    c = ciphertext.raw
    gxd = tf_big.pow(c, dk.d1, dk.nn)
    xd = (gxd - 1) // dk.n
    x = (xd * dk.d2) % dk.n
    return tf_big.convert_from_tensor(x, dtype=dtype) if dtype else x


def refresh(ek, ciphertext):
    c = ciphertext.raw
    s = gen_nonce(ek, c.shape).raw
    sn = tf_big.pow(s, ek.n, ek.nn)
    d = (c * sn) % ek.nn
    return Ciphertext(d)


def add(ek, ciphertext_lhs, ciphertext_rhs, do_refresh=True):
    c0 = ciphertext_lhs.raw
    c1 = ciphertext_rhs.raw
    c = (c0 * c1) % ek.nn
    c = Ciphertext(c)
    return refresh(ek, c) if do_refresh else c


def mul(ek, ciphertext, plaintext, do_refresh=True):
    c = ciphertext.raw
    k = tf_big.convert_to_tensor(plaintext)
    d = tf_big.pow(c, k) % ek.nn
    d = Ciphertext(d)
    return refresh(ek, d) if do_refresh else d
