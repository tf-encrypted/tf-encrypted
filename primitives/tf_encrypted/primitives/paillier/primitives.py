import tf_big

tf_big.set_secure_default(True)


class EncryptionKey:
    def __init__(self, n):
        self.n = n
        self.nn = n * n
        self.g = 1 + n


class DecryptionKey:
    def __init__(self, p, q):
        pass


def gen_keypair():
    pass


class Nonce:
    def __init__(self, raw_nonce):
        self.raw = raw_nonce


def gen_nonce():
    raw_nonce = tf_big.random_uniform(1, )


def encrypt(ek, x, r):
    gx = tf_big.pow(ek.g, x, ek.nn)
    rn = tf_big.pow(r, ek.n, ek.nn)
    return (gx * rn) % ek.nn


def decrypt(dk, c):
    pass


def add(ek, c1, c2):
    return (c1 * c2) % ek.nn


def mul(ek, c1, x2):
    return tf_big.pow(c1, x2, ek.nn)
