import os
from typing import Tuple

import tensorflow as tf

# SO_FILE = "{current_dir}/sodium_module_tf_{tf_version}.so".format(
#     current_dir=os.path.dirname(__file__), tf_version=tf.__version__,
# )
SO_FILE = "{current_dir}/_sodium_module.so".format(
    current_dir=os.path.dirname(__file__),
)
assert os.path.exists(SO_FILE), (
    "Could not find required 'sodium' module, '{so_file}'"
).format(so_file=SO_FILE)

sodium_module = tf.load_op_library(SO_FILE)
assert sodium_module, "Could not load required 'sodium' module"


class PublicKey:
    def __init__(self, raw_pk):
        self.raw = raw_pk


class SecretKey:
    def __init__(self, raw_sk):
        self.raw = raw_sk


def gen_keypair() -> Tuple[PublicKey, SecretKey]:
    raw_pk, raw_sk = sodium_module.sodium_easy_box_gen_keypair()
    return PublicKey(raw_pk), SecretKey(raw_sk)


class Nonce:
    def __init__(self, raw_nonce):
        self.raw = raw_nonce


def gen_nonce() -> Nonce:
    raw_nonce = sodium_module.sodium_easy_box_gen_nonce()
    return Nonce(raw_nonce)


class Ciphertext:
    def __init__(self, raw_ciphertext):
        self.raw = raw_ciphertext


class Mac:
    def __init__(self, raw_mac):
        self.raw = raw_mac


def seal_detached(
    plaintext: tf.Tensor,
    nonce: Nonce,
    publickey_receiver: PublicKey,
    secretkey_sender: SecretKey,
) -> Tuple[Ciphertext, Mac]:
    ciphertext, mac = sodium_module.sodium_easy_box_seal_detached(
        plaintext, nonce.raw, publickey_receiver.raw, secretkey_sender.raw
    )
    return Ciphertext(ciphertext), Mac(mac)


def open_detached(
    ciphertext: Ciphertext,
    mac: Mac,
    nonce: Nonce,
    public_sender: PublicKey,
    secretkey_receiver: SecretKey,
) -> tf.Tensor:
    plaintext = sodium_module.sodium_easy_box_open_detached(
        ciphertext.raw, mac.raw, nonce.raw, public_sender.raw, secretkey_receiver.raw
    )
    return plaintext
