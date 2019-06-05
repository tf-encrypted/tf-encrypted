"""GMP"""
import os
import tensorflow as tf
import tf_encrypted as tfe


dirname = os.path.dirname(tfe.__file__)
so_name = '{dn}/operations/gmp_variant/variant_gmp_op.so'
shared_object = so_name.format(dn=dirname)

mpz_module = tf.load_op_library(shared_object)

def create(val):
  return mpz_module.create_mpz_variant(val)

def add(a, b):
  return mpz_module.add_mpz(a, b)

def convert_to_str(val):
  return mpz_module.mpz_to_string(val)
