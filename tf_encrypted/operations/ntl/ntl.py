import os
import tf_encrypted as tfe
import tensorflow as tf

dirname = os.path.dirname(tfe.__file__)
so_name = "/operations/ntl/_ntl_ops.so"
shared_object = dirname + so_name
ntl_ops = tf.load_op_library(shared_object)

create_ntl_matrix = ntl_ops.create_ntl_matrix
ntl_to_native = ntl_ops.ntl_to_native
matmul_ntl = ntl_ops.mat_mul_ntl
