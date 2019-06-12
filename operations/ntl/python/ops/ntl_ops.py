"""NTL Ops"""
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

ntl_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_ntl_ops.so'))
create_ntl_matrix = ntl_ops.create_ntl_matrix
ntl_to_native = ntl_ops.ntl_to_native
matmul_ntl = ntl_ops.mat_mul_ntl
