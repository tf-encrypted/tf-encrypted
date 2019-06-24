from generics import build_tfe_op


# NOTE: Can optionally code-gen this from protocol kernels
#       or from the main tensorflow namespace.
_TFE_OP_NAMES = ["add", "mul"]

for op_name in _TFE_OP_NAMES:
  globals()[op_name] = build_tfe_op(op_name)
