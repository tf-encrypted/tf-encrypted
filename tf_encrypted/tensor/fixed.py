"""A fixed-point configuration to support various tensor types."""
from __future__ import absolute_import

from math import ceil, log2

from .factory import AbstractFactory


# NOTE the assumption in encoding/decoding is that encoded numbers will fit
#      into signed int32
class FixedpointConfig:
  """
  Helper class containing various parameters of fixed-point precision
  tensors.
  """

  def __init__(
      self,
      scaling_base: int,
      precision_integral: int,
      precision_fractional: int,
      matmul_threshold: int,
      truncation_gap: int,
      use_noninteractive_truncation: bool,
  ) -> None:
    self.scaling_base = scaling_base
    self.precision_integral = precision_integral
    self.precision_fractional = precision_fractional
    self.matmul_threshold = matmul_threshold
    self.truncation_gap = truncation_gap
    self.use_noninteractive_truncation = use_noninteractive_truncation

  @property
  def bound_single_precision(self) -> int:
    total_precision = self.precision_integral + self.precision_fractional
    return self.scaling_base ** (total_precision)

  @property
  def bound_double_precision(self) -> int:
    total_precision = self.precision_integral + 2 * self.precision_fractional
    return self.scaling_base ** (total_precision)

  @property
  def bound_intermediate_results(self) -> int:
    return self.bound_double_precision * self.matmul_threshold

  @property
  def scaling_factor(self) -> int:
    return self.scaling_base ** self.precision_fractional


fixed100 = FixedpointConfig(
    scaling_base=2,
    precision_integral=14,
    precision_fractional=16,
    matmul_threshold=1024,
    truncation_gap=40,
    use_noninteractive_truncation=False,
)

fixed100_ni = FixedpointConfig(
    scaling_base=2,
    precision_integral=14,
    precision_fractional=16,
    matmul_threshold=1024,
    truncation_gap=20,
    use_noninteractive_truncation=True,
)

# TODO[Morten] make sure values in int64 configs make sense

fixed64 = FixedpointConfig(
    scaling_base=3,
    precision_integral=7,
    precision_fractional=8,
    matmul_threshold=256,
    truncation_gap=20,
    use_noninteractive_truncation=False,
)

fixed64_ni = FixedpointConfig(
    scaling_base=2,
    precision_integral=10,
    precision_fractional=13,
    matmul_threshold=256,
    truncation_gap=20,
    use_noninteractive_truncation=True,
)


def _validate_fixedpoint_config(config: FixedpointConfig,
                                tensor_factory: AbstractFactory) -> bool:
  """
  Ensure the given FixedpointConfig is compatible with the current
  tensor_factory, preventing silent errors.
  """
  no_issues = True

  check_32bit = ceil(log2(config.bound_single_precision)) > 31
  check_64bit = ceil(log2(config.bound_single_precision)) > 63
  trunc_over_mod = (ceil(log2(config.bound_double_precision))
                    + config.truncation_gap
                    >= log2(tensor_factory.modulus))

  if check_32bit:
    print("WARNING: Plaintext values won't fit in 32bit tensors")
    no_issues = False

  if check_64bit:
    print("WARNING: Plaintext values won't fit in 64bit values")
    no_issues = False

  if trunc_over_mod:
    print("WARNING: Modulus is too small for truncation")
    no_issues = False

  # TODO[Morten] test for intermediate size wrt native type

  # TODO[Morten] in decoding we assume that x + bound fits within the native
  #              type of the backing tensor

  # TODO[Morten] truncation gap is statistical security for interactive
  #              truncation; write assertions

  return no_issues
