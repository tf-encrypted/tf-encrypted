"""Ntl Ops Tests"""
import numpy as np

from tensorflow.python.platform import test
try:
  from ntl.python.ops.ntl_ops import create_ntl_matrix, ntl_to_string, matmul_ntl
except ImportError:
  from ntl_ops import create_ntl_matrix, ntl_to_string, matmul_ntl


class NTLMatrixTest(test.TestCase):
  """NTLMatrix test"""

  def test_ntl_matrix(self):
    with self.test_session():
      var1 = create_ntl_matrix([["5", "5"], ["5", "5"]], 555666)
      var2 = create_ntl_matrix([["5", "5"], ["5", "5"]], 555666)

      res = matmul_ntl(var1, var2)

      s = ntl_to_string(res)

      np.testing.assert_array_equal([[b"50", b"50"], [b"50", b"50"]], s.eval())


if __name__ == '__main__':
  test.main()
