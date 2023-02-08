import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.operations import tf_i128


class I128TestCase(unittest.TestCase):
    def setUp(self):
        self.precision_bits = 50
        self.scale = 2**self.precision_bits

        self._dim1_scalar = tf.constant([10, 0], dtype=tf.int64)
        self._dim2_scalar = tf.constant([[11, 0]], dtype=tf.int64)

        self._A = np.random.rand(5, 3) - 0.5
        self.A = tf_i128.to_i128(
            tf.convert_to_tensor(self._A, dtype=tf.float64), self.scale
        )

        self._B = np.random.rand(self._A.shape[0], self._A.shape[1]) - 0.5
        self.B = tf_i128.to_i128(
            tf.convert_to_tensor(self._B, dtype=tf.float64), self.scale
        )

        self._C = np.random.rand(self._A.shape[1], 20) - 0.5
        self.C = tf_i128.to_i128(
            tf.convert_to_tensor(self._C, dtype=tf.float64), self.scale
        )

        self._D = np.random.randint(low=-100, high=100, size=(3, 10))
        self.D = tf_i128.to_i128(tf.convert_to_tensor(self._D, dtype=tf.float64))

        self._E = np.random.rand(self._A.shape[1]) - 0.5
        self.E = tf_i128.to_i128(
            tf.convert_to_tensor(self._E, dtype=tf.float64), self.scale
        )

        self._F = np.random.rand(1) - 0.5
        self.F = tf_i128.to_i128(
            tf.convert_to_tensor(self._F, dtype=tf.float64), self.scale
        )

        self._A_3d = np.random.rand(5, 3, 3) - 0.5
        self.A_3d = tf_i128.to_i128(
            tf.convert_to_tensor(self._A_3d, dtype=tf.float64), self.scale
        )

    def test_addsub(self):
        ground_list = list()
        ground_list.append(self._A + self._B)
        ground_list.append(self._A + self._F)
        ground_list.append(self._F + self._A)
        ground_list.append(self._A + 10)
        ground_list.append(self._A + 11)
        ground_list.append(self._B + 10)
        ground_list.append(self._B + 11)
        ground_list.append(np.array([10 + 11]))
        ground_list.append(np.array([10 + 11]))
        ground_list.append(self._A + self._E)
        ground_list.append(self._E - self._A)

        ground_list.append(self._B - self._A)
        ground_list.append(self._E - self._A)
        ground_list.append(self._A - self._E)
        ground_list.append(10 - self._A)
        ground_list.append(11 - self._A)
        ground_list.append(self._B - 10)
        ground_list.append(self._B - 11)
        ground_list.append(np.array([10 - 11]))
        ground_list.append(np.array([11 - 10]))

        computed_list = list()
        computed_list.append(tf_i128.from_i128(tf_i128.add(self.A, self.B), self.scale))
        computed_list.append(tf_i128.from_i128(tf_i128.add(self.A, self.F), self.scale))
        computed_list.append(tf_i128.from_i128(tf_i128.add(self.F, self.A), self.scale))
        computed_list.append(
            tf_i128.from_i128(
                tf_i128.add(self.A, self._dim1_scalar * tf.constant(self.scale)),
                self.scale,
            )
        )
        computed_list.append(
            tf_i128.from_i128(
                tf_i128.add(self.A, self._dim2_scalar * tf.constant(self.scale)),
                self.scale,
            )
        )
        computed_list.append(
            tf_i128.from_i128(
                tf_i128.add(self._dim1_scalar * tf.constant(self.scale), self.B),
                self.scale,
            )
        )
        computed_list.append(
            tf_i128.from_i128(
                tf_i128.add(self._dim2_scalar * tf.constant(self.scale), self.B),
                self.scale,
            )
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.add(self._dim1_scalar, self._dim2_scalar), 1)
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.add(self._dim2_scalar, self._dim1_scalar), 1)
        )
        computed_list.append(tf_i128.from_i128(tf_i128.add(self.A, self.E), self.scale))
        computed_list.append(tf_i128.from_i128(tf_i128.sub(self.E, self.A), self.scale))

        computed_list.append(tf_i128.from_i128(tf_i128.sub(self.B, self.A), self.scale))
        computed_list.append(tf_i128.from_i128(tf_i128.sub(self.E, self.A), self.scale))
        computed_list.append(tf_i128.from_i128(tf_i128.sub(self.A, self.E), self.scale))
        computed_list.append(
            tf_i128.from_i128(
                tf_i128.sub(self._dim1_scalar * tf.constant(self.scale), self.A),
                self.scale,
            )
        )
        computed_list.append(
            tf_i128.from_i128(
                tf_i128.sub(self._dim2_scalar * tf.constant(self.scale), self.A),
                self.scale,
            )
        )
        computed_list.append(
            tf_i128.from_i128(
                tf_i128.sub(self.B, self._dim1_scalar * tf.constant(self.scale)),
                self.scale,
            )
        )
        computed_list.append(
            tf_i128.from_i128(
                tf_i128.sub(self.B, self._dim2_scalar * tf.constant(self.scale)),
                self.scale,
            )
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.sub(self._dim1_scalar, self._dim2_scalar), 1)
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.sub(self._dim2_scalar, self._dim1_scalar), 1)
        )

        for c, g in zip(computed_list, ground_list):
            self.assertTrue(np.allclose(c, g), "wrong add_sub")
            self.assertTrue(c.shape == g.shape, "wrong add_sub shape")

    def test_unary(self):
        computed_list = list()
        computed_list.append(tf_i128.from_i128(tf_i128.negate(self.A), self.scale))
        computed_list.append(tf_i128.from_i128(tf_i128.negate(self.B), self.scale))
        computed_list.append(tf_i128.from_i128(tf_i128.i128_abs(self.A), self.scale))
        computed_list.append(tf_i128.from_i128(tf_i128.i128_abs(self.B), self.scale))
        computed_list.append(tf_i128.equal(self.A, self.B))
        computed_list.append(tf_i128.equal(self.B, self.B))
        computed_list.append(
            tf_i128.equal(
                self.A,
                tf.constant(
                    tf_i128.encode(np.array([self._A[0, 0]]) * self.scale),
                    dtype=tf.int64,
                ),
            )
        )
        computed_list.append(
            tf_i128.equal(
                self.B,
                tf.constant(
                    tf_i128.encode(np.array([self._B[1, 1]]) * self.scale),
                    dtype=tf.int64,
                ),
            )
        )
        computed_list.append(
            tf_i128.equal(
                tf.constant(
                    tf_i128.encode(np.array([self._A[1, 0]]) * self.scale),
                    dtype=tf.int64,
                ),
                self.A,
            )
        )
        computed_list.append(
            tf_i128.equal(
                tf.constant(
                    tf_i128.encode(np.array([self._B[0, 1]]) * self.scale),
                    dtype=tf.int64,
                ),
                self.B,
            )
        )

        computed_list.append(
            tf_i128.equal(
                tf.constant(
                    tf_i128.encode(np.array([1.0]) * self.scale), dtype=tf.int64
                ),
                tf.constant(
                    tf_i128.encode(np.array([2.0]) * self.scale), dtype=tf.int64
                ),
            )
        )

        computed_list.append(
            tf_i128.equal(
                tf.constant(
                    tf_i128.encode(np.array([[1.0]]) * self.scale), dtype=tf.int64
                ),
                tf.constant(
                    tf_i128.encode(np.array([2.0]) * self.scale), dtype=tf.int64
                ),
            )
        )

        computed_list.append(
            tf_i128.equal(
                tf.constant(
                    tf_i128.encode(np.array([1.0]) * self.scale), dtype=tf.int64
                ),
                tf.constant(
                    tf_i128.encode(np.array([[1.0]]) * self.scale), dtype=tf.int64
                ),
            )
        )

        computed_list.append(
            tf_i128.equal(
                tf.constant(
                    tf_i128.encode(np.array([[1.0]]) * self.scale), dtype=tf.int64
                ),
                tf.constant(
                    tf_i128.encode(np.array([[2.0]]) * self.scale), dtype=tf.int64
                ),
            )
        )

        ground_list = list()
        ground_list.append(-self._A)
        ground_list.append(-self._B)
        ground_list.append(np.abs(self._A))
        ground_list.append(np.abs(self._B))
        ground_list.append(self._A == self._B)
        ground_list.append(self._B == self._B)
        ground_list.append(self._A == self._A[0, 0])
        ground_list.append(self._B == self._B[1, 1])
        ground_list.append(self._A == self._A[1, 0])
        ground_list.append(self._B == self._B[0, 1])
        ground_list.append(np.array([False]))
        ground_list.append(np.array([[False]]))
        ground_list.append(np.array([[True]]))
        ground_list.append(np.array([[False]]))

        for c, g in zip(computed_list, ground_list):
            self.assertTrue(np.allclose(c, g), "wrong unary")
            self.assertTrue(c.shape == g.shape, "wrong unary shape")

    def test_scalar_mul(self):
        computed_list = list()
        # left-hand-side is tensor
        computed_list.append(
            tf_i128.from_i128(tf_i128.mul(self.A, self._dim1_scalar), self.scale)
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.mul(self.A, self._dim2_scalar), self.scale)
        )
        # left-hand-side is scalar
        computed_list.append(
            tf_i128.from_i128(tf_i128.mul(self._dim1_scalar, self.B), self.scale)
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.mul(self._dim2_scalar, self.B), self.scale)
        )
        # element-wise product
        computed_list.append(
            tf_i128.from_i128(
                tf_i128.right_shift(tf_i128.mul(self.A, self.B), self.precision_bits),
                self.scale,
            )
        )
        computed_list.append(
            tf_i128.from_i128(
                tf_i128.right_shift(tf_i128.mul(self.A, self.E), self.precision_bits),
                self.scale,
            )
        )
        # scalar * scalar
        computed_list.append(
            tf_i128.from_i128(tf_i128.mul(self._dim1_scalar, self._dim1_scalar), 1)
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.mul(self._dim2_scalar, self._dim2_scalar), 1)
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.mul(self._dim1_scalar, self._dim2_scalar), 1)
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.mul(self._dim2_scalar, self._dim1_scalar), 1)
        )

        ground_list = list()
        ground_list.append(self._A * 10)
        ground_list.append(self._A * 11)
        ground_list.append(self._B * 10)
        ground_list.append(self._B * 11)
        ground_list.append(self._A * self._B)
        ground_list.append(self._A * self._E)
        ground_list.append(np.array(10 * 10))
        ground_list.append(np.array([11 * 11]))
        ground_list.append(np.array([10 * 11]))
        ground_list.append(np.array([10 * 11]))

        for c, g in zip(computed_list, ground_list):
            np.testing.assert_allclose(c, g, rtol=0.0, atol=0.01)
            self.assertTrue(c.shape == g.shape, "wrong scalar_mul shape")

    def test_matmul(self):
        computed = tf_i128.from_i128(
            tf_i128.right_shift(tf_i128.matmul(self.A, self.C), self.precision_bits),
            self.scale,
        )
        ground = self._A.dot(self._C)

        self.assertTrue(np.allclose(computed, ground), "wrong mat_mul")
        self.assertTrue(computed.shape == ground.shape, "wrong mat_mul shape")

    def test_reduce_sum(self):
        computed_list = list()
        ground_list = list()
        for axis in [None, 0, 1]:
            for kd in [True, False]:
                ground_list.append(self._A.sum(axis=axis, keepdims=kd))
                computed_list.append(
                    tf_i128.from_i128(
                        tf_i128.reduce_sum(self.A, axis=axis, keepdims=kd), self.scale
                    )
                )

        for c, g in zip(computed_list, ground_list):
            np.testing.assert_allclose(c, g, rtol=0.0, atol=0.01)
            self.assertTrue(c.shape == g.shape, "wrong reduce_sum shape")

    def test_reduce_sum_3d(self):
        computed_list = list()
        ground_list = list()
        for axis in [None, 0, 1, 2]:
            for kd in [True, False]:
                ground_list.append(self._A_3d.sum(axis=axis, keepdims=kd))
                computed_list.append(
                    tf_i128.from_i128(
                        tf_i128.reduce_sum(self.A_3d, axis=axis, keepdims=kd),
                        self.scale,
                    )
                )

        for c, g in zip(computed_list, ground_list):
            np.testing.assert_allclose(c, g, rtol=0.0, atol=0.01)
            self.assertTrue(c.shape == g.shape, "wrong reduce_sum shape")

    def test_shift(self):
        computed_list = list()
        ground_list = list()

        def rshift(val, n):
            val = val.astype(object)
            mask = (1 << (128 - n)) - 1
            return ((val >> n) & mask).astype(np.double)

        computed_list.append(
            tf_i128.from_i128(tf_i128.left_shift(self.A, 50), self.scale)
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.left_shift(self.B, 50), self.scale)
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.right_shift(self.A, 10), self.scale)
        )
        computed_list.append(
            tf_i128.from_i128(tf_i128.right_shift(self.B, 10), self.scale)
        )
        computed_list.append(tf_i128.from_i128(tf_i128.logic_right_shift(self.D, 3)))
        computed_list.append(tf_i128.from_i128(tf_i128.logic_right_shift(self.D, 10)))

        ground_list.append(self._A * 2.0**50)
        ground_list.append(self._B * 2.0**50)
        ground_list.append(self._A / 2.0**10)
        ground_list.append(self._B / 2.0**10)
        ground_list.append(rshift(self._D.copy(), 3))
        ground_list.append(rshift(self._D.copy(), 10))

        for c, g in zip(computed_list, ground_list):
            np.testing.assert_allclose(c, g, rtol=1e-05, atol=1e-08)
            self.assertTrue(c.shape == g.shape, "wrong shift shape")

    def test_bit_reverse(self):
        a = tf.Variable(np.random.randint(1, 100, size=(1, 2)), dtype=tf.int64)
        a_rev = tf_i128.i128_bit_reverse(a)
        _a = tf_i128.i128_bit_reverse(a_rev)
        self.assertTrue(np.allclose(a, _a), "wrong bit reverse")

    def test_encode_decode(self):
        # inter-change encode/decode using python codes and tf.operations
        self._test_encode()
        self._test_decode()

        encoded_list = list()
        encoded_list.append(tf_i128.encode(self._A * self.scale))
        encoded_list.append(tf_i128.encode(self._B * self.scale))
        encoded_list.append(tf_i128.encode(self._C * self.scale))
        encoded_list.append(tf_i128.encode(self._D * self.scale))
        ground_list = [self._A, self._B, self._C, self._D]
        for e, g in zip(encoded_list, ground_list):
            e = tf_i128.decode(e, self.scale)
            self.assertTrue(np.allclose(e, g), "wrong encode_decode")

    def _test_encode(self):
        # encode ndarray to int64 array before define tf.tensor
        computed_list = list()
        ground_list = list()

        encoded_A = tf_i128.encode(self._A * self.scale)
        encoded_B = tf_i128.encode(self._B * self.scale)
        encoded_C = tf_i128.encode(self._C * self.scale)
        encoded_D = tf_i128.encode(self._D * self.scale)
        encoded_E = tf_i128.encode(self._E * self.scale)

        computed_list.append(
            tf_i128.from_i128(tf.Variable(encoded_A, dtype=tf.int64), self.scale)
        )
        computed_list.append(
            tf_i128.from_i128(tf.Variable(encoded_B, dtype=tf.int64), self.scale)
        )
        computed_list.append(
            tf_i128.from_i128(tf.Variable(encoded_C, dtype=tf.int64), self.scale)
        )
        computed_list.append(
            tf_i128.from_i128(tf.Variable(encoded_D, dtype=tf.int64), self.scale)
        )
        computed_list.append(
            tf_i128.from_i128(tf.Variable(encoded_E, dtype=tf.int64), self.scale)
        )

        ground_list.append(self._A)
        ground_list.append(self._B)
        ground_list.append(self._C)
        ground_list.append(self._D)
        ground_list.append(self._E)

        for c, g in zip(computed_list, ground_list):
            self.assertTrue(np.allclose(c, g), "wrong encode")

    def _test_decode(self):
        computed_list = list()
        ground_list = list()

        computed_list.append(self.A)
        computed_list.append(self.B)
        computed_list.append(self.C)
        computed_list.append(self.E)

        ground_list.append(self._A)
        ground_list.append(self._B)
        ground_list.append(self._C)
        ground_list.append(self._E)

        for c, g in zip(computed_list, ground_list):
            c = tf_i128.decode(c.numpy(), self.scale)
            self.assertTrue(np.allclose(c, g), "wrong decode")

    def test_128_sample_uniform(self):
        from tf_encrypted.tensor import int128factory

        r = int128factory.sample_uniform([3, 3], minval=0, maxval=2)
        x = tf_i128.from_i128(r.value).numpy()
        x[x == 0] = 1
        np.testing.assert_array_equal(x, np.ones([3, 3]))

    def test_128_mul_3d(self):
        a = tf_i128.to_i128(tf.ones([128, 1, 1], dtype=tf.float64) * 2)
        b = tf_i128.to_i128(tf.ones([128, 2, 3], dtype=tf.float64) * 3)
        c = tf_i128.to_i128(tf.ones([128, 2, 3], dtype=tf.float64) * 6)
        d = tf_i128.mul(a, b)
        np.testing.assert_array_equal(c, d)

    def test_128_mul_4d(self):
        a = tf_i128.to_i128(tf.ones([128, 2, 1, 4], dtype=tf.float64) * 5)
        b = tf_i128.to_i128(tf.ones([128, 2, 1, 4], dtype=tf.float64) * 2)
        c = tf_i128.to_i128(tf.ones([128, 2, 1, 4], dtype=tf.float64) * 10)
        d = tf_i128.mul(a, b)
        d_value = tf_i128.from_i128(d)
        c_value = tf_i128.from_i128(c)
        np.testing.assert_array_equal(c_value, d_value)

    def test_128_add_3d(self):
        a = tf_i128.to_i128(tf.ones([128, 1, 1], dtype=tf.float64) * 2)
        b = tf_i128.to_i128(tf.ones([128, 2, 3], dtype=tf.float64) * 3)
        c = tf_i128.to_i128(tf.ones([128, 2, 3], dtype=tf.float64) * 5)
        d = tf_i128.add(a, b)
        np.testing.assert_array_equal(c, d)

    def test_128_sub_3d(self):
        a = tf_i128.to_i128(tf.ones([128, 1, 1], dtype=tf.float64) * 2)
        b = tf_i128.to_i128(tf.ones([128, 2, 3], dtype=tf.float64) * 3)
        c = tf_i128.to_i128(tf.ones([128, 2, 3], dtype=tf.float64) * -1)
        d = tf_i128.sub(a, b)
        np.testing.assert_array_equal(c, d)

    def test_128_right_shift_3d(self):
        a = tf_i128.to_i128(tf.ones([3, 2, 3], dtype=tf.float64) * 5)
        b = tf.ones([3, 1, 1], dtype=tf.int64)
        c = tf_i128.from_i128(tf_i128.right_shift(a, b))
        d = tf_i128.from_i128(tf_i128.right_shift(a, 1))
        e = tf_i128.from_i128(tf_i128.right_shift(a, -7))
        np.testing.assert_allclose(c, np.ones([3, 2, 3]) * 2, rtol=0.0, atol=0.01)
        np.testing.assert_allclose(d, np.ones([3, 2, 3]) * 2, rtol=0.0, atol=0.01)
        np.testing.assert_allclose(e, np.ones([3, 2, 3]) * 5, rtol=0.0, atol=0.01)

    def test_128_left_shift_3d(self):
        a = tf_i128.to_i128(tf.ones([3, 2, 3], dtype=tf.float64) * 5)
        b = tf.ones([3, 1, 1], dtype=tf.int64)
        c = tf_i128.from_i128(tf_i128.left_shift(a, b))
        d = tf_i128.from_i128(tf_i128.left_shift(a, 1))
        np.testing.assert_allclose(c, np.ones([3, 2, 3]) * 10, rtol=0.0, atol=0.01)
        np.testing.assert_allclose(d, np.ones([3, 2, 3]) * 10, rtol=0.0, atol=0.01)

    def test_128_reduce_sum_3d(self):
        a = tf_i128.to_i128(tf.ones([4, 2, 3], dtype=tf.float64) * 5)
        b = tf_i128.from_i128(tf_i128.reduce_sum(a, axis=0, keepdims=False))
        c = tf_i128.from_i128(tf_i128.reduce_sum(a, axis=0, keepdims=True))
        d = tf_i128.from_i128(tf_i128.reduce_sum(a, axis=1, keepdims=False))
        e = tf_i128.from_i128(tf_i128.reduce_sum(a, axis=1, keepdims=True))
        np.testing.assert_allclose(b, np.ones([2, 3]) * 20, rtol=0.0, atol=0.01)
        np.testing.assert_allclose(c, np.ones([1, 2, 3]) * 20, rtol=0.0, atol=0.01)
        np.testing.assert_allclose(d, np.ones([4, 3]) * 10, rtol=0.0, atol=0.01)
        np.testing.assert_allclose(e, np.ones([4, 1, 3]) * 10, rtol=0.0, atol=0.01)


if __name__ == "__main__":
    unittest.main()
