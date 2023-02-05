# pylint: disable=missing-docstring
import os
import unittest

import numpy as np
import tensorflow as tf

from tf_encrypted.operations import dataset


class TestWriteRecord(unittest.TestCase):
    def test_write_record_simgle(self):
        a = tf.random.uniform(shape=[100, 100], maxval=10000, dtype=tf.int64)
        sa = tf.io.serialize_tensor(a)
        dataset.write_record("./temp_write_single.TFRecord", sa, append=False)

        temp_dataset = iter(tf.data.TFRecordDataset(["./temp_write_single.TFRecord"]))
        rsa = next(temp_dataset)
        ra = tf.io.parse_tensor(rsa, out_type=tf.int64)
        os.remove("./temp_write_single.TFRecord")

        np.testing.assert_array_equal(a, ra)

    def test_append_record_simgle(self):
        a = tf.random.uniform(shape=[100, 100], maxval=10000, dtype=tf.int64)
        sa = tf.io.serialize_tensor(a)
        dataset.write_record("./temp_append_single.TFRecord", sa)

        temp_dataset = iter(tf.data.TFRecordDataset(["./temp_append_single.TFRecord"]))
        rsa = next(temp_dataset)
        ra = tf.io.parse_tensor(rsa, out_type=tf.int64)
        os.remove("./temp_append_single.TFRecord")

        np.testing.assert_array_equal(a, ra)

    def test_append_record_multip(self):
        a = tf.random.uniform(shape=[100, 100], maxval=10000, dtype=tf.int64)
        b = tf.random.uniform(shape=[100, 100], maxval=10000, dtype=tf.int64)
        c = tf.random.uniform(shape=[100, 100], maxval=10000, dtype=tf.int64)
        sa = tf.io.serialize_tensor(a)
        sb = tf.io.serialize_tensor(b)
        sc = tf.io.serialize_tensor(c)
        dataset.write_record("./temp_append_multip.TFRecord", sa)
        dataset.write_record("./temp_append_multip.TFRecord", sb)
        dataset.write_record("./temp_append_multip.TFRecord", sc)

        temp_dataset = iter(tf.data.TFRecordDataset(["./temp_append_multip.TFRecord"]))
        rsa = next(temp_dataset)
        ra = tf.io.parse_tensor(rsa, out_type=tf.int64)
        np.testing.assert_array_equal(a, ra)

        rsb = next(temp_dataset)
        rb = tf.io.parse_tensor(rsb, out_type=tf.int64)
        np.testing.assert_array_equal(b, rb)

        rsc = next(temp_dataset)
        rc = tf.io.parse_tensor(rsc, out_type=tf.int64)
        np.testing.assert_array_equal(c, rc)
        os.remove("./temp_append_multip.TFRecord")

    def test_write_record_multip(self):
        a = tf.random.uniform(shape=[100, 100], maxval=10000, dtype=tf.int64)
        b = tf.random.uniform(shape=[100, 100], maxval=10000, dtype=tf.int64)
        c = tf.random.uniform(shape=[100, 100], maxval=10000, dtype=tf.int64)
        sa = tf.io.serialize_tensor(a)
        sb = tf.io.serialize_tensor(b)
        sc = tf.io.serialize_tensor(c)
        dataset.write_record("./temp_write_multip.TFRecord", sa)
        dataset.write_record("./temp_write_multip.TFRecord", sb)
        dataset.write_record("./temp_write_multip.TFRecord", sc, append=False)

        temp_dataset = iter(tf.data.TFRecordDataset(["./temp_write_multip.TFRecord"]))
        rsc = next(temp_dataset)
        rc = tf.io.parse_tensor(rsc, out_type=tf.int64)
        np.testing.assert_array_equal(c, rc)
        os.remove("./temp_write_multip.TFRecord")


if __name__ == "__main__":
    unittest.main()
