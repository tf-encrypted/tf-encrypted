import numpy as np
import tensorflow as tf
from tensorflow_encrypted.tensor.int100 import (
    Int100Tensor,
    Int100Constant,
    Int100Variable,
    Int100Placeholder
)

x = Int100Tensor(np.array([1, 2, 3]))
y = Int100Tensor(np.array([1, 2, 3]))

z = x + y
print(z)

z = x - y
print(z)

z = x * y
print(z)

c = Int100Constant(np.array([4, 4, 4]))
v = Int100Variable(np.array([1, 1, 1]))
p = Int100Placeholder((3, ))

with tf.Session() as sess:

    print('Constant')
    print(c.eval(sess).to_int32())

    print('Variable')
    sess.run(v.initializer)
    print(v.eval(sess).to_int32())

    print('Placeholder')
    print(p.eval(sess, feed_dict=p.feed_from_native(np.array([5, 5, 5]))).to_int32())

    print('Assignment')
    w = c - p
    sess.run(v.assign_from_same(w), feed_dict=p.feed_from_native(np.array([5, 5, 5])))
    print(v.eval(sess).to_bigint() - Int100Tensor.modulus)
