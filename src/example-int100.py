import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.tensor.int100 import (
    Int100Tensor, Int100Variable, Int100Placeholder, to_native
)

x = Int100Tensor(np.array([1,2,3]))
y = Int100Tensor(np.array([1,2,3]))
z = x + y; print(z)
z = x - y; print(z)
z = x * y; print(z)

v = Int100Variable(np.array([0,0,0]))
p = Int100Placeholder((3,))

with tf.Session() as sess:

    sess.run(v.initializer)
    print to_native(sess.run(v.value))

    sess.run(v.assign_from_int100(p), feed_dict=p.feed_from_native(np.array([5,5,5])))
    print to_native(sess.run(v.value))
    