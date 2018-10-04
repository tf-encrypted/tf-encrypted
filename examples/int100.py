import numpy as np
import tensorflow_encrypted as tfe
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

with tfe.Session() as sess:

    print('Constant')
    print(sess.run(c))

    print('Variable')
    sess.run(v.initializer)
    print(sess.run(v))

    print('Placeholder')
    print(sess.run(p, feed_dict=p.feed_from_native(np.array([5, 5, 5]))))

    print('Assignment')
    w = c - p
    sess.run(v.assign_from_same(w), feed_dict=p.feed_from_native(np.array([5, 5, 5])))
    print(sess.run(v))
