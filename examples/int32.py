import numpy as np
import tf_encrypted as tfe
from tf_encrypted.tensor import int32factory as int32

x = int32.tensor(np.array([1, 2, 3]))
y = int32.tensor(np.array([1, 2, 3]))

z = x + y

z = x - y

z = x * y

c = int32.constant(np.array([4, 4, 4]))
v = int32.variable(np.array([1, 1, 1]))
p = int32.placeholder((3, ))

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
