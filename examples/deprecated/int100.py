import numpy as np

import tf_encrypted as tfe
from tf_encrypted.tensor.int100 import int100factory as int100

x = int100.tensor(np.array([1, 2, 3]))
y = int100.tensor(np.array([1, 2, 3]))

z = x + y
print(z)

z = x - y
print(z)

z = x * y
print(z)

c = int100.constant(np.array([4, 4, 4]))
v = int100.variable(np.array([1, 1, 1]))
p = int100.placeholder((3,))

with tfe.Session() as sess:

    print("Constant")
    print(sess.run(c))

    print("Variable")
    sess.run(v.initializer)
    print(sess.run(v))

    print("Placeholder")
    print(sess.run(p, feed_dict=p.feed(np.array([5, 5, 5]))))

    print("Assignment")
    w = c - p
    sess.run(v.assign_from_same(w), feed_dict=p.feed(np.array([5, 5, 5])))
    print(sess.run(v))
