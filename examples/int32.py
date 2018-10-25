import numpy as np
import tf_encrypted as tfe
import tf_encrypted.tensor.int32 as tensor_type

Int32Tensor = tensor_type.Int32Tensor
Int32Constant = tensor_type.Int32Constant
Int32Variable = tensor_type.Int32Variable
Int32Placeholder = tensor_type.Int32Placeholder

x = Int32Tensor(np.array([1, 2, 3]))
y = Int32Tensor(np.array([1, 2, 3]))

z = x + y

z = x - y

z = x * y

c = Int32Constant(np.array([4, 4, 4]))
v = Int32Variable(np.array([1, 1, 1]))
p = Int32Placeholder((3, ))

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
