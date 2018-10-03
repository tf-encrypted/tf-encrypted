import numpy as np
import tensorflow_encrypted as tfe
import tensorflow_encrypted.tensor.int32 as tensor_type

Int32Tensor = tensor_type.Int32Tensor
Int32Constant = tensor_type.Int32Constant
Int32Variable = tensor_type.Int32Variable
Int32Placeholder = tensor_type.Int32Placeholder

x = Int32Tensor(np.array([1, 2, 3]))
y = Int32Tensor(np.array([1, 2, 3]))

z = x + y
print(z)

z = x - y
print(z)

z = x * y
print(z)

c = Int32Constant(np.array([4, 4, 4]))
v = Int32Variable(np.array([1, 1, 1]))
p = Int32Placeholder((3, ))

with tfe.Session() as sess:

    print('Constant')
    print(c.eval(sess).to_native())

    print('Variable')
    sess.run(v.initializer)
    print(v.eval(sess).to_native())

    print('Placeholder')
    print(p.eval(sess, feed_dict=p.feed_from_native(np.array([5, 5, 5]))).to_native())

    print('Assignment')
    w = c - p
    sess.run(v.assign_from_same(w), feed_dict=p.feed_from_native(np.array([5, 5, 5])))
    print(v.eval(sess).to_native())
