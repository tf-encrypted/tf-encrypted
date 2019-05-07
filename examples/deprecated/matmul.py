import numpy as np
import tf_encrypted as tfe


a = np.ones((10, 10))

x = tfe.define_private_variable(a)

b = a
y = x
for _ in range(2):
  b = np.dot(b, b)
  y = y.matmul(y)

with tfe.Session() as sess:
  sess.run(tfe.global_variables_initializer(), tag='init')
  actual = sess.run(y.reveal(), tag='reveal')

  expected = b
  np.testing.assert_allclose(actual, expected)
