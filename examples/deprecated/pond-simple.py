import numpy as np
import tf_encrypted as tfe

prot = tfe.protocol.Pond()

# a = prot.define_constant(np.array([4, 3, 2, 1]).reshape(2,2))
# b = prot.define_constant(np.array([4, 3, 2, 1]).reshape(2,2))
# c = a * b

a = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2, 2))
b = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2, 2))
c = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2, 2))

x = (a * b)
y = (a * c)
z = x + y

w = prot.define_private_variable(np.zeros((2, 2)))

with tfe.Session() as sess:
  # print(sess.run(c, tag='c'))

  sess.run(tfe.global_variables_initializer(), tag='init')
  sess.run(prot.assign(w, z), tag='assign')
  sess.run(prot.assign(w, z), tag='assign')

  print(sess.run(w.reveal(), tag='reveal'))
