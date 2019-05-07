"""Secure training."""
import numpy as np
import tensorflow as tf
import tf_encrypted as tfe

from data import gen_training_input, gen_test_input

tf.set_random_seed(1)

def main():
  # Parameters
  learning_rate = 0.01
  training_set_size = 2000
  test_set_size = 100
  training_epochs = 10
  batch_size = 100
  nb_feats = 10

  def mock_training_data():
    return gen_training_input(training_set_size, nb_feats, batch_size)

  def mock_test_data():
    return gen_test_input(test_set_size, nb_feats, batch_size)

  xp, yp = tfe.define_private_input('input-provider', mock_training_data())
  xp_test, yp_test = tfe.define_private_input(
      'input-provider',
      mock_test_data())

  w = tfe.define_private_variable(tf.random_uniform([nb_feats, 1], -0.01, 0.01))
  b = tfe.define_private_variable(tf.zeros([1]))

  # Training model
  out = tfe.matmul(xp, w) + b
  pred = tfe.sigmoid(out)
  # Due to missing log function approximation, we need to compute the cost in numpy
  # cost = -tfe.sum(y * tfe.log(pred) + (1 - y) * tfe.log(1 - pred)) * (1/train_batch_size)

  # Backprop
  dc_dout = pred - yp
  dw = tfe.matmul(tfe.transpose(xp), dc_dout) * (1 / batch_size)
  db = tfe.reduce_sum(1. * dc_dout, axis=0) * (1 / batch_size)
  ops = [
      tfe.assign(w, w - dw * learning_rate),
      tfe.assign(b, b - db * learning_rate)
  ]

  # Testing model
  pred_test = tfe.sigmoid(tfe.matmul(xp_test, w) + b)


  def print_accuracy(pred_test_tf, y_test_tf: tf.Tensor) -> tf.Operation:
    correct_prediction = tf.equal(tf.round(pred_test_tf), y_test_tf)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return tf.print("Accuracy", accuracy)


  print_acc_op = tfe.define_output(
      'input-provider', [pred_test, yp_test], print_accuracy)

  total_batch = training_set_size // batch_size
  with tfe.Session() as sess:
    sess.run(tfe.global_variables_initializer(), tag='init')

    for epoch in range(training_epochs):
      avg_cost = 0.

      for _ in range(total_batch):
        _, y_out, p_out = sess.run(
            [ops, yp.reveal(), pred.reveal()], tag='optimize')
        # Our sigmoid function is an approximation
        # it can have values outside of the range [0, 1], we remove them and add/substract an epsilon to compute the cost
        p_out = p_out * (p_out > 0) + 0.001
        p_out = p_out * (p_out < 1) + (p_out >= 1) * 0.999
        c = -np.mean(y_out * np.log(p_out) + (1 - y_out) * np.log(1 - p_out))
        avg_cost += c / total_batch

      print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    sess.run(print_acc_op)

if __name__ == '__main__':
  main()
