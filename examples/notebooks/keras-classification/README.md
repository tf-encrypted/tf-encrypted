# Private Prediction with tfe.keras

This directory contains a list of notebooks demonstrating how you train a model with tf.keras, then start serving private predictions with tfe.keras. 

To start serving private predictions you can run the notebooks in the following order:
- *a - Secure Classification with TFE Keras - Public Training without DP*
- *b - Secure Classification with TFE Keras - Secure Model Serving*
- *c - Secure Classification with TFE Keras - Private Prediction Client*

Before serving your model, you can also train the model with differential privacy: *a - Secure Classification with TFE Keras - Public Training - with DP*. 
By training your model with differential privacy, you can ensure that the model is not memorizing sensitive information about the training set. If the model is not trained with differential privacy, attackers could reveal some private imformation by just querying the deployed model. Two common attacks are [membership inference](https://www.cs.cornell.edu/~shmat/shmat_oak17.pdf) and [model inversion](https://www.cs.cmu.edu/~mfredrik/papers/fjr2015ccs.pdf).

To learn more about [TensorFlow Privacy](https://github.com/tensorflow/privacy) you can read this [excellent blog post](http://www.cleverhans.io/privacy/2019/03/26/machine-learning-with-differential-privacy-in-tensorflow.html). 