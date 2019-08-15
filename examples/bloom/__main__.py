"""Main entrypoint for running Bloom regression example."""
import tensorflow as tf

from regressor import BloomRegressor, DataOwner

NUM_FEATURES = 100
TRAINING_SET_SIZE = 10000
TEST_SET_SIZE = 1000

genebanks = [
    DataOwner("genebank-0", NUM_FEATURES, TRAINING_SET_SIZE, TEST_SET_SIZE),
    DataOwner("genebank-1", NUM_FEATURES, TRAINING_SET_SIZE, TEST_SET_SIZE),
    DataOwner("genebank-3", NUM_FEATURES, TRAINING_SET_SIZE, TEST_SET_SIZE),
    DataOwner("genebank-4", NUM_FEATURES, TRAINING_SET_SIZE, TEST_SET_SIZE),
    DataOwner("genebank-5", NUM_FEATURES, TRAINING_SET_SIZE, TEST_SET_SIZE),
    DataOwner("genebank-6", NUM_FEATURES, TRAINING_SET_SIZE, TEST_SET_SIZE),
]

model = BloomRegressor()
model.fit(genebanks)

# report results of training
with tf.Graph().as_default() as g:
  output = model.coefficients
  print(tf.Session(graph=g).run(output))
