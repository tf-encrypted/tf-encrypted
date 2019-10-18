import pytest
import tensorflow as tf

@pytest.fixture(scope="session", autouse=True)
def v1():
  tf.compat.v1.disable_v2_behavior()