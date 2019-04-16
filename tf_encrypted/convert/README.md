Convert
===========
A Converter for constructing TFE Sessions from TensorFlow Graphs.

## Contents
1. [Reserved scopes](#reserved-scopes)
2. [General workflow](#general-workflow)
2. [Adding a conversion](#adding-an-op)
 - [Adding the conversion function](#adding-the-conversion-function)
 - [Adding the conversion test](#adding-the-conversion-test)
3. [Adding a special op](#adding-a-special-op)
 - [Registering intermediate nodes](#registering-intermediate-nodes)

## Reserved scopes
The following name scopes are reserved for use by the TF Encrypted Converter.  If you don't see the one you want, please file a feature request or submit a PR [implementing a conversion for it](#adding-a-special-op).

Reserved Scope | TF Counterpart
---------------|---------------
`conv2d`|[tf.keras.layers.Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
`flatten`|[tf.keras.layers.Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten)
`required_space_to_batch_paddings`|[tf.required_space_to_batch_paddings](https://www.tensorflow.org/api_docs/python/tf/required_space_to_batch_paddings)


## General workflow
1. Train a model in TensorFlow or Keras, or import one into TensorFlow through [`onnx-tensorflow`](https://github.com/onnx/onnx-tensorflow).
2. Export in SavedModel format.
3. Run the `freeze_graph.py` script to produce a GraphDef protobuf  with weights included (e.g. as `model.pb`).
4. Ensure that conversion is possible, which means either:
  1. All TensorFlow Ops in the graph have implementations in TF Encrypted and conversion functions registered in `register.py`.
  2. All special ops, i.e. higher-level subgraphs, have corresponding Ops in TF Encrypted and conversion functions registered in `specops.yaml`.
5. Feed your frozen GraphDef as the `--model_url` flag in `bin/run`, or use the `tfe.convert.Converter.convert` method in your own TF Encrypted script.

## Adding a conversion
Suppose you wanted to add a conversion for the [`tf.squeeze`](https://www.tensorflow.org/api_docs/python/tf/squeeze) op from TensorFlow.

#### Adding the conversion function
First, you'll write the conversion function.  These functions accept a `converter` instance, a NodeDef `node`, and a list of `inputs` NodeDefs. The function should construct and output a valid TF Encrypted operation for `node` NodeDefs from `inputs` using the current TFE protocol, config, and session information specified by the `converter`. You can use the `converter.outputs` dictionary to find and use previously constructed TFE tensors for your list of `inputs`. For example,
```python
def squeeze(converter, node, inputs) -> Any:
    input = converter.outputs[inputs[0]]
    axis = node.attr["squeeze_dims"].list.i
    return converter.protocol.squeeze(input, list(axis))
```

Once you've defined this function, place it in the dictionary provided by the `registry` in register.py.  It should be keyed by the corresponding Op name (for normal ops) or the reserved scope of the special op (see below).

#### Adding the conversion test
1. Write a function called `export_squeeze` accepting args `filename` and `input_shape` and returning a call to the `export` function from `test_convert.py`.  The `export` function accepts the output tf.Operation of the graph, a string representing the filename (you don't need to worry about this here), and an optional tf.Session object to use when exporting.
2. Write a function called `run_squeeze` accepting an `input` ndarray and produces the `output` of the op as an ndarray.
3. Write a function `test_squeeze_convert` that instantiates an example numpy array and feeds it to a call to `_test_with_ndarray_input_fn`. Be mindful here: the name you pass as the first argument needs to match the name in your `run_*` and `export_*` functions exactly in order for the test to run successfully.

Here is an example for `tf.squeeze` in full:

```python
class TestConvert(unittest.TestCase):
    [...]
    def test_squeeze_convert(self):
        test_input = np.ones([1, 2, 3, 1])
        self._test_with_ndarray_input_fn('squeeze', test_input, protocol='Pond')

[...]

def run_squeeze(input):
    a = tf.placeholder(tf.float32, shape=input.shape, name="input")
    x = tf.squeeze(a, axis=[0, 3])
    with tf.Session() as sess:
        output = sess.run(x, feed_dict=dict(a=input))
    return output

def export_squeeze(filename, input_shape):
    a = tf.placeholder(tf.float32, shape=input_shape, name="input")
    x = tf.squeeze(a, axis=[0, 3])
    return export(x, filename)
```

## Adding a special op
Special ops (or `specops` in our code) are higher-level scopes from either TensorFlow or Keras that do not have individual Ops defined in the TF backend, usually because they were written in Python. In a GraphDef, these often manifest as subgraphs containing Ops that we do not wish to replace with secure equivalents themselves, but as a whole can be replaced with secure versions of various ops from TF Encrypted.

A good example is the Keras Conv2D layer: although TF Encrypted implements the Conv2D Op from TensorFlow (which Keras's Conv2D depends on), there are additional Cond Ops that are only used during graph construction and should be ignored during inference.

Adding a special op requires three steps:
1. [Registering the special op](#registering-interior-nodes). See below.  Note that reserved scopes must match up with those in the `registry` from `register.py`.
2. [Writing a conversion function](#adding-the-conversion-function). This is similar to the previous section, except the `node` argument will be replaced with an OrderedDict called `interiors`.  You can use this argument to extract NodeDefs of ops that are interior to the special op subgraph. See the `keras_conv2d` function in `register.py` for an example.
3. [Writing a conversion test](#adding-the-conversion-test).  This is done identically to the single op case

**Developer's Note:** If you're submitting the conversion in a PR, ensure that you run `make` (or just `make lint`) before committing your code.  This will auto-populate the above table with your new special op.


#### Registering the special op
Special ops are registered in `specops.yaml`.  This file has a two-level structure:
```yaml
scope:
    interiors:
        - Op1    # optional array of interior Op names
        - ...
    tf-name:
        name     # required value
    hyperlink:
        URL      # required value
```

The top level of the yaml entry must match with a [reserved scope](#reserved-scopes); these are scopes that capture the entire subgraph of the special op you wish to convert.  These also must match with a key of the `registry` in `register.py`, otherwise the converter will fail to identify and provide the correct NodeDefs for your conversion function.  Meanwhile, `tf-name` and `hyperlink` values help users identify which higher-level TensorFlow Op or layer have existing conversion implementations.
