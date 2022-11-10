Convert
===========
A Converter for constructing TF Encrypted models from TensorFlow/ONNX models.

## General workflow
1. Train a model in TensorFlow or Keras, or any other machine learning framework and convert it to a onnx model.
2. Optionally save model as file in tensorflow(e.g. as model.pb) or in onnx(e.g. as model.onnx).
3. Ensure that conversion is possible, which means either:
   1. All ONNX nodes in the model's graph have implementations registered in `nodes.py`.
   2. If using model for inference, only implement `forward` is enough. To realize model training, both `forward` and `backward` must be implemented.
4. Using a Converter's `convert` method to convert tensorflow/onnx model to TFE model.

## Adding a node
Suppose you wanted to add a node for the [`tf.squeeze`](https://www.tensorflow.org/api_docs/python/tf/squeeze) op from TensorFlow.
### Adding the node definition
First, you'll define a SqueezeNode class inherited from BaseNode.  
Its constructor accept a NodeProto `node`, a list of `inputs` ordered by `node.inputs` and a `model_provider` player.
`inputs` include this node's all inputs, if a input is other node's output, it will be substituted with a tensor shape.
`model_provider` is the player who will act as the model provider, or a string identifier for the player.
This class should implement encrypted `forward` and `backward`(optional) using the current TFE protocol, 
config and information specified by the `node` and `inputs`.
For example,
```python
class SqueezeNode(BaseNode):

    def __init__(self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player) -> None:
        super(SqueezeNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        if len(inputs) == 2:
            self.axes = list(inputs[1].numpy())
        else:
            self.axes = None
        
        output_shape = []
        for index, dim in enumerate(self._input_shapes[0]):
            if dim == 1 and (self.axes is None or index in self.axes):
                continue
            output_shape.append(dim)
        self._output_shapes.append(output_shape)
    
    def forward(self, x):
        return [tfe.squeeze(x[0], axis=self.axes)]
    
    def backward(self, d_y):
        return [], [tfe.reshape(d_y[0], self._input_shapes[0])]
```

Once you've implement this class, place it in the dictionary provided by the `nodes_dict` in `nodes.py`.
It should be keyed by the corresponding node name.

### Adding the node test
1. Write a function called `squeeze_model` accepting args `inputs` which is a list of tf.Tensor 
and returning a tensorflow model include `squeeze` op that will be tested
1. Write a function called `test_squeeze` that instantiates an example tensor and feeds it to a call to `_build_test`. Be mindful here: the name you pass as the first argument needs to match the name in your `*_model` functions exactly in order for the test to run successfully.

Here is an example for `tf.squeeze` in full:

```python
class TestConvert(unittest.TestCase):
    [...]
    def test_squeeze(self):
        test_input = [tf.random.uniform([1, 2, 3, 1])]
        self._build_test("squeeze", test_input)

[...]

def squeeze_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.squeeze(x, axis=[0, 3])
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model
```
