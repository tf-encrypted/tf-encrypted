import grpc
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def cb(result_future):
    exception = result_future.exception()
    if exception:
        print(exception)
        return

        response = np.array(result_future.result().outputs['reg_out'].float_val)
        prediction = np.argmax(response)
        print(prediction)


input_data = np.array([.1, -.1, .2, -.2]).reshape(2, 2)

channel = grpc.insecure_channel('0.0.0.0:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'logreg'
request.model_spec.signature_name = 'predict_reg'
request.inputs['data'].CopyFrom(
    tf.contrib.util.make_tensor_proto(input_data, shape=[2, 2])
)

result_future = stub.Predict.future(request, 5.0)  # 5 seconds
result_future.add_done_callback(cb)
