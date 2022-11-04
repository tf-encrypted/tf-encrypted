"""An example of performing secure inference with model converted from a plain TF model.
"""
import argparse
import os
import unittest

import _pickle as pickle
import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.convert import convert
from tf_encrypted.performance import Performance
from tf_encrypted.protocol import ABY3  # noqa:F403,F401
from tf_encrypted.protocol import Pond  # noqa:F403,F401
from tf_encrypted.protocol import SecureNN  # noqa:F403,F401
from tf_encrypted.utils import print_banner

directory = os.path.dirname(os.path.abspath(__file__))


class TestConvertInference(unittest.TestCase):
    def load_imagenet_images(self, preprocess_input=None):
        with open(os.path.join(directory, "n02109961_36_enc.pkl"), "rb") as ff:
            images = np.array(pickle.load(ff))
        if preprocess_input is not None:
            images = preprocess_input(images)
        images = tf.convert_to_tensor(images)
        return images

    def tfe_model_predict(self, tf_model, tf_data, decode_pre):

        model_name = tf_model.name.capitalize()
        batch_size = tf_data.shape[0]
        tfe_data = tfe.define_private_input("prediction-client", lambda: tf_data)

        Performance.time_log("Model conversion")
        c = convert.Converter(config=config)
        tfe_model = c.convert(
            tf_model,
            list(tf_data.shape),
            model_provider=config.get_player("weights-provider"),
        )
        Performance.time_log("Model conversion")

        Performance.time_log(model_name + " Prediction 1st run")
        preds = tfe_model.predict(tfe_data, batch_size=batch_size)
        Performance.time_log(model_name + " Prediction 1st run")

        Performance.time_log(model_name + " Prediction 2nd run")
        preds = tfe_model.predict(tfe_data, batch_size=batch_size)
        Performance.time_log(model_name + " Prediction 2nd run")

        print("Predicted:", decode_pre(preds, top=10)[0])

    def test_vgg19(self):
        from tensorflow.keras.applications.vgg19 import decode_predictions
        from tensorflow.keras.applications.vgg19 import preprocess_input

        images = self.load_imagenet_images(preprocess_input)
        model = tf.keras.applications.VGG19(weights="imagenet")
        preds = model.predict(images)
        print_banner(
            "Convert Plain "
            + model.name.capitalize()
            + " to TFE "
            + model.name.capitalize()
        )
        print("Plain model predicted:", decode_predictions(preds, top=10)[0])
        self.tfe_model_predict(model, images, decode_predictions)

    def test_densenet121(self):
        from tensorflow.keras.applications.densenet import decode_predictions
        from tensorflow.keras.applications.densenet import preprocess_input

        images = self.load_imagenet_images(preprocess_input)
        model = tf.keras.applications.DenseNet121(weights="imagenet")
        preds = model.predict(images)
        print_banner(
            "Convert Plain "
            + model.name.capitalize()
            + " to TFE "
            + model.name.capitalize()
        )
        print("Plain model predicted:", decode_predictions(preds, top=10)[0])
        self.tfe_model_predict(model, images, decode_predictions)

    def test_resnet50(self):
        from tensorflow.keras.applications.resnet_v2 import decode_predictions
        from tensorflow.keras.applications.resnet_v2 import preprocess_input

        images = self.load_imagenet_images(preprocess_input)
        model = tf.keras.applications.ResNet50V2(weights="imagenet")
        preds = model.predict(images)
        print_banner(
            "Convert Plain "
            + model.name.capitalize()
            + " to TFE "
            + model.name.capitalize()
        )
        print("Plain model predicted:", decode_predictions(preds, top=10)[0])
        self.tfe_model_predict(model, images, decode_predictions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a TF Encrypted model inference")
    parser.add_argument(
        "model_name",
        metavar="MODEL NAME",
        type=str,
        help="name of model to be trained",
    )
    parser.add_argument(
        "--protocol",
        metavar="PROTOCOL",
        type=str,
        default="ABY3",
        help="MPC protocol TF Encrypted used",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        default="./config.json",
        help="path to configuration file",
    )
    args = parser.parse_args()

    # set tfe config
    if args.config != "local":
        # config file was specified
        config_file = args.config
        config = tfe.RemoteConfig.load(config_file)
        config.connect_servers()
        tfe.set_config(config)
    else:
        # Always best practice to preset all players to avoid invalid device errors
        config = tfe.LocalConfig(
            player_names=[
                "server0",
                "server1",
                "server2",
                "training-client",
                "prediction-client",
            ]
        )
        tfe.set_config(config)

    # set tfe protocol
    tfe.set_protocol(globals()[args.protocol]())

    test = "test_" + args.model_name
    singletest = unittest.TestSuite()
    singletest.addTest(TestConvertInference(test))
    unittest.TextTestRunner().run(singletest)
