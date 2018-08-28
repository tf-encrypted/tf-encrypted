
# Overview

- two compute servers
- a crypto producer
- a model trainer
- a prediction client

# Running

## Preparations

Although the usual MNIST dataset is used it has to be converted to the TFRecord format before it can be used by this example. To do so simply run
```shell
python3 download.py
```
from the `mnist` directory; this will place the converted files in `data` subdirectory.


## Using LocalConfig

When using this configuration simply run
```shell
python3 run.py
```
from the `mnist` directory.


## Using RemoteConfig

To launch the required servers for this configuration run
```shell
python3 run.py server0
python3 run.py server1
python3 run.py crypto_producer
python3 run.py model_trainer
python3 run.py prediction_client
```
in *five different processes*. Then, in yet another separate process run
```shell
python3 run.py
```
to kick things off.