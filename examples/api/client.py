from typing import List

import argparse
import json
import requests
import numpy as np


class Client():
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.base_url = "http://%s:%d" % (self.host, self.port)

    def send_random_inputs(self, shape: List[int]) -> str:
        data = np.random.standard_normal(shape).tolist()

        r = requests.post("%s/predict" % self.base_url, json=data)
        data = json.loads(r.text)

        return data['request_id']

    def poll(self, request_id: str) -> str:
        r = requests.post("%s/poll" % self.base_url, json=request_id)
        return json.loads(r.text)


def main(config):
    c = Client(config.host, config.port)

    if config.poll:
        request_id = config.request_id
        if request_id is None:
            print("you need to provide a request_id")
            return

        output = c.poll(request_id)
        print(output)
    else:
        shape = list(map(lambda x: int(x), config.shape.split(',')))
        request_id = c.send_random_inputs(shape)
        print(request_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", type=str, default="localhost", help="increase output verbosity")
    parser.add_argument(
        '--port', type=int, default=5000, help='port API endpoint for the prediction')
    parser.add_argument(
        '--poll', type=bool, default=False, help='Perform pooling')
    parser.add_argument(
        '--request_id', type=str, help='id used to poll an output')
    parser.add_argument(
        '--shape', type=str, default="1", help='shape as a string array: 1,2,3')
    args = parser.parse_args()

    main(args)
