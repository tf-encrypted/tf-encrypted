from typing import Dict, Tuple, List, Any, Union
import tensorflow as tf
from collections import Iterable
import numpy as np

from ..io import InputProvider
from ..player import Player
from ..protocol.pond import Pond
from ..config import Config, get_config


class ConvertInputProvider(InputProvider):

    def __init__(self, player: Union[str, Player], input: Union[np.ndarray, tf.Tensor]) -> None:
        self.input = input
        if isinstance(player, str):
            self.player = get_config().get_player(player)
        else:
            self.player = player

    def provide_input(self) -> tf.Tensor:
        if isinstance(self.input, tf.Tensor):
            return self.input
        return tf.constant(self.input)


class Converter():

    def __init__(self, config: Config, protocol: Pond,
                 player: Union[str, Player]) -> None:
        self.config = config if config is not None else get_config()
        self.protocol = protocol
        if isinstance(player, str):
            self.model_provider = self.config.get_player(player)
        else:
            self.model_provider = player
        self.outputs = {}

    def convert(self, graph_def: Any, input: Union[List[InputProvider], InputProvider],
                register: Dict[str, Any]) -> Any:
        name_to_input_name, name_to_node = extract_graph_summary(graph_def)

        if isinstance(input, InputProvider):
            i = [input]
        elif isinstance(input, Iterable):
            i = input
        else:
            i = []

        iter = enumerate(i)

        for output, inputs in name_to_input_name.items():
            node = name_to_node[output]

            if node.op == "Placeholder":
                try:
                    count, item = iter.__next__()
                except StopIteration:
                    raise InvalidArgumentError("Not enough placeholders supplied")

                x = self.protocol.define_private_input(item)

                self.outputs[output] = x
                continue

            self.outputs[output] = register[node.op](self, node, inputs)

        return self.outputs[graph_def.node[-1].name]


def node_name(n: str) -> str:
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]


def extract_graph_summary(graph_def: Any) -> Tuple[Dict[str, List[str]],
                                                   Dict[str, Any]]:
    """Extracts useful information from the graph and returns them."""
    name_to_input_name = {}  # Keyed by the dest node name.
    name_to_node = {}  # Keyed by node name.

    for node in graph_def.node:
        name = node.name
        n = node_name(name)
        name_to_node[n] = node
        name_to_input_name[n] = [node_name(x) for x in node.input]

    return name_to_input_name, name_to_node


class InvalidArgumentError(Exception):
    pass
