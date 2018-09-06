from typing import Dict, Tuple, List, Any, Union
import tensorflow as tf
from collections import Iterable

from ..io import InputProvider
from ..player import Player
from ..protocol.pond import Pond
from ..config import Config


class ConvertInputProvider(InputProvider):
    input: tf.Tensor

    def __init__(self, player: Player, input: tf.Tensor) -> None:
        self.input = input
        self.player = player

    def provide_input(self) -> tf.Tensor:
        return tf.identity(self.input)


class Converter():
    config: Config
    protocol: Pond
    outputs: Dict[str, Any] = {}
    weights_provider: Player

    def __init__(self, config: Config, protocol: Pond,
                 weights_provider: Player) -> None:
        self.config = config
        self.protocol = protocol
        self.weights_provider = weights_provider

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
        name: str = node.name
        n = node_name(name)
        name_to_node[n] = node
        name_to_input_name[n] = [node_name(x) for x in node.input]

    return name_to_input_name, name_to_node


class InvalidArgumentError(Exception):
    pass
