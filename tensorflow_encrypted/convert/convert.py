from typing import Dict, Tuple, List, Any
import tensorflow as tf

from ..layers import Layer
from ..protocol.pond import PondPrivateTensor
from ..io import InputProvider
from ..protocol.protocol import get_protocol
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

    def convert(self, graph_def: Any, input: InputProvider, register: Dict[str, Any]) -> Dict[str, Any]:
        name_to_input_name, name_to_node = extract_graph_summary(graph_def)

        if graph_def.node[0].op != "Placeholder":
            raise AttributeError("First node in graph must be placeholder for now")

        for output, inputs in name_to_input_name.items():
            node = name_to_node[output]

            # just take the input passed into this function for now
            if node.op == "Placeholder":
                x = self.protocol.define_private_input(input)

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
