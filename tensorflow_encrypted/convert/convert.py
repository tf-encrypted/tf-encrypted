from typing import Dict, Tuple, List, Any, Union, Optional

from ..player import Player
from ..protocol import Pond, get_protocol
from ..protocol.pond import TFEInputter
from ..config import Config, get_config


class Converter():

    def __init__(
        self,
        config: Optional[Config] = None,
        protocol: Optional[Pond] = None,
        player: Optional[Union[str, Player]] = None
    ) -> None:
        self.config = config if config is not None else get_config()
        self.protocol = protocol if protocol is not None else get_protocol()
        if player is None:
            self.model_provider = self.config.get_player('model-provider')
        elif isinstance(player, str):
            self.model_provider = self.config.get_player(player)
        else:
            self.model_provider = player
        self.outputs = {}

    def convert(
        self,
        graph_def: Any,
        register: Dict[str, Any],
        input_player: Union[str, Player],
        inputter_fn: Optional[Union[TFEInputter, List[TFEInputter]]]=None
    ) -> Any:
        if type(input_player) is str:
            input_player = get_config().get_player('input-provider')
        assert isinstance(input_player, Player)

        name_to_input_name, name_to_node = extract_graph_summary(graph_def)

        if inputter_fn is None:
            inputs = []
        elif type(inputter_fn) is list:
            inputs = inputter_fn
        else:
            inputs = [inputter_fn]
        inputs_iterable = enumerate(inputs)

        for output, inputs in name_to_input_name.items():
            node = name_to_node[output]

            if node.op == "Placeholder":
                try:
                    count, item = inputs_iterable.__next__()
                except StopIteration:
                    raise InvalidArgumentError("Not enough placeholders supplied")

                x = self.protocol.define_private_input(input_player, item)

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
