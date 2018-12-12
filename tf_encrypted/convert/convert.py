from typing import Dict, List, Any, Union, Optional
from collections import OrderedDict

from ..player import Player
from ..protocol import Protocol, get_protocol
from ..protocol.pond import TFEInputter
from ..config import Config, get_config


special_ops = ['required_space_to_batch_paddings']


class Converter():

    def __init__(
        self,
        config: Optional[Config] = None,
        protocol: Optional[Protocol] = None,
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
        inputter_fn: Optional[Union[TFEInputter, List[TFEInputter]]] = None
    ) -> Any:
        if type(input_player) is str:
            input_player = get_config().get_player('input-provider')
        assert isinstance(input_player, Player)

        if inputter_fn is None:
            inputs = []
        elif type(inputter_fn) is list:
            inputs = inputter_fn
        else:
            inputs = [inputter_fn]
        inputs_iterable = enumerate(inputs)

        # Identify if there are special ops in pb file, e.g. required_space_to_batch_paddings
        # If yes, identify the inputs and outputs of this special ops.
        special_op_dict, special_op_inputs, special_op_outputs = find_special_ops(special_ops,
                                                                                  graph_def)

        # Create a dictionary excluding all the sub ops related to required_space_to_batch_paddings
        # Except the sub ops related to the input or output of this special ops.
        pb_trimmed = select_relevant_ops(special_ops,
                                         special_op_inputs,
                                         special_op_outputs,
                                         graph_def)

        node_list = pb_trimmed.values()

        # If the ops are not related to the special ops, use the existing approach to register them.
        # Otherwise for the special ops replace the output from the sub ops by the output from the
        # high level operation then register.
        for node in node_list:
            if node.name not in special_op_outputs:

                output = node_name(node.name)
                inputs = [node_name(x) for x in node.input]
                if node.op == "Placeholder":
                    try:
                        count, item = inputs_iterable.__next__()
                    except StopIteration:
                        raise InvalidArgumentError("Not enough placeholders supplied")

                    x = self.protocol.define_private_input(input_player, item)

                    self.outputs[output] = x
                    continue

                self.outputs[output] = register[node.op](self, node, inputs)
            else:
                # Register high level special operations
                for s in special_op_dict.keys():
                    input_list = special_op_dict[s]['inputs']
                    output_list = special_op_dict[s]['outputs']

                    # Handle edge cased if the ops return multiple outputs
                    outs = register[special_op_dict[s]['op']](self, node, input_list)

                    for i, x in enumerate(outs):
                        self.outputs[output_list[i]] = x

        return self.outputs[graph_def.node[-1].name]


def node_name(n: str) -> str:
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]


class InvalidArgumentError(Exception):
    pass


def find_leaves(special_ops, graph_def, lookahead):
    potential_leaf_ops = []
    graph_plusminus_special = []

    for node in graph_def.node:
        if not lookahead:
            if special_ops in node.name:
                potential_leaf_ops += node.input
            else:
                graph_plusminus_special.append(node.name)
        else:
            if special_ops not in node.name.split('/'):
                potential_leaf_ops += node.input
            else:
                graph_plusminus_special.append(node.name)

    potential_leaf_ops_unique = OrderedDict.fromkeys(potential_leaf_ops)
    uniques = OrderedDict.fromkeys(graph_plusminus_special)

    def gen_leaf_keys():
        for x in uniques:
            if x in potential_leaf_ops_unique:
                yield x

    final_leaves = OrderedDict.fromkeys(key for key in gen_leaf_keys())
    final_list = list(final_leaves.keys())
    return final_list


def special_ops_namespace(special_ops_name: str, graph_def: Any):

    special_op_name_space = set()
    for n in graph_def.node:
        name = n.name.split('/')

        special_op_idx = 0

        for j in range(len(name)):
            if special_ops_name in name[j]:
                special_op_name_space.add('/'.join(name[:special_op_idx + 1]))

    return list(special_op_name_space)


def find_special_ops(special_ops_list: list, graph_def: Any):

    special_ops_dict = OrderedDict()
    all_special_op_inputs = []
    all_special_op_outputs = []

    for s in special_ops_list:

        special_ops_namespace_list = special_ops_namespace(s, graph_def)

        for n in special_ops_namespace_list:

            special_ops_dict[n] = OrderedDict()
            special_ops_dict[n]['op'] = s

            inputs = find_leaves(n, graph_def, lookahead=False)
            special_ops_dict[n]['inputs'] = inputs
            all_special_op_inputs += inputs

            outputs = find_leaves(n, graph_def, lookahead=True)
            special_ops_dict[n]['outputs'] = outputs
            all_special_op_outputs += outputs

    return special_ops_dict, all_special_op_inputs, all_special_op_outputs


def select_relevant_ops(special_ops: list,
                        all_special_op_inputs: list,
                        all_special_op_outputs: list,
                        graph_def: Any):

    trimmed_graph = OrderedDict()

    for i in range(len(special_ops)):
        for n in graph_def.node:
            if special_ops[i] in n.name:
                if n.name in all_special_op_inputs or n.name in all_special_op_outputs:
                    trimmed_graph[n.name] = n
            else:
                trimmed_graph[n.name] = n

    return trimmed_graph
