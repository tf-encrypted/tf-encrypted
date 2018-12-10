from typing import Dict, List, Any, Union, Optional
from collections import OrderedDict

from ..player import Player
from ..protocol import Protocol, get_protocol
from ..protocol.pond import TFEInputter
from ..config import Config, get_config

special_ops = ['required_space_to_batch_paddings']
special_op_two_outputs = ['required_space_to_batch_paddings']

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
        if are_there_special_ops(special_ops[0], graph_def):
            input_special_op = find_inputs(special_ops[0], graph_def)
            output_special_op = find_outputs(special_ops[0], graph_def)

            # Create a dictionary excluding all the sub ops related to required_space_to_batch_paddings
            # Except the sub ops related to the input or output of this special ops.
            pb_trimmed = OrderedDict()

            for n in graph_def.node:
                if special_ops[0] in n.name: 
                    if  n.name in input_special_op or n.name in output_special_op:
                        pb_trimmed[n.name] = n
                else:
                    pb_trimmed[n.name] = n

            node_list = pb_trimmed.values()


            # If the ops are not related to the special ops, use the existing approach to regist the ops. 
            # Otherwise for the special ops replace the output from the sub ops by the output from the 
            # high level operation then register. 
            for node in node_list:
                if node.name not in output_special_op:

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
                    # Handle edge cased if the ops return two outputs
                    if special_ops[0] in special_op_two_outputs :
                        a, b = register[special_ops[0]](self, node, input_special_op)
                        self.outputs[output_special_op[0]] = a
                        self.outputs[output_special_op[1]] = b
                    else:
                        self.outputs[node.name] = register[special_ops[0]](self, node, inputs)

        else:
            for node in graph_def.node:
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

        return self.outputs[graph_def.node[-1].name]


def node_name(n: str) -> str:
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]


class InvalidArgumentError(Exception):
    pass


def find_inputs(special_ops, graph_def):
    potential_ops_input = []
    graph_nodes_not_in_special_ops = []

    for node in graph_def.node:
        if special_ops in node.name:
            potential_ops_input += node.input
        else:
            graph_nodes_not_in_special_ops.append(node.name)

    potential_ops_input_unique = OrderedDict.fromkeys(potential_ops_input)
    graph_nodes_not_in_special_ops_unique = OrderedDict.fromkeys(graph_nodes_not_in_special_ops)

    final_input = OrderedDict.fromkeys(x for x in graph_nodes_not_in_special_ops_unique if x in potential_ops_input_unique)
    final_list = list(final_input.keys())
    return final_list


def find_outputs(special_ops, graph_def):
    potential_ops_output = []
    graph_nodes_in_special_ops = []

    for node in graph_def.node:
        if special_ops not in node.name.split('/'):
            potential_ops_output += node.input
        else:
            graph_nodes_in_special_ops.append(node.name)

    potential_ops_output_unique = OrderedDict.fromkeys(potential_ops_output)
    graph_nodes_in_special_ops_unique = OrderedDict.fromkeys(graph_nodes_in_special_ops)

    final_output = OrderedDict.fromkeys(x for x in graph_nodes_in_special_ops_unique if x in potential_ops_output_unique)
    final_list = list(final_output.keys())
    return final_list

def are_there_special_ops(special_op, graph_def):
    for n in graph_def.node:
        if special_op in n.name:
            return True
        
    return False