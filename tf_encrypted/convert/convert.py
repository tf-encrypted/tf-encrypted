import re
from typing import Dict, List, Any, Union, Optional
from collections import OrderedDict

from ..player import Player
from ..protocol import Protocol, get_protocol
from ..protocol.pond import TFEInputter
from ..config import Config, get_config
from .register import REGISTERED_SPECOPS


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
        specop_dict, specop_inputs, specop_outputs = find_specops(graph_def,
                                                                  graph_def.node[-1].name)

        # Create a dictionary excluding all the sub ops related to required_space_to_batch_paddings
        # Except the sub ops related to the input or output of this special ops.
        pb_trimmed = select_relevant_ops(specop_inputs,
                                         specop_outputs,
                                         graph_def)
        node_list = pb_trimmed.values()

        # If the ops are not related to the special ops, use the existing approach to register them.
        # Otherwise for the special ops replace the output from the sub ops by the output from the
        # high level operation then register.
        for node in node_list:
            if node.name not in specop_outputs:

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
                for s in specop_dict:
                    input_list = specop_dict[s]['inputs']
                    output_list = specop_dict[s]['outputs']

                    # Handle edge cases if the ops return multiple outputs
                    op_handler = register[specop_dict[s]['op']]

                    nodes = specop_dict[s]['interiors']
                    if not nodes:
                        nodes = node
                    outs = op_handler(self, nodes, input_list)
                    if isinstance(outs, list) or isinstance(outs, tuple):
                        for i, x in enumerate(outs):
                            self.outputs[output_list[i]] = x
                    else:
                        self.outputs[output_list[0]] = outs

        return self.outputs[graph_def.node[-1].name]


def select_relevant_ops(all_specop_inputs, all_specop_outputs, graph_def):
    """
    Prune out subgraphs that have been identified as special ops from the
    graph_def, and return the pruned graph_def.
    """

    trimmed_graph = OrderedDict()
    for n in graph_def.node:
        for op in REGISTERED_SPECOPS:

            matched = False
            # If the node falls under a specop scope,
            # only add if it's an input or output to the specop.
            if match_numbered_scope(op, n.name, return_group=False):
                matched = True
                is_input = n.name in all_specop_inputs
                is_output = n.name in all_specop_outputs
                if is_input or is_output:
                    trimmed_graph[n.name] = n
                break
        # Otherwise, just add it
        if not matched:
            trimmed_graph[n.name] = n

    return trimmed_graph


def find_specops(graph_def, output_name):
    """
    For special ops defined in REGISTERED_SPECOPS, assemble necessary info to
    convert them and place into the specops_dict. Also returns these ops'
    collective inputs and outputs together in separate arrays.
    """

    specops_dict = OrderedDict()
    all_specop_inputs = []
    all_specop_outputs = []
    namespace = specop_namespace(graph_def)

    for scope, subscope_map in namespace.items():
        specops_dict[scope] = {}
        specops_dict[scope]['op'] = specop_from_numberedscope(scope)
        specops_dict[scope]['interiors'] = get_interiors(scope, namespace)
        inputs, outputs = find_leaves(scope, subscope_map)

        specops_dict[scope]['inputs'] = inputs
        all_specop_inputs += inputs
        # if no outputs found assume output to model is the special op output
        if len(outputs) == 0:
            outputs = [output_name]
        specops_dict[scope]['outputs'] = outputs
        all_specop_outputs += outputs

    return specops_dict, all_specop_inputs, all_specop_outputs


def specop_namespace(graph_def):
    """
    Gathers all subgraphs corresponding to registered special ops.

    For each specop scope matching `{specop}_[0-9]+/`, assemble all ops
    falling under that scope into a map of op name to op node, and add the
    map to the namespace keyed by its scope.

    Returns an OrderedDict[scope --> ops_map],
    where ops_map is an OrderedDict[NodeDef.name --> NodeDef].
    """

    namespace = OrderedDict()
    for node in graph_def.node:
        for specop in REGISTERED_SPECOPS:
            node_name = node.name
            this_scope = match_numbered_scope(specop, node_name)
            if this_scope is None:
                continue
            if this_scope not in namespace:
                namespace[this_scope] = OrderedDict()
            namespace[this_scope][node_name] = node

    return namespace


def get_interiors(specop_scope, subscope_map):
    """
    An interior (op/node) of a subgraph is an op that is neither
    an input or an output for all ops outside of that subgraph.

    Given a specop_scope, look for registered interior ops in the
    corresponding value of subscope_map and collect their NodeDefs into a
    OrderedDict keyed by the registered interior op name.
    """
    specop = specop_from_numberedscope(specop_scope)
    if specop is None:
        return None
    interior_names = REGISTERED_SPECOPS[specop]["interiors"]
    interiors = OrderedDict()
    if interior_names is None:
        return interiors
    subscope_ops_map = subscope_map[specop_scope]
    for op in interior_names:
        for node_name in subscope_ops_map:
            if match_numbered_leaf(op, node_name) is not None:
                interiors[op] = subscope_ops_map[node_name]
    return interiors


def find_leaves(scope, subscope_map):
    """
    Assembles input and output leaf nodes of the subgraph represented by subscope_map.
    """

    input_leaves = []
    output_leaves = list(subscope_map.keys())
    for name, node in subscope_map.items():
        for input in node.input:
            if match_numbered_scope(scope, input) is None:
                input_leaves.append(input)
            if input in output_leaves:
                output_leaves.remove(input)
    seen = set()
    adder = seen.add
    input_leaves = [x for x in input_leaves if not (x in seen or adder(x))]

    return input_leaves, output_leaves


def match_numbered_scope(specop, search_string, return_group=True):
    """
    Find a numbered scope matching a specop from REGISTERED_SPECOPS,
    and return it if found

    Example: 'conv2d' will match '...conv2d_345/...' and return 'conv2d_345'.
    """
    expr = '({0})/|({0}_[0-9]+)/'.format(specop)
    match = re.search(expr, search_string)
    if match is not None:
        if not return_group:
            return match
        return match.group(1)


def match_numbered_leaf(leaf_to_match, search_string):
    """
    Find a numbered leaf matching a tf.Operation and return it if found.

    Example: 'Conv2D' will match '.../Conv2D_5' and return 'Conv2D_5'
    """
    expr = '/({0})|/({0}_[0-9]+)'.format(leaf_to_match)
    match = re.search(expr, search_string)
    if match is not None:
        return match.group(1)


def specop_from_numberedscope(scope):
    """
    An inverse for `match_numbered_scope`.

    Example: 'conv2d_4' will produce 'conv2d'.
    """
    expr = '[_a-zA-Z0-9]+(?=_[0-9]+)'
    match = re.search(expr, scope)
    if match is not None:
        return match.group(0)
    else:
        return scope


def node_name(n: str) -> str:
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]


class InvalidArgumentError(Exception):
    pass
