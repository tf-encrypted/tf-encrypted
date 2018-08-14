from typing import Dict, Tuple, List, Any
from .register import register


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


def convert(graph_def: Any) -> Dict[str, Any]:
    name_to_input_name, name_to_node = extract_graph_summary(graph_def)

    output_vars: Dict[str, Any] = {}

    for output, inputs in name_to_input_name.items():
        node = name_to_node[output]

        reg = register()
        output_vars[output] = reg[node.op](node, inputs, output_vars)

    return output_vars
