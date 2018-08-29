from typing import Any, Optional, Dict, List


def import_meta_graph(path: str) -> Any:
    pass

# Saver class defined here
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/training/saver.py#L1075


class Server:
    def __init__(self, clusterSpec: 'ClusterSpec',
                 job_name: str, task_index: int) -> None:
        ...

    @property
    def target(self)-> str:
        ...


class ClusterSpec:
    def __init__(self, cluster: Dict[str, List[str]]) -> None:
        ...


class Saver:
    def restore(self, session: Any, path: str) -> None:
        pass

    # TODO: parameter types:
    #    sess is of type Session
    #    global_step is of type Tensor or integer
    def save(self,
             sess: Any,
             save_path: str,
             global_step: Any = None,
             latest_filename: Optional[str] = None,
             meta_graph_suffix: str = "meta",
             write_meta_graph: bool = True,
             write_state: bool = True,
             strip_default_attrs: bool = False
             ) -> Optional[str]:
        ...
