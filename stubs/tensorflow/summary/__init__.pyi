from typing import Optional
import tensorflow as tf


class FileWriter:
    def __init__(self, tag: str, graph: Optional[tf.Graph]) -> None:
        ...

    def add_run_metadata(self,
                         metadata: tf.RunMetadata,
                         session_tag: str) -> None:
        ...

    def close(self) -> None:
        ...
