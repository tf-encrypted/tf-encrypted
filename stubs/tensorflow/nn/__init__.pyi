from typing import Any, Optional, Tuple


def ctc_beam_search_decoder(inputs: Any,
                            sequence_length: [int],
                            beam_width: int=...,
                            top_paths: int=...,
                            merge_repeated: bool=...
                            ) -> Tuple[Any, Any]:
    # ctc_beam_search_decoder implemented here:
    # https://github.com/tensorflow/tensorflow/blob/bb4e724f429ae5c9afad3a343dc1f483ecde1f74/tensorflow/python/ops/ctc_ops.py#L234
    ...


def bidirectional_dynamic_rnn(cell_fw: 'RNNCell',
                              cell_bw: 'RNNCell',
                              inputs: Any,
                              sequence_length: Any=...,
                              initial_state_fw: Any=...,
                              initial_state_bw: Any=...,
                              dtype: Any=...,
                              parallel_iterations: Optional[int]=...,
                              swap_memory: Optional[bool]=...,
                              time_major: Optional[bool]=...,
                              scope: Any=...
                              ) -> Tuple[Any, Any]:
    # bidirectional_dynamic_rnn implemented here:
    # https://github.com/tensorflow/tensorflow/blob/d8f9538ab48e3c677aaf532769d29bc29a05b76e/tensorflow/python/ops/rnn.py#L314
    # TODO: types
    # scope VariableScope
    ...


def ctc_loss(labels: 'SparseTensor',
             inputs: Any,
             sequence_length: Any,
             preprocess_collapse_repeated: bool=...,
             ctc_merge_repeated: bool=...,
             ignore_longer_outputs_than_inputs: bool=...,
             time_major: bool=...
             ) -> Any:
    # ctc_loss implemented here:
    # https://github.com/tensorflow/tensorflow/blob/bb4e724f429ae5c9afad3a343dc1f483ecde1f74/tensorflow/python/ops/ctc_ops.py#L32
    # TODO: types
    ...


def log_softmax(logits: Any,
                axis: Optional[int]=...,
                name: Optional[str]=...,
                dim: Optional[int]=...
                ) -> Any:
    # log_softmax implemented here:
    # https://github.com/tensorflow/tensorflow/blob/95c8f92947c6a420b70759d9d0d7825f2f5de368/tensorflow/python/ops/nn_ops.py#L1741
    # TODO: types
    # Returns Tensor
    ...
