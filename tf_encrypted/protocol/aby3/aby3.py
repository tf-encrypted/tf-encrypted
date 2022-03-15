# flake8: noqa
# pylint: disable=all
"""
Implementation of the ABY3 framework.
"""
from __future__ import absolute_import

import abc
import sys
from functools import reduce
from functools import wraps
from math import ceil
from math import log2
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf

from ...config import get_config
from ...operations import secure_random as crypto
from ...player import Player
from ...tensor import fixed64
from ...tensor import fixed64_ni
from ...tensor.boolfactory import bool_factory
from ...tensor.factory import AbstractConstant
from ...tensor.factory import AbstractFactory
from ...tensor.factory import AbstractTensor
from ...tensor.fixed import FixedpointConfig
from ...tensor.fixed import _validate_fixedpoint_config
from ...tensor.helpers import inverse
from ...tensor.native import native_factory
from ..protocol import Protocol
from ..protocol import memoize

from .aby3_tensors import *

TFEInputter = Callable[[], Union[List[tf.Tensor], tf.Tensor]]
TF_NATIVE_TYPES = [tf.bool, tf.int8, tf.int16, tf.int32, tf.int64]

_THISMODULE = sys.modules[__name__]


class ABY3(Protocol):
    """ABY3 framework."""

    def __init__(
        self,
        server_0=None,
        server_1=None,
        server_2=None,
    ):
        self._initializers = list()
        config = get_config()
        self.servers = [None, None, None]
        self.servers[0] = config.get_player(server_0 if server_0 else "server0")
        self.servers[1] = config.get_player(server_1 if server_1 else "server1")
        self.servers[2] = config.get_player(server_2 if server_2 else "server2")

        self.fixedpoint_config = fixed64_ni

        self.factories = {
            0 : bool_factory(),
            1 : native_factory(tf.int8),
            2 : native_factory(tf.int8),
            4 : native_factory(tf.int8),
            8 : native_factory(tf.int8),
            16: native_factory(tf.int16),
            32: native_factory(tf.int32),
            64: native_factory(tf.int64)
        }
        self.factories.update({
            tf.bool: self.factories[0],
            tf.int8: self.factories[8],
            tf.int16: self.factories[16],
            tf.int32: self.factories[32],
            tf.int64: self.factories[64]
        })
        self.default_nbits = 64
        self.default_factory = self.factories[self.default_nbits]

        self.pairwise_keys, self.pairwise_nonces = self.setup_pairwise_randomness()
        self.b2a_keys_1, self.b2a_keys_2, self.b2a_nonce = self.setup_b2a_generator()


    def setup_pairwise_randomness(self):
        """
    Initial setup for pairwise randomness: Every two parties hold a shared key.
    """

        keys = [[None, None], [None, None], [None, None]]

        if crypto.supports_seeded_randomness():
            with tf.device(self.servers[0].device_name):
                seed_0 = crypto.secure_seed()
            with tf.device(self.servers[1].device_name):
                seed_1 = crypto.secure_seed()
            with tf.device(self.servers[2].device_name):
                seed_2 = crypto.secure_seed()
        else:
            # Shape and Type are kept consistent with the 'secure_seed' version
            with tf.device(self.servers[0].device_name):
                seed_0 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[1].device_name):
                seed_1 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[2].device_name):
                seed_2 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )

        # Replicated keys
        # NOTE: The following `with` contexts do NOT have any impact for the Python-only operations.
        #       We use them here only for indicating "which server has which seed".
        #       In other words, `keys[0][1] = seed_1` only stores the TF graph node `seed_1` in the
        #       Python list `keys`, but does NOT actually "send" `seed_1` to server 0, which only happens
        #       when a future TF operation on server 0 uses `keys[0][1]`.
        # The same NOTE applies to other places where we use Python list to store TF graph nodes in the
        # `with` context.
        with tf.device(self.servers[0].device_name):
            keys[0][0] = seed_0
            keys[0][1] = seed_1
        with tf.device(self.servers[1].device_name):
            keys[1][0] = seed_1
            keys[1][1] = seed_2
        with tf.device(self.servers[2].device_name):
            keys[2][0] = seed_2
            keys[2][1] = seed_0

        # nonces[0] for server 0 and 1, nonces[1] for server 1 and 2, nonces[2] for server 2 and 0
        nonces = np.array([0, 0, 0], dtype=np.int)

        return keys, nonces

    def setup_b2a_generator(self):
        """
    Initial setup for generating shares during the conversion
    from boolean sharing to arithmetic sharing
    """

        # Type 1: Server 0 and 1 hold three keys, while server 2 holds two
        b2a_keys_1 = [[None, None, None], [None, None, None], [None, None, None]]

        if crypto.supports_seeded_randomness():
            with tf.device(self.servers[0].device_name):
                seed_0 = crypto.secure_seed()
            with tf.device(self.servers[1].device_name):
                seed_1 = crypto.secure_seed()
            with tf.device(self.servers[2].device_name):
                seed_2 = crypto.secure_seed()
        else:
            # Shape and Type are kept consistent with the 'secure_seed' version
            with tf.device(self.servers[0].device_name):
                seed_0 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[1].device_name):
                seed_1 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[2].device_name):
                seed_2 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )

        with tf.device(self.servers[0].device_name):
            b2a_keys_1[0][0] = seed_0
            b2a_keys_1[0][1] = seed_1
            b2a_keys_1[0][2] = seed_2
        with tf.device(self.servers[1].device_name):
            b2a_keys_1[1][0] = seed_0
            b2a_keys_1[1][1] = seed_1
            b2a_keys_1[1][2] = seed_2
        with tf.device(self.servers[2].device_name):
            b2a_keys_1[2][0] = seed_0
            b2a_keys_1[2][2] = seed_2

        # Type 2: Server 1 and 2 hold three keys, while server 0 holds two
        b2a_keys_2 = [[None, None, None], [None, None, None], [None, None, None]]

        if crypto.supports_seeded_randomness():
            with tf.device(self.servers[0].device_name):
                seed_0 = crypto.secure_seed()
            with tf.device(self.servers[1].device_name):
                seed_1 = crypto.secure_seed()
            with tf.device(self.servers[2].device_name):
                seed_2 = crypto.secure_seed()
        else:
            # Shape and Type are kept consistent with the 'secure_seed' version
            with tf.device(self.servers[0].device_name):
                seed_0 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[1].device_name):
                seed_1 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )
            with tf.device(self.servers[2].device_name):
                seed_2 = tf.random.uniform(
                    [2], minval=tf.int64.min, maxval=tf.int64.max, dtype=tf.int64
                )

        with tf.device(self.servers[0].device_name):
            b2a_keys_2[0][0] = seed_0
            b2a_keys_2[0][1] = seed_1
        with tf.device(self.servers[1].device_name):
            b2a_keys_2[1][0] = seed_0
            b2a_keys_2[1][1] = seed_1
            b2a_keys_2[1][2] = seed_2
        with tf.device(self.servers[2].device_name):
            b2a_keys_2[2][0] = seed_0
            b2a_keys_2[2][1] = seed_1
            b2a_keys_2[2][2] = seed_2

        b2a_nonce = 0
        return b2a_keys_1, b2a_keys_2, b2a_nonce

    def define_constant(
        self,
        value: Union[np.ndarray, int, float],
        apply_scaling: bool = True,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """
    Define a constant to use in computation.

    .. code-block:: python

        x = prot.define_constant(np.array([1,2,3,4]), apply_scaling=False)

    :See: tf.constant

    :param bool apply_scaling: Whether or not to scale the value.
    :param str name: What name to give to this node in the graph.
    :param AbstractFactory factory: Which tensor type to represent this value with.
    """
        assert isinstance(value, (np.ndarray, int, float))

        if isinstance(value, (int, float)):
            value = np.array([value])

        factory = factory or self.default_factory

        value = self._encode(value, apply_scaling)
        with tf.name_scope("constant{}".format("-" + name if name else "")):
            with tf.device(self.servers[0].device_name):
                x_on_0 = factory.constant(value)

            with tf.device(self.servers[1].device_name):
                x_on_1 = factory.constant(value)

            with tf.device(self.servers[2].device_name):
                x_on_2 = factory.constant(value)

        return ABY3Constant(self, [x_on_0, x_on_1, x_on_2], apply_scaling)

    def define_private_variable(
        self,
        initial_value,
        apply_scaling: bool = True,
        share_type=ShareType.ARITHMETIC,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """
    Define a private variable.

    This will take the passed value and construct shares that will be split up
    between those involved in the computation.

    For example, in a three party replicated sharing, this will split the value into
    three shares and transfer two shares to each party in a secure manner.

    :see tf.Variable

    :param Union[np.ndarray,tf.Tensor,ABY3PublicTensor] initial_value: The initial value.
    :param bool apply_scaling: Whether or not to scale the value.
    :param str name: What name to give to this node in the graph.
    :param AbstractFactory factory: Which tensor type to represent this value with.
    """
        init_val_types = (np.ndarray, tf.Tensor, ABY3PrivateTensor)
        assert isinstance(initial_value, init_val_types), type(initial_value)

        factory = factory or self.default_factory
        suffix = "-" + name if name else ""

        with tf.name_scope("private-var{}".format(suffix)):

            if isinstance(initial_value, np.ndarray):
                initial_value = self._encode(initial_value, apply_scaling)
                v = factory.tensor(initial_value)
                shares = self._share(v, share_type=share_type)

            elif isinstance(initial_value, tf.Tensor):
                initial_value = self._encode(initial_value, apply_scaling)
                v = factory.tensor(initial_value)
                shares = self._share(v, share_type=share_type)

            elif isinstance(initial_value, ABY3PrivateTensor):
                shares = initial_value.unwrapped

            else:
                raise TypeError(
                    ("Don't know how to turn {} " "into private variable").format(
                        type(initial_value)
                    )
                )

            # The backing factory for the shares might have changed after the sharing step
            factory = shares[0][0].factory
            x = [[None, None], [None, None], [None, None]]
            with tf.device(self.servers[0].device_name):
                x[0][0] = factory.variable(shares[0][0])
                x[0][1] = factory.variable(shares[0][1])

            with tf.device(self.servers[1].device_name):
                x[1][0] = factory.variable(shares[1][0])
                x[1][1] = factory.variable(shares[1][1])

            with tf.device(self.servers[2].device_name):
                x[2][0] = factory.variable(shares[2][0])
                x[2][1] = factory.variable(shares[2][1])

        x = ABY3PrivateVariable(self, x, apply_scaling, share_type)
        return x

    def local_computation(
            self,
            player_name=None,
            **kwargs
    ):
        """Annotate a function `compute_func` for local computation.

        This decorator can be used to pin a function's code to a specific player's
        device for remote execution.  This is useful when defining player-specific
        handlers for e.g. providing model weights, or input and output tensors.

        The decorator can handle global functions, normal object methods, or
        classmethods. If wrapping a method, it's presumed that the method's object
        has an attribute named `player_name`, or that the user will provide the
        `player_name` later on as a kwarg to the `compute_func`.

        Example:
            ```
            @tfe.local_computation('input-provider')
            def provide_input():
                return tf.random.normal((3, 3))

            @tfe.local_computation
            def receive_output(logits):
                return tf.print(tf.argmax(logits, axis=-1))

            x = provide_input()
            y = model(x)
            receive_op = receive_output(y, player_name='output-receiver')
            with tfe.Session():
                sess.run(receive_op)
            ```

        Arguments:
            player_name: Name of the player who should execute the function.
            kwargs: Keyword arguments to use when encoding or encrypting
                inputs/outputs to compute_func: see tfe.define_local_computation for
                details.

        Returns:
            The compute_func, but decorated for remote execution.
        """
        if callable(player_name):
            # The user has called us as a standard decorator:
            #
            # @tfe.local_computation
            # def provide_input():
            #   return tf.zeros((2, 2))
            actual_compute_func = player_name
            player_name = None
        else:
            # The user has called us as a function, maybe with non-default args:
            #
            # @tfe.local_computation('input-provider')
            # def provide_input():
            #   return tf.zeros((2, 2))
            actual_compute_func = None

        def decorator(compute_func):

            @wraps(compute_func)
            def compute_func_wrapper(*compute_func_args, **compute_func_kwargs):

                # Assumer user has passed player_name to decorator. If not, try to recover.
                actual_player_name = player_name
                if actual_player_name is None:
                    # Maybe user has passed player_name to compute_func as a kwarg
                    actual_player_name = compute_func_kwargs.get("player_name", None)
                if actual_player_name is None:
                    # Assume compute_func is a method and its instance has some attribute
                    # 'player_name'
                    if compute_func_args:
                        parent_instance = compute_func_args[0]
                        actual_player_name = getattr(parent_instance, 'player_name', None)
                if actual_player_name is None:
                    # Fallback to error
                    raise ValueError("'player_name' not provided. Please provide "
                                                      "'player_name' as a keyword argument to this "
                                                      "function, or as an argument to the "
                                                      "tfe.local_computation decorator.")

                return self.define_local_computation(
                        actual_player_name,
                        compute_func,
                        arguments=compute_func_args,
                        **kwargs,
                )

            return compute_func_wrapper

        if actual_compute_func is None:
            # User has not yet passed a compute_func, so we'll expect them to
            # pass it outside of this function's scope (e.g. as a decorator).
            return decorator

        # User has already passed a compute_func, so return the decorated version.
        return decorator(actual_compute_func)

    def define_local_computation(
        self,
        player,
        computation_fn,
        arguments=None,
        apply_scaling=True,
        share_type=ShareType.ARITHMETIC,
        name=None,
        factory=None,
    ):
        """
    Define a local computation that happens on plaintext tensors.

    :param player: Who performs the computation and gets to see the values in plaintext.
    :param apply_scaling: Whether or not to scale the outputs.
    :param name: Optional name to give to this node in the graph.
    :param factory: Backing tensor type to use for outputs.
    """

        factory = factory or self.default_factory

        if isinstance(player, str):
            player = get_config().get_player(player)
        assert isinstance(player, Player)

        def share_output(v: tf.Tensor):
            assert (
                v.shape.is_fully_defined()
            ), "Shape of return value '{}' on '{}' not fully defined".format(
                name if name else "", player.name,
            )

            v = self._encode(v, apply_scaling)
            w = factory.tensor(v)
            x = self._share_and_wrap(w, apply_scaling, share_type, player)

            return x

        def reconstruct_input(x, player):

            if isinstance(x, tf.Tensor):
                return x

            if isinstance(x, ABY3PublicTensor):
                w, _ = x.unwrapped
                v = self._decode(w, x.is_scaled)
                return v

            if isinstance(x, ABY3PrivateTensor):
                shares = x.unwrapped
                w = self._reconstruct(shares, player, share_type)
                v = self._decode(w, x.is_scaled)
                return v

            raise TypeError(
                ("Don't know how to process input argument " "of type {}").format(
                    type(x)
                )
            )

        with tf.name_scope(name if name else "local-computation"):

            with tf.device(player.device_name):
                if arguments is None:
                    inputs = []
                else:
                    if not isinstance(arguments, (list, tuple)):
                        arguments = [arguments]

                    inputs = [reconstruct_input(x, player) for x in arguments]

                outputs = computation_fn(*inputs)

                if isinstance(outputs, tf.Operation):
                    return outputs

                if isinstance(outputs, tf.Tensor):
                    return share_output(outputs)

                if isinstance(outputs, (list, tuple)):
                    return [share_output(output) for output in outputs]

                raise TypeError(
                    "Don't know how to handle results of "
                    "type {}".format(type(outputs))
                )

    def define_private_input(
        self,
        player,
        inputter_fn,
        apply_scaling: bool = True,
        share_type=ShareType.ARITHMETIC,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """
    Define a private input.

    This represents a `private` input owned by the specified player into the graph.

    :param Union[str,Player] player: Which player owns this input.
    :param bool apply_scaling: Whether or not to scale the value.
    :param str name: What name to give to this node in the graph.
    :param AbstractFactory factory: Which backing type to use for this input
        (e.g. `int100` or `int64`).
    """
        suffix = "-" + name if name else ""

        return self.define_local_computation(
            player=player,
            computation_fn=inputter_fn,
            arguments=[],
            apply_scaling=apply_scaling,
            share_type=share_type,
            name="private-input{}".format(suffix),
            factory=factory,
        )

    def define_public_input(
        self,
        player: Union[str, Player],
        inputter_fn: TFEInputter,
        apply_scaling: bool = True,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """
    Define a public input.

    This represents a `public` input owned by the specified player into the
    graph.

    :param Union[str,Player] player: Which player owns this input.
    :param bool apply_scaling: Whether or not to scale the value.
    :param str name: What name to give to this node in the graph.
    """
        if isinstance(player, str):
            player = get_config().get_player(player)
        assert isinstance(player, Player)

        factory = factory or self.default_factory
        suffix = "-" + name if name else ""

        def helper(v: tf.Tensor) -> "ABY3PublicTensor":
            assert (
                v.shape.is_fully_defined()
            ), "Shape of input '{}' on '{}' is not fully defined".format(
                name if name else "", player.name,
            )
            v = self._encode(v, apply_scaling)
            w = factory.tensor(v)
            return ABY3PublicTensor(self, [w, w, w], apply_scaling)

        with tf.name_scope("public-input{}".format(suffix)):

            with tf.device(player.device_name):

                inputs = inputter_fn()

                if isinstance(inputs, tf.Tensor):
                    # single input -> single output
                    v = inputs
                    return helper(v)

                if isinstance(inputs, (list, tuple)):
                    # multiple inputs -> multiple outputs
                    return [helper(v) for v in inputs]

                raise TypeError(
                    ("Don't know how to handle inputs of type {}").format(type(inputs))
                )

    def define_public_tensor(
        self,
        tensor: tf.Tensor,
        apply_scaling: bool = True,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ):
        """
    Convert a tf.Tensor to an ABY3PublicTensor.
    """
        assert isinstance(tensor, tf.Tensor)
        assert (
            tensor.shape.is_fully_defined()
        ), "Shape of input '{}' is not fully defined".format(name if name else "")

        factory = factory or self.defaul_factory

        with tf.name_scope("public-tensor"):
            tensor = self._encode(tensor, apply_scaling)
            w = factory.tensor(tensor)
            return ABY3PublicTensor(self, [w, w, w], apply_scaling)

    def define_output(
        self, player, arguments, outputter_fn, name=None,
    ):
        """
    Define an output for this graph.

    :param player: Which player this output will be sent to.
    """

        def result_wrapper(*args):
            op = outputter_fn(*args)
            # wrap in tf.group to prevent sending back any tensors (which might hence
            # be leaked)
            return tf.group(op)

        return self.define_local_computation(
            player=player,
            computation_fn=result_wrapper,
            arguments=arguments,
            name="output{}".format("-" + name if name else ""),
        )

    def _encode(
        self,
        rationals: Union[tf.Tensor, np.ndarray],
        apply_scaling: bool,
        factory=None,
    ) -> Union[tf.Tensor, np.ndarray]:
        """
    Encode tensor of rational numbers into tensor of ring elements. Output is
    of same type as input to allow function to be used for constants.
    """

        with tf.name_scope("encode"):

            # we first scale as needed
            if apply_scaling:
                scaled = rationals * self.fixedpoint_config.scaling_factor
            else:
                scaled = rationals

            # and then we round to integers

            if isinstance(scaled, np.ndarray):
                integers = scaled.astype(int).astype(object)

            elif isinstance(scaled, tf.Tensor):
                factory = factory or self.default_factory
                tf_native_type = factory.native_type
                assert tf_native_type in TF_NATIVE_TYPES
                integers = tf.cast(scaled, dtype=tf_native_type)

            else:
                raise TypeError("Don't know how to encode {}".format(type(rationals)))

            assert type(rationals) == type(integers)
            return integers

    @memoize
    def _decode(self, elements: AbstractTensor, is_scaled: bool) -> tf.Tensor:
        """Decode tensor of ring elements into tensor of rational numbers."""

        with tf.name_scope("decode"):
            scaled = elements.to_native()
            if not is_scaled:
                return scaled
            return scaled / self.fixedpoint_config.scaling_factor

    def _share(self, secret: AbstractTensor, share_type: str, player=None):
        """Secret-share an AbstractTensor.

    Args:
      secret: `AbstractTensor`, the tensor to share.

    Returns:
      A pair of `AbstractTensor`, the shares.
    """

        with tf.name_scope("share"):
            if share_type == ShareType.ARITHMETIC or share_type == ShareType.BOOLEAN:
                share0 = secret.factory.sample_uniform(secret.shape)
                share1 = secret.factory.sample_uniform(secret.shape)
                if share_type == ShareType.ARITHMETIC:
                    share2 = secret - share0 - share1
                elif share_type == ShareType.BOOLEAN:
                    share2 = secret ^ share0 ^ share1
                # Replicated sharing
                shares = ((share0, share1), (share1, share2), (share2, share0))
                return shares

            else:
                raise NotImplementedError("Unknown share type.")

    def _share_and_wrap(
        self, secret: AbstractTensor, is_scaled: bool, share_type: str, player=None,
    ) -> "ABY3PrivateTensor":
        shares = self._share(secret, share_type, player)
        return ABY3PrivateTensor(self, shares, is_scaled, share_type)

    def _reconstruct(self, shares, player, share_type):
        """
    Reconstruct the plaintext value at a specified player.
    The shares might locate at three different players, so we need the 'player' argument
    in order to optimally use two local shares and one (probably) remote share to
    minimize communication.

    :param shares:
    :param player: Where to reconstruct
    :return:
    """

        def helper(s0, s1, s2):
            if share_type == ShareType.ARITHMETIC:
                return s0 + s1 + s2
            elif share_type == ShareType.BOOLEAN:
                return s0 ^ s1 ^ s2
            else:
                raise NotImplementedError(
                    "Only arithmetic and boolean sharings are supported."
                )

        with tf.name_scope("reconstruct"):
            if share_type == ShareType.ARITHMETIC or share_type == ShareType.BOOLEAN:
                if player == self.servers[0]:
                    return helper(shares[0][0], shares[0][1], shares[2][0])
                elif player == self.servers[1]:
                    return helper(shares[1][0], shares[1][1], shares[0][0])
                elif player == self.servers[2]:
                    return helper(shares[2][0], shares[2][1], shares[1][0])
                else:
                    # The player is not any of the three ABY3 servers, so
                    # we just let each server give one share to this player
                    # in order to have a fair communication cost for each server
                    return helper(shares[0][0], shares[1][0], shares[2][0])

            else:
                raise NotImplementedError("Unknown share type.")

    def _gen_zero_sharing(self, shape, share_type=ShareType.ARITHMETIC, factory=None):
        def helper(f0, f1):
            if share_type == ShareType.ARITHMETIC:
                return f0 - f1
            elif share_type == ShareType.BOOLEAN:
                return f0 ^ f1
            else:
                raise NotImplementedError(
                    "Only arithmetic and boolean sharings are supported."
                )

        factory = factory or self.default_factory
        with tf.name_scope("zero-sharing"):
            with tf.device(self.servers[0].device_name):
                f00 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[0][0] + self.pairwise_nonces[2]
                )  # yapf: disable
                f01 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[0][1] + self.pairwise_nonces[0]
                )  # yapf: disable
                a0 = helper(f00, f01)
            with tf.device(self.servers[1].device_name):
                f10 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[1][0] + self.pairwise_nonces[0]
                )  # yapf: disable
                f11 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[1][1] + self.pairwise_nonces[1]
                )  # yapf: disable
                a1 = helper(f10, f11)
            with tf.device(self.servers[2].device_name):
                f20 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[2][0] + self.pairwise_nonces[1]
                )  # yapf: disable
                f21 = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[2][1] + self.pairwise_nonces[2]
                )  # yapf: disable
                a2 = helper(f20, f21)

        self.pairwise_nonces = self.pairwise_nonces + 1
        return a0, a1, a2

    def _gen_random_sharing(self, shape, share_type=ShareType.ARITHMETIC, factory=None):

        r = [[None] * 2 for _ in range(3)]
        factory = factory or self.default_factory
        with tf.name_scope("random-sharing"):
            with tf.device(self.servers[0].device_name):
                r[0][0] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[0][0] + self.pairwise_nonces[2]
                )  # yapf: disable
                r[0][1] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[0][1] + self.pairwise_nonces[0]
                )  # yapf: disable
            with tf.device(self.servers[1].device_name):
                r[1][0] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[1][0] + self.pairwise_nonces[0]
                )  # yapf: disable
                r[1][1] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[1][1] + self.pairwise_nonces[1]
                )  # yapf: disable
            with tf.device(self.servers[2].device_name):
                r[2][0] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[2][0] + self.pairwise_nonces[1]
                )  # yapf: disable
                r[2][1] = factory.sample_seeded_uniform(
                    shape=shape, seed=self.pairwise_keys[2][1] + self.pairwise_nonces[2]
                )  # yapf: disable

        self.pairwise_nonces = self.pairwise_nonces + 1

        return ABY3PrivateTensor(self, r, True, share_type)

    def _gen_b2a_sharing(self, shape, b2a_keys, factory=None):
        factory = factory or self.default_factory
        shares = [[None, None], [None, None], [None, None]]
        with tf.device(self.servers[0].device_name):
            shares[0][0] = factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[0][0] + self.b2a_nonce
            )  # yapf: disable
            shares[0][1] = factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[0][1] + self.b2a_nonce
            )  # yapf: disable
            x_on_0 = None
            if b2a_keys[0][2] is not None:
                share_2 = factory.sample_seeded_uniform(
                    shape=shape, seed=b2a_keys[0][2] + self.b2a_nonce
                )  # yapf: disable
                x_on_0 = shares[0][0] ^ shares[0][1] ^ share_2

        with tf.device(self.servers[1].device_name):
            shares[1][0] = factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[1][1] + self.b2a_nonce
            )  # yapf: disable
            shares[1][1] = factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[1][2] + self.b2a_nonce
            )  # yapf: disable
            x_on_1 = None
            if b2a_keys[1][0] is not None:
                share_0 = factory.sample_seeded_uniform(
                    shape=shape, seed=b2a_keys[1][0] + self.b2a_nonce
                )  # yapf: disable
                x_on_1 = share_0 ^ shares[1][0] ^ shares[1][1]

        with tf.device(self.servers[2].device_name):
            shares[2][0] = factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[2][2] + self.b2a_nonce
            )  # yapf: disable
            shares[2][1] = factory.sample_seeded_uniform(
                shape=shape, seed=b2a_keys[2][0] + self.b2a_nonce
            )  # yapf: disable
            x_on_2 = None
            if b2a_keys[2][1] is not None:
                share_1 = factory.sample_seeded_uniform(
                    shape=shape, seed=b2a_keys[2][1] + self.b2a_nonce
                )  # yapf: disable
                x_on_2 = share_1 ^ shares[2][0] ^ shares[2][1]

        self.b2a_nonce = self.b2a_nonce + 1
        return x_on_0, x_on_1, x_on_2, shares

    def _ot(
        self,
        sender,
        receiver,
        helper,
        m0,
        m1,
        c_on_receiver,
        c_on_helper,
        key_on_sender,
        key_on_helper,
        nonce,
    ):
        """
    Three-party OT protocol.

    'm0' and 'm1' are the two messages located on the sender.
    'c_on_receiver' and 'c_on_helper' should be the same choice bit, located on receiver and helper respectively.
    'key_on_sender' and 'key_on_helper' should be the same key, located on sender and helper respectively.
    'nonce' is a non-repeating ID for this call of the OT protocol.
    """
        assert m0.shape == m1.shape, "m0 shape {}, m1 shape {}".format(
            m0.shape, m1.shape
        )
        assert c_on_receiver.factory == self.factories[tf.bool]
        assert c_on_helper.factory == self.factories[tf.bool]
        assert m0.factory == m1.factory

        factory = m0.factory

        with tf.name_scope("OT"):
            with tf.device(sender.device_name):
                w_on_sender = factory.sample_seeded_uniform(
                    shape=[2] + m0.shape.as_list(), seed=key_on_sender + nonce
                )
                masked_m0 = m0 ^ w_on_sender[0]
                masked_m1 = m1 ^ w_on_sender[1]
            with tf.device(helper.device_name):
                w_on_helper = factory.sample_seeded_uniform(
                    shape=[2] + m0.shape.as_list(), seed=key_on_helper + nonce
                )
                w_c = factory.where(
                    c_on_helper.value, w_on_helper[1], w_on_helper[0], v2=False
                )
            with tf.device(receiver.device_name):
                masked_m_c = factory.where(
                    c_on_receiver.value, masked_m1, masked_m0, v2=False
                )
                m_c = masked_m_c ^ w_c

        return m_c

    @memoize
    def assign(self, variable: "ABY3PrivateVariable", value) -> tf.Operation:
        """See tf.assign."""
        assert isinstance(variable, ABY3PrivateVariable), type(variable)
        assert isinstance(value, ABY3PrivateTensor), type(value)
        assert (
            variable.is_scaled == value.is_scaled
        ), "Scaling must match: {}, {}".format(variable.is_scaled, value.is_scaled,)

        var_shares = variable.unwrapped
        val_shares = value.unwrapped

        with tf.name_scope("assign"):

            # Having this control_dependencies is important in order to avoid that
            # computationally-dependent shares are updated in different pace
            # (e.g., share0 is computed from share1, and we need to make sure that
            # share1 is NOT already updated).
            with tf.control_dependencies(
                [
                    val_shares[0][0].value,
                    val_shares[0][1].value,
                    val_shares[1][0].value,
                    val_shares[1][1].value,
                    val_shares[2][0].value,
                    val_shares[2][1].value,
                ]
            ):

                with tf.device(self.servers[0].device_name):
                    op00 = var_shares[0][0].assign_from_same(val_shares[0][0])
                    op01 = var_shares[0][1].assign_from_same(val_shares[0][1])

                with tf.device(self.servers[1].device_name):
                    op10 = var_shares[1][0].assign_from_same(val_shares[1][0])
                    op11 = var_shares[1][1].assign_from_same(val_shares[1][1])

                with tf.device(self.servers[2].device_name):
                    op20 = var_shares[2][0].assign_from_same(val_shares[2][0])
                    op21 = var_shares[2][1].assign_from_same(val_shares[2][1])

                op = tf.group(op00, op01, op10, op11, op20, op21)

        return op

    @memoize
    def add(self, x, y):
        """
    Adds two tensors `x` and `y`.

    :param ABY3Tensor x: The first operand.
    :param ABY3Tensor y: The second operand.
    """
        x, y = self.lift(x, y)
        return self.dispatch("add", x, y)

    def lift(self, x, y=None):
        """
    Convenience method for working with mixed typed tensors in programs:
    combining any of the ABY3 objects together with e.g. ints and floats
    will automatically lift the latter into ABY3 objects.

    Lifting will guarantee the two outputs are both scaled or unscaled if at
    least one of them is lifted from int or float.
    """

        if y is None:

            if isinstance(x, (np.ndarray, int, float)):
                return self.define_constant(x)

            if isinstance(x, tf.Tensor):
                return self.define_public_tensor(x)

            if isinstance(x, ABY3Tensor):
                return x

            raise TypeError("Don't know how to lift {}".format(type(x)))

        if isinstance(x, (np.ndarray, int, float)):

            if isinstance(y, (np.ndarray, int, float)):
                x = self.define_constant(x)
                y = self.define_constant(y)
                return x, y

            if isinstance(y, tf.Tensor):
                x = self.define_constant(x)
                y = self.define_public_tensor(y)
                return x, y

            if isinstance(y, ABY3Tensor):
                x = self.define_constant(
                    x,
                    apply_scaling=y.is_scaled,
                    factory=y.backing_dtype,
                )
                return x, y

            raise TypeError(
                ("Don't know how to lift " "{}, {}").format(type(x), type(y))
            )

        if isinstance(x, tf.Tensor):

            if isinstance(y, (np.ndarray, int, float)):
                x = self.define_public_tensor(x)
                y = self.define_constant(y)
                return x, y

            if isinstance(y, tf.Tensor):
                x = self.define_public_tensor(x)
                y = self.define_public_tensor(y)
                return x, y

            if isinstance(y, ABY3Tensor):
                x = self.define_public_tensor(
                    x,
                    apply_scaling=y.is_scaled,
                    factory=y.backing_dtype,
                )
                return x, y

            raise TypeError(
                ("Don't know how to lift " "{}, {}").format(type(x), type(y))
            )

        if isinstance(x, ABY3Tensor):

            if isinstance(y, (np.ndarray, int, float)):
                y = self.define_constant(
                    y,
                    apply_scaling=x.is_scaled,
                    factory=x.backing_dtype,
                )
                return x, y

            if isinstance(y, tf.Tensor):
                y = self.define_public_tensor(
                    y,
                    apply_scaling=x.is_scaled,
                    factory=x.backing_dtype,
                )
                return x, y

            if isinstance(y, ABY3Tensor):
                return x, y

        raise TypeError(("Don't know how to lift " "{}, {}").format(type(x), type(y)))

    @memoize
    def add_n(self, tensors):
        # TODO(Morten) we could optimize by doing lazy reductions, potentially
        #              segmenting as needed
        return reduce(lambda x, y: x + y, tensors)

    @memoize
    def sub(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("sub", x, y)

    @memoize
    def negative(self, x):
        x = self.lift(x)
        return self.dispatch("negative", x)

    @memoize
    def mul(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("mul", x, y)

    @memoize
    def mul_trunc2(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("mul_trunc2", x, y)

    @memoize
    def div(self, x, y):
        """
    Performs a true division of `x` by `y` where `y` is public.

    No flooring is performing if `y` is an integer type as it is implicitly
    treated as a float.
    """

        assert isinstance(x, ABY3Tensor)

        if isinstance(y, float):
            y_inverse = 1.0 / y
        elif isinstance(y, int):
            y_inverse = 1.0 / float(y)
        elif isinstance(y, ABY3PublicTensor):
            y_inverse = 1.0 / y.decode()
        else:
            raise TypeError("Don't know how to divide by type {}".format(type(y)))

        return self.mul(x, y_inverse)

    @memoize
    def pow(self, x, p):
        x = self.lift(x)
        return self.dispatch("pow", x, p)

    @memoize
    def matmul(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("matmul", x, y)

    def gather_bit(self, x, even):
        assert x.share_type is ShareType.BOOLEAN
        return self.dispatch("gather_bit", x, even)

    def xor_indices(self, x):
        assert x.share_type is ShareType.BOOLEAN
        return self.dispatch("xor_indices", x)

    @memoize
    def transpose(self, x, perm=None):
        x = self.lift(x)
        return self.dispatch("transpose", x, perm)

    def indexer(self, x: "ABY3Tensor", slc) -> "ABY3Tensor":
        return self.dispatch("indexer", x, slc)

    def reshape(self, x: "ABY3Tensor", axe) -> "ABY3Tensor":
        return self.dispatch("reshape", x, axe)

    @memoize
    def concat(self, xs, axis):
        """See tf.concat"""
        if all(isinstance(x, ABY3PublicTensor) for x in xs):
            return _concat_public(self, xs, axis=axis)

        if all(isinstance(x, ABY3PrivateTensor) for x in xs):
            return _concat_private(self, xs, axis=axis)

        raise TypeError("Don't know how to do a concat {}".format(type(xs)))

    @memoize
    def stack(self, xs, axis):
        """See tf.stack"""

        if all([isinstance(x, ABY3PublicTensor) for x in xs]):
            xs_stack = _stack_public(self, xs, axis=axis)

        elif all([isinstance(x, ABY3PrivateTensor) for x in xs]):
            xs_stack = _stack_private(self, xs, axis=axis)

        else:
            raise TypeError("Don't know how to do a stack {}".format(type(xs)))

        return xs_stack

    @memoize
    def expand_dims(self, x, axis):
        """See tf.expand_dims."""
        return self.dispatch("expand_dims", x, axis)

    @memoize
    def squeeze(self, x, axis=None):
        """See tf.squeeze"""
        return self.dispatch("squeeze", x, axis)


    @memoize
    def strided_slice(self, x, *args, **kwargs):
        """
        See tf.strided_slice
        """
        return self.dispatch("strided_slice", x, args, kwargs)

    @memoize
    def reduce_sum(self, x, axis=None, keepdims=False):
        x = self.lift(x)
        return self.dispatch("reduce_sum", x, axis=axis, keepdims=keepdims)

    @memoize
    def truncate(self, x: "ABY3Tensor", method="heuristic"):
        """
        @param method: "local", or "heuristic". "local" truncation always has a small
            probability to fail with big error, while "heuristic" truncation relies on
            an assumption of the maximum plain value and will not fail if this assumption
            holds.
        """
        return self.dispatch("truncate_" + method, x)

    @memoize
    def truncate_msb0(self, x, method="secureq8"):
        """
        @param method: "cheetah" or "secureq8". "secureq8" is a little more efficient.
            "cheetah" is a 3pc truncation protocol inspired by the Cheetah paper.
            "secureq8" is a 3pc truncation protocol inspired by the SequceQ8 paper.
        """
        return self.dispatch("truncate_msb0_" + method, x)

    @memoize
    def reveal(self, x):
        return self.dispatch("reveal", x)

    @memoize
    def xor(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("xor", x, y)

    @memoize
    def and_(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("and", x, y)

    @memoize
    def or_(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("or", x, y)

    @memoize
    def not_(self, x):
        x = self.lift(x)
        return self.dispatch("not", x)

    @memoize
    def ppa(self, x, y, n_bits=None, topology="kogge_stone"):
        x, y = self.lift(x, y)
        return self.dispatch("ppa", x, y, n_bits, topology)

    @memoize
    def b_add(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("b_add", x, y)

    @memoize
    def b_sub(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("b_sub", x, y)

    @memoize
    def lshift(self, x, steps):
        return self.dispatch("lshift", x, steps)

    @memoize
    def rshift(self, x, steps):
        return self.dispatch("rshift", x, steps)

    @memoize
    def logical_rshift(self, x, steps):
        return self.dispatch("logical_rshift", x, steps)

    @memoize
    def a2b(self, x, nbits=None):
        return self.dispatch("a2b", x, nbits)

    @memoize
    def b2a(self, x, nbits=None):
        return self.dispatch("b2a", x, nbits)

    @memoize
    def b2a_single(self, x):
        return self.dispatch("b2a_single", x)

    @memoize
    def mul_ab(self, x, y):
        """
    Callers should make sure y is boolean sharing whose backing TF native type is `tf.bool`.
    There is no automatic lifting for boolean sharing in the mixed-protocol multiplication.
    """
        x = self.lift(x)
        return self.dispatch("mul_ab", x, y)

    @memoize
    def bit_extract(self, x, i):
        if x.share_type == ShareType.BOOLEAN or x.share_type == ShareType.ARITHMETIC:
            return self.dispatch("bit_extract", x, i)
        else:
            raise ValueError("unsupported share type: {}".format(x.share_type))

    @memoize
    def msb(self, x):
        x = self.lift(x)
        return self.dispatch("msb", x)

    @memoize
    def polynomial(self, x, coeffs):
        x = self.lift(x)
        return self.dispatch("polynomial", x, coeffs)

    @memoize
    def polynomial_piecewise(self, x, c, coeffs):
        return self.dispatch("polynomial_piecewise", x, c, coeffs)

    @memoize
    def sigmoid(self, x, approx_type="piecewise_linear"):
        return self.dispatch("sigmoid", x, approx_type)

    @memoize
    def gather(self, x, indices, axis):
        raise NotImplementedError("Unsupported share type: {}".format(x.share_type))

    def write(self, x, filename_prefix):
        if not isinstance(x, ABY3PrivateTensor):
            raise TypeError("Only support writing ABY3PrivateTensor to disk.")
        return self.dispatch("write", x, filename_prefix)

    def read(self, filename_prefix, batch_size, n_columns):
        return self.dispatch("read", filename_prefix, batch_size, n_columns)

    def iterate(
        self,
        tensor: "ABY3PrivateTensor",
        batch_size: int,
        repeat=True,
        shuffle=True,
        seed: int = None,
    ):
        if not isinstance(tensor, ABY3PrivateTensor):
            raise TypeError("Only support iterating ABY3PrivateTensor.")
        return self.dispatch("iterate", tensor, batch_size, repeat, shuffle, seed)

    def blinded_shuffle(self, tensor: "ABY3PrivateTensor"):
        """
    Shuffle the rows of the given tenosr privately.
    After the shuffle, none of the share holder could know the exact shuffle order.
    """
        if not isinstance(tensor, ABY3PrivateTensor):
            raise TypeError(
                (
                    "Only support blindly shuffle ABY3PrivateTensor. "
                    "For public tensor, use the shuffle() method"
                )
            )
        return self.dispatch("blinded_shuffle", tensor)

    @memoize
    def equal_zero(self, x):
        return self.dispatch("equal_zero", x)

    @memoize
    def equal(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("equal", x, y)

    @memoize
    def greater_than(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("greater_than", x, y)

    @memoize
    def greater_equal(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("greater_equal", x, y)

    @memoize
    def less_than(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("less_than", x, y)

    @memoize
    def less_equal(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("less_equal", x, y)

    @memoize
    def bit_gather(self, x, start, stride):
        return self.dispatch("bit_gather", x, start, stride)

    @memoize
    def cast(self, x, factory):
        return self.dispatch("cast", x, factory)

    @memoize
    def carry(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("carry", x, y)


    def dispatch(self, base_name, *args, container=None, **kwargs):
        """
    Finds the correct protocol logicto perform based on the dispatch_id
    attribute of the input tensors in args.
    """
        suffix = "_".join(
            [arg.dispatch_id for arg in args if hasattr(arg, "dispatch_id")]
        )
        func_name = "_{}_{}".format(base_name, suffix)

        if container is None:
            container = _THISMODULE

        func = getattr(container, func_name, None)
        if func is not None:
            return func(self, *args, **kwargs)  # pylint: disable=not-callable
        raise TypeError(
            ("Don't know how to {}: {}").format(base_name, [type(arg) for arg in args])
        )


#
# reveal helpers
#


def _reveal_private(prot, x):
    assert isinstance(x, ABY3PrivateTensor), type(x)

    with tf.name_scope("reveal"):

        shares = x.unwrapped

        with tf.device(prot.servers[0].device_name):
            z_on_0 = prot._reconstruct(shares, prot.servers[0], x.share_type)

        with tf.device(prot.servers[1].device_name):
            z_on_1 = prot._reconstruct(shares, prot.servers[1], x.share_type)

        with tf.device(prot.servers[2].device_name):
            z_on_2 = prot._reconstruct(shares, prot.servers[2], x.share_type)

    return ABY3PublicTensor(prot, [z_on_0, z_on_1, z_on_2], x.is_scaled)


def _bit_gather_private(prot, x, start, stride):
    assert x.share_type == ShareType.BOOLEAN

    shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("bit-gather-private"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = shares[i][0].bit_gather(start, stride)
                z[i][1] = shares[i][1].bit_gather(start, stride)

        z = ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)
    return z


def _bit_gather_public(prot, x, start, stride):
    x_ = x.unwrapped

    z = [None, None, None]
    with tf.name_scope("bit-gather-public"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i] = x_[i].bit_gather(start, stride)

        z = ABY3PublicTensor(prot, z, x.is_scaled)
    return z



def _cast_private(prot, x, factory):
    assert isinstance(x, ABY3PrivateTensor), type(x)

    if x.backing_dtype == factory:
        return x

    shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("cast-private"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = shares[i][0].cast(factory)
                z[i][1] = shares[i][1].cast(factory)

        z = ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)
    return z


def _cast_public(prot, x, factory):
    assert isinstance(x, ABY3PublicTensor), type(x)

    if x.backing_dtype == factory:
        return x

    x_ = x.unwrapped

    z = [None, None, None]
    with tf.name_scope("cast-public"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i] = x_[i].cast(factory)

        z = ABY3PublicTensor(prot, z, x.is_scaled)
    return z

#
# add helpers
#


def _add_private_private(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("add"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x.shares[i][0] + y.shares[i][0]
                z[i][1] = x.shares[i][1] + y.shares[i][1]

    return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


def _add_private_public(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PublicTensor), type(y)
    assert x.is_scaled == y.is_scaled, (
        "Cannot mix different encodings: " "{} {}"
    ).format(x.is_scaled, y.is_scaled)

    shares = x.unwrapped
    y_on_0, _, y_on_2 = y.unwrapped

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("add"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0] + y_on_0
            z[0][1] = shares[0][1]

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0]
            z[1][1] = shares[1][1]

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0]
            z[2][1] = shares[2][1] + y_on_2
    return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


def _add_public_private(prot, x, y):
    assert isinstance(x, ABY3PublicTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)
    assert x.is_scaled == y.is_scaled, (
        "Cannot mix different encodings: " "{} {}"
    ).format(x.is_scaled, y.is_scaled)

    x_on_0, _, x_on_2 = x.unwrapped
    shares = y.unwrapped

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("add"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0] + x_on_0
            z[0][1] = shares[0][1]

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0]
            z[1][1] = shares[1][1]

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0]
            z[2][1] = shares[2][1] + x_on_2

    return ABY3PrivateTensor(prot, z, x.is_scaled, y.share_type)


def _add_public_public(prot, x, y):
    assert isinstance(x, ABY3PublicTensor), type(x)
    assert isinstance(y, ABY3PublicTensor), type(y)
    assert x.is_scaled == y.is_scaled, "Cannot add tensors with different scales"

    x_shares = x.unwrapped
    y_shares = y.unwrapped

    z = [None] * 3
    with tf.name_scope("add"):
        for i in range(3):
            z[i] = x_shares[i] + y_shares[i]

    return ABY3PublicTensor(prot, z, x.is_scaled)


#
# sub helpers
#


def _sub_private_private(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)
    assert x.is_scaled == y.is_scaled

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("sub"):
        x_shares = x.unwrapped
        y_shares = y.unwrapped
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0] - y_shares[i][0]
                z[i][1] = x_shares[i][1] - y_shares[i][1]

    return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


def _sub_private_public(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PublicTensor), type(y)
    assert x.is_scaled == y.is_scaled

    shares = x.unwrapped
    y_on_0, _, y_on_2 = y.unwrapped

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("sub"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0] - y_on_0
            z[0][1] = shares[0][1]

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0]
            z[1][1] = shares[1][1]

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0]
            z[2][1] = shares[2][1] - y_on_2

    return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


def _sub_public_private(prot, x, y):
    assert isinstance(x, ABY3PublicTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    x_on_0, _, x_on_2 = x.unwrapped
    shares = y.unwrapped

    z = [[None] * 2 for _ in range(3)]
    with tf.name_scope("sub"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = x_on_0 - shares[0][0]
            z[0][1] = -shares[0][1]

        with tf.device(prot.servers[1].device_name):
            z[1][0] = -shares[1][0]
            z[1][1] = -shares[1][1]

        with tf.device(prot.servers[2].device_name):
            z[2][0] = -shares[2][0]
            z[2][1] = x_on_2 - shares[2][1]

    return ABY3PrivateTensor(prot, z, x.is_scaled, y.share_type)


#
# negative helpers
#


def _negative_private(prot, x):
    assert isinstance(x, ABY3PrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("negative"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = -x_shares[i][0]
                z[i][1] = -x_shares[i][1]

        z = ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)
    return z


def _negative_public(prot, x):
    assert isinstance(x, ABY3PublicTensor), type(x)

    x_on_0, x_on_1, x_on_2 = x.unwrapped

    with tf.name_scope("negative"):
        with tf.device(prot.servers[0].device_name):
            x_on_0_neg = -x_on_0
        with tf.device(prot.servers[1].device_name):
            x_on_1_neg = -x_on_1
        with tf.device(prot.servers[2].device_name):
            x_on_2_neg = -x_on_2
        x_neg = ABY3PublicTensor(
            prot, [x_on_0_neg, x_on_1_neg, x_on_2_neg], x.is_scaled
        )
    return x_neg


#
# mul helpers
#


def _mul_public_private(prot, x, y):
    assert isinstance(x, ABY3PublicTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    x_on_0, x_on_1, x_on_2 = x.unwrapped
    shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("mul"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0] * x_on_0
            z[0][1] = shares[0][1] * x_on_0

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0] * x_on_1
            z[1][1] = shares[1][1] * x_on_1

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0] * x_on_2
            z[2][1] = shares[2][1] * x_on_2

        z = ABY3PrivateTensor(prot, z, x.is_scaled or y.is_scaled, y.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _mul_private_public(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PublicTensor), type(y)

    shares = x.unwrapped
    y_on_0, y_on_1, y_on_2 = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("mul"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0] * y_on_0
            z[0][1] = shares[0][1] * y_on_0

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0] * y_on_1
            z[1][1] = shares[1][1] * y_on_1

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0] * y_on_2
            z[2][1] = shares[2][1] * y_on_2

        z = ABY3PrivateTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _mul_private_private(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    x_shares = x.unwrapped
    y_shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("mul"):
        a0, a1, a2 = prot._gen_zero_sharing(x.shape)
        with tf.device(prot.servers[0].device_name):
            z0 = (
                x_shares[0][0] * y_shares[0][0]
                + x_shares[0][0] * y_shares[0][1]
                + x_shares[0][1] * y_shares[0][0]
                + a0
            )

        with tf.device(prot.servers[1].device_name):
            z1 = (
                x_shares[1][0] * y_shares[1][0]
                + x_shares[1][0] * y_shares[1][1]
                + x_shares[1][1] * y_shares[1][0]
                + a1
            )

        with tf.device(prot.servers[2].device_name):
            z2 = (
                x_shares[2][0] * y_shares[2][0]
                + x_shares[2][0] * y_shares[2][1]
                + x_shares[2][1] * y_shares[2][0]
                + a2
            )
        # Re-sharing
        with tf.device(prot.servers[0].device_name):
            z[0][0] = z0
            z[0][1] = z1
        with tf.device(prot.servers[1].device_name):
            z[1][0] = z1
            z[1][1] = z2
        with tf.device(prot.servers[2].device_name):
            z[2][0] = z2
            z[2][1] = z0

        z = ABY3PrivateTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _mul_trunc2_private_private(prot, x, y):
    """
  Multiplication with the Trunc2 protocol in the ABY3 paper.
  This is more efficient (in terms of communication rounds)
  than `mul` in the onlline phase only when pre-computation
  is left out of consideration.
  """
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    # If there will not be any truncation, then just call the simple multiplication protocol.
    if not (x.is_scaled and y.is_scaled):
        return _mul_private_private(prot, x, y)

    x_shares = x.unwrapped
    y_shares = y.unwrapped
    shape = x_shares[0][0].shape
    amount = prot.fixedpoint_config.precision_fractional

    with tf.name_scope("mul_trunc2"):
        # Step 1: Generate a Random Truncation Pair
        # If TF is smart enough, this part is supposed to be pre-computation.
        r = prot._gen_random_sharing(shape, share_type=ShareType.BOOLEAN)
        r_trunc = r.arith_rshift(amount)
        r = prot.b2a(r)
        r_trunc = prot.b2a(r_trunc)

        # Step 2: Compute 3-out-of-3 sharing of (x*y - r)
        a0, a1, a2 = prot._gen_zero_sharing(x.shape)
        with tf.device(prot.servers[0].device_name):
            z0 = (
                x_shares[0][0] * y_shares[0][0]
                + x_shares[0][0] * y_shares[0][1]
                + x_shares[0][1] * y_shares[0][0]
                + a0
                - r.shares[0][0]
            )

        with tf.device(prot.servers[1].device_name):
            z1 = (
                x_shares[1][0] * y_shares[1][0]
                + x_shares[1][0] * y_shares[1][1]
                + x_shares[1][1] * y_shares[1][0]
                + a1
                - r.shares[1][0]
            )

        with tf.device(prot.servers[2].device_name):
            z2 = (
                x_shares[2][0] * y_shares[2][0]
                + x_shares[2][0] * y_shares[2][1]
                + x_shares[2][1] * y_shares[2][0]
                + a2
                - r.shares[2][0]
            )

        # Step 3: Reveal (x*y - r) / 2^d
        # xy_minus_r = z0 + z1 + z2
        # xy_minus_r_trunc = xy_minus_r.right_shift(amount)
        # z = ABY3PublicTensor(prot, [xy_minus_r_trunc, xy_minus_r_trunc, xy_minus_r_trunc], True, ShareType.ARITHMETIC)
        xy_minus_r_trunc = [None] * 3
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                xy_minus_r_trunc[i] = z0 + z1 + z2
                xy_minus_r_trunc[i] = xy_minus_r_trunc[i].right_shift(amount)
        z = ABY3PublicTensor(prot, xy_minus_r_trunc, True)

        # Step 4: Final addition
        z = z + r_trunc

        return z


def _matmul_public_private(prot, x, y):
    assert isinstance(x, ABY3PublicTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    x_on_0, x_on_1, x_on_2 = x.unwrapped
    shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("matmul"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = x_on_0.matmul(shares[0][0])
            z[0][1] = x_on_0.matmul(shares[0][1])

        with tf.device(prot.servers[1].device_name):
            z[1][0] = x_on_1.matmul(shares[1][0])
            z[1][1] = x_on_1.matmul(shares[1][1])

        with tf.device(prot.servers[2].device_name):
            z[2][0] = x_on_2.matmul(shares[2][0])
            z[2][1] = x_on_2.matmul(shares[2][1])

        z = ABY3PrivateTensor(prot, z, x.is_scaled or y.is_scaled, y.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _matmul_private_public(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PublicTensor), type(y)

    shares = x.unwrapped
    y_on_0, y_on_1, y_on_2 = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("matmul"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = shares[0][0].matmul(y_on_0)
            z[0][1] = shares[0][1].matmul(y_on_0)

        with tf.device(prot.servers[1].device_name):
            z[1][0] = shares[1][0].matmul(y_on_1)
            z[1][1] = shares[1][1].matmul(y_on_1)

        with tf.device(prot.servers[2].device_name):
            z[2][0] = shares[2][0].matmul(y_on_2)
            z[2][1] = shares[2][1].matmul(y_on_2)

        z = ABY3PrivateTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _matmul_private_private(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    x_shares = x.unwrapped
    y_shares = y.unwrapped

    # Tensorflow supports matmul for more than 2 dimensions,
    # with the inner-most 2 dimensions specifying the 2-D matrix multiplication
    result_shape = tf.TensorShape((*x.shape[:-1], y.shape[-1]))

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("matmul"):
        a0, a1, a2 = prot._gen_zero_sharing(result_shape)

        with tf.device(prot.servers[0].device_name):
            z0 = (
                x_shares[0][0].matmul(y_shares[0][0])
                + x_shares[0][0].matmul(y_shares[0][1])
                + x_shares[0][1].matmul(y_shares[0][0])
                + a0
            )

        with tf.device(prot.servers[1].device_name):
            z1 = (
                x_shares[1][0].matmul(y_shares[1][0])
                + x_shares[1][0].matmul(y_shares[1][1])
                + x_shares[1][1].matmul(y_shares[1][0])
                + a1
            )

        with tf.device(prot.servers[2].device_name):
            z2 = (
                x_shares[2][0].matmul(y_shares[2][0])
                + x_shares[2][0].matmul(y_shares[2][1])
                + x_shares[2][1].matmul(y_shares[2][0])
                + a2
            )
        # Re-sharing
        with tf.device(prot.servers[0].device_name):
            z[0][0] = z0
            z[0][1] = z1
        with tf.device(prot.servers[1].device_name):
            z[1][0] = z1
            z[1][1] = z2
        with tf.device(prot.servers[2].device_name):
            z[2][0] = z2
            z[2][1] = z0

        z = ABY3PrivateTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _truncate_heuristic_private(prot: ABY3, x: ABY3PrivateTensor) -> ABY3PrivateTensor:
    with tf.name_scope("truncate-heuristic"):
        amount = prot.fixedpoint_config.precision_fractional

        heuristic_bound_bits = x.backing_dtype.nbits - 4
        y = x + (1 << (heuristic_bound_bits - amount)) # Lifted to make msb 0
        z = prot.truncate_msb0(y, "secureq8")
        z = z - (1 << (heuristic_bound_bits - 2*amount)) # Reverse the effect of lifting
    return z


def _truncate_local_private(
    prot: ABY3, x: ABY3PrivateTensor,
) -> ABY3PrivateTensor:

    base = prot.fixedpoint_config.scaling_base
    amount = prot.fixedpoint_config.precision_fractional
    shares = x.unwrapped

    y = [[None, None], [None, None], [None, None]]
    with tf.name_scope("truncate-local"):

        # Local computation
        with tf.device(prot.servers[2].device_name):
            r_on_2 = x.backing_dtype.sample_seeded_uniform(
                shares[2][0].shape, prot.pairwise_keys[2][0] + prot.pairwise_nonces[1]
            )

        # Local computation
        with tf.device(prot.servers[0].device_name):
            y0 = shares[0][0].truncate(amount, base)

        # Local computation
        with tf.device(prot.servers[1].device_name):
            r_on_1 = x.backing_dtype.sample_seeded_uniform(
                shares[1][0].shape, prot.pairwise_keys[1][1] + prot.pairwise_nonces[1]
            )
            t = shares[1][0] + shares[1][1]
            tmp = t.truncate(amount, base)
            y1 = tmp - r_on_1

        prot.pairwise_nonces[1] = prot.pairwise_nonces[1] + 1

        # Replicate shares

        # server 1 sends `y1`
        with tf.device(prot.servers[0].device_name):
            y[0][0] = y0
            y[0][1] = y1
        # Local
        with tf.device(prot.servers[1].device_name):
            y[1][0] = y1
            y[1][1] = r_on_1
        # server 0 sends `y0`
        with tf.device(prot.servers[2].device_name):
            y[2][0] = r_on_2
            y[2][1] = y0

    return ABY3PrivateTensor(prot, y, x.is_scaled, x.share_type)


def _truncate_msb0_cheetah_private(prot: ABY3, x: ABY3PrivateTensor) -> ABY3PrivateTensor:
    assert x.share_type == ShareType.ARITHMETIC, x.share_type

    bfactory = prot.factories[tf.bool]

    amount = prot.fixedpoint_config.precision_fractional
    shape = x.shape
    x_shares = x.unwrapped
    with tf.device(prot.servers[0].device_name):
        x_alice = x_shares[0][0] + x_shares[0][1]
        # sample w_A
        w_alice = bfactory.sample_uniform(shape)
        msb_alice = x_alice.logical_rshift(x.backing_dtype.nbits-1).cast(bfactory)
        s0 = w_alice ^ msb_alice
        s1 = w_alice ^ 1
        # x_A >> f
        x_alice_trunc = x_alice.logical_rshift(amount)

    with tf.device(prot.servers[1].device_name):
        x_bob_on_1 = x_shares[1][1]
        msb_bob_on_1 = x_bob_on_1.logical_rshift(x.backing_dtype.nbits-1).cast(bfactory)
        # x_B >> f
        x_bob_on_1_trunc = x_bob_on_1.logical_rshift(amount)
    with tf.device(prot.servers[2].device_name):
        x_bob_on_2 = x_shares[2][0]
        msb_bob_on_2 = x_bob_on_2.logical_rshift(x.backing_dtype.nbits-1).cast(bfactory)
        # x_B >> f
        x_bob_on_2_trunc = x_bob_on_2.logical_rshift(amount)


    # Alice inputs: (s0, s1), Bob inputs: msb(x_B)
    w_bob = prot._ot(prot.servers[0], prot.servers[1], prot.servers[2],
            s0, s1,
            msb_bob_on_1, msb_bob_on_2,
            prot.pairwise_keys[0][0], prot.pairwise_keys[2][1],
            prot.pairwise_nonces[2])

    w = [[None, None], [None, None], [None, None]]
    with tf.device(prot.servers[0].device_name):
        w[0][0] = w_alice
        w[0][1] = bfactory.sample_seeded_uniform(
                shape,
                prot.pairwise_keys[0][1] + prot.pairwise_nonces[0])

    with tf.device(prot.servers[1].device_name):
        w[1][0] = bfactory.sample_seeded_uniform(
                shape,
                prot.pairwise_keys[1][0] + prot.pairwise_nonces[0])
        w[1][1] = w_bob ^ w[1][0]

    with tf.device(prot.servers[2].device_name):
        w[2][0] = w[1][1]
        w[2][1] = w_alice


    # b2a. TODO: communication could be optimized from l to f
    w = prot.b2a_single(ABY3PrivateTensor(prot, w, False, ShareType.BOOLEAN))

    # Fix potential big error: w * 2^{l-f}
    error = (w << (x.backing_dtype.nbits - amount)).unwrapped
    z = [[None, None], [None, None], [None, None]]
    with tf.device(prot.servers[0].device_name):
        z[0][0] = x_alice_trunc - error[0][0]
        z[0][1] = -error[0][1]
    with tf.device(prot.servers[1].device_name):
        z[1][0] = -error[1][0]
        z[1][1] = x_bob_on_1_trunc - error[1][1]
    with tf.device(prot.servers[2].device_name):
        z[2][0] = x_bob_on_2_trunc - error[2][0]
        z[2][1] = z[0][0]

    z = ABY3PrivateTensor(prot, z, x.is_scaled, ShareType.ARITHMETIC)
    return z


def _truncate_msb0_secureq8_private(prot: ABY3, x: ABY3PrivateTensor) -> ABY3PrivateTensor:

    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert x.share_type == ShareType.ARITHMETIC, x.share_type

    ifactory = x.backing_dtype
    bfactory = prot.factories[tf.bool]

    amount = prot.fixedpoint_config.precision_fractional
    shape = x.shape
    x_shares = x.unwrapped
    msb_mask = 1 << (x.backing_dtype.nbits - 1)
    mod_mask = (1 << (x.backing_dtype.nbits - amount - 1)) - 1

    # P2: Generate random values, compute intermediate results, and share them as 2PC sharings among P0 and P1
    with tf.device(prot.servers[2].device_name):
        r = ifactory.sample_uniform(shape)
        h = r & msb_mask
        s = r.logical_rshift(amount) - h.logical_rshift(amount)
        r_msb = h.cast(bfactory).cast(ifactory)

        r0 = ifactory.sample_uniform(shape)
        r1 = r - r0

        s0 = ifactory.sample_uniform(shape)
        s1 = s - s0

        r_msb0 = ifactory.sample_uniform(shape)
        r_msb1 = r_msb - r_msb0

        y0 = ifactory.sample_uniform(shape)
        y2 = ifactory.sample_uniform(shape)

    # Proceed as 2PC computation between P0 and P1

    # P0: mask x0
    with tf.device(prot.servers[0].device_name):
        x0 = x_shares[0][0] + x_shares[0][1]
        c0 = x0 + r0

    # P1: mask x1
    with tf.device(prot.servers[1].device_name):
        x1 = x_shares[1][1]
        c1 = x1 + r1

    with tf.device(prot.servers[0].device_name):
        c_on_0 = c0 + c1
        c_prime_on_0 = (c_on_0 >> amount) & mod_mask
        c_msb_on_0 = (c_on_0 & msb_mask).cast(bfactory)
        b0 = ifactory.where(c_msb_on_0.value, 1 - r_msb0, r_msb0, v2=False)
        y_prime0 = c_prime_on_0 - s0 + b0 * (mod_mask + 1)
        y_tilde0 = y_prime0 - y0

    with tf.device(prot.servers[1].device_name):
        c_on_1 = c0 + c1
        c_prime_on_1 = (c_on_1 >> amount) & mod_mask
        c_msb_on_1 = (c_on_1 & msb_mask).cast(bfactory)
        b1 = ifactory.where(c_msb_on_1.value, -r_msb1, r_msb1, v2=False)
        y_prime1 = - s1 + b1 * (mod_mask + 1)
        y_tilde1 = y_prime1 - y2


    y = [[None, None], [None, None], [None, None]]
    with tf.device(prot.servers[0].device_name):
        y[0][0] = y0
        y[0][1] = y_tilde0 + y_tilde1

    with tf.device(prot.servers[1].device_name):
        y[1][0] = y_tilde0 + y_tilde1
        y[1][1] = y2

    with tf.device(prot.servers[2].device_name):
        y[2][0] = y2
        y[2][1] = y0

    y = ABY3PrivateTensor(prot, y, x.is_scaled, ShareType.ARITHMETIC)
    return y


def _greater_than_private_private(prot, x, y):
    return _less_than_private_private(prot, y, x)


def _greater_than_private_public(prot, x, y):
    return _less_than_public_private(prot, y, x)


def _greater_than_public_private(prot, x, y):
    return _less_than_private_public(prot, y, x)

def _greater_equal_private_private(prot, x, y):
    return ~_less_than_private_private(prot, x, y)


def _greater_equal_private_public(prot, x, y):
    return ~_less_than_private_public(prot, x, y)


def _greater_equal_public_private(prot, x, y):
    return ~_less_than_public_private(prot, x, y)

def _less_equal_private_private(prot, x, y):
    return ~_greater_than_private_private(prot, x, y)


def _less_equal_private_public(prot, x, y):
    return ~_greater_than_private_public(prot, x, y)


def _less_equal_public_private(prot, x, y):
    return ~_greater_than_public_private(prot, x, y)

def _less_than_private_private(prot, x, y):
    assert x.is_arithmetic() and y.is_arithmetic(), \
            "Unexpected share type: x {}, y {}".format(x.share_type, y.share_type)

    return _less_than_computation(prot, x, y)


def _less_than_private_public(prot, x, y):
    assert x.is_arithmetic(), \
            "Unexpected share type: x {}".format(x.share_type)

    return _less_than_computation(prot, x, y)


def _less_than_public_private(prot, x, y):
    assert y.is_arithmetic(), \
            "Unexpected share type: y {}".format(y.share_type)

    return _less_than_computation(prot, x, y)


def _less_than_computation(prot, x, y):
    z = x - y
    result = prot.msb(z)

    return result


def _equal_private_private(prot, x, y):
    assert x.is_arithmetic() and y.is_arithmetic(), \
            "Unexpected share type: x {}, y {}".format(x.share_type, y.share_type)
    return _equal_zero_private(prot, x-y)

def _equal_private_public(prot, x, y):
    assert x.is_arithmetic(), \
            "Unexpected share type: x {}".format(x.share_type)
    return _equal_zero_private(prot, x-y)

def _equal_public_private(prot, x, y):
    assert y.is_arithmetic(), \
            "Unexpected share type: y {}".format(y.share_type)
    return _equal_zero_private(prot, x-y)


def _equal_zero_private(prot, x):
    neg_x = -x
    pack = prot.stack([x, neg_x], axis=0)
    msb = prot.msb(pack)
    result = (~msb[0]) & (~msb[1])
    return result


def _xor_private_private(prot: ABY3, x: ABY3PrivateTensor, y: ABY3PrivateTensor):
    assert x.share_type == ShareType.BOOLEAN
    assert y.share_type == ShareType.BOOLEAN
    assert x.backing_dtype == y.backing_dtype

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("xor"):

        with tf.device(prot.servers[0].device_name):
            z[0][0] = x.shares[0][0] ^ y.shares[0][0]
            z[0][1] = x.shares[0][1] ^ y.shares[0][1]

        with tf.device(prot.servers[1].device_name):
            z[1][0] = x.shares[1][0] ^ y.shares[1][0]
            z[1][1] = x.shares[1][1] ^ y.shares[1][1]

        with tf.device(prot.servers[2].device_name):
            z[2][0] = x.shares[2][0] ^ y.shares[2][0]
            z[2][1] = x.shares[2][1] ^ y.shares[2][1]

    return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


def _xor_private_public(prot: ABY3, x: ABY3PrivateTensor, y: ABY3PublicTensor):
    assert x.share_type == ShareType.BOOLEAN
    assert x.backing_dtype == y.backing_dtype

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("xor"):
        y_on_0, y_on_1, y_on_2 = y.unwrapped
        with tf.device(prot.servers[0].device_name):
            z[0][0] = x.shares[0][0] ^ y_on_0
            z[0][1] = x.shares[0][1] ^ y_on_0

        with tf.device(prot.servers[1].device_name):
            z[1][0] = x.shares[1][0] ^ y_on_1
            z[1][1] = x.shares[1][1] ^ y_on_1

        with tf.device(prot.servers[2].device_name):
            z[2][0] = x.shares[2][0] ^ y_on_2
            z[2][1] = x.shares[2][1] ^ y_on_2

    return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


def _and_private_private(prot: ABY3, x: ABY3PrivateTensor, y: ABY3PrivateTensor):
    assert x.share_type == ShareType.BOOLEAN
    assert y.share_type == ShareType.BOOLEAN
    assert x.backing_dtype == y.backing_dtype

    x_shares = x.unwrapped
    y_shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("and"):
        a0, a1, a2 = prot._gen_zero_sharing(
            x.shape, share_type=ShareType.BOOLEAN, factory=x.backing_dtype
        )

        with tf.device(prot.servers[0].device_name):
            tmp0 = x_shares[0][0] & y_shares[0][0]
            tmp1 = x_shares[0][0] & y_shares[0][1]
            tmp2 = x_shares[0][1] & y_shares[0][0]
            z0 = tmp0 ^ tmp1 ^ tmp2 ^ a0

        with tf.device(prot.servers[1].device_name):
            tmp0 = x_shares[1][0] & y_shares[1][0]
            tmp1 = x_shares[1][0] & y_shares[1][1]
            tmp2 = x_shares[1][1] & y_shares[1][0]
            z1 = tmp0 ^ tmp1 ^ tmp2 ^ a1

        with tf.device(prot.servers[2].device_name):
            tmp0 = x_shares[2][0] & y_shares[2][0]
            tmp1 = x_shares[2][0] & y_shares[2][1]
            tmp2 = x_shares[2][1] & y_shares[2][0]
            z2 = tmp0 ^ tmp1 ^ tmp2 ^ a2

        # Re-sharing
        with tf.device(prot.servers[0].device_name):
            z[0][0] = z0
            z[0][1] = z1
        with tf.device(prot.servers[1].device_name):
            z[1][0] = z1
            z[1][1] = z2
        with tf.device(prot.servers[2].device_name):
            z[2][0] = z2
            z[2][1] = z0

        z = ABY3PrivateTensor(prot, z, x.is_scaled or y.is_scaled, x.share_type)
        return z


def _and_private_public(prot, x, y):
    assert x.share_type == ShareType.BOOLEAN
    assert x.backing_dtype == y.backing_dtype

    x_shares = x.unwrapped
    y_on_0, y_on_1, y_on_2 = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("and"):
        with tf.device(prot.servers[0].device_name):
            z[0][0] = x_shares[0][0] & y_on_0
            z[0][1] = x_shares[0][1] & y_on_0
        with tf.device(prot.servers[1].device_name):
            z[1][0] = x_shares[1][0] & y_on_1
            z[1][1] = x_shares[1][1] & y_on_1
        with tf.device(prot.servers[2].device_name):
            z[2][0] = x_shares[2][0] & y_on_2
            z[2][1] = x_shares[2][1] & y_on_2

    z = ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)
    return z


def _and_public_private(prot, x, y):
    assert y.share_type == ShareType.BOOLEAN
    assert x.backing_dtype == y.backing_dtype

    x_on_0, x_on_1, x_on_2 = x.unwrapped
    y_shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("and"):
        with tf.device(prot.servers[0].device_name):
            z[0][0] = x_on_0 & y_shares[0][0]
            z[0][1] = x_on_0 & y_shares[0][1]
        with tf.device(prot.servers[1].device_name):
            z[1][0] = x_on_1 & y_shares[1][0]
            z[1][1] = x_on_1 & y_shares[1][1]
        with tf.device(prot.servers[2].device_name):
            z[2][0] = x_on_2 & y_shares[2][0]
            z[2][1] = x_on_2 & y_shares[2][1]

    z = ABY3PrivateTensor(prot, z, y.is_scaled, y.share_type)
    return z


def _or_private_private(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    with tf.name_scope("or"):
        z = (x ^ y) ^ (x & y)

    return z


def _not_private(prot, x):
    assert isinstance(x, ABY3PrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("not"):
        with tf.device(prot.servers[0].device_name):
            # We use the `~` operator instead of XORing a constant, because we want it to work for both
            # the int_factory and the bool_factory
            z[0][0] = ~x_shares[0][0]
            z[0][1] = x_shares[0][1]
        with tf.device(prot.servers[1].device_name):
            z[1][0] = x_shares[1][0]
            z[1][1] = x_shares[1][1]
        with tf.device(prot.servers[2].device_name):
            z[2][0] = x_shares[2][0]
            z[2][1] = ~x_shares[2][1]
        z = ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)
    return z


def _lshift_private(prot, x, steps):
    """
  Left shift.
  """
    assert isinstance(x, ABY3PrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("lshift"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0] << steps
                z[i][1] = x_shares[i][1] << steps

        z = ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)

    return z


def _rshift_private(prot, x, steps):
    """
  Arithmetic right shift.
  """
    assert isinstance(x, ABY3PrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("rshift"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0] >> steps
                z[i][1] = x_shares[i][1] >> steps

        z = ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)

    return z


def _logical_rshift_private(prot, x, steps):
    """
  Logical right shift.
  """
    assert isinstance(x, ABY3PrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("logical-rshift"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0].logical_rshift(steps)
                z[i][1] = x_shares[i][1].logical_rshift(steps)

        z = ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)

    return z


def _b_add_private_private(prot, x, y):
    raise NotImplementedError(
        "Addition with boolean sharing is not implemented, and not recommended."
    )


def _b_sub_private_private(prot, x, y):
    raise NotImplementedError(
        "Sbustraction with boolean sharing is not implemented, and not recommended."
    )


def _ppa_private_private(prot, x, y, n_bits, topology="kogge_stone"):
    """
  Parallel prefix adder (PPA). This adder can be used for addition of boolean sharings.

  `n_bits` can be passed as an optimization to constrain the computation for least significant
  `n_bits` bits.

  AND Depth: log(k)
  Total gates: klog(k)
  """

    if topology == "kogge_stone":
        return _ppa_kogge_stone_private_private(prot, x, y, n_bits)
    elif topology == "sklansky":
        return _ppa_sklansky_private_private(prot, x, y, n_bits)
    else:
        raise NotImplementedError("Unknown adder topology.")


def _ppa_sklansky_private_private(prot, x, y, n_bits):
    """
  Parallel prefix adder (PPA), using the Sklansky adder topology.
  """

    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    if x.backing_dtype.native_type != tf.int64:
        raise NotImplementedError(
            "Native type {} not supported".format(x.backing_dtype.native_type)
        )

    with tf.name_scope("ppa"):
        keep_masks = [
            0x5555555555555555,
            0x3333333333333333,
            0x0F0F0F0F0F0F0F0F,
            0x00FF00FF00FF00FF,
            0x0000FFFF0000FFFF,
            0x00000000FFFFFFFF,
        ]  # yapf: disable
        copy_masks = [
            0x5555555555555555,
            0x2222222222222222,
            0x0808080808080808,
            0x0080008000800080,
            0x0000800000008000,
            0x0000000080000000,
        ]  # yapf: disable

        G = x & y
        P = x ^ y

        k = x.backing_dtype.nbits
        if n_bits is not None:
            k = n_bits
        for i in range(ceil(log2(k))):
            c_mask = prot.define_constant(
                np.ones(x.shape, dtype=np.object) * copy_masks[i],
                apply_scaling=False,
            )
            k_mask = prot.define_constant(
                np.ones(x.shape, dtype=np.object) * keep_masks[i],
                apply_scaling=False,
            )
            # Copy the selected bit to 2^i positions:
            # For example, when i=2, the 4-th bit is copied to the (5, 6, 7, 8)-th bits
            G1 = (G & c_mask) << 1
            P1 = (P & c_mask) << 1
            for j in range(i):
                G1 = (G1 << (2 ** j)) ^ G1
                P1 = (P1 << (2 ** j)) ^ P1
            """
      Two-round impl. using algo. that assume using OR gate is free, but in fact,
      here using OR gate cost one round.
      The PPA operator 'o' is defined as:
      (G, P) o (G1, P1) = (G + P*G1, P*P1), where '+' is OR, '*' is AND
      """
            # G1 and P1 are 0 for those positions that we do not copy the selected bit to.
            # Hence for those positions, the result is: (G, P) = (G, P) o (0, 0) = (G, 0).
            # In order to keep (G, P) for these positions so that they can be used in the future,
            # we need to let (G1, P1) = (G, P) for these positions, because (G, P) o (G, P) = (G, P)
            #
            # G1 = G1 ^ (G & k_mask)
            # P1 = P1 ^ (P & k_mask)
            #
            # G = G | (P & G1)
            # P = P & P1
            """
      One-round impl. by modifying the PPA operator 'o' as:
      (G, P) o (G1, P1) = (G ^ (P*G1), P*P1), where '^' is XOR, '*' is AND
      This is a valid definition: when calculating the carry bit c_i = g_i + p_i * c_{i-1},
      the OR '+' can actually be replaced with XOR '^' because we know g_i and p_i will NOT take '1'
      at the same time.
      And this PPA operator 'o' is also associative. BUT, it is NOT idempotent: (G, P) o (G, P) != (G, P).
      This does not matter, because we can do (G, P) o (0, P) = (G, P), or (G, P) o (0, 1) = (G, P)
      if we want to keep G and P bits.
      """
            # Option 1: Using (G, P) o (0, P) = (G, P)
            # P1 = P1 ^ (P & k_mask)
            # Option 2: Using (G, P) o (0, 1) = (G, P)
            P1 = P1 ^ k_mask

            G = G ^ (P & G1)
            P = P & P1

        # G stores the carry-in to the next position
        C = G << 1
        P = x ^ y
        z = C ^ P

    return z


def _ppa_kogge_stone_private_private(prot, x, y, n_bits):
    """
  Parallel prefix adder (PPA), using the Kogge-Stone adder topology.
  """

    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    if x.backing_dtype.native_type != tf.int64:
        raise NotImplementedError(
            "Native type {} not supported".format(x.backing_dtype.native_type)
        )

    with tf.name_scope("ppa"):
        keep_masks = []
        for i in range(ceil(log2(x.backing_dtype.nbits))):
            keep_masks.append((1 << (2 ** i)) - 1)
        """
    For example, if nbits = 64, then keep_masks is:
    keep_masks = [0x0000000000000001, 0x0000000000000003, 0x000000000000000f,
                  0x00000000000000ff, 0x000000000000ffff, 0x00000000ffffffff]
    """

        G = x & y
        P = x ^ y
        k = x.backing_dtype.nbits if n_bits is None else n_bits
        for i in range(ceil(log2(k))):
            k_mask = prot.define_constant(
                np.ones(x.shape, dtype=np.object) * keep_masks[i],
                apply_scaling=False,
            )

            G1 = G << (2 ** i)
            P1 = P << (2 ** i)
            """
      One-round impl. by modifying the PPA operator 'o' as:
      (G, P) o (G1, P1) = (G ^ (P*G1), P*P1), where '^' is XOR, '*' is AND
      This is a valid definition: when calculating the carry bit c_i = g_i + p_i * c_{i-1},
      the OR '+' can actually be replaced with XOR '^' because we know g_i and p_i will NOT take '1'
      at the same time.
      And this PPA operator 'o' is also associative. BUT, it is NOT idempotent: (G, P) o (G, P) != (G, P).
      This does not matter, because we can do (G, P) o (0, P) = (G, P), or (G, P) o (0, 1) = (G, P)
      if we want to keep G and P bits.
      """
            # Option 1: Using (G, P) o (0, P) = (G, P)
            # P1 = P1 ^ (P & k_mask)
            # Option 2: Using (G, P) o (0, 1) = (G, P)
            P1 = P1 ^ k_mask

            G = G ^ (P & G1)
            P = P & P1

        # G stores the carry-in to the next position
        C = G << 1
        P = x ^ y
        z = C ^ P
    return z


def _carry_private_public(prot, x, y):
    assert x.share_type == ShareType.BOOLEAN, x.share_type
    return _carry_computation(prot, x, y)


def _carry_public_private(prot, x, y):
    assert y.share_type == ShareType.BOOLEAN, y.share_type
    return _carry_computation(prot, x, y)


def _carry_private_private(prot, x, y):
    assert x.share_type == ShareType.BOOLEAN, x.share_type
    assert y.share_type == ShareType.BOOLEAN, y.share_type
    return _carry_computation(prot, x, y)

def _carry_computation(prot, x, y):
    """
    Carry circuit, using the Kogge-Stone adder topology.
    """
    assert x.backing_dtype == y.backing_dtype

    with tf.name_scope("carry"):
        G = x & y
        P = x ^ y
        k = x.backing_dtype.nbits
        while k > 1:
            G1 = prot.bit_gather(G, 1, 2).cast(prot.factories[k // 2])
            G2 = prot.bit_gather(G, 0, 2).cast(prot.factories[k // 2])
            P1 = prot.bit_gather(P, 1, 2).cast(prot.factories[k // 2])
            P2 = prot.bit_gather(P, 0, 2).cast(prot.factories[k // 2])

            G = G1 ^ (G2 & P1)
            P = P1 & P2
            k = k // 2

        # G stores the carry-in to the next position
        G = G & prot.define_constant(1, apply_scaling=False, factory=G.backing_dtype)
        G.is_scaled = False
        return G


def _a2b_private(prot, x, nbits):
    """
  Bit decomposition: Convert an arithmetic sharing to a boolean sharing.
  """
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert x.share_type == ShareType.ARITHMETIC

    x_shares = x.unwrapped
    zero = prot.define_constant(
        np.zeros(x.shape, dtype=np.int64), apply_scaling=False
    )
    zero_on_0, zero_on_1, zero_on_2 = zero.unwrapped
    a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=ShareType.BOOLEAN)

    operand1 = [[None, None], [None, None], [None, None]]
    operand2 = [[None, None], [None, None], [None, None]]
    with tf.name_scope("a2b"):
        # Step 1: We know x = ((x0, x1), (x1, x2), (x2, x0))
        # We need to reshare it into two operands that will be fed into an addition circuit:
        # operand1 = (((x0+x1) XOR a0, a1), (a1, a2), (a2, (x0+x1) XOR a0)), meaning boolean sharing of x0+x1
        # operand2 = ((0, 0), (0, x2), (x2, 0)), meaning boolean sharing of x2
        with tf.device(prot.servers[0].device_name):
            x0_plus_x1 = x_shares[0][0] + x_shares[0][1]
            operand1[0][0] = x0_plus_x1 ^ a0
            operand1[0][1] = a1

            operand2[0][0] = zero_on_0
            operand2[0][1] = zero_on_0

        with tf.device(prot.servers[1].device_name):
            operand1[1][0] = a1
            operand1[1][1] = a2

            operand2[1][0] = zero_on_1
            operand2[1][1] = x_shares[1][1]

        with tf.device(prot.servers[2].device_name):
            operand1[2][0] = a2
            operand1[2][1] = operand1[0][0]

            operand2[2][0] = x_shares[2][0]
            operand2[2][1] = zero_on_2

        operand1 = ABY3PrivateTensor(prot, operand1, x.is_scaled, ShareType.BOOLEAN)
        operand2 = ABY3PrivateTensor(prot, operand2, x.is_scaled, ShareType.BOOLEAN)

        # Step 2: Parallel prefix adder that requires log(k) rounds of communication
        result = prot.ppa(operand1, operand2, nbits)

    return result


def _bit_extract_private(prot, x, i):
    """
  Bit extraction: Extracts the `i`-th bit of an arithmetic sharing or boolean sharing
  to a single-bit boolean sharing.
  """
    assert isinstance(x, ABY3PrivateTensor), type(x)

    with tf.name_scope("bit_extract"):
        if x.share_type == ShareType.ARITHMETIC:
            with tf.name_scope("a2b_partial"):
                x_shares = x.unwrapped
                zero = prot.define_constant(
                    np.zeros(x.shape, dtype=np.int64),
                    apply_scaling=False
                )
                zero_on_0, zero_on_1, zero_on_2 = zero.unwrapped
                a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=ShareType.BOOLEAN)

                operand1 = [[None, None], [None, None], [None, None]]
                operand2 = [[None, None], [None, None], [None, None]]
                # Step 1: We know x = ((x0, x1), (x1, x2), (x2, x0))
                # We need to reshare it into two operands that will be fed into an addition circuit:
                # operand1 = (((x0+x1) XOR a0, a1), (a1, a2), (a2, (x0+x1) XOR a0)), meaning boolean sharing of x0+x1
                # operand2 = ((0, 0), (0, x2), (x2, 0)), meaning boolean sharing of x2
                with tf.device(prot.servers[0].device_name):
                    x0_plus_x1 = x_shares[0][0] + x_shares[0][1]
                    operand1[0][0] = x0_plus_x1 ^ a0
                    operand1[0][1] = a1

                    operand2[0][0] = zero_on_0
                    operand2[0][1] = zero_on_0

                with tf.device(prot.servers[1].device_name):
                    operand1[1][0] = a1
                    operand1[1][1] = a2

                    operand2[1][0] = zero_on_1
                    operand2[1][1] = x_shares[1][1]

                with tf.device(prot.servers[2].device_name):
                    operand1[2][0] = a2
                    operand1[2][1] = operand1[0][0]

                    operand2[2][0] = x_shares[2][0]
                    operand2[2][1] = zero_on_2

                operand1 = ABY3PrivateTensor(prot, operand1, x.is_scaled, ShareType.BOOLEAN)
                operand2 = ABY3PrivateTensor(prot, operand2, x.is_scaled, ShareType.BOOLEAN)

                # Step 2: Parallel prefix adder that requires log(i+1) rounds of communication
                x = prot.ppa(operand1, operand2, i + 1)

        # Take out the i-th bit
        #
        # NOTE: Don't use x = x & 0x1. Even though we support automatic lifting of 0x1
        # to an ABY3Tensor, but it also includes automatic scaling to make the two operands have
        # the same scale, which is not what want here.
        #
        mask = prot.define_constant(
            np.array([0x1 << i]), apply_scaling=False
        )
        x = x & mask

        x_shares = x.unwrapped
        result = [[None, None], [None, None], [None, None]]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                result[i][0] = x_shares[i][0].cast(prot.factories[tf.bool])
                result[i][1] = x_shares[i][1].cast(prot.factories[tf.bool])
        result = ABY3PrivateTensor(prot, result, False, ShareType.BOOLEAN)

    return result

def _msb_private(prot, x):

    with tf.name_scope("msb"):
        mask = prot.define_constant(1, apply_scaling=False, factory=x.backing_dtype)

        if x.share_type == ShareType.BOOLEAN:
            z = (x >> (x.backing_dtype.nbits-1)) & mask
            z = z.cast(prot.factories[tf.bool])

        elif x.share_type == ShareType.ARITHMETIC:
            x_shares = x.unwrapped
            zero = prot.define_constant(
                np.zeros(x.shape, dtype=np.int64),
                apply_scaling=False
            )
            zero_on_0, zero_on_1, zero_on_2 = zero.unwrapped
            a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=ShareType.BOOLEAN)

            operand1 = [[None, None], [None, None], [None, None]]
            operand2 = [[None, None], [None, None], [None, None]]
            # Step 1: We know x = ((x0, x1), (x1, x2), (x2, x0))
            # We need to reshare it into two operands that will be fed into an addition circuit:
            # operand1 = (((x0+x1) XOR a0, a1), (a1, a2), (a2, (x0+x1) XOR a0)), meaning boolean sharing of x0+x1
            # operand2 = ((0, 0), (0, x2), (x2, 0)), meaning boolean sharing of x2
            with tf.device(prot.servers[0].device_name):
                x0_plus_x1 = x_shares[0][0] + x_shares[0][1]
                operand1[0][0] = x0_plus_x1 ^ a0
                operand1[0][1] = a1

                operand2[0][0] = zero_on_0
                operand2[0][1] = zero_on_0

            with tf.device(prot.servers[1].device_name):
                operand1[1][0] = a1
                operand1[1][1] = a2

                operand2[1][0] = zero_on_1
                operand2[1][1] = x_shares[1][1]

            with tf.device(prot.servers[2].device_name):
                operand1[2][0] = a2
                operand1[2][1] = operand1[0][0]

                operand2[2][0] = x_shares[2][0]
                operand2[2][1] = zero_on_2

            operand1 = ABY3PrivateTensor(prot, operand1, x.is_scaled, ShareType.BOOLEAN)
            operand2 = ABY3PrivateTensor(prot, operand2, x.is_scaled, ShareType.BOOLEAN)

            # Step 2: Carry circuit that requires log(i+1) rounds of communication
            carry = prot.carry(operand1 << 1, operand2 << 1)
            P = (((operand1 ^ operand2) >> (x.backing_dtype.nbits - 1)) & mask).cast(carry.backing_dtype)
            z = (carry ^ P).cast(prot.factories[tf.bool])

        z.is_scaled=False

    return z


def _b2a_private(prot, x, nbits):
    """
  Bit composition: Convert a boolean sharing to an arithmetic sharing.
  """
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert x.share_type == ShareType.BOOLEAN

    # In semi-honest, the following two calls can be further optimized because we don't
    # need the boolean shares of x1 and x2. We only need their original values on intended servers.
    x1_on_0, x1_on_1, x1_on_2, x1_shares = prot._gen_b2a_sharing(
        x.shape, prot.b2a_keys_1
    )
    assert x1_on_2 is None
    x2_on_0, x2_on_1, x2_on_2, x2_shares = prot._gen_b2a_sharing(
        x.shape, prot.b2a_keys_2
    )
    assert x2_on_0 is None

    a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=ShareType.BOOLEAN)

    with tf.name_scope("b2a"):
        # Server 1 reshares (-x1-x2) as private input
        neg_x1_neg_x2 = [[None, None], [None, None], [None, None]]
        with tf.device(prot.servers[1].device_name):
            value = -x1_on_1 - x2_on_1
            neg_x1_neg_x2[1][0] = value ^ a1
            neg_x1_neg_x2[1][1] = a2
        with tf.device(prot.servers[0].device_name):
            neg_x1_neg_x2[0][0] = a0
            neg_x1_neg_x2[0][1] = neg_x1_neg_x2[1][0]
        with tf.device(prot.servers[2].device_name):
            neg_x1_neg_x2[2][0] = a2
            neg_x1_neg_x2[2][1] = a0
        neg_x1_neg_x2 = ABY3PrivateTensor(prot, neg_x1_neg_x2, x.is_scaled, ShareType.BOOLEAN)

        # Compute x0 = x + (-x1-x2) using the parallel prefix adder
        x0 = prot.ppa(x, neg_x1_neg_x2, nbits)

        # Reveal x0 to server 0 and 2
        with tf.device(prot.servers[0].device_name):
            x0_on_0 = prot._reconstruct(x0.unwrapped, prot.servers[0], ShareType.BOOLEAN)
        with tf.device(prot.servers[2].device_name):
            x0_on_2 = prot._reconstruct(x0.unwrapped, prot.servers[2], ShareType.BOOLEAN)

        # Construct the arithmetic sharing
        result = [[None, None], [None, None], [None, None]]
        with tf.device(prot.servers[0].device_name):
            result[0][0] = x0_on_0
            result[0][1] = x1_on_0
        with tf.device(prot.servers[1].device_name):
            result[1][0] = x1_on_1
            result[1][1] = x2_on_1
        with tf.device(prot.servers[2].device_name):
            result[2][0] = x2_on_2
            result[2][1] = x0_on_2
        result = ABY3PrivateTensor(prot, result, x.is_scaled, ShareType.ARITHMETIC)

    return result


def _b2a_single_private(prot, x):
    assert x.share_type == ShareType.BOOLEAN
    assert x.backing_dtype == prot.factories[tf.bool]

    # TODO: this can be improved with 3pc COT

    a = prot.define_constant(np.ones(x.shape), apply_scaling=False)
    return _mul_ab_public_private(prot, a, x)


def _mul_ab_public_private(prot, x, y):
    assert isinstance(x, ABY3PublicTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(x)
    assert y.share_type == ShareType.BOOLEAN

    x_on_0, x_on_1, x_on_2 = x.unwrapped

    with tf.name_scope("mul_ab"):
        z = __mul_ab_routine(prot, x_on_2, y, 2)
        z = ABY3PrivateTensor(prot, z, x.is_scaled, ShareType.ARITHMETIC)

    return z


def _mul_ab_private_private(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)
    assert x.share_type == ShareType.ARITHMETIC
    assert y.share_type == ShareType.BOOLEAN

    x_shares = x.unwrapped

    with tf.name_scope("mul_ab"):
        with tf.name_scope("term0"):
            w = __mul_ab_routine(prot, x_shares[0][0], y, 0)
            w = ABY3PrivateTensor(prot, w, x.is_scaled, ShareType.ARITHMETIC)

        with tf.name_scope("term1"):
            with tf.device(prot.servers[1].device_name):
                a = x_shares[1][0] + x_shares[1][1]
            z = __mul_ab_routine(prot, a, y, 1)
            z = ABY3PrivateTensor(prot, z, x.is_scaled, ShareType.ARITHMETIC)
        z = w + z

    return z


def __mul_ab_routine(prot, a, b, sender_idx):
    """
    A sub routine for multiplying a value 'a' (located at servers[sender_idx]) with a boolean sharing 'b'.
    """
    assert isinstance(a, AbstractTensor), type(a)
    assert isinstance(b, ABY3PrivateTensor), type(b)

    with tf.name_scope("__mul_ab_routine"):
        b_shares = b.unwrapped
        s = [None, None, None]
        s[0], s[1], s[2] = prot._gen_zero_sharing(a.shape, ShareType.ARITHMETIC)

        z = [[None, None], [None, None], [None, None]]
        idx0 = sender_idx
        idx1 = (sender_idx + 1) % 3
        idx2 = (sender_idx + 2) % 3
        with tf.device(prot.servers[idx0].device_name):
            z[idx0][0] = s[idx2]
            z[idx0][1] = s[idx1]
            tmp = (b_shares[idx0][0] ^ b_shares[idx0][1]).cast(a.factory) * a
            m0 = tmp + s[idx0]
            m1 = -tmp + a + s[idx0]

        with tf.device(prot.servers[idx1].device_name):
            z[idx1][0] = s[idx1]
            z[idx1][1] = prot._ot(
                prot.servers[idx0],
                prot.servers[idx1],
                prot.servers[idx2],
                m0,
                m1,
                b_shares[idx1][1],
                b_shares[idx2][0],
                prot.pairwise_keys[idx0][0],
                prot.pairwise_keys[idx2][1],
                prot.pairwise_nonces[idx2],
            )
            prot.pairwise_nonces[idx2] = prot.pairwise_nonces[idx2] + 1

        with tf.device(prot.servers[idx2].device_name):
            z[idx2][0] = prot._ot(
                prot.servers[idx0],
                prot.servers[idx2],
                prot.servers[idx1],
                m0,
                m1,
                b_shares[idx2][0],
                b_shares[idx1][1],
                prot.pairwise_keys[idx0][1],
                prot.pairwise_keys[idx1][0],
                prot.pairwise_nonces[idx0],
            )
            z[idx2][1] = s[idx2]
            prot.pairwise_nonces[idx0] = prot.pairwise_nonces[idx0] + 1

    return z


def _pow_private(prot, x, p):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert x.share_type == ShareType.ARITHMETIC
    assert p >= 1, "Exponent should be >= 0"

    # NOTE: pow should be able to use the `memoir` memoization

    with tf.name_scope("pow"):
        result = 1
        tmp = x
        while p > 0:
            bit = p & 0x1
            if bit > 0:
                result = result * tmp
            p >>= 1
            if p > 0:
                tmp = tmp * tmp
    return result


def _polynomial_private(prot, x, coeffs):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert x.share_type == ShareType.ARITHMETIC

    with tf.name_scope("polynomial"):
        result = prot.define_constant(np.zeros(x.shape), apply_scaling=x.is_scaled)
        for i in range(len(coeffs)):
            if i == 0:
                result = result + coeffs[i]
            elif coeffs[i] == 0:
                continue
            elif (coeffs[i] - int(coeffs[i])) == 0:
                # Optimization when coefficient is integer: mulitplication can be performed
                # locally without interactive truncation
                tmp = prot.define_constant(np.array([coeffs[i]]), apply_scaling=False)
                tmp = tmp * (x ** i)
                result = result + tmp
            else:
                tmp = coeffs[i] * (x ** i)
                result = result + tmp
    return result


def _polynomial_piecewise_private(prot, x, c, coeffs):
    """
  :param prot:
  :param x:
  :param c: A list of splitting points between pieces
  :param coeffs: Two-dimensional list: 1st dimension is the polynomial index, 2nd dimension is the coefficient index
  :return:
  """
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert len(c) + 1 == len(coeffs), "# of pieces do not match # of polynomials"

    with tf.name_scope("polynomial_piecewise"):
        # Compute the selection bit for each polynomial
        with tf.name_scope("polynomial-selection-bit"):
            msbs = [None] * len(c)
            for i in range(len(c)):
                msbs[i] = prot.msb(x - c[i])
            b = [None] * len(coeffs)
            b[0] = msbs[0]
            for i in range(len(c) - 1):
                b[i + 1] = ~msbs[i] & msbs[i + 1]
            b[len(c)] = ~msbs[len(c) - 1]

        # Compute the piecewise combination result
        result = 0
        for i in range(len(coeffs)):
            fi = prot.polynomial(x, coeffs[i])
            result = result + prot.mul_ab(fi, b[i])
    return result


def _sigmoid_private(prot, x, approx_type):
    assert isinstance(x, ABY3PrivateTensor), type(x)

    with tf.name_scope("sigmoid"):
        if approx_type == "piecewise_linear":
            c = (-2.5, 2.5)
            coeffs = ((1e-4,), (0.50, 0.17), (1 - 1e-4,))
        else:
            raise NotImplementedError(
                "Only support piecewise linear approximation of sigmoid."
            )

        result = prot.polynomial_piecewise(x, c, coeffs)
    return result


#
# transpose helpers
#


def _transpose_private(prot, x, perm=None):
    assert isinstance(x, ABY3PrivateTensor)

    x_shares = x.unwrapped

    with tf.name_scope("transpose"):
        z = [[None, None], [None, None], [None, None]]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0].transpose(perm=perm)
                z[i][1] = x_shares[i][1].transpose(perm=perm)

        return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


def _transpose_public(prot, x, perm=None):
    assert isinstance(x, ABY3PublicTensor)

    x_on_0, x_on_1, x_on_2 = x.unwrapped

    with tf.name_scope("transpose"):

        with tf.device(prot.servers[0].device_name):
            x_on_0_t = x_on_0.transpose(perm=perm)

        with tf.device(prot.servers[1].device_name):
            x_on_1_t = x_on_1.transpose(perm=perm)

        with tf.device(prot.servers[2].device_name):
            x_on_2_t = x_on_2.transpose(perm=perm)

        return ABY3PublicTensor(
            prot, [x_on_0_t, x_on_1_t, x_on_2_t], x.is_scaled
        )


#
# reduce_sum helpers
#


def _reduce_sum_public(prot, x, axis=None, keepdims=False):

    x_on_0, x_on_1, x_on_2 = x.unwrapped

    with tf.name_scope("reduce_sum"):

        with tf.device(prot.servers[0].device_name):
            y_on_0 = x_on_0.reduce_sum(axis, keepdims)

        with tf.device(prot.servers[1].device_name):
            y_on_1 = x_on_1.reduce_sum(axis, keepdims)

        with tf.device(prot.servers[2].device_name):
            y_on_2 = x_on_2.reduce_sum(axis, keepdims)

    return ABY3PublicTensor(prot, [y_on_0, y_on_1, y_on_2], x.is_scaled)


def _reduce_sum_private(prot, x, axis=None, keepdims=False):

    x_shares = x.unwrapped

    with tf.name_scope("reduce_sum"):
        z = [[None, None], [None, None], [None, None]]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = x_shares[i][0].reduce_sum(axis, keepdims)
                z[i][1] = x_shares[i][1].reduce_sum(axis, keepdims)
    return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


#
# concat helpers
#


def _concat_public(prot, xs, axis):
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    xs_on_0, xs_on_1, xs_on_2 = zip(*(x.unwrapped for x in xs))

    with tf.name_scope("concat"):

        with tf.device(prot.servers[0].device_name):
            x_on_0_concat = factory.concat(xs_on_0, axis=axis)

        with tf.device(prot.servers[1].device_name):
            x_on_1_concat = factory.concat(xs_on_1, axis=axis)

        with tf.device(prot.servers[2].device_name):
            x_on_2_concat = factory.concat(xs_on_2, axis=axis)

        return ABY3PublicTensor(
            prot,
            [x_on_0_concat, x_on_1_concat, x_on_2_concat],
            is_scaled
        )


def _concat_private(prot, xs, axis):
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    share_type = xs[0].share_type

    xs_shares = [x.unwrapped for x in xs]
    z = [[None, None], [None, None], [None, None]]
    for i in range(3):
        z[i][0] = [x_shares[i][0] for x_shares in xs_shares]
        z[i][1] = [x_shares[i][1] for x_shares in xs_shares]

    with tf.name_scope("concat"):

        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = factory.concat(z[i][0], axis=axis)
                z[i][1] = factory.concat(z[i][1], axis=axis)

        return ABY3PrivateTensor(prot, z, is_scaled, share_type)


def _stack_public(prot, xs, axis):
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    xs_on_0, xs_on_1, xs_on_2 = zip(*(x.unwrapped for x in xs))

    with tf.name_scope("stack-public"):

        with tf.device(prot.servers[0].device_name):
            x_on_0_stack = factory.stack(xs_on_0, axis=axis)

        with tf.device(prot.servers[1].device_name):
            x_on_1_stack = factory.stack(xs_on_1, axis=axis)

        with tf.device(prot.servers[2].device_name):
            x_on_2_stack = factory.stack(xs_on_2, axis=axis)

        return ABY3PublicTensor(
            prot,
            [x_on_0_stack, x_on_1_stack, x_on_2_stack],
            is_scaled
        )


def _stack_private(prot, xs, axis):
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    share_type = xs[0].share_type

    xs_shares = [x.unwrapped for x in xs]
    z = [[None, None], [None, None], [None, None]]
    for i in range(3):
        z[i][0] = [x_shares[i][0] for x_shares in xs_shares]
        z[i][1] = [x_shares[i][1] for x_shares in xs_shares]

    with tf.name_scope("stack-private"):

        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = factory.stack(z[i][0], axis=axis)
                z[i][1] = factory.stack(z[i][1], axis=axis)

        return ABY3PrivateTensor(prot, z, is_scaled, share_type)


def _expand_dims_public(prot, x, axis):

    xs = x.unwrapped

    with tf.name_scope("expand-dims-public"):
        z = [None, None, None]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i] = xs[i].expand_dims(axis=axis)

        return ABY3PublicTensor(prot, z, x.is_scaled)


def _expand_dims_private(prot, x, axis):

    xs = x.unwrapped

    with tf.name_scope("expand-dims-private"):
        z = [[None, None], [None, None], [None, None]]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = xs[i][0].expand_dims(axis=axis)
                z[i][1] = xs[i][1].expand_dims(axis=axis)

        return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


def _squeeze_public(prot, x, axis):

    xs = x.unwrapped

    with tf.name_scope("squeeze-public"):
        z = [None, None, None]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i] = xs[i].squeeze(axis=axis)

        return ABY3PublicTensor(prot, z, x.is_scaled)


def _squeeze_private(prot, x, axis):

    xs = x.unwrapped

    with tf.name_scope("squeeze-private"):
        z = [[None, None], [None, None], [None, None]]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = xs[i][0].squeeze(axis=axis)
                z[i][1] = xs[i][1].squeeze(axis=axis)

        return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


def _strided_slice_public(prot, x, args, kwargs):

    xs = x.unwrapped

    with tf.name_scope("strided-slice-public"):
        z = [None, None, None]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i] = xs[i].strided_slice(args, kwargs)

        return ABY3PublicTensor(prot, z, x.is_scaled)


def _strided_slice_private(prot, x, args, kwargs):

    xs = x.unwrapped

    with tf.name_scope("strided-slice-private"):
        z = [[None, None], [None, None], [None, None]]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                z[i][0] = xs[i][0].strided_slice(args, kwargs)
                z[i][1] = xs[i][1].strided_slice(args, kwargs)

        return ABY3PrivateTensor(prot, z, x.is_scaled, x.share_type)


def _write_private(prot, x, filename_prefix):
    assert isinstance(x, ABY3PrivateTensor), type(x)

    def encode(feature_row):
        # Converting a row to a string seems to be the only way of writing out
        # the dataset in a distributed way
        feature = tf.strings.reduce_join(
            tf.dtypes.as_string(tf.reshape(feature_row, [-1])), separator=","
        )
        return feature

    x_shares = x.unwrapped
    ops = []
    for i in range(3):
        with tf.device(prot.servers[i].device_name):
            for j in range(2):
                data = tf.data.Dataset.from_tensor_slices(x_shares[i][j].value).map(
                    encode
                )
                writer = tf.data.experimental.TFRecordWriter(
                    "{}_share{}{}".format(filename_prefix, i, j)
                )
                ops.append(writer.write(data))

    return tf.group(*ops)


def _read_(prot, filename_prefix, batch_size, n_columns):

    row_shape = [n_columns]

    def decode(line):
        fields = tf.string_split([line], ",").values
        fields = tf.strings.to_number(fields, tf.int64)
        fields = tf.reshape(fields, row_shape)
        return fields

    batch = [[None] * 2 for _ in range(3)]
    for i in range(3):
        with tf.device(prot.servers[i].device_name):
            for j in range(2):
                data = (
                    tf.data.TFRecordDataset(
                        ["{}_share{}{}".format(filename_prefix, i, j)]
                    )
                    .map(decode)
                    .repeat()
                    .batch(batch_size=batch_size)
                )
                it = data.make_one_shot_iterator()
                batch[i][j] = it.get_next()
                batch[i][j] = tf.reshape(batch[i][j], [batch_size] + row_shape)
                batch[i][j] = prot.default_factory.tensor(batch[i][j])

    return ABY3PrivateTensor(prot, batch, True, ShareType.ARITHMETIC)


def _iterate_private(
    prot,
    tensor: "ABY3PrivateTensor",
    batch_size: int,
    repeat=True,
    shuffle=True,
    seed: int = None,
):

    assert isinstance(tensor, ABY3PrivateTensor)
    shares = tensor.unwrapped
    iterators = [[None] * 2 for _ in range(3)]
    results = [[None] * 2 for _ in range(3)]

    if seed is None:
        seed = np.random.randint(1, 1 << 32)  # this seed is publicly known.
    batch_size = max(1, batch_size)

    def helper(idx):
        with tf.device(prot.servers[idx].device_name):
            out_shape = shares[idx][0].value.shape.as_list()
            out_shape[0] = batch_size
            for i in range(2):
                dataset = tf.data.Dataset.from_tensor_slices(shares[idx][i].value)

                if repeat:
                    dataset = dataset.repeat()

                if shuffle:
                    dataset = dataset.shuffle(buffer_size=512, seed=seed)

                dataset = dataset.batch(batch_size)

                # NOTE: initializable_iterator needs to run initializer.
                iterators[idx][i] = tf.compat.v1.data.make_initializable_iterator(
                    dataset
                )
                batch = iterators[idx][i].get_next()
                # Wrap the tf.tensor as a dense tensor (no extra encoding is needed)
                results[idx][i] = prot.default_factory.tensor(tf.reshape(batch, out_shape))

            prot._initializers.append(
                tf.group(iterators[idx][0].initializer, iterators[idx][1].initializer)
            )

    for idx in range(3):
        helper(idx)

    # Synchronize the reading of all 6 dataset iterators
    with tf.control_dependencies(
        [share.value for result in results for share in result]
    ):
        for i in range(3):
            results[i][0] = results[i][0].identity()
            results[i][1] = results[i][1].identity()

    return ABY3PrivateTensor(prot, results, tensor.is_scaled, tensor.share_type)


def _indexer_private(prot: ABY3, tensor: ABY3PrivateTensor, slc) -> "ABY3PrivateTensor":
    shares = tensor.unwrapped
    results = [[None] * 2 for _ in range(3)]
    with tf.name_scope("index"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                results[i][0] = shares[i][0][slc]
                results[i][1] = shares[i][1][slc]
    return ABY3PrivateTensor(prot, results, tensor.is_scaled, tensor.share_type)


def _reshape_private(prot: ABY3, tensor: ABY3PrivateTensor, axe):
    shares = tensor.unwrapped
    results = [[None] * 2 for _ in range(3)]
    with tf.name_scope("reshape"):
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                results[i][0] = shares[i][0].reshape(axe)
                results[i][1] = shares[i][1].reshape(axe)
    return ABY3PrivateTensor(prot, results, tensor.is_scaled, tensor.share_type)
