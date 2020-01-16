"""
Implementation of the ABY3 framework.
"""
from __future__ import absolute_import
from typing import Tuple, List, Union, Optional, Callable
import abc
import sys
from math import log2, ceil
from functools import reduce

import numpy as np
import tensorflow as tf

from ...tensor.factory import (
    AbstractFactory,
    AbstractTensor,
    AbstractConstant,
)
from ...tensor.helpers import inverse
from ...tensor.fixed import FixedpointConfig, _validate_fixedpoint_config
from ...tensor import fixed64, fixed64_ni
from ...tensor.native import native_factory
from ...tensor.boolfactory import bool_factory
from ...player import Player
from ...config import get_config
from ..protocol import Protocol, memoize
from ...operations import secure_random as crypto

TFEInputter = Callable[[], Union[List[tf.Tensor], tf.Tensor]]
TF_NATIVE_TYPES = [tf.bool, tf.int8, tf.int16, tf.int32, tf.int64]

_THISMODULE = sys.modules[__name__]

# ===== Share types =====
ARITHMETIC = 0
BOOLEAN = 1

# ===== Factory =====
i64_factory = native_factory(tf.int64)
b_factory = bool_factory()


class ABY3(Protocol):
  """ABY3 framework."""

  def __init__(
      self,
      server_0=None,
      server_1=None,
      server_2=None,
      use_noninteractive_truncation=True,
  ):
    self._initializers = list()
    config = get_config()
    self.servers = [None, None, None]
    self.servers[0] = config.get_player(server_0 if server_0 else "server0")
    self.servers[1] = config.get_player(server_1 if server_1 else "server1")
    self.servers[2] = config.get_player(server_2 if server_2 else "server2")

    int_factory = i64_factory

    if use_noninteractive_truncation:
      fixedpoint_config = fixed64_ni
    else:
      fixedpoint_config = fixed64

    self.fixedpoint_config = fixedpoint_config
    self.int_factory = int_factory
    self.bool_factory = b_factory

    self.pairwise_keys, self.pairwise_nonces = self.setup_pairwise_randomness()
    self.b2a_keys_1, self.b2a_keys_2, self.b2a_nonce = self.setup_b2a_generator()

  @property
  def nbits(self):
    return self.int_factory.nbits

  def setup_pairwise_randomness(self):
    """
    Initial setup for pairwise randomness: Every two parties hold a shared key.
    """
    if not crypto.supports_seeded_randomness():
      raise NotImplementedError("Secure randomness implementation is not available.")

    keys = [[None, None], [None, None], [None, None]]
    with tf.device(self.servers[0].device_name):
      seed_0 = crypto.secure_seed()
    with tf.device(self.servers[1].device_name):
      seed_1 = crypto.secure_seed()
    with tf.device(self.servers[2].device_name):
      seed_2 = crypto.secure_seed()

    # Replicated keys
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

    # TODO: Think about the security: Do we really need PRF for zero sharing?
    #
    # According to the discussion here:
    # https://crypto.stackexchange.com/questions/5333/difference-between-stream-cipher-and-block-cipher
    # stream ciphers can also be treated as a keyed pseudorandom function family, like block ciphers.
    # And the underlying C++ implementation of the secure_random.py uses exactly chacha20 stream cipher from
    # the libsodium library. Therefore, if we can treated stream ciphers as pseudorandom functions, then it
    # should be fine to directly use G(k + id) to generate the random numbers, where the seed "k + id" is
    # used as the key of the stream cipher in the C++ code: generators.h
    #
    # Otherwise, the absolutely secure way is to implement our own PRF by using block ciphers.
    # In practice, people normally use stream ciphers for PRG, and use block ciphers for PRF.

    return keys, nonces

  def setup_b2a_generator(self):
    """
    Initial setup for generating shares during the conversion
    from boolean sharing to arithmetic sharing

    TODO: Think about the security: Do we really need PRF?
    """

    if not crypto.supports_seeded_randomness():
      raise NotImplementedError("Secure randomness implementation is not available.")

    # Type 1: Server 0 and 1 hold three keys, while server 2 holds two
    b2a_keys_1 = [[None, None, None], [None, None, None], [None, None, None]]
    with tf.device(self.servers[0].device_name):
      seed_0 = crypto.secure_seed()
    with tf.device(self.servers[1].device_name):
      seed_1 = crypto.secure_seed()
    with tf.device(self.servers[2].device_name):
      seed_2 = crypto.secure_seed()

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
    with tf.device(self.servers[0].device_name):
      seed_0 = crypto.secure_seed()
    with tf.device(self.servers[1].device_name):
      seed_1 = crypto.secure_seed()
    with tf.device(self.servers[2].device_name):
      seed_2 = crypto.secure_seed()

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
      share_type=ARITHMETIC,
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

    factory = factory or self.int_factory

    value = self._encode(value, apply_scaling)
    with tf.name_scope("constant{}".format("-" + name if name else "")):
      with tf.device(self.servers[0].device_name):
        x_on_0 = factory.constant(value)

      with tf.device(self.servers[1].device_name):
        x_on_1 = factory.constant(value)

      with tf.device(self.servers[2].device_name):
        x_on_2 = factory.constant(value)

    return ABY3Constant(self, x_on_0, x_on_1, x_on_2, apply_scaling, share_type)

  def define_private_variable(
      self,
      initial_value,
      apply_scaling: bool = True,
      share_type=ARITHMETIC,
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

            if isinstance(initial_value, np.ndarray):
                initial_value = self._encode(initial_value, apply_scaling)
                v = factory.tensor(initial_value)
                shares = self._share(v, share_type=share_type)
    factory = factory or self.int_factory
    suffix = "-" + name if name else ""

            elif isinstance(initial_value, tf.Tensor):
                initial_value = self._encode(initial_value, apply_scaling)
                v = factory.tensor(initial_value)
                shares = self._share(v, share_type=share_type)
    with tf.name_scope("private-var{}".format(suffix)):

            elif isinstance(initial_value, ABY3PrivateTensor):
                shares = initial_value.unwrapped

            else:
                raise TypeError(("Don't know how to turn {} "
                                 "into private variable").format(type(initial_value)))

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
        self._initializers.append(x.initializer)
        return x

    def define_local_computation(
            self,
            player,
            computation_fn,
            arguments=None,
            apply_scaling=True,
            share_type=ARITHMETIC,
            name=None,
            factory=None):
        """
        Define a local computation that happens on plaintext tensors.

        :param player: Who performs the computation and gets to see the values in
            plaintext.
        :param apply_scaling: Whether or not to scale the outputs.
        :param name: Optional name to give to this node in the graph.
        :param factory: Backing tensor type to use for outputs.
        """

        factory = factory or self.int_factory

        if isinstance(player, str):
            player = get_config().get_player(player)
        assert isinstance(player, Player)

        def share_output(v: tf.Tensor):
            assert v.shape.is_fully_defined(), ("Shape of return value '{}' on '{}' "
                                                "not fully defined").format(
                name if name else "",
                player.name,
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

            raise TypeError(("Don't know how to process input argument "
                             "of type {}").format(type(x)))

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

                raise TypeError("Don't know how to handle results of "
                                "type {}".format(type(outputs)))

    def define_private_input(
            self,
            player,
            inputter_fn,
            apply_scaling: bool = True,
            share_type=ARITHMETIC,
            name: Optional[str] = None,
            factory: Optional[AbstractFactory] = None,
    ):
        """
        Define a private input.

        This represents a `private` input owned by the specified player into the
        graph.

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
            share_type = share_type,
            name="private-input{}".format(suffix),
            factory=factory,
        )

    def define_public_input(
            self,
            player: Union[str, Player],
            inputter_fn: TFEInputter,
            apply_scaling: bool = True,
            share_type=ARITHMETIC,
            name: Optional[str] = None,
            factory: Optional[AbstractFactory] = None
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

        factory = factory or self.int_factory
        suffix = "-" + name if name else ""

        def helper(v: tf.Tensor) -> "ABY3PublicTensor":
            assert v.shape.is_fully_defined(), ("Shape of input '{}' on '{}' is not "
                                                "fully defined").format(
                name if name else "",
                player.name,
            )
            v = self._encode(v, apply_scaling)
            w = factory.tensor(v)
            return ABY3PublicTensor(self, w, w, w, apply_scaling, share_type)

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

                raise TypeError(("Don't know how to handle inputs "
                                 "of type {}").format(type(inputs)))

    def define_public_tensor(
            self,
            tensor: tf.Tensor,
            apply_scaling: bool = True,
            share_type=ARITHMETIC,
            name: Optional[str] = None,
            factory: Optional[AbstractFactory] = None
    ):
        """
        Convert a tf.Tensor to an ABY3PublicTensor.
        """
        assert isinstance(tensor, tf.Tensor)
        assert tensor.shape.is_fully_defined(), ("Shape of input '{}' is not "
                                                 "fully defined").format(name if name else "")

        factory = factory or self.int_factory

        with tf.name_scope("public-tensor"):
            tensor = self._encode(tensor, apply_scaling)
            w = factory.tensor(tensor)
            return ABY3PublicTensor(self, w, w, w, apply_scaling, share_type)

    def define_output(
            self,
            player,
            arguments,
            outputter_fn,
            name=None,
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

    @property
    def initializer(self) -> tf.Operation:
        return tf.group(*self._initializers)

    def add_initializers(self, *initializers):
        self._initializers.append(tf.group(*initializers))

    def clear_initializers(self) -> None:
        del self._initializers[:]

    def _encode(self,
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
                factory = factory or self.int_factory
                tf_native_type = factory.native_type
                assert tf_native_type in TF_NATIVE_TYPES
                integers = tf.cast(scaled, dtype=tf_native_type)

            else:
                raise TypeError("Don't know how to encode {}".format(type(rationals)))

            assert type(rationals) == type(integers)
            return integers

    @memoize
    def _decode(self,
                elements: AbstractTensor,
                is_scaled: bool) -> tf.Tensor:
        """Decode tensor of ring elements into tensor of rational numbers."""

        with tf.name_scope("decode"):
            scaled = elements.to_native()
            if not is_scaled:
                return scaled
            return scaled / self.fixedpoint_config.scaling_factor

    def _share(
            self,
            secret: AbstractTensor,
            share_type: str,
            player=None
    ):
        """Secret-share an AbstractTensor.

        Args:
          secret: `AbstractTensor`, the tensor to share.

        Returns:
          A pair of `AbstractTensor`, the shares.
        """

        with tf.name_scope("share"):
            if share_type == ARITHMETIC or share_type == BOOLEAN:
                share0 = secret.factory.sample_uniform(secret.shape)
                share1 = secret.factory.sample_uniform(secret.shape)
                if share_type == ARITHMETIC:
                    share2 = secret - share0 - share1
                elif share_type == BOOLEAN:
                    share2 = secret ^ share0 ^ share1
                # Replicated sharing
                shares = ((share0, share1), (share1, share2), (share2, share0))
                return shares

            else:
                raise NotImplementedError("Unknown share type.")

    def _share_and_wrap(self,
                        secret: AbstractTensor,
                        is_scaled: bool,
                        share_type: str,
                        player=None) -> "ABY3PrivateTensor":
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
            if share_type == ARITHMETIC:
                return s0 + s1 + s2
            elif share_type == BOOLEAN:
                return s0 ^ s1 ^ s2
            else:
                raise NotImplementedError("Only arithmetic and boolean sharings are supported.")

        with tf.name_scope("reconstruct"):
            if share_type == ARITHMETIC or share_type == BOOLEAN:
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

    def _gen_zero_sharing(self, shape, share_type=ARITHMETIC, factory=None):

        def helper(f0, f1):
            if share_type == ARITHMETIC:
                return f0 - f1
            elif share_type == BOOLEAN:
                return f0 ^ f1
            else:
                raise NotImplementedError("Only arithmetic and boolean sharings are supported.")

        # TODO: (Zico) Think about "graph building" vs "session execution". Are we generating the same zero sharings
        # TODO: in every session execution? ...NO, because the seeds are actually "Operation" and will be
        # TODO: fresh random in every session.
        factory = factory or self.int_factory
        with tf.name_scope("zero-sharing"):
            with tf.device(self.servers[0].device_name):
                f00 = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[0][0] + self.pairwise_nonces[2]
                )
                f01 = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[0][1] + self.pairwise_nonces[0]
                )
                a0 = helper(f00, f01)
            with tf.device(self.servers[1].device_name):
                f10 = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[1][0] + self.pairwise_nonces[0]
                )
                f11 = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[1][1] + self.pairwise_nonces[1]
                )
                a1 = helper(f10, f11)
            with tf.device(self.servers[2].device_name):
                f20 = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[2][0] + self.pairwise_nonces[1]
                )
                f21 = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[2][1] + self.pairwise_nonces[2]
                )
                a2 = helper(f20, f21)

        self.pairwise_nonces = self.pairwise_nonces + 1
        return a0, a1, a2

    def _gen_random_sharing(self, shape, share_type=ARITHMETIC, factory=None):

        r = [[None]*2 for _ in range(3)]
        factory = factory or self.int_factory
        with tf.name_scope("random-sharing"):
            with tf.device(self.servers[0].device_name):
                r[0][0] = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[0][0] + self.pairwise_nonces[2]
                )
                r[0][1] = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[0][1] + self.pairwise_nonces[0]
                )
            with tf.device(self.servers[1].device_name):
                r[1][0] = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[1][0] + self.pairwise_nonces[0]
                )
                r[1][1] = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[1][1] + self.pairwise_nonces[1]
                )
            with tf.device(self.servers[2].device_name):
                r[2][0] = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[2][0] + self.pairwise_nonces[1]
                )
                r[2][1] = factory.sample_seeded_uniform(
                    shape = shape,
                    seed = self.pairwise_keys[2][1] + self.pairwise_nonces[2]
                )

        self.pairwise_nonces = self.pairwise_nonces + 1

        return ABY3PrivateTensor(self, r, True, share_type)

    def _gen_b2a_sharing(self, shape, b2a_keys):
        shares = [[None, None], [None, None], [None, None]]
        with tf.device(self.servers[0].device_name):
            shares[0][0] = self.int_factory.sample_seeded_uniform(
                shape = shape,
                seed = b2a_keys[0][0] + self.b2a_nonce
            )
            shares[0][1] = self.int_factory.sample_seeded_uniform(
                shape = shape,
                seed = b2a_keys[0][1] + self.b2a_nonce
            )
            x_on_0 = None
            if b2a_keys[0][2] is not None:
                share_2 = self.int_factory.sample_seeded_uniform(
                    shape = shape,
                    seed = b2a_keys[0][2] + self.b2a_nonce
                )
                x_on_0 = shares[0][0] ^ shares[0][1] ^ share_2

        with tf.device(self.servers[1].device_name):
            shares[1][0] = self.int_factory.sample_seeded_uniform(
                shape = shape,
                seed = b2a_keys[1][1] + self.b2a_nonce
            )
            shares[1][1] = self.int_factory.sample_seeded_uniform(
                shape = shape,
                seed = b2a_keys[1][2] + self.b2a_nonce
            )
            x_on_1 = None
            if b2a_keys[1][0] is not None:
                share_0 = self.int_factory.sample_seeded_uniform(
                    shape = shape,
                    seed = b2a_keys[1][0] + self.b2a_nonce
                )
                x_on_1 = share_0 ^ shares[1][0] ^ shares[1][1]

        with tf.device(self.servers[2].device_name):
            shares[2][0] = self.int_factory.sample_seeded_uniform(
                shape = shape,
                seed = b2a_keys[2][2] + self.b2a_nonce
            )
            shares[2][1] = self.int_factory.sample_seeded_uniform(
                shape = shape,
                seed = b2a_keys[2][0] + self.b2a_nonce
            )
            x_on_2 = None
            if b2a_keys[2][1] is not None:
                share_1 = self.int_factory.sample_seeded_uniform(
                    shape = shape,
                    seed = b2a_keys[2][1] + self.b2a_nonce
                )
                x_on_2 = share_1 ^ shares[2][0] ^ shares[2][1]

        self.b2a_nonce = self.b2a_nonce + 1
        return x_on_0, x_on_1, x_on_2, shares

    def _ot(self,
            sender, receiver, helper,
            m0, m1,
            c_on_receiver,
            c_on_helper,
            key_on_sender,
            key_on_helper,
            nonce):
        """
        Three-party OT protocol.
        'm0' and 'm1' are the two messages located on the sender.
        'c_on_receiver' and 'c_on_helper' should be the same choice bit, located on receiver and helper respectively.
        'key_on_sender' and 'key_on_helper' should be the same key, located on sender and helper respectively.
        'nonce' is a non-repeating ID for this call of the OT protocol.
        """
        assert m0.shape == m1.shape, "m0 shape {}, m1 shape {}".format(m0.shape, m1.shape)
        assert m0.factory == self.int_factory
        assert m1.factory == self.int_factory
        assert c_on_receiver.factory == self.bool_factory
        assert c_on_helper.factory == self.bool_factory

        with tf.name_scope("OT"):
            int_factory = self.int_factory
            with tf.device(sender.device_name):
                w_on_sender = int_factory.sample_seeded_uniform(
                    shape = [2] + m0.shape.as_list(),
                    seed = key_on_sender + nonce
                )
                masked_m0 = m0 ^ w_on_sender[0]
                masked_m1 = m1 ^ w_on_sender[1]
            with tf.device(helper.device_name):
                w_on_helper = int_factory.sample_seeded_uniform(
                    shape = [2] + m0.shape.as_list(),
                    seed = key_on_helper + nonce
                )
                # w_c = w_on_helper[0] * (1-c_on_helper.cast(int_factory)) \
                #       + w_on_helper[1] * c_on_helper.cast(int_factory)
                w_c = int_factory.where(c_on_helper.value, w_on_helper[1], w_on_helper[0], v2=False)
            with tf.device(receiver.device_name):
                # masked_m_c = masked_m0 * (1-c_on_receiver.cast(int_factory)) \
                #              + masked_m1 * c_on_receiver.cast(int_factory)
                masked_m_c = int_factory.where(c_on_receiver.value, masked_m1, masked_m0, v2=False)
                m_c = masked_m_c ^ w_c

        return m_c

    @memoize
    def assign(self, variable: "ABY3PrivateVariable", value) -> tf.Operation:
        """See tf.assign."""
        assert isinstance(variable, ABY3PrivateVariable), type(variable)
        assert isinstance(value, ABY3PrivateTensor), type(value)
        assert (variable.is_scaled == value.is_scaled), ("Scaling must match: "
                                                         "{}, {}").format(
            variable.is_scaled,
            value.is_scaled,
        )

        var_shares = variable.unwrapped
        val_shares = value.unwrapped

        with tf.name_scope("assign"):

            # Having this control_dependencies is important in order to avoid that
            # computationally-dependent shares are updated in different pace
            # (e.g., share0 is computed from share1, and we need to make sure that
            # share1 is NOT already updated).
            with tf.control_dependencies([val_shares[0][0].value,
                                          val_shares[0][1].value,
                                          val_shares[1][0].value,
                                          val_shares[1][1].value,
                                          val_shares[2][0].value,
                                          val_shares[2][1].value]):

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

    def lift(self, x, y=None, share_type=ARITHMETIC):
        """
        Convenience method for working with mixed typed tensors in programs:
        combining any of the ABY3 objects together with e.g. ints and floats
        will automatically lift the latter into ABY3 objects.

        Lifting will guarantee the two outputs are both scaled or unscaled if at
        least one of them is lifted from int or float.
        """

        if y is None:

            if isinstance(x, (np.ndarray, int, float)):
                return self.define_constant(x, share_type=share_type)

            if isinstance(x, tf.Tensor):
                return self.define_public_tensor(x, share_type=share_type)

            if isinstance(x, ABY3Tensor):
                return x

            raise TypeError("Don't know how to lift {}".format(type(x)))

        if isinstance(x, (np.ndarray, int, float)):

            if isinstance(y, (np.ndarray, int, float)):
                x = self.define_constant(x, share_type=share_type)
                y = self.define_constant(y, share_type=share_type)
                return x, y

            if isinstance(y, tf.Tensor):
                x = self.define_constant(x, share_type=share_type)
                y = self.define_public_tensor(y, share_type=share_type)
                return x, y

            if isinstance(y, ABY3Tensor):
                x = self.define_constant(
                    x,
                    apply_scaling=y.is_scaled,
                    share_type=share_type,
                    factory=y.backing_dtype,
                )
                return x, y

            raise TypeError(("Don't know how to lift "
                             "{}, {}").format(type(x), type(y)))

        if isinstance(x, tf.Tensor):

            if isinstance(y, (np.ndarray, int, float)):
                x = self.define_public_tensor(x, share_type=share_type)
                y = self.define_constant(y, share_type=share_type)
                return x, y

            if isinstance(y, tf.Tensor):
                x = self.define_public_tensor(x, share_type=share_type)
                y = self.define_public_tensor(y, share_type=share_type)
                return x, y

            if isinstance(y, ABY3Tensor):
                x = self.define_public_tensor(
                    x,
                    apply_scaling=y.is_scaled,
                    share_type=share_type,
                    factory=y.backing_dtype,
                )
                return x, y

            raise TypeError(("Don't know how to lift "
                             "{}, {}").format(type(x), type(y)))

        if isinstance(x, ABY3Tensor):

            if isinstance(y, (np.ndarray, int, float)):
                y = self.define_constant(
                    y,
                    apply_scaling=x.is_scaled,
                    share_type=share_type,
                    factory=x.backing_dtype,
                )
                return x, y

            if isinstance(y, tf.Tensor):
                y = self.define_public_tensor(
                    y,
                    apply_scaling=x.is_scaled,
                    share_type=share_type,
                    factory=x.backing_dtype,
                )
                return x, y

            if isinstance(y, ABY3Tensor):
                return x, y

        raise TypeError(("Don't know how to lift "
                         "{}, {}").format(type(x), type(y)))

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
    def mul2(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("mul2", x, y)

    @memoize
    def div(self, x, y):
        """
        Performs a true division of `x` by `y` where `y` is public.

        No flooring is performing if `y` is an integer type as it is implicitly
        treated as a float.
        """

        assert isinstance(x, ABY3Tensor)

        if isinstance(y, float):
            y_inverse = 1. / y
        elif isinstance(y, int):
            y_inverse = 1. / float(y)
        elif isinstance(y, ABY3PublicTensor):
            y_inverse = 1. / y.decode()
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
        assert x.share_type is BOOLEAN
        return self.dispatch("gather_bit", x, even)

    def xor_indices(self, x):
        assert x.share_type is BOOLEAN
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
        if all(isinstance(x, ABY3PublicTensor) for x in xs):
            return _concat_public(self, xs, axis=axis)

        if all(isinstance(x, ABY3PrivateTensor) for x in xs):
            return _concat_private(self, xs, axis=axis)

        raise TypeError("Don't know how to do a concat {}".format(type(xs)))

    @memoize
    def reduce_sum(self, x, axis=None, keepdims=False):
        x = self.lift(x)
        return self.dispatch("reduce_sum", x, axis=axis, keepdims=keepdims)

    @memoize
    def truncate(self, x: "ABY3Tensor"):
        return self.dispatch("truncate", x)

    @memoize
    def reveal(self, x):
        return self.dispatch("reveal", x)

    @memoize
    def B_xor(self, x, y):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_xor", x, y)

    @memoize
    def B_and(self, x, y):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_and", x, y)

    @memoize
    def B_or(self, x, y):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_or", x, y)

    @memoize
    def B_not(self, x):
        x = self.lift(x, share_type=BOOLEAN)
        return self.dispatch("B_not", x)

    @memoize
    def B_ppa(self, x, y, n_bits=None, topology="kogge_stone"):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_ppa", x, y, n_bits, topology)

    @memoize
    def B_add(self, x, y):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_add", x, y)

    @memoize
    def B_sub(self, x, y):
        x, y = self.lift(x, y, share_type=BOOLEAN)
        return self.dispatch("B_sub", x, y)

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
    def A2B(self, x, nbits=None):
        return self.dispatch("A2B", x, nbits)

    @memoize
    def B2A(self, x, nbits=None):
        return self.dispatch("B2A", x, nbits)

    @memoize
    def mul_AB(self, x, y):
        """
        Callers should make sure y is boolean sharing whose backing TF native type is `tf.bool`.
        There is no automatic lifting for boolean sharing in the mixed-protocol multiplication.
        """
        x = self.lift(x)
        return self.dispatch("mul_AB", x, y)

    @memoize
    def bit_extract(self, x, i):
        if x.share_type == BOOLEAN or x.share_type == ARITHMETIC:
            return self.dispatch("bit_extract", x, i)
        else:
            raise ValueError("unsupported share type: {}".format(x.share_type))

    @memoize
    def msb(self, x):
        return self.bit_extract(x, self.nbits-1)

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

    @memoize
    def stack(self, xs, axis):
        raise TypeError("Don't know how to do a stack {}".format(type(xs)))

    def write(self, x, filename_prefix):
        if not isinstance(x, ABY3PrivateTensor):
            raise TypeError("Only support writing ABY3PrivateTensor to disk.")
        return self.dispatch("write", x, filename_prefix)

    def read(self, filename_prefix, batch_size, n_columns):
        return self.dispatch("read", filename_prefix, batch_size, n_columns)

    def iterate(self, tensor: "ABY3PrivateTensor", batch_size: int, repeat=True, shuffle=True, seed: int=None):
        if not isinstance(tensor, ABY3PrivateTensor):
            raise TypeError("Only support iterating ABY3PrivateTensor.")
        return self.dispatch("iterate", tensor, batch_size, repeat, shuffle, seed)

    def blinded_shuffle(self, tensor: "ABY3PrivateTensor"):
        """
        Shuffle the rows of the given tenosr privately.
        After the shuffle, none of the share holder could know the exact shuffle order.
        """
        if not isinstance(tensor, ABY3PrivateTensor):
            raise TypeError("Only support blindly shuffle ABY3PrivateTensor. For public tensor, use the shuffle() method")
        return self.dispatch("blinded_shuffle", tensor)

    def dispatch(self, base_name, *args, container=None, **kwargs):
        """
        Finds the correct protocol logicto perform based on the dispatch_id
        attribute of the input tensors in args.
        """
        suffix = "_".join([arg.dispatch_id
                           for arg in args if hasattr(arg, "dispatch_id")])
        func_name = "_{}_{}".format(base_name, suffix)

        if container is None:
            container = _THISMODULE

        func = getattr(container, func_name, None)
        if func is not None:
            return func(self, *args, **kwargs)  # pylint: disable=not-callable
        raise TypeError(("Don't know how to {}: "
                         "{}").format(base_name, [type(arg) for arg in args]))


#
# Classes representing the base values in the ABY3 protocol.
#


class ABY3Tensor(abc.ABC):
    """
    This class functions mostly as a convenient way of exposing operations
    directly on the various tensor objects, ie allowing one to write `x + y`
    instead of `prot.add(x, y)`. Since this functionality is shared among all
    tensors we put it in this superclass.

    This class should never be instantiated on its own.
    Instead you should use your chosen protocols factory methods::

        x = prot.define_private_input(tf.constant(np.array([1,2,3,4])))
        y = prot.define_public_input(tf.constant(np.array([4,5,6,7])))

        z = x + y

        with config.Session() as sess:
            answer = z.reveal().eval(sess)

            print(answer) # => [5, 7, 9, 11]
    """

    def __init__(self, prot, is_scaled, share_type):
        self.prot = prot
        self.is_scaled = is_scaled
        self.share_type = share_type

    @property
    @abc.abstractmethod
    def shape(self) -> List[int]:
        """
        :rtype: List[int]
        :returns: The shape of this tensor.
        """

    @property
    @abc.abstractmethod
    def unwrapped(self) -> Tuple[AbstractTensor, ...]:
        pass

    def add(self, other):
        """
        Add `other` to this ABY3Tensor.  This can be another tensor with the same
        backing or a primitive.

        This function returns a new ABY3Tensor and does not modify this one.

        :param ABY3Tensor other: a or primitive (e.g. a float)
        :return: A new ABY3Tensor with `other` added.
        :rtype: ABY3Tensor
        """
        if self.share_type == ARITHMETIC:
            return self.prot.add(self, other)
        else:
            raise ValueError("unsupported share type for add: {}".format(self.share_type))

    def __add__(self, other):
        """
        See :meth:`~tf_encrypted.protocol.aby3.ABY3Tensor.add`
        """
        return self.add(other)

    def __radd__(self, other):
        return self + other

    def reduce_sum(self, axis=None, keepdims=False):
        """
        Like :meth:`tensorflow.reduce_sum`

        :param int axis:  The axis to reduce along
        :param bool keepdims: If true, retains reduced dimensions with length 1.
        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.reduce_sum(self, axis, keepdims)

    def sum(self, axis=None, keepdims=False):
        """
        See :meth:`ABY3Tensor.reduce_sum`
        """
        return self.reduce_sum(axis, keepdims)

    def sub(self, other):
        """
        Subtract `other` from this tensor.

        :param ABY3Tensor other: to subtract
        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        if self.share_type == ARITHMETIC:
            return self.prot.sub(self, other)
        else:
            raise ValueError("unsupported share type for sub: {}".format(self.share_type))

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        if self.share_type == ARITHMETIC:
            return self.prot.sub(other, self)
        else:
            raise ValueError("unsupported share type for sub: {}".format(self.share_type))

    def mul(self, other):
        """
        Multiply this tensor with `other`

        :param ABY3Tensor other: to multiply
        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.mul(self, other)

    def __mul__(self, other):
        return self.prot.mul(self, other)

    def __rmul__(self, other):
        return self.prot.mul(other, self)

    def __truediv__(self, other):
        return self.prot.div(self, other)

    def __mod__(self, other):
        return self.prot.mod(self, other)

    def __pow__(self, p):
        return self.prot.pow(self, p)

    def square(self):
        """
        Square this tensor.

        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.square(self)

    def matmul(self, other):
        """
        MatMul this tensor with `other`.  This will perform matrix multiplication,
        rather than elementwise like
        :meth:`~tf_encrypted.protocol.aby3.ABY3Tensor.mul`

        :param ABY3Tensor other: to mul
        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.matmul(self, other)

    def dot(self, other):
        """
        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.matmul(other)

    def __getitem__(self, slc):
        return self.prot.indexer(self, slc)

    def transpose(self, perm=None):
        """
        Transpose this tensor.

        See :meth:`tensorflow.transpose`

        :param List[int]: A permutation of the dimensions of this tensor.

        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.transpose(self, perm)

    def truncate(self):
        """
        Truncate this tensor.

        `TODO`

        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.truncate(self)

    def expand_dims(self, axis=None):
        """
        :See: tf.expand_dims

        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.expand_dims(self, axis=axis)

    def reshape(self, shape: List[int]) -> "ABY3Tensor":
        """
        :See: tf.reshape

        :param List[int] shape: The new shape of the tensor.
        :rtype: ABY3Tensor
        :returns: A new tensor with the contents of this tensor, but with the new
            specified shape.
        """
        return self.prot.reshape(self, shape)

    def __neg__(self):
        return self.prot.negative(self)

    def negative(self) -> "ABY3Tensor":
        """
        :See: tf.negative

        :rtype: ABY3Tensor
        :returns: A new tensor with numerical negative value element-wise computed.
        """
        return self.prot.negative(self)

    def reduce_max(self, axis: int) -> "ABY3Tensor":
        """
        :See: tf.reduce_max

        :param int axis: The axis to take the max along
        :rtype: ABY3Tensor
        :returns: A new ABY3 tensor with the max value from each axis.
        """
        return self.prot.reduce_max(self, axis)

    def consistency_check(self, shares, inst_type):
        pass

    def xor(self, other):
        if self.share_type == BOOLEAN:
            return self.prot.B_xor(self, other)
        else:
            raise ValueError("Unsupported share type for xor: {}".format(self.share_type))

    def __xor__(self, other):
        return self.xor(other)

    def and_(self, other):
        if self.share_type == BOOLEAN:
            return self.prot.B_and(self, other)
        else:
            raise ValueError("unsupported share type for and: {}".format(self.share_type))

    def __and__(self, other):
        return self.and_(other)

    def or_(self, other):
        if self.share_type == BOOLEAN:
            return self.prot.B_or(self, other)
        else:
            raise ValueError("unsupported share type for and: {}".format(self.share_type))

    def __or__(self, other):
        return self.or_(other)

    def invert(self):
        if self.share_type == BOOLEAN:
            return self.prot.B_not(self)
        else:
            raise ValueError("unsupported share type for and: {}".format(self.share_type))

    def __invert__(self):
        return self.invert()

    def __lshift__(self, steps):
        return self.prot.lshift(self, steps)

    def lshift(self, steps):
        return self.prot.lshift(self, steps)

    def __rshift__(self, steps):
        return self.prot.rshift(self, steps)

    def rshift(self, steps):
        return self.prot.rshift(self, steps)

    def arith_rshift(self, steps):
        return self.rshift(steps)

    def logical_rshift(self, steps):
        return self.prot.logical_rshift(self, steps)

    def write(self, filename_prefix):
        return self.prot.write(self, filename_prefix)


class ABY3PublicTensor(ABY3Tensor):
    """
    This class represents a public tensor, known by at least by the three servers
    but potentially known by more. Although there is only a single value we
    replicate it on both servers to avoid sending it from one to the other
    in the operations where it's needed by both (eg multiplication).
    """

    dispatch_id = "public"

    def __init__(
            self,
            prot: ABY3,
            value_on_0: AbstractTensor,
            value_on_1: AbstractTensor,
            value_on_2: AbstractTensor,
            is_scaled: bool,
            share_type
    ) -> None:
        assert isinstance(value_on_0, AbstractTensor), type(value_on_0)
        assert isinstance(value_on_1, AbstractTensor), type(value_on_1)
        assert isinstance(value_on_2, AbstractTensor), type(value_on_2)
        assert value_on_0.shape == value_on_1.shape
        assert value_on_0.shape == value_on_2.shape

        super(ABY3PublicTensor, self).__init__(prot, is_scaled, share_type)
        self.value_on_0 = value_on_0
        self.value_on_1 = value_on_1
        self.value_on_2 = value_on_2

    def __repr__(self) -> str:
        return "ABY3PublicTensor(shape={}, share_type={})".format(self.shape, self.share_type)

    @property
    def shape(self) -> List[int]:
        return self.value_on_0.shape

    @property
    def backing_dtype(self):
        return self.value_on_0.factory

    @property
    def unwrapped(self) -> Tuple[AbstractTensor, ...]:
        """
        Unwrap the tensor.

        This will return the value for each of the parties that collectively own
        the tensor.

        In most cases, this will be the same value on each device.

        .. code-block:: python

            x_0, y_0, z_0 = tensor.unwrapped
            # x_0 == 10 with the value pinned to player_0's device.
            # y_0 == 10 with the value pinned to player_1's device.
            # z_0 == 10 with the value pinned to player_2's device.

        In most cases you will want to work on this data on the specified device.

        .. code-block:: python

            x_0, y_0, z_0= tensor.unwrapped

            with tf.device(prot.player_0.device_name):
                # act on x_0

            with tf.device(prot.player_1.device_name):
                # act on y_0

            with tf.device(prot.player_2.device_name):
                # act on z_0

        In most cases you will not need to use this method.  All funtions
        will hide this functionality for you (e.g. `add`, `mul`, etc).
        """
        return self.value_on_0, self.value_on_1, self.value_on_2

    def decode(self) -> Union[np.ndarray, tf.Tensor]:
        return self.prot._decode(self.value_on_0, self.is_scaled)  # pylint: disable=protected-access

    def to_native(self):
        return self.decode()


class ABY3Constant(ABY3PublicTensor):
    """
    This class essentially represents a public value, however it additionally
    records the fact that the underlying value was declared as a constant.
    """

    def __init__(self, prot, constant_on_0, constant_on_1, constant_on_2, is_scaled, share_type):
        assert isinstance(constant_on_0, AbstractConstant), type(constant_on_0)
        assert isinstance(constant_on_1, AbstractConstant), type(constant_on_1)
        assert isinstance(constant_on_2, AbstractConstant), type(constant_on_2)
        assert constant_on_0.shape == constant_on_1.shape
        assert constant_on_0.shape == constant_on_2.shape

        super(ABY3Constant, self).__init__(
            prot, constant_on_0, constant_on_1, constant_on_2, is_scaled, share_type
        )
        self.constant_on_0 = constant_on_0
        self.constant_on_1 = constant_on_1
        self.constant_on_2 = constant_on_2

    def __repr__(self) -> str:
        return "ABY3Constant(shape={}, share_type={})".format(self.shape, self.share_type)

class ABY3PrivateTensor(ABY3Tensor):
    """
    This class represents a private value that may be unknown to everyone.
    """

    dispatch_id = "private"

    def __init__(self, prot, shares, is_scaled, share_type):
        assert len(shares) == 3
        shape = shares[0][0].shape
        for i in range(len(shares)):
            for j in range(len(shares[i])):
                msg = "Shares have different shapes: Expected {}, but share[{}][{}] has {}".format(shape, i, j, shares[i][j].shape)
                assert shares[i][j].shape == shape, msg
        super(ABY3PrivateTensor, self).__init__(prot, is_scaled, share_type)
        self.shares = shares

    def __repr__(self) -> str:
        return "ABY3PrivateTensor(shape={}, share_type={})".format(self.shape, self.share_type)

    @property
    def shape(self) -> List[int]:
        return self.shares[0][0].shape

    @property
    def backing_dtype(self):
        return self.shares[0][0].factory

    @property
    def unwrapped(self):
        return self.shares

    def reveal(self) -> ABY3PublicTensor:
        return self.prot.reveal(self)


class ABY3PrivateVariable(ABY3PrivateTensor):
    """
    This class essentially represents a private value, however it additionally
    records the fact that the backing tensor was declared as a variable in
    order to allow treating it as a variable itself.
    """

    def __init__(self, prot, shares, is_scaled, share_type):

        super(ABY3PrivateVariable, self).__init__(
            prot, shares, is_scaled, share_type
        )
        self.shares = shares
        self.initializer = tf.group(
            *[var.initializer for share in shares for var in share]
        )

    def __repr__(self) -> str:
        return "ABY3PrivateVariable(shape={}, share_type={})".format(self.shape, self.share_type)


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

    return ABY3PublicTensor(prot, z_on_0, z_on_1, z_on_2, x.is_scaled, x.share_type)

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
    assert x.is_scaled == y.is_scaled, ("Cannot mix different encodings: "
                                        "{} {}").format(x.is_scaled, y.is_scaled)

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
    assert x.is_scaled == y.is_scaled, ("Cannot mix different encodings: "
                                        "{} {}").format(x.is_scaled, y.is_scaled)

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

    return ABY3PublicTensor(prot, z[0], z[1], z[2], x.is_scaled, x.share_type)

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
        x_neg = ABY3PublicTensor(prot, x_on_0_neg, x_on_1_neg, x_on_2_neg, x.is_scaled, x.share_type)
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
            z0 = x_shares[0][0] * y_shares[0][0] \
                 + x_shares[0][0] * y_shares[0][1] \
                 + x_shares[0][1] * y_shares[0][0] \
                 + a0

        with tf.device(prot.servers[1].device_name):
            z1 = x_shares[1][0] * y_shares[1][0] \
                 + x_shares[1][0] * y_shares[1][1] \
                 + x_shares[1][1] * y_shares[1][0] \
                 + a1

        with tf.device(prot.servers[2].device_name):
            z2 = x_shares[2][0] * y_shares[2][0] \
                 + x_shares[2][0] * y_shares[2][1] \
                 + x_shares[2][1] * y_shares[2][0] \
                 + a2
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


def _mul2_private_private(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    # If there will not be any truncation, then just call the simple multiplication protocol.
    if not (x.is_scaled and y.is_scaled):
        return _mul_private_private(prot, x, y)

    x_shares = x.unwrapped
    y_shares = y.unwrapped
    shape = x_shares[0][0].shape
    amount = prot.fixedpoint_config.precision_fractional

    with tf.name_scope("mul2"):
        # Step 1: Generate a Random Truncation Pair
        # If TF is smart enough, this part is supposed to be pre-computation.
        r = prot._gen_random_sharing(shape, share_type=BOOLEAN)
        r_trunc = r.arith_rshift(amount)
        r = prot.B2A(r)
        r_trunc = prot.B2A(r_trunc)

        # Step 2: Compute 3-out-of-3 sharing of (x*y - r)
        a0, a1, a2 = prot._gen_zero_sharing(x.shape)
        with tf.device(prot.servers[0].device_name):
            z0 = x_shares[0][0] * y_shares[0][0] \
                 + x_shares[0][0] * y_shares[0][1] \
                 + x_shares[0][1] * y_shares[0][0] \
                 + a0 - r.shares[0][0]

        with tf.device(prot.servers[1].device_name):
            z1 = x_shares[1][0] * y_shares[1][0] \
                 + x_shares[1][0] * y_shares[1][1] \
                 + x_shares[1][1] * y_shares[1][0] \
                 + a1 - r.shares[1][0]

        with tf.device(prot.servers[2].device_name):
            z2 = x_shares[2][0] * y_shares[2][0] \
                 + x_shares[2][0] * y_shares[2][1] \
                 + x_shares[2][1] * y_shares[2][0] \
                 + a2 - r.shares[2][0]

        # Step 3: Reveal (x*y - r) / 2^d
        # xy_minus_r = z0 + z1 + z2
        # xy_minus_r_trunc = xy_minus_r.right_shift(amount)
        # z = ABY3PublicTensor(prot, xy_minus_r_trunc, xy_minus_r_trunc, xy_minus_r_trunc, True, ARITHMETIC)
        xy_minus_r_trunc = [None] * 3
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                xy_minus_r_trunc[i] = z0 + z1 + z2
                xy_minus_r_trunc[i] = xy_minus_r_trunc[i].right_shift(amount)
        z = ABY3PublicTensor(prot, xy_minus_r_trunc[0], xy_minus_r_trunc[1], xy_minus_r_trunc[2], True, ARITHMETIC)

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
            z0 = x_shares[0][0].matmul(y_shares[0][0]) \
                 + x_shares[0][0].matmul(y_shares[0][1]) \
                 + x_shares[0][1].matmul(y_shares[0][0]) \
                 + a0

        with tf.device(prot.servers[1].device_name):
            z1 = x_shares[1][0].matmul(y_shares[1][0]) \
                 + x_shares[1][0].matmul(y_shares[1][1]) \
                 + x_shares[1][1].matmul(y_shares[1][0]) \
                 + a1

        with tf.device(prot.servers[2].device_name):
            z2 = x_shares[2][0].matmul(y_shares[2][0]) \
                 + x_shares[2][0].matmul(y_shares[2][1]) \
                 + x_shares[2][1].matmul(y_shares[2][0]) \
                 + a2
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


def _truncate_private(prot: ABY3, x: ABY3PrivateTensor) -> ABY3PrivateTensor:
    assert isinstance(x, ABY3PrivateTensor)

    if prot.fixedpoint_config.use_noninteractive_truncation:
        return _truncate_private_noninteractive(prot, x)

    return _truncate_private_interactive(prot, x)


def _truncate_private_noninteractive(
        prot: ABY3, x: ABY3PrivateTensor) -> ABY3PrivateTensor:
    assert isinstance(x, ABY3PrivateTensor), type(x)

    base = prot.fixedpoint_config.scaling_base
    amount = prot.fixedpoint_config.precision_fractional
    shares = x.unwrapped

    y = [[None, None], [None, None], [None, None]]
    with tf.name_scope("truncate"):

        # First step: compute new shares

        with tf.device(prot.servers[2].device_name):
            r_on_2 = prot.int_factory.sample_seeded_uniform(
                shares[2][0].shape,
                prot.pairwise_keys[2][0] + prot.pairwise_nonces[1]
            )

        with tf.device(prot.servers[0].device_name):
            y0 = shares[0][0].truncate(amount, base)

        with tf.device(prot.servers[1].device_name):
            r_on_1 = prot.int_factory.sample_seeded_uniform(
                shares[1][0].shape,
                prot.pairwise_keys[1][1] + prot.pairwise_nonces[1]
            )
            t = shares[1][0] + shares[1][1]
            # tmp = 0 - (0 - t).truncate(amount, base)
            tmp = t.truncate(amount, base)
            y1 = tmp - r_on_1

        prot.pairwise_nonces[1] = prot.pairwise_nonces[1] + 1

        # Second step: replicate shares

        with tf.device(prot.servers[0].device_name):
            y[0][0] = y0
            y[0][1] = y1
        with tf.device(prot.servers[1].device_name):
            y[1][0] = y1
            y[1][1] = r_on_1
        with tf.device(prot.servers[2].device_name):
            y[2][0] = r_on_2
            y[2][1] = y0

    return ABY3PrivateTensor(prot, y, x.is_scaled, x.share_type)


def _truncate_private_interactive(
        prot: ABY3, a: ABY3PrivateTensor) -> ABY3PrivateTensor:
    """
    See protocol TruncPr (3.1) in
      "Secure Computation With Fixed-Point Numbers" by Octavian Catrina and Amitabh
      Saxena, FC'10.

    We call it "interactive" to keep consistent with the 2pc setting,
    but in fact, our protocol uses only one round communication, exactly the same as
    that in the "non-interactive" one.
    """
    assert isinstance(a, ABY3PrivateTensor), type(a)

    with tf.name_scope("truncate-i"):
        scaling_factor = prot.fixedpoint_config.scaling_factor
        scaling_factor_inverse = inverse(
            prot.fixedpoint_config.scaling_factor, prot.int_factory.modulus
        )

        # we first rotate `a` to make sure reconstructed values fall into
        # a non-negative interval `[0, 2B)` for some bound B; this uses an
        # assumption that the values originally lie in `[-B, B)`, and will
        # leak private information otherwise

        # 'a + bound' will automatically lift 'bound' by another scaling factor,
        # so we should first divide bound by the scaling factor if we want to
        # use this convenient '+' operation.
        bound = prot.fixedpoint_config.bound_double_precision
        b = a + (bound / scaling_factor)

        # next step is for servers to add a statistical mask to `b`, reveal
        # it to server1 and server2, and compute the lower part
        trunc_gap = prot.fixedpoint_config.truncation_gap
        mask_bitlength = ceil(log2(bound)) + 2 + trunc_gap

        b_shares = b.unwrapped
        a_shares = a.unwrapped
        shape = a.shape


        # NOTE: The following algorithm has an assumption to ensure the correctness:
        # c = a + bound + r0 + r1  SHOULD be positively smaller than
        # the max int64 number 2^{63} - 1. This is necessary to ensure the correctness of
        # the modulo operation 'c % scaling_factor'.
        # As a simple example, consider a 4-bit number '1111', when we think of it as a signed
        # number, it is '-1', and '-1 % 3 = 2'. But when we think of it as an unsigned number,
        # then '15 % 3 = 0'. AND the following works only if c is a positive number that is within
        # 63-bit, because 64-bit becomes a negative number.
        # Therefore, 'mask_bitlength' is better <= 61 if we use int64 as the underlying type, because
        # r0 is 61-bit, r1 is 61-bit, bound is much smaller, and (assuming) a is much smaller than bound.

        d = [[None] * 2 for _ in range(3)]
        with tf.device(prot.servers[0].device_name):
            r0_on_0 = prot.int_factory.sample_seeded_bounded(
                shape,
                prot.pairwise_keys[0][0] + prot.pairwise_nonces[2],
                mask_bitlength)
            r1_on_0 = prot.int_factory.sample_seeded_bounded(
                shape,
                prot.pairwise_keys[0][1] + prot.pairwise_nonces[0],
                mask_bitlength)
            c0_on_0 = b_shares[0][0] + r0_on_0
            c1_on_0 = b_shares[0][1] + r1_on_0

            r0_lower_on_0 = r0_on_0 % scaling_factor
            r1_lower_on_0 = r1_on_0 % scaling_factor

            a_lower0_on_0 = -r0_lower_on_0
            a_lower1_on_0 = -r1_lower_on_0


            d[0][0] = (a_shares[0][0] - a_lower0_on_0) * scaling_factor_inverse
            d[0][1] = (a_shares[0][1] - a_lower1_on_0) * scaling_factor_inverse

        with tf.device(prot.servers[1].device_name):
            r1_on_1 = prot.int_factory.sample_seeded_bounded(
                shape,
                prot.pairwise_keys[1][0] + prot.pairwise_nonces[0],
                mask_bitlength)
            c1_on_1 = b_shares[1][0] + r1_on_1
            c2_on_1 = b_shares[1][1]

            # server0 sends c0 to server1, revealing c to server1
            c_on_1 = c0_on_0 + c1_on_1 + c2_on_1

            r1_lower_on_1 = r1_on_1 % scaling_factor

            a_lower1_on_1 = -r1_lower_on_1
            a_lower2_on_1 = c_on_1 % scaling_factor

            d[1][0] = (a_shares[1][0] - a_lower1_on_1) * scaling_factor_inverse
            d[1][1] = (a_shares[1][1] - a_lower2_on_1) * scaling_factor_inverse

        with tf.device(prot.servers[2].device_name):
            r0_on_2 = prot.int_factory.sample_seeded_bounded(
                shape,
                prot.pairwise_keys[2][1] + prot.pairwise_nonces[2],
                mask_bitlength)
            c0_on_2 = b_shares[2][1] + r0_on_2
            c2_on_2 = b_shares[2][0]

            # server1 sends c1 to server2, revealing c to server2
            c_on_2 = c0_on_2 + c1_on_1 + c2_on_2

            r0_lower_on_2 = r0_on_2 % scaling_factor

            a_lower0_on_2 = -r0_lower_on_2
            a_lower2_on_2 = c_on_2 % scaling_factor

            d[2][0] = (a_shares[2][0] - a_lower2_on_2) * scaling_factor_inverse
            d[2][1] = (a_shares[2][1] - a_lower0_on_2) * scaling_factor_inverse

        prot.pairwise_nonces[0] += 1
        prot.pairwise_nonces[2] += 1

    return ABY3PrivateTensor(prot, d, a.is_scaled, a.share_type)


def _B_xor_private_private(
        prot: ABY3,
        x: ABY3PrivateTensor,
        y: ABY3PrivateTensor):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("b_xor"):

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


def _B_xor_private_public(
        prot: ABY3,
        x: ABY3PrivateTensor,
        y: ABY3PublicTensor):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PublicTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("b_xor"):
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


def _B_and_private_private(
        prot: ABY3,
        x: ABY3PrivateTensor,
        y: ABY3PrivateTensor):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    x_shares = x.unwrapped
    y_shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("b_and"):
        a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN, factory=x.backing_dtype)

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


def _B_and_private_public(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PublicTensor), type(x)
    assert x.backing_dtype == y.backing_dtype

    x_shares = x.unwrapped
    y_on_0, y_on_1, y_on_2 = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("B_and"):
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


def _B_and_public_private(prot, x, y):
    assert isinstance(x, ABY3PublicTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)
    assert x.backing_dtype == y.backing_dtype

    x_on_0, x_on_1, x_on_2 = x.unwrapped
    y_shares = y.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("B_and"):
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


def _B_or_private_private(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    with tf.name_scope("B_or"):
        z = (x ^ y) ^ (x & y)

    return z


def _B_not_private(prot, x):
    assert isinstance(x, ABY3PrivateTensor), type(x)

    x_shares = x.unwrapped

    z = [[None, None], [None, None], [None, None]]
    with tf.name_scope("B_not"):
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


def _B_add_private_private(prot, x, y):
    pass


def _B_sub_private_private(prot, x, y):
    pass


def _B_ppa_private_private(prot, x, y, n_bits, topology="kogge_stone"):
    """
    Parallel prefix adder (PPA). This adder can be used for addition of boolean sharings.

    `n_bits` can be passed as an optimization to constrain the computation for least significant
    `n_bits` bits.

    AND Depth: log(k)
    Total gates: klog(k)
    """

    if topology == "kogge_stone":
        return _B_ppa_kogge_stone_private_private(prot, x, y, n_bits)
    elif topology == "sklansky":
        return _B_ppa_sklansky_private_private(prot, x, y, n_bits)
    else:
        raise NotImplementedError("Unknown adder topology.")

def _B_ppa_sklansky_private_private(prot, x, y, n_bits):
    """
    Parallel prefix adder (PPA), using the Sklansky adder topology.
    """

    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    if x.backing_dtype.native_type != tf.int64:
        raise NotImplementedError("Native type {} not supported".format(x.backing_dtype.native_type))

    with tf.name_scope("B_ppa"):
        if prot.nbits == 64:
            keep_masks = [0x5555555555555555, 0x3333333333333333, 0x0f0f0f0f0f0f0f0f,
                          0x00ff00ff00ff00ff, 0x0000ffff0000ffff, 0x00000000ffffffff]
            copy_masks = [0x5555555555555555, 0x2222222222222222, 0x0808080808080808,
                          0x0080008000800080, 0x0000800000008000, 0x0000000080000000]
        elif prot.nbits == 128:
            keep_masks = [
                    0x55555555555555555555555555555555, 0x33333333333333333333333333333333,
                    0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f, 0x00ff00ff00ff00ff00ff00ff00ff00ff,
                    0x0000ffff0000ffff0000ffff0000ffff, 0x00000000ffffffff00000000ffffffff,
                    0x0000000000000000ffffffffffffffff]
            copy_masks = [
                    0x55555555555555555555555555555555, 0x22222222222222222222222222222222,
                    0x08080808080808080808080808080808, 0x00800080008000800080008000800080,
                    0x00008000000080000000800000008000, 0x00000000800000000000000080000000,
                    0x00000000000000008000000000000000]

        G = x & y
        P = x ^ y

        k = prot.nbits
        if n_bits is not None:
            k = n_bits
        for i in range(ceil(log2(k))):
            c_mask = prot.define_constant(np.ones(x.shape, dtype=np.object) * copy_masks[i],
                                          apply_scaling=False, share_type=BOOLEAN)
            k_mask = prot.define_constant(np.ones(x.shape, dtype=np.object) * keep_masks[i],
                                          apply_scaling=False, share_type=BOOLEAN)
            # Copy the selected bit to 2^i positions:
            # For example, when i=2, the 4-th bit is copied to the (5, 6, 7, 8)-th bits
            G1 = (G & c_mask) << 1
            P1 = (P & c_mask) << 1
            for j in range(i):
                G1 = (G1 << (2**j)) ^ G1
                P1 = (P1 << (2**j)) ^ P1
            '''
            Two-round impl. using algo. specified in the slides that assume using OR gate is free, but in fact,
            here using OR gate cost one round.
            The PPA operator 'o' is defined as:
            (G, P) o (G1, P1) = (G + P*G1, P*P1), where '+' is OR, '*' is AND
            '''
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


            '''
            One-round impl. by modifying the PPA operator 'o' as:
            (G, P) o (G1, P1) = (G ^ (P*G1), P*P1), where '^' is XOR, '*' is AND
            This is a valid definition: when calculating the carry bit c_i = g_i + p_i * c_{i-1},
            the OR '+' can actually be replaced with XOR '^' because we know g_i and p_i will NOT take '1'
            at the same time.
            And this PPA operator 'o' is also associative. BUT, it is NOT idempotent: (G, P) o (G, P) != (G, P).
            This does not matter, because we can do (G, P) o (0, P) = (G, P), or (G, P) o (0, 1) = (G, P)
            if we want to keep G and P bits.
            '''
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


def _B_ppa_kogge_stone_private_private(prot, x, y, n_bits):
    """
    Parallel prefix adder (PPA), using the Kogge-Stone adder topology.
    """

    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)

    if x.backing_dtype.native_type != tf.int64:
        raise NotImplementedError("Native type {} not supported".format(x.backing_dtype.native_type))

    with tf.name_scope("B_ppa"):
        keep_masks = []
        for i in range(ceil(log2(prot.nbits))):
            keep_masks.append((1 << (2**i)) - 1)
        '''
        For example, if prot.nbits = 64, then keep_masks is:
        keep_masks = [0x0000000000000001, 0x0000000000000003, 0x000000000000000f,
                      0x00000000000000ff, 0x000000000000ffff, 0x00000000ffffffff]
        '''

        G = x & y
        P = x ^ y
        k = prot.nbits if n_bits is None else n_bits
        for i in range(ceil(log2(k))):
            k_mask = prot.define_constant(np.ones(x.shape, dtype=np.object) * keep_masks[i],
                                          apply_scaling=False, share_type=BOOLEAN)

            G1 = G << (2**i)
            P1 = P << (2**i)

            '''
            One-round impl. by modifying the PPA operator 'o' as:
            (G, P) o (G1, P1) = (G ^ (P*G1), P*P1), where '^' is XOR, '*' is AND
            This is a valid definition: when calculating the carry bit c_i = g_i + p_i * c_{i-1},
            the OR '+' can actually be replaced with XOR '^' because we know g_i and p_i will NOT take '1'
            at the same time.
            And this PPA operator 'o' is also associative. BUT, it is NOT idempotent: (G, P) o (G, P) != (G, P).
            This does not matter, because we can do (G, P) o (0, P) = (G, P), or (G, P) o (0, 1) = (G, P)
            if we want to keep G and P bits.
            '''
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


def _A2B_private(prot, x, nbits):
    """
    Bit decomposition: Convert an arithmetic sharing to a boolean sharing.
    """
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert x.share_type == ARITHMETIC

    x_shares = x.unwrapped
    zero = prot.define_constant(np.zeros(x.shape, dtype=np.int64),
                                apply_scaling=False, share_type=BOOLEAN)
    zero_on_0, zero_on_1, zero_on_2 = zero.unwrapped
    a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)

    # Method 1: Resist semi-honest adversary
    operand1 = [[None, None], [None, None], [None, None]]
    operand2 = [[None, None], [None, None], [None, None]]
    with tf.name_scope("A2B"):
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

        operand1 = ABY3PrivateTensor(prot, operand1, x.is_scaled, BOOLEAN)
        operand2 = ABY3PrivateTensor(prot, operand2, x.is_scaled, BOOLEAN)

        # Step 2: Parallel prefix adder that requires log(k) rounds of communication
        result = prot.B_ppa(operand1, operand2, nbits)

    # TODO Method 2: Resist malicious adversary.

    return result


def _bit_extract_private(prot, x, i):
    """
    Bit extraction: Extracts the `i`-th bit of an arithmetic sharing or boolean sharing
    to a single-bit boolean sharing.
    """
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert x.backing_dtype == prot.int_factory

    with tf.name_scope("bit_extract"):
        if x.share_type == ARITHMETIC:
            with tf.name_scope("A2B_partial"):
                x_shares = x.unwrapped
                zero = prot.define_constant(np.zeros(x.shape, dtype=np.int64),
                                            apply_scaling=False, share_type=BOOLEAN)
                zero_on_0, zero_on_1, zero_on_2 = zero.unwrapped
                a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)

                # Method 1: Resist semi-honest adversary
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

                operand1 = ABY3PrivateTensor(prot, operand1, x.is_scaled, BOOLEAN)
                operand2 = ABY3PrivateTensor(prot, operand2, x.is_scaled, BOOLEAN)

                # Step 2: Parallel prefix adder that requires log(i+1) rounds of communication
                x = prot.B_ppa(operand1, operand2, i+1)

        # Take out the i-th bit
        #
        # Method 1:
        # mask = prot.define_constant(np.array([0x1]), apply_scaling=False, share_type=BOOLEAN)
        # x = x >> i
        # x = x & mask
        # # NOTE: Don't use x = x & 0x1. Even though we support automatic lifting of 0x1
        # # to an ABY3Tensor, but it also includes automatic scaling to make the two operands have
        # # the same scale, which is not what want here.
        #
        # Method 2:
        mask = prot.define_constant(np.array([0x1 << i]), apply_scaling=False, share_type=BOOLEAN)
        x = x & mask

        x_shares = x.unwrapped
        result = [[None, None], [None, None], [None, None]]
        for i in range(3):
            with tf.device(prot.servers[i].device_name):
                result[i][0] = x_shares[i][0].cast(prot.bool_factory)
                result[i][1] = x_shares[i][1].cast(prot.bool_factory)
        result = ABY3PrivateTensor(prot, result, False, BOOLEAN)

    return result


def _B2A_private(prot, x, nbits):
    """
    Bit composition: Convert a boolean sharing to an arithmetic sharing.
    """
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert x.share_type == BOOLEAN

    # Method 1: Resist semi-honest adversary
    # In semi-honest, the following two calls can be further optimized because we don't
    # need the boolean shares of x1 and x2. We only need their original values on intended servers.
    x1_on_0, x1_on_1, x1_on_2, x1_shares = prot._gen_b2a_sharing(x.shape, prot.b2a_keys_1)
    assert (x1_on_2 is None)
    x2_on_0, x2_on_1, x2_on_2, x2_shares = prot._gen_b2a_sharing(x.shape, prot.b2a_keys_2)
    assert (x2_on_0 is None)

    a0, a1, a2 = prot._gen_zero_sharing(x.shape, share_type=BOOLEAN)

    with tf.name_scope("B2A"):
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
        neg_x1_neg_x2 = ABY3PrivateTensor(prot, neg_x1_neg_x2, x.is_scaled, BOOLEAN)

        # Compute x0 = x + (-x1-x2) using the parallel prefix adder
        x0 = prot.B_ppa(x, neg_x1_neg_x2, nbits)

        # Reveal x0 to server 0 and 2
        with tf.device(prot.servers[0].device_name):
            x0_on_0 = prot._reconstruct(x0.unwrapped, prot.servers[0], BOOLEAN)
        with tf.device(prot.servers[2].device_name):
            x0_on_2 = prot._reconstruct(x0.unwrapped, prot.servers[2], BOOLEAN)

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
        result = ABY3PrivateTensor(prot, result, x.is_scaled, ARITHMETIC)

    return result


def _mul_AB_public_private(prot, x, y):
    assert isinstance(x, ABY3PublicTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(x)
    assert x.share_type == ARITHMETIC
    assert y.share_type == BOOLEAN

    x_on_0, x_on_1, x_on_2 = x.unwrapped

    with tf.name_scope("mul_AB"):
        z = __mul_AB_routine(prot, x_on_2, y, 2)
        z = ABY3PrivateTensor(prot, z, x.is_scaled, ARITHMETIC)

    return z


def _mul_AB_private_private(prot, x, y):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert isinstance(y, ABY3PrivateTensor), type(y)
    assert x.share_type == ARITHMETIC
    assert y.share_type == BOOLEAN

    x_shares = x.unwrapped

    with tf.name_scope("mul_AB"):
        with tf.name_scope("term0"):
            w = __mul_AB_routine(prot, x_shares[0][0], y, 0)
            w = ABY3PrivateTensor(prot, w, x.is_scaled, ARITHMETIC)

        with tf.name_scope("term1"):
            with tf.device(prot.servers[1].device_name):
                a = x_shares[1][0] + x_shares[1][1]
            z = __mul_AB_routine(prot, a, y, 1)
            z = ABY3PrivateTensor(prot, z, x.is_scaled, ARITHMETIC)
        z = w + z

    return z


def __mul_AB_routine(prot, a, b, sender_idx):
    """
    A sub routine for multiplying a value 'a' (located at servers[sender_idx]) with a boolean sharing 'b'.
    """
    assert isinstance(a, AbstractTensor), type(a)
    assert isinstance(b, ABY3PrivateTensor), type(b)

    with tf.name_scope("__mul_AB_routine"):
        b_shares = b.unwrapped
        s = [None, None, None]
        s[0], s[1], s[2] = prot._gen_zero_sharing(a.shape, ARITHMETIC)

        z = [[None, None], [None, None], [None, None]]
        idx0 = sender_idx
        idx1 = (sender_idx + 1) % 3
        idx2 = (sender_idx + 2) % 3
        with tf.device(prot.servers[idx0].device_name):
            # TODO: Think about the security:
            # "z[idx0][0] = s[idx2]" doesn't make z[idx0][0] a Tensor on servers[idx0], it is just a reference to
            # s[idx2] which is still on servers[idx2] (just print out z[idx0][0]'s device will verify this).
            # Consider later in some code, if we use z[idx0][0], then servers[idx2] will send the value
            # (actually only happens when we reconstruct on servers[idx1]),
            # but not servers[idx0], which essentially means, the value is never sent from servers[idx2] to
            # servers[idx0]. Will this lead to any vulnerability? If not, then this saves some communication
            # cost, and gives us another reason to use TensorFlow.
            z[idx0][0] = s[idx2]
            z[idx0][1] = s[idx1]
            tmp = (b_shares[idx0][0] ^ b_shares[idx0][1]).cast(a.factory) * a
            m0 = tmp + s[idx0]
            m1 = - tmp + a + s[idx0]

        with tf.device(prot.servers[idx1].device_name):
            z[idx1][0] = s[idx1]
            z[idx1][1] = prot._ot(prot.servers[idx0], prot.servers[idx1], prot.servers[idx2],
                                  m0, m1,
                                  b_shares[idx1][1], b_shares[idx2][0],
                                  prot.pairwise_keys[idx0][0], prot.pairwise_keys[idx2][1],
                                  prot.pairwise_nonces[idx2])
            prot.pairwise_nonces[idx2] = prot.pairwise_nonces[idx2] + 1

        with tf.device(prot.servers[idx2].device_name):
            z[idx2][0] = prot._ot(prot.servers[idx0], prot.servers[idx2], prot.servers[idx1],
                                  m0, m1,
                                  b_shares[idx2][0], b_shares[idx1][1],
                                  prot.pairwise_keys[idx0][1], prot.pairwise_keys[idx1][0],
                                  prot.pairwise_nonces[idx0])
            z[idx2][1] = s[idx2]
            prot.pairwise_nonces[idx0] = prot.pairwise_nonces[idx0] + 1

    return z


def _pow_private(prot, x, p):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert x.share_type == ARITHMETIC
    assert p >= 1, "Exponent should be >= 0"

    # NOTE: pow should be able to use the `memoir` memoization

    with tf.name_scope("pow"):
        result = 1
        tmp = x
        while p > 0:
            bit = (p & 0x1)
            if bit > 0:
                result = result * tmp
            p >>= 1
            if p > 0:
                tmp = tmp * tmp
    return result


def _polynomial_private(prot, x, coeffs):
    assert isinstance(x, ABY3PrivateTensor), type(x)
    assert x.share_type == ARITHMETIC

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
            for i in range(len(c)-1):
                b[i+1] = ~msbs[i] & msbs[i+1]
            b[len(c)] = ~msbs[len(c)-1]

        # Compute the piecewise combination result
        result = 0
        for i in range(len(coeffs)):
            fi = prot.polynomial(x, coeffs[i])
            result = result + prot.mul_AB(fi, b[i])
    return result


def _sigmoid_private(prot, x, approx_type):
    assert isinstance(x, ABY3PrivateTensor), type(x)

    with tf.name_scope("sigmoid"):
        if approx_type == "piecewise_linear":
            c = (-2.5, 2.5)
            coeffs = ((1e-4, ), (0.50, 0.17), (1-1e-4, ))
        else:
            raise NotImplementedError("Only support piecewise linear approximation of sigmoid.")

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

        return ABY3PublicTensor(prot, x_on_0_t, x_on_1_t, x_on_2_t, x.is_scaled, x.share_type)


#
# reduce_sum helpers
#


def _reduce_sum_public(prot, x, axis = None, keepdims = False):

    x_on_0, x_on_1, x_on_2 = x.unwrapped

    with tf.name_scope("reduce_sum"):

        with tf.device(prot.servers[0].device_name):
            y_on_0 = x_on_0.reduce_sum(axis, keepdims)

        with tf.device(prot.servers[1].device_name):
            y_on_1 = x_on_1.reduce_sum(axis, keepdims)

        with tf.device(prot.servers[2].device_name):
            y_on_2 = x_on_2.reduce_sum(axis, keepdims)

    return ABY3PublicTensor(prot, y_on_0, y_on_1, y_on_2, x.is_scaled, x.share_type)


def _reduce_sum_private(prot, x, axis = None, keepdims = False):

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

        return ABY3PublicTensor(prot, x_on_0_concat, x_on_1_concat, x_on_2_concat,
                                is_scaled, xs[0].share_type)


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


def _write_private(prot, x, filename_prefix):
    assert isinstance(x, ABY3PrivateTensor), type(x)

    def encode(feature_row):
        # Converting a row to a string seems to be the only way of writing out
        # the dataset in a distributed way
        feature = tf.strings.reduce_join(tf.dtypes.as_string(tf.reshape(feature_row, [-1])), separator=",")
        return feature

    x_shares = x.unwrapped
    ops = []
    for i in range(3):
        with tf.device(prot.servers[i].device_name):
            for j in range(2):
                data = tf.data.Dataset.from_tensor_slices(x_shares[i][j].value) \
                        .map(encode)
                writer = tf.data.experimental.TFRecordWriter("{}_share{}{}".format(filename_prefix, i, j))
                ops.append(writer.write(data))

    return tf.group(*ops)


def _read_(prot, filename_prefix, batch_size, n_columns):

    if prot.nbits == 64:
        row_shape = [n_columns]
    elif prot.nbits == 128:
        row_shape = [n_columns, 2]

    def decode(line):
        fields = tf.string_split([line], ",").values
        fields = tf.strings.to_number(fields, tf.int64)
        fields = tf.reshape(fields, row_shape)
        return fields

    batch = [[None] * 2 for _ in range(3)]
    for i in range(3):
            with tf.device(prot.servers[i].device_name):
                for j in range(2):
                    data = tf.data.TFRecordDataset(["{}_share{}{}".format(filename_prefix, i, j)]) \
                            .map(decode) \
                            .repeat() \
                            .batch(batch_size=batch_size)
                    it = data.make_one_shot_iterator()
                    batch[i][j] = it.get_next()
                    batch[i][j] = tf.reshape(batch[i][j], [batch_size] + row_shape)
                    batch[i][j] = prot.int_factory.convert_to_dense_tensor(batch[i][j])

    return ABY3PrivateTensor(prot, batch, True, ARITHMETIC)


def _iterate_private(prot, tensor: "ABY3PrivateTensor",
        batch_size: int, repeat=True, shuffle=True, seed:int=None):

        assert isinstance(tensor, ABY3PrivateTensor)
        shares = tensor.unwrapped
        iterators = [[None] * 2 for _ in range(3)]
        results = [[None] * 2 for _ in range(3)]

        if seed is None:
            seed = np.random.randint(1, 1<<32) # this seed is publicly known.
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
                        dataset = dataset.shuffle(
                                buffer_size=512,
                                seed=seed)

                    dataset = dataset.batch(batch_size)

                    # TODO (juhou): initializable_iterator needs to run initializer.
                    # See how to switch to make_one_shot_iterator.
                    iterators[idx][i] = tf.compat.v1.data.make_initializable_iterator(dataset)
                    batch = iterators[idx][i].get_next()
                    # Wrap the tf.tensor as a dense tensor (no extra encoding is needed)
                    results[idx][i] = prot.int_factory.convert_to_dense_tensor(tf.reshape(batch, out_shape))

                prot.add_initializers(*[iterators[idx][i].initializer for i in range(2)])

        for idx in range(3): helper(idx)

        # Synchronize the reading of all 6 dataset iterators
        with tf.control_dependencies([share.value for result in results for share in result]):
            for i in range(3):
                results[i][0] = prot.int_factory.convert_to_dense_tensor(tf.identity(results[i][0].value))
                results[i][1] = prot.int_factory.convert_to_dense_tensor(tf.identity(results[i][1].value))

        return ABY3PrivateTensor(prot, results, tensor.is_scaled, tensor.share_type)


def _indexer_private(prot: ABY3,
                     tensor: ABY3PrivateTensor,
                     slc) -> "ABY3PrivateTensor":
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

