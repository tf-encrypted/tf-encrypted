"""Implementation of the DataOwner abstraction."""
import tensorflow as tf

import tf_encrypted as tfe


class DataOwner:
    def __init__(self, player, generator_builder, num_samples=None):
        """
        An abstraction for data owners to share dataset in model training or inference

        :param str or Player player: Player or Name of the player
        :param callable generator_builder: Function of building dataset generator
        :param int num_samples: How many samples in this generator
        """
        if isinstance(player, str):
            self.player = tfe.get_config().get_player(player)
        else:
            self.player = player
        assert isinstance(self.player, tfe.player.Player)

        self.num_samples = num_samples
        with tf.device(self.player.device_name):
            self.generator = generator_builder()
            self.first_element = next(self.generator)

            if isinstance(self.first_element, (list, tuple)):
                self.first_data_x = self.first_element[0]
                self.first_data_y = self.first_element[1]
            else:
                self.first_data_x = self.first_element
                self.first_data_y = None

    @property
    def batch_shape(self):
        return self.first_data_x.shape

    @property
    def batch_size(self):
        return self.first_data_x.shape[0]

    @property
    def num_classes(self):
        return self.first_data_y.shape[1]

    def provide_data(self):
        def share(plain_iter):

            if isinstance(self.first_element, (list, tuple)):
                yield (
                    tfe.define_private_input(self.player.name, lambda: data)
                    for data in self.first_element
                )
            else:
                yield tfe.define_private_input(
                    self.player.name, lambda: self.first_element
                )

            for plain_data in plain_iter:
                if isinstance(plain_data, (list, tuple)):
                    yield (
                        tfe.define_private_input(self.player.name, lambda: data)
                        for data in plain_data
                    )
                else:
                    yield tfe.define_private_input(self.player.name, lambda: plain_data)

        return share(self.generator)
