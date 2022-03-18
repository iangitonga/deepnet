import abc
from abc import ABC


class Layer(abc.ABC):
    def _check_shape(self, actual, expected):
        """Checks if the actual shape is equal to expected one.

        Args:
            actual: A tuple of integers representing the actual shape.
            expected: A tuple of integers representing the expected shape.

        Raises:
             ValueError: If the actual shape is not equal to the expected one.
        """
        if expected != actual:
            raise ValueError(
                f'the given value has shape {actual} instead of {expected}.'
            )

    @property
    @abc.abstractmethod
    def inp_shape(self):
        pass

    @property
    @abc.abstractmethod
    def out_shape(self):
        pass

    @abc.abstractmethod
    def forward(self, *args):
        """Computes the output of the layer in the forward pass."""

    @abc.abstractmethod
    def forward_fast(self, *args):
        """Computes the output of the layer in the forward pass without making caches."""

    @abc.abstractmethod
    def backward(self, *args):
        """Computes the gradients of the layer in the backward pass."""

    @abc.abstractmethod
    def get_save_data(self):
        """Returns a dictionary whose objects are JSON-serializable. The
          dictionary has all the data in the layer and all parameters."""

    @classmethod
    @abc.abstractmethod
    def from_saved_data(cls, data):
        """Constructs a layer object from the given data.

        Args:
            data: A dictionary of all the data and all parameters in the layer.
              The data is assumed to be created by `get_save_data` method.

        Returns:
            A new Layer object.
        """

    def is_fully_connected_layer(self):
        return False

    def is_conv_layer(self):
        return False

    def is_pool_layer(self):
        return False

    def is_non_linearity_layer(self):
        return False


class ParameterizedLayer(Layer, ABC):
    """A layer with internal parameters."""

    @property
    @abc.abstractmethod
    def weights(self):
        pass

    @weights.setter
    @abc.abstractmethod
    def weights(self, weights):
        pass

    @property
    @abc.abstractmethod
    def weights_grad(self):
        pass

    @property
    @abc.abstractmethod
    def bias(self):
        pass

    @bias.setter
    @abc.abstractmethod
    def bias(self, bias):
        pass

    @property
    @abc.abstractmethod
    def bias_grad(self):
        pass
