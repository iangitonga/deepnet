import numpy as np

from .base import Layer


class FlattenLayer(Layer):
    """Flattens the input into a vector."""
    def __init__(self,):
        self._inp_shape = None
        self._out_shape = None

    def __repr__(self):
        r = f'FlattenLayer(in={self._inp_shape}, out={self._out_shape})'
        return r

    @property
    def inp_shape(self):
        return self._inp_shape

    @property
    def out_shape(self):
        return self._out_shape

    def configure(self, inp_shape):
        """Initializes all the values needed for this layer to function.

        Args:
            inp_shape: A tuple of length 4 that represents input batch shape.
        """
        self._inp_shape = inp_shape
        self._out_shape = inp_shape[0], int(np.prod(inp_shape[1:]))

    def forward(self, X):
        self._check_shape(X.shape, self._inp_shape)
        result = X.reshape(self._out_shape)
        return result

    def forward_fast(self, X):
        result = X.reshape((X.shape[0], np.prod(X.shape[1:])))
        return result

    def backward(self, top_grad):
        result = top_grad.reshape(self._inp_shape)
        return result

    def get_save_data(self):
        data = {
            'name': 'flatten',
            'inp_shape': self._inp_shape
        }
        return data

    @classmethod
    def from_saved_data(cls, data):
        inp_shape = tuple(data['inp_shape'])
        layer = cls()
        layer.configure(inp_shape)
        return layer
