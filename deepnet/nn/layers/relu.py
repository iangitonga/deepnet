import numpy as np

from .base import Layer


class ReluLayer(Layer):
    def __init__(self):
        self._inp_shape = None
        self._cache_input = None
        self._cache_output = None
        self._relu_grad = np.vectorize(self._relu_derivative, otypes=[np.float64])

    def __repr__(self):
        r = f'ReluLayer(in={self._inp_shape})'
        return r

    def _relu(self, X):
        clipped = X.clip(0)
        return clipped

    def _relu_derivative(self, z):
        return 0 if z <= 0 else 1

    def _output_grad_wrt_input(self):
        grad = self._relu_grad(self._cache_input)
        return grad

    def _cost_grad_wrt_input(self, top_grad):
        grad = top_grad * self._output_grad_wrt_input()
        return grad

    @property
    def inp_shape(self):
        return self._inp_shape

    @property
    def out_shape(self):
        return self._inp_shape

    def configure(self, inp_shape):
        """Initializes all the values needed for this layer to function.

        Args:
            inp_shape: A tuple of length 4 that represents input batch shape.
        """
        self._inp_shape = inp_shape

    def forward(self, X):
        self._check_shape(X.shape, self._inp_shape)
        self._cache_input = X
        result = self._relu(X)
        self._cache_output = result
        return result

    def forward_fast(self, X):
        return self._relu(X)

    def backward(self, top_grad):
        self._check_shape(top_grad.shape, self._inp_shape)
        grad = top_grad * self._output_grad_wrt_input()
        return grad

    def is_non_linearity_layer(self):
        return True

    def get_save_data(self):
        data = {
            'name': 'relu',
            'inp_shape': self._inp_shape,
        }
        return data

    @classmethod
    def from_saved_data(cls, data):
        inp_shape = tuple(data['inp_shape'])
        layer = cls()
        layer.configure(inp_shape)
        return layer
