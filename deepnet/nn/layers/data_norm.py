import numpy as np

from .base import Layer


class InputNormLayer(Layer):
    """Preprocessed a batch of color images by normalizing and centering it."""
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std
        self._inp_shape = None

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
        X = X.astype(np.float64) - self._mean
        X /= self._std
        return X

    def forward_fast(self, X):
        X = X.astype(np.float64) - self._mean
        X /= self._std
        return X

    def backward(self, top_grad):
        return top_grad

    def get_save_data(self):
        data = {
            'name': 'norm',
            'inp_shape': self._inp_shape,
            'mean': self._mean.tolist(),
            'std': self._std.tolist(),
        }
        return data

    @classmethod
    def from_saved_data(cls, data):
        inp_shape = tuple(data['inp_shape'])
        mean = np.array(data['mean'])
        std = np.array(data['std'])
        layer = cls(mean, std)
        layer.configure(inp_shape)
        return layer
