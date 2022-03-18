import numpy as np

from . import base


class Dropout(base.Layer):
    """Implements dropout regularization.

    This implementation of dropout acts as a layer object and not a regular
      regularization object.

    Forward:
        This layer receives a tensor of activations and drops activations
        randomly according to the given probability of dropping any activation.
    Backward:
        Receives a tensor that represents gradient of cost w.r.t output and
        returns a tensor that represents gradient of cost w.r.t input tensor.
    """
    def __init__(self, prob):
        """Layer initialization.

        Args:
            prob: An integer in range (0, 1] that represents the probability
              of keeping, i.e not dropping any activation received in this
              layer.
        """
        self._prob = prob
        self._inp_shape = None
        self._cache_mask = None

    def __repr__(self):
        r = f'Dropout(in={self._inp_shape}, prob={self._prob})'
        return r

    def configure(self, inp_shape):
        """Initializes all the values needed for this layer to function.

        Args:
            inp_shape: A tuple of length 4 that represents input batch shape.
        """
        self._inp_shape = inp_shape

    @property
    def inp_shape(self):
        return self._inp_shape

    @property
    def out_shape(self):
        out_shape = self._inp_shape
        return out_shape

    def forward(self, X):
        # The divide allows for scaling the activations to avoid scaling them
        # when doing prediction.
        mask = (np.random.rand(*X.shape) < self._prob) / self._prob
        self._cache_mask = mask
        X *= mask
        return X

    def forward_fast(self, X):
        return X

    def backward(self, top_grad):
        top_grad *= self._cache_mask
        return top_grad

    def get_save_data(self):
        data = {
            'name': 'dropout',
            'inp_shape': self._inp_shape,
            'prob': self._prob,
        }
        return data

    @classmethod
    def from_saved_data(cls, data):
        inp_shape = tuple(data['inp_shape'])
        prob = data['prob']
        layer = cls(prob)
        layer.configure(inp_shape)
        return layer
