import numpy as np

from .base import Layer


class SoftmaxCrossEntropyLayer(Layer):
    """Softmax output layer coupled with cross-entropy loss function.

    Forward:
        This layer takes a matrix of shape (M, D) where M is the batch size
        and D is the number of units in the previous layer. It computes the
        softmax of the input and then the cross entropy loss of the batch.
    Backward:
        This layer computes a matrix of shape (M, D) that represents the
        gradient of the loss wrt input tensor.
    """
    def __init__(self):
        self._inp_shape = None
        self._cache_input = None
        self._cache_label = None

    def __repr__(self):
        r = f'SoftmaxCrossEntropyLayer(in={self._inp_shape})'
        return r

    def _softmax(self, batch):
        batch = batch - batch.max(axis=1).reshape((batch.shape[0], 1))
        # batch = batch.clip(-1000, None, out=batch)
        exp = np.exp(batch)
        result = exp / np.sum(exp, axis=1).reshape((batch.shape[0], 1))
        return result

    def _loss(self, X):
        indices = self._cache_label.argmax(axis=1)
        result1 = -1 * (X[np.arange(X.shape[0]), indices]).transpose()
        result2 = (np.log(np.sum(np.exp(X), axis=1))).transpose()
        result = np.sum(result1 + result2)
        return result

    def _gradient(self):
        sm = self._softmax(self._cache_input)
        mask = self._cache_label
        grad = sm - mask
        return grad

    @property
    def inp_shape(self):
        return self._inp_shape

    @property
    def out_shape(self):
        return 1,

    def configure(self, inp_shape):
        """Initializes all the values needed for this layer to function.

        Args:
            inp_shape: A tuple of length 4 that represents input batch shape.
        """
        self._inp_shape = inp_shape

    def forward(self, X, Y):
        self._check_shape(X.shape, self._inp_shape)
        self._cache_input = X
        self._cache_label = Y
        result = self._loss(X)
        return result

    def forward_fast(self, X):
        result = self._softmax(X)
        return result

    def backward(self):
        grad = self._gradient()
        return grad

    def get_save_data(self):
        data = {
            'name': 'softmax_crossentropy',
            'inp_shape': self._inp_shape,
        }
        return data

    @classmethod
    def from_saved_data(cls, data):
        inp_shape = tuple(data['inp_shape'])
        layer = cls()
        layer.configure(inp_shape)
        return layer
