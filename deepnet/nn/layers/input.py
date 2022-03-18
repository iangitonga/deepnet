from .base import Layer


class InputLayer(Layer):
    def __init__(self, inp_shape):
        """Initializes the network.

        Args:
            inp_shape: A tuple of integers that represent the shape of input
              ndarray.
        Raises:
            TypeError - if input shape is not an integer.
        """
        self._inp_shape = inp_shape

    def __repr__(self):
        r = f'InputLayer(in={self._inp_shape})'
        return r

    @property
    def inp_shape(self):
        return self._inp_shape

    @property
    def out_shape(self):
        return self._inp_shape

    def forward(self, X):
        self._check_shape(X.shape, self._inp_shape)
        return X

    def backward(self, top_grad):
        return

    def forward_fast(self, X):
        return X

    def get_save_data(self):
        data = {
            'name': 'input',
            'inp_shape': self._inp_shape,
        }
        return data

    @classmethod
    def from_saved_data(cls, data):
        inp_shape = tuple(data['inp_shape'])
        layer = cls(inp_shape)
        return layer
