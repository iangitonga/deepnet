import numpy as np

from .base import ParameterizedLayer


class FullyConnectedLayer(ParameterizedLayer):
    """Implements Linear layer.
    
    Forward:
        This layer takes an input batch of shape (M, D) where M is the batch
        size and D is the number of units in the previous layer. It produces
        an output batch of shape (M, N) where N is the number of the units in
        this layer.
    Backward:
        This layer expects a matrix of shape (M, N) that represents the
        gradient of cost wrt the output of this function. It computes a matrix
        of shape (M, D) that represents the gradient of the cost wrt the input
        to this layer. It also computes the gradient of the cost wrt the
        internal parameters.
    """
    def __init__(self, n_units):
        """Initializes FullyConnectedLayer.

        Args:
            n_units: A positive integer denoting the number of units in this layer.
        """
        self._n_units = n_units
        self._out_shape = None
        self._inp_shape = None
        self._weights = None
        self._weights_grad = None
        self._bias = None
        self._bias_grad = None
        self._cache_inp = None
        self._cache_out = None

    def __repr__(self):
        r = f'FullyConnectedLayer(in={self._inp_shape}, out={self._out_shape}, H={self._n_units})'
        return r

    @property
    def n_units(self):
        return self._n_units

    @property
    def inp_shape(self):
        return self._inp_shape

    @property
    def out_shape(self):
        return self._out_shape

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._check_shape(weights.shape, self._weights.shape)
        self._weights = weights

    @property
    def weights_grad(self):
        return self._weights_grad

    @weights_grad.setter
    def weights_grad(self, weights_grad):
        self._check_shape(weights_grad.shape, self._weights.shape)
        self._weights_grad = weights_grad

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._check_shape(bias.shape, self._bias.shape)
        self._bias = bias

    @property
    def bias_grad(self):
        return self._bias_grad

    @bias_grad.setter
    def bias_grad(self, bias_grad):
        self._check_shape(bias_grad.shape, self._bias.shape)
        self._bias_grad = bias_grad

    def _initialize_weights(self):
        inp_dim, out_dim = self._inp_shape[1], self._n_units
        self._weights = np.random.randn(out_dim, inp_dim) / np.sqrt(inp_dim / 2)

    def _initialize_bias(self):
        self._bias = np.full((self._n_units, 1), 0.1)

    def _cost_grad_wrt_weights(self, top_grad):
        """Gradient of the cost function with respect to this layer's weights."""
        out_grad = self._output_grad_wrt_weights()
        grad = (top_grad.reshape((top_grad.shape[0], 1, top_grad.shape[1])) * out_grad).transpose((0, 2, 1))
        grad = grad.sum(axis=0)
        return grad

    def _output_grad_wrt_weights(self):
        """Gradient of this layers output with respect to its weights."""
        M, N = self._inp_shape
        ext_cache = self._cache_inp.reshape((M, N, 1))
        grad = np.full((M, N, self._n_units), ext_cache)
        return grad

    def _cost_grad_wrt_bias(self, top_grad):
        """Gradient of the cost function with respect to this layer's bias."""
        grad = np.sum(top_grad.transpose(), axis=1).reshape((self._n_units, 1))
        return grad

    def _cost_grad_wrt_input(self, top_grad):
        """Gradient of the cost function with respect to this layer's input."""
        grad = np.matmul(self._weights.transpose(), top_grad.transpose()).transpose()
        return grad

    def _linear(self, X):
        return (np.matmul(self._weights, X.transpose()) + self._bias).transpose()

    def configure(self, inp_shape):
        """Initializes all the values needed for this layer to function.

        Args:
            inp_shape: A tuple of length 4 that represents input batch shape.
        """
        self._inp_shape = inp_shape
        self._out_shape = self._inp_shape[0], self._n_units
        self._initialize_weights()
        self._initialize_bias()

    def forward(self, X):
        self._check_shape(X.shape, self._inp_shape)
        self._cache_inp = X
        result = self._linear(X)
        self._cache_out = result
        return result

    def forward_fast(self, X):
        result = self._linear(X)
        return result

    def backward(self, top_grad):
        self._check_shape(top_grad.shape, self._out_shape)
        self.weights_grad = self._cost_grad_wrt_weights(top_grad)
        self.bias_grad = self._cost_grad_wrt_bias(top_grad)
        grad_wrt_inp = self._cost_grad_wrt_input(top_grad)
        return grad_wrt_inp

    def is_fully_connected_layer(self):
        return True

    def get_save_data(self):
        data = {
            'name': 'fcl',
            'n_units': self._n_units,
            'inp_shape': self._inp_shape,
            'out_shape': self._out_shape,
            'weights': self._weights.tolist(),
            'bias': self.bias.tolist(),
        }
        return data

    @classmethod
    def from_saved_data(cls, data):
        n_units = data['n_units']
        inp_shape = tuple(data['inp_shape'])
        weights = np.array(data['weights'])
        bias = np.array(data['bias'])
        layer = cls(n_units)
        layer.configure(inp_shape)
        layer.weights = weights
        layer.bias = bias
        return layer
