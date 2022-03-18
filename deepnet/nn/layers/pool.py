import numpy as np

from .base import Layer


class MaxPoolLayer(Layer):
    """Implements max pooling.

    Forward:
        This layer takes an input batch of shape (M, D, H, W) where M is the
        batch size, D is the depth of each volume in the batch, H is the height
        of the volume and W is the width of the volume. It produces an output
        batch of shape (M, D, H1, W1) where H1 is the new height of the volume
        and W2 is the new width of the volume.

    Backward:
        This layer expects a tensor of shape (M, D, H1, W1) which represents
        the gradient of the cost wrt the output of this layer. It computes and
        returns a tensor of shape (M, D, H, W) which represents the gradient of
        the cost wrt the input tensor to this layer .It also computes the
        gradients wrt internal parameters.
    """
    def __init__(self, spacial_extent, stride):
        """Initialize MaxPoolLayer.

        Args:
            spacial_extent: A positive integer which represents the height and
              the width of the pooling kernel.
            stride: A positive integer which represents the number of columns
              and rows to skip when sliding the pooling kernel.
        """
        self._spacial_extent = spacial_extent
        self._stride = stride
        self._inp_shape = None
        self._cache_inp = None
        self._out_shape = None
        # Stores the indices for the where the max values in the output are
        # located in the input.
        self._cache_idx = None

    def __repr__(self):
        r = f'MaxPool(in={self._inp_shape}, out={self._out_shape}, spacial_extent={self._spacial_extent},' \
            f' stride={self._stride})'
        return r

    def configure(self, inp_shape):
        """Initializes all the values needed for this layer to function.

        Args:
            inp_shape: A tuple of length 4 that represents input batch shape.
        """
        self._inp_shape = inp_shape
        M, D, H, W = self._inp_shape
        F = self._spacial_extent
        S = self._stride
        self._out_shape = (M, D, int(((H - F) / S) + 1), int(((W - F) / S) + 1))

    def _max_pool(self, X):
        """Performs max pooling of the input batch.

        Args:
            X: A numpy array of shape (M, D, H, W).

        Returns:
            A numpy array of shape (M, D, H1, W1).
        """
        M = X.shape[0]
        _, D, H1, W1 = self._out_shape
        F = self._spacial_extent
        S = self._stride
        # M, D, H1, W1, F, F
        pool = np.lib.stride_tricks.sliding_window_view(X, (F, F), axis=(2, 3))[::1, ::1, ::S, ::S]
        pool = pool.reshape((M, D, H1*W1, F*F))
        self._cache_idx = pool.argmax(axis=3)
        # M, D, H1*W1 * 1
        out = pool.max(axis=3)
        out = out.reshape((M, D, H1, W1))
        return out

    def _grad_max_pool(self, top_grad):
        """Computes the gradient of the cost wrt input to this layer.

        Args:
            top_grad: A numpy array of shape (M, D, H1, W1) that represents
             the gradient of the cost wrt the output of this layer.

        Returns:
            A numpy array of shape (M, D, H, W) that represents the gradient
            of the cost wrt the input to this layer.
        """
        M = top_grad.shape[0]
        _, _, H, W = self._inp_shape
        _, D, H1, W1 = self._out_shape
        F = self._spacial_extent
        S = self._stride
        grad = np.zeros((M*D*H1*W1, F*F))
        idx = self._cache_idx.flatten()
        grad[np.arange(grad.shape[0]), idx] = top_grad.flatten()
        pool_shape = int(W / F), F
        row_step, col_step = int(W / F), S
        pool = np.lib.stride_tricks.sliding_window_view(grad, pool_shape)[::row_step, ::col_step]
        r = pool.shape[0] * pool.shape[1]
        c = pool.shape[2] * pool.shape[3]
        pool = pool.reshape((r, c))
        grad = pool.reshape((M, D, H, W))
        return grad

    @property
    def inp_shape(self):
        return self._inp_shape

    @property
    def out_shape(self):
        return self._out_shape

    def forward(self, X):
        self._check_shape(X.shape, self._inp_shape)
        self._cache_inp = X
        out = self._max_pool(X)
        return out

    def forward_fast(self, X):
        out = self._max_pool(X)
        return out

    def backward(self, top_grad):
        self._check_shape(top_grad.shape, self._out_shape)
        out = self._grad_max_pool(top_grad)
        return out

    def is_pool_layer(self):
        return True

    def get_save_data(self):
        data = {
            'name': 'pool',
            'inp_shape': self._inp_shape,
            'out_shape': self._out_shape,
            'spacial_extent': self._spacial_extent,
            'stride': self._stride,
        }
        return data

    @classmethod
    def from_saved_data(cls, data):
        inp_shape = tuple(data['inp_shape'])
        spacial_extent = data['spacial_extent']
        stride = data['stride']
        layer = cls(spacial_extent, stride)
        layer.configure(inp_shape)
        return layer
