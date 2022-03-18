import numpy as np

from .base import ParameterizedLayer


# noinspection PyUnresolvedReferences
class ConvolutionLayer(ParameterizedLayer):
    """Implements 3D Convolution layer.

    Forward:
        This layer takes an input batch of shape (M, D, H, W) where M is the
        batch size, D is the depth of each volume in the batch, H is the height
        of the volume and W is the width of the volume. It produces an output
        batch of shape (M, N, H1, W1) where N is the number of filters, H1 is
        the new height of each volume and W2 is the new width of each volume.

    Backward:
        This layer expects a tensor of shape (M, N, H1, W1) which represents
        the gradient of the cost wrt the output of this layer. It computes and
        returns a tensor of shape (M, D, H, W) which represents the gradient of
        the cost wrt the input tensor to this layer. It also computes the
        gradient wrt internal parameters.

    Notes:
        Supports convolution where:
          - Input volume height and width are equal to output volume height and
              width.
          - Filter height and width are equal.
          - Stride is 1.
    """
    def __init__(self, n_filters, filter_size, padding):
        """Initializes the Convolution Layer.

        Args:
            n_filters: A positive integer that denotes the number of filters to
              include in this layer.
            filter_size: A tuple of length 2, (f1, f2) where f1 is the height
              of filters and f2 is the width of the filters.

        Notes:
            This implementation supports stride 1 only.
        """
        self._n_filters = n_filters
        self._filter_shape = filter_size
        self._padding = padding
        self._stride = 1
        self._inp_shape = None
        self._out_shape = None
        self.full_filters_shape = None
        # Params
        self._filters = None
        self._filters_grad = None
        self._bias = None
        self._bias_grad = None
        # Caches
        self._cache_inp = None

    def __repr__(self):
        r = f'ConvLayer(in={self._inp_shape}, out={self._out_shape}, stride={self._stride},' \
            f' n_filters={self._n_filters}, filter_size={self._filter_shape}, padding={self._padding})'
        return r

    def _initialize_filters(self):
        self._filters = np.empty(self.full_filters_shape)
        inp_size = self._inp_shape[1] * self._inp_shape[2] * self._inp_shape[3]
        depth = self._inp_shape[1]
        for i in range(self._n_filters):
            for j in range(depth):
                self._filters[i][j] = np.random.randn(*self._filter_shape) / np.sqrt(inp_size/2)

    def _initialize_bias(self):
        self._bias = np.full((self._n_filters, 1), 0.1)

    def _convolve(self, X):
        """Computes the convolution of a batch of data with the tensor of
          filters in this layer."""
        M, D, H, W = X.shape
        N = self._n_filters
        P = self._padding
        pad_width = ((0, 0), (0, 0), (P, P), (P, P))
        X_padded = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)
        X_stretched = self._stretch_batch(X_padded, self._filters[0].shape, (1, 2, 3))
        # Here, we convert the filters tensor into a matrix where the rows are
        # formed by flattening each filter into a row vector and then performing
        # a transpose on that matrix.
        F_stretched = self._filters.reshape((N, np.prod(self._filters.shape[1:]))).transpose()
        # The operation below performs a matrix multiplication between each
        # matrix in the 'X_stretched' and 'F_stretched' matrix. This
        # corresponds to performing a dot product between each window and each
        # filter for every example.
        out = np.matmul(X_stretched, F_stretched)
        # Since we transposed 'F_stretched' matrix to allow for matrix
        # multiplication, we transpose the above matrix to correctly
        # rearrange it.
        out = out.swapaxes(1, 2)
        # Filter shape must be square
        F = self._filter_shape[0]
        S = self._stride
        H_out, W_out = int(((H - F) / S) + 1 + P*2), int(((W - F) / S) + 1 + P*2)
        out = out.reshape((M, N, H_out, W_out))
        return out

    def _stretch_batch(self, X, window_shape, axis):
        """Constructs an output tensor where all the window views in each batch
         example are arranged in a matrix.

        Essentially, this method arranges the window views in the given axis
        in a matrix where each row is a window view(same shape as window_shape)
        that is flattened into a row vector.

        Args:
            X: A numpy array that is 4-dimensional.
            window_shape: A tuple that represents the shape of the sliding
              window.
            axis: A tuple that represents the axes in which to perform slide
              operation.

        Returns:
            An ndarray of shape (M, N, S) where M is the batch size, N is the
              total number of window views in each example and S is the length
              of each window view if the length of given axis tuple is 3. If
              the the length of given axis tuple is 2, it returns An ndarray
              of shape (M, D, N, S) where D is the depth of each volume in X.
        """
        # Out shape: (M, 1, H1, W1, f1, f2, f3)
        windows = np.lib.stride_tricks.sliding_window_view(X, window_shape, axis=axis)
        # Multiplies the height and the width of the matrix whose elements are
        # the window views. This gives us the number of the rows we need to
        # store the flattened window views.
        M = windows.shape[0]
        # Only useful when window shape is 2 dimensional. It is always equal to
        # 1 if window shape is 3 dimensional.
        N = windows.shape[1]
        n_rows = windows.shape[2] * windows.shape[3]
        # Multiplies the dimensions of the window views which allows us to get
        # the length of the row vector which is the flattened window view.
        n_cols = np.prod(windows.shape[4:])
        if len(axis) == 2:
            out = windows.reshape((M, N, n_rows, n_cols))
        else:
            out = windows.reshape((M, n_rows, n_cols))
        return out

    def _grad_cost_wrt_inp_vol(self, top_grad):
        M, D, H, W = self._inp_shape
        N = self._n_filters
        P = self._padding
        pad_width = ((0, 0), (0, 0), (P, P), (P, P))
        top_grad = np.pad(top_grad, pad_width=pad_width, mode='constant', constant_values=0)
        # For each gradient volume in 'top_grad', we want to perform a
        # convolution between each gradient matrix in the volume and the
        # corresponding filter matrix in the filters volume. Each filter matrix
        # has to be rotated by 180 degrees.
        top_grad_stretched = self._stretch_batch(top_grad, self._filter_shape, (2, 3))
        # Rotates each filter matrix in the filters tensor by 180 degrees.
        filters_rot = np.rot90(self._filters, 2, axes=(2, 3))
        # Stretches each filter in the filters tensor into a flat array.
        # Swapping axes allows matrix multiplication.
        filters_stretched = filters_rot.reshape((N, D, np.prod(self._filter_shape))).swapaxes(1, 2)
        # Perform convolution and swap axes to rearrange the swapped axes.
        result = np.matmul(top_grad_stretched, filters_stretched).swapaxes(2, 3)
        grad = result.reshape((M, N, D, H, W))
        # Sum the gradients of all the filters in this layer.
        grad = np.sum(grad, axis=1)
        return grad

    def _grad_cost_wrt_filters(self, top_grad):
        M, D, H, W = self._inp_shape
        N = self._n_filters
        P = self._padding
        pad_width = ((0, 0), (0, 0), (P, P), (P, P))
        # For each gradient matrix in each gradient volume in 'top_grad', we
        # want to compute a convolution between itself and each input matrix
        # in the corresponding input volume. Each input matrix in each input
        # volume must be rotated bt 180 degrees. Also, each gradient matrix in
        # each gradient volume must be rotated by 180 degrees.
        top_grad = np.pad(top_grad, pad_width=pad_width, mode='constant', constant_values=0)
        top_grad_rot = np.rot90(top_grad, 2, axes=(2, 3))
        top_grad_stretched = self._stretch_batch(top_grad_rot, (H, W), axis=(2, 3))
        inp_rot = np.rot90(self._cache_inp, 2, axes=(2, 3))
        inp_stretched = inp_rot.reshape((M, 1, D, H*W)).swapaxes(2, 3)
        result = np.matmul(top_grad_stretched, inp_stretched).swapaxes(2, 3)
        result = result.reshape((M, N, D, *self._filter_shape))
        # Sum the gradients of all the examples in the batch.
        grad = result.sum(axis=0)
        return grad

    def _grad_cost_wrt_bias(self, top_grad):
        M = self._inp_shape[0]
        N = self._n_filters
        grad = np.empty((M, N))
        # TODO: Avoid loops.
        for i in range(M):
            for j in range(N):
                grad[i][j] = np.sum(top_grad[i][j])
        grad = (grad.sum(axis=0)).reshape((N, 1))
        return grad

    @property
    def inp_shape(self):
        return self._inp_shape

    @property
    def out_shape(self):
        return self._out_shape

    @property
    def weights(self):
        return self._filters

    @weights.setter
    def weights(self, w):
        self._check_shape(w.shape, self.full_filters_shape)
        self._filters = w

    @property
    def weights_grad(self):
        return self._filters_grad

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, b):
        self._check_shape(b.shape, self._bias.shape)
        self._bias = b

    @property
    def bias_grad(self):
        return self._bias_grad

    def configure(self, inp_shape):
        """Initializes all the values needed for this layer to function.

        Args:
            inp_shape: A tuple of length 4 that represents input batch shape.
        """
        self._inp_shape = inp_shape
        H, W = self._inp_shape[2], self._inp_shape[3]
        F = self._filter_shape[0]
        if self._filter_shape[0] != self._filter_shape[1]:
            e = 'Convolutions with non-square filters is currently not supported.'
            raise NotImplementedError(e)
        S = self._stride
        P = self._padding
        H_out, W_out = int(((H - F) / S) + 1 + 2*P), int(((W - F) / S) + 1 + 2*P)
        self._out_shape = self._inp_shape[0], self._n_filters, H_out, W_out
        if self._out_shape[2] != H or self._out_shape[3] != W:
            e = 'Convolutions where input volume height and width are not the equal to output volume' \
                ' height and width are currently not supported. Use padding to make them equal.'
            raise NotImplementedError(e)
        self.full_filters_shape = (self._n_filters, self._inp_shape[1], *self._filter_shape)
        self._initialize_filters()
        self._initialize_bias()

    def forward(self, X):
        self._check_shape(X.shape, self._inp_shape)
        self._cache_inp = X
        out = self._convolve(X)
        return out

    def forward_fast(self, X):
        out = self._convolve(X)
        return out

    def backward(self, top_grad):
        self._check_shape(top_grad.shape, self._out_shape)
        self._filters_grad = self._grad_cost_wrt_filters(top_grad)
        self._bias_grad = self._grad_cost_wrt_bias(top_grad)
        out_grad = self._grad_cost_wrt_inp_vol(top_grad)
        return out_grad

    def is_conv_layer(self):
        return True

    def get_save_data(self):
        data = {
            'name': 'conv',
            'inp_shape': self._inp_shape,
            'out_shape': self._out_shape,
            'n_filters': self._n_filters,
            'filter_size': self._filter_shape,
            'padding': self._padding,
            'weights': self._filters.tolist(),
            'bias': self._bias.tolist(),
        }
        return data

    @classmethod
    def from_saved_data(cls, data):
        n_filters = data['n_filters']
        filter_size = tuple(data['filter_size'])
        padding = data['padding']
        layer = cls(n_filters, filter_size, padding)
        inp_shape = tuple(data['inp_shape'])
        layer.configure(inp_shape)
        layer.weights = np.array(data['weights'])
        layer.bias = np.array(data['bias'])
        return layer
