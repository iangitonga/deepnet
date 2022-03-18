import abc


class Regularizer(abc.ABC):
    """Base class for all regularizers.

    Forward:
        Receives a list of weights in the network and compute the
          regularization loss.
    Backward:
        Computes and return the gradient of the regularization loss w.r.t the
          input weights.
    """
    def __init__(self, lambda_):
        """Initialize regularizer.

        Args:
            lambda_: A non-negative integer that represents the value that the
             regularization loss is scaled by.
        """
        self._lambda = lambda_
        self._cache_weights = None

    @property
    def lambda_(self):
        return self._lambda

    @abc.abstractmethod
    def forward(self, weights):
        """Compute the regularization loss.

        Args:
            weights: A list of numpy arrays that represents the weights of a
              particular network. The ordering of the weights is preserved when
              computing the gradient.

        Returns:
            A non-negative integer that represents regularization loss.
        """

    @abc.abstractmethod
    def backward(self):
        """Compute the gradient of the regularization loss w.r.t the input
        weights.

        Returns:
            A list of numpy arrays representing the computed gradient. The
              gradients are ordered in the same way as the weights.
        """
