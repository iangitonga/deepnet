import numpy as np

from . import base


class L2Regularizer(base.Regularizer):
    """Implements L2 loss. Also known as weight decay."""
    def __repr__(self):
        return f'L2Regularizer(lambda={self._lambda})'
    
    def forward(self, weights):
        self._cache_weights = weights
        s = 0
        for w in weights:
            s += np.sum(np.square(w))
        # 0.5 is included to allow nicer derivative.
        loss = self._lambda * 0.5 * s
        return loss

    def backward(self):
        # A good idea may be to multiply the weights by lambda while
        # in the layers in-place.
        for i in range(len(self._cache_weights)):
            self._cache_weights[i] = self._lambda * self._cache_weights[i]
        return self._cache_weights
