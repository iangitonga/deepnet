import numpy as np

from . import base


class MomentumTrainer(base.Trainer):
    """Implements SGD with regular momentum."""
    def __init__(self, mu, *args, **kwargs):
        super(MomentumTrainer, self).__init__(*args, **kwargs)
        self._mu = mu
        # Keeps the accumulated velocity for the weights and biases.
        self._v_weights = {}
        self._v_bias = {}
        self._layers = self._net.layers_with_internal_params()
        for i in range(len(self._layers)):
            layer = self._layers[i]
            if layer.is_fully_connected_layer():
                self._v_weights[i] = np.zeros((layer.n_units, layer.inp_shape[1]))
                self._v_bias[i] = np.zeros((layer.n_units, 1))
            elif layer.is_conv_layer():
                self._v_weights[i] = np.zeros(layer.full_filters_shape)
                self._v_bias[i] = np.zeros((layer.full_filters_shape[0], 1))

    def _update_net_parameters(self):
        layers = self._net.layers_with_internal_params()
        eta = self._learning_rate / self._batch_size
        for i in range(len(layers)):
            layer = layers[i]
            weights_grad = layer.weights_grad
            bias_grad = layer.bias_grad
            self._v_weights[i] = (self._mu * self._v_weights[i]) - (eta * weights_grad)
            self._v_bias[i] = (self._mu * self._v_bias[i]) - (eta * bias_grad)
            layer.weights += self._v_weights[i]
            layer.bias += self._v_bias[i]
