import numpy as np

from .base import Trainer


class VanillaSgd(Trainer):
    def _update_net_parameters(self):
        layers = self._net.layers_with_internal_params()
        for layer in layers:
            eta = self._learning_rate / self._batch_size
            update_weights = eta * layer.weights_grad
            layer.weights += -update_weights
            update_bias = eta * layer.bias_grad
            layer.bias += -update_bias


class NesterovSgd(Trainer):
    def __init__(self, mu, *args, **kwargs):
        super(NesterovSgd, self).__init__(*args, **kwargs)
        self._mu = mu
        self._v_weights = {}
        self._v_bias = {}
        self._layers = self._net.layers_with_internal_params()
        for i in range(len(self._layers)):
            layer = self._layers[i]
            if layer.is_fully_connected_layer():
                self._v_weights[i] = np.zeros((layer.n_units, layer.inp_shape[1]))
                self._v_bias[i] = np.zeros((layer.n_units, 1))
            if layer.is_conv_layer():
                self._v_weights[i] = np.zeros(layer.full_filters_shape)
                self._v_bias[i] = np.zeros((layer.full_filters_shape[0], 1))

    def _update_net_parameters(self):
        for i in range(len(self._layers)):
            layer = self._layers[i]
            eta = self._learning_rate / self._batch_size
            weights_grad = layer.weights_grad
            bias_grad = layer.bias_grad
            v_weights_prev = self._v_weights[i]
            v_bias_prev = self._v_bias[i]
            self._v_weights[i] = (self._mu * self._v_weights[i]) - (eta * weights_grad)
            self._v_bias[i] = (self._mu * self._v_bias[i]) - (eta * bias_grad)
            # x += -mu * v_prev + (1 + mu) * v
            layer.weights += -self._mu * v_weights_prev + (1 + self._mu) * self._v_weights[i]
            layer.bias += -self._mu * v_bias_prev + (1 + self._mu) * self._v_bias[i]              
