import datetime
import json
import os

import nn


class Sequential:
    """Represents a sequential network."""
    def __init__(self, layers, bias_init=.1):
        """Initializes the network.

        Args:
            layers: A list of `cifarvision.nn.layers.Layer` subclasses arranged
              from first layer to the last.
            bias_init: The value to initialize biases.
        """
        self._layers = layers
        self._bias_init = bias_init

    def configure(self):
        for i in range(1, len(self._layers)):
            self._layers[i].configure(self._layers[i-1].out_shape)

    def forward(self, batch):
        """Forward propagate the batch to compute loss.

        Args:
            batch: A tuple of length 2 where the first object is an ndarray of
              shape (M, D, H, W) which represents M images, each of with
              depth D, height H and width W while the second object is an
              ndarray of shape (M, 10) which represents labels for the images.

        Returns:
            A positive integer, which includes zero, which represents the loss.
        """
        output = batch[0]
        for i in range(len(self._layers) - 1):
            output = self._layers[i].forward(output)
        cost = self._layers[-1].forward(output, batch[1])
        return cost

    def forward_fast(self, X_batch):
        """Forward propagate the X_batch through the network without making
        any caches and return the scores of the batch

        Args:
            X_batch: an ndarray of shape (M, D, H, W) which represents M
              images, each of depth D, height H and width W.

        Returns:
            An ndarray of shape (M, 10) which represents labels for the X_batch.
        """
        # Output vector of the previous vector.
        current_out = X_batch
        for layer in self._layers:
            current_out = layer.forward_fast(current_out)
        return current_out

    def predict(self, X_batch):
        return self.forward_fast(X_batch)

    def backward(self):
        """Perform back-propagation through the network to compute gradients."""
        top_grad = self._layers[-1].backward()
        grad = top_grad
        for i in range(len(self._layers) - 2, -1, -1):
            grad = self._layers[i].backward(grad)

    def layers_with_internal_params(self):
        """Returns a list of layers with weights and biases."""
        layers = []
        for layer in self._layers:
            if layer.is_conv_layer() or layer.is_fully_connected_layer():
                layers.append(layer)
        return layers

    def get_layer_weights(self):
        weights = []
        layers = self.layers_with_internal_params()
        for layer in layers:
            weights.append(layer.weights)
        return weights

    def _save_layout(self, filename):
        with open(filename, 'w') as f:
            for layer in self._layers:
                f.write(str(layer) + '\n')

    def save(self, base_dir, trainer_data):
        date = datetime.datetime.today()
        directory = f"{base_dir}\\model_{date.strftime('%Y%m%d%H%M%S')}"
        os.mkdir(directory)
        filename = f'{directory}\\stats.txt'
        with open(filename, 'w') as f:
            for k, v in trainer_data.items():
                f.write(f'{k}: {v}\n')
        self._save_layout(f'{directory}\\architecture.txt')
        layers = {}
        for i in range(len(self._layers)):
            layers[i] = self._layers[i].get_save_data()
        data = {
            'saved': str(date),
            'layers': layers,
        }
        filename = f"{directory}\\model.json"
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f'Model successfully saved as {filename}')

    @classmethod
    def from_saved_file(cls, file_dir):
        with open(file_dir, 'r') as f:
            data = json.load(f)
        layers = []
        for config in data['layers'].values():
            name = config['name']
            if name == 'input':
                layers.append(nn.InputLayer.from_saved_data(config))
            elif name == 'conv':
                layers.append(nn.ConvolutionLayer.from_saved_data(config))
            elif name == 'norm':
                layers.append(nn.InputNormLayer.from_saved_data(config))
            elif name == 'dropout':
                layers.append(nn.Dropout.from_saved_data(config))
            elif name == 'fcl':
                layers.append(nn.FullyConnectedLayer.from_saved_data(config))
            elif name == 'flatten':
                layers.append(nn.FlattenLayer.from_saved_data(config))
            elif name == 'pool':
                layers.append(nn.MaxPoolLayer.from_saved_data(config))
            elif name == 'relu':
                layers.append(nn.ReluLayer.from_saved_data(config))
            elif name == 'softmax_crossentropy':
                layers.append(nn.SoftmaxCrossEntropyLayer.from_saved_data(config))
            else:
                raise ValueError(f'Unknown layer: {name}')
        net = cls(layers)
        return net
