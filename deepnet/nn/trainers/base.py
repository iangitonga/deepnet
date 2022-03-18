import abc
import datetime

import numpy as np


class Trainer(abc.ABC):
    """A base class for trainers which are classes whose instances can train a
    net object with a given dataset.

    A trainer class must inherit from this class and implement
     `_update_net_parameters` function.
    """
    def __init__(self, net, num_epochs, batch_size, learning_rate, lr_decay, iter_per_epoch=None):
        """Initializes the trainer.

        Args:
            net: A network object.
            num_epochs: The number of epochs to train.
            batch_size: The number of examples to compute the gradient.
            learning_rate: A float which scales the update weights and biases.
            lr_decay: A positive float in (0, 1] that is multiplied by the
              learning rate after every epoch to decay it.
            iter_per_epoch: The number of iterations to run for each epoch. If
              not provided it is calculated.

        Raises:
            TypeError: If the given `net` object is of the wrong type.
            TypeError: If any of  `n_iter` or `iter_per_epoch` or
              `learning_rate` is not an integer.
            ValueError: If any of  `n_iter` or `learning_rate` is a negative
              integer.
            ValueError: If `iter_per_epoch` is less that one.
        """
        self._net = net
        self._net.configure()
        self._num_epochs = num_epochs
        self._iter_per_epoch = iter_per_epoch
        self._learning_rate = learning_rate
        self._lr_decay = lr_decay
        self._batch_size = batch_size
        self._regularizer = None
        self._rng = np.random.default_rng()

    def _get_batch(self, X, Y, size=None):
        size = size or self._batch_size
        batch_indices = self._rng.choice(np.arange(X.shape[0]), size, replace=False)
        X_batch = X[batch_indices]
        Y_batch = Y[batch_indices]
        return X_batch, Y_batch

    def train(self, X_train, Y_train, X_val, Y_val, regularizer=None, verbose=True):
        """Trains the network.

        Args:
            X_train: A numpy array of shape (N, D) where N is the number of
              training examples and D is the dimensionality of each example.
            Y_train: A numpy array of shape (N, C) where C is the number of
              classes. It represents the labels.
            X_val: A numpy array that represents the validation features.
            Y_val: A numpy array that represents validation labels.
            regularizer: A 'Regularizer' object that regularizes the network.
              Defaults to `None` meaning no regularization.
            verbose: A boolean that controls whether to print out relevant
              information after every epoch. Defaults to `True`.
        """
        import time
        self._regularizer = regularizer
        iter_per_epoch = self._iter_per_epoch or int(X_train.shape[0] / self._batch_size)
        start_time = datetime.datetime.today()
        # Average loss per example.
        loss = 0
        # An approximation for the model's accuracy on the training dataset.
        train_acc = 0
        # An approximation for the model's accuracy on the validation dataset.
        val_acc = 0
        for i in range(self._num_epochs):
            for j in range(iter_per_epoch):
                # time.sleep(.5)
                train_batch = self._get_batch(X_train, Y_train)
                loss = (loss + self._net.forward(train_batch)) / self._batch_size
                # We divide by two to average the sum above.
                loss /= 2
                if regularizer and regularizer.lambda_ != 0:
                    weights_list = self._net.get_layer_weights()
                    regularizer_loss = regularizer.forward(weights_list)
                    loss += regularizer_loss
                self._net.backward()
                self._update_net_parameters()
                if self._regularizer and self._regularizer.lambda_ != 0:
                    self._update_net_regularizer()

                # We compute the accuracy for this iteration by averaging the
                # previous iteration accuracy and the current iteration
                # accuracy. The computed accuracy is equivalent to an
                # exponentially decaying weighted average where the size.
                val_batch = self._get_batch(X_val, Y_val)
                if j == 0:
                    train_acc = self.get_batch_accuracy(train_batch)
                    val_acc = self.get_batch_accuracy(val_batch)
                train_acc = (train_acc + self.get_batch_accuracy(train_batch)) / 2
                val_acc = (val_acc + self.get_batch_accuracy(val_batch)) / 2
                if verbose:
                    p = np.round(((j + 1) / iter_per_epoch) * 100)
                    train_acc = np.round(train_acc, 4)
                    val_acc = np.round(val_acc, 4)
                    loss = np.round(loss, 4)
                    s = f'epoch={i + 1}, %={p} loss={loss}, train_acc={train_acc}, val_acc={val_acc}'
                    if j != iter_per_epoch - 1:
                        print(s, end='\r', flush=True)
                    else:
                        print(s, flush=True)
            self._decay_learning_rate()

        trainer_data = {
            'start_time': str(start_time),
            'end_time': str(datetime.datetime.today()),
            'num_epochs': self._num_epochs,
            'learning_rate': self._learning_rate,
            'lr_decay': self._lr_decay,
            'regularizer': str(self._regularizer),
            'batch_size': self._batch_size,
            'train_accuracy(%)': train_acc,
            'val_accuracy(%)': val_acc,
        }
        return trainer_data

    def get_batch_accuracy(self, batch):
        X, Y = batch
        correct = 0
        Y_pred = self._net.predict(X)
        correct += np.count_nonzero(Y.argmax(axis=1) == Y_pred.argmax(axis=1))
        acc = correct / X.shape[0]
        return acc

    def get_data_accuracy(self, data):
        X, Y = data
        correct = 0
        Y_pred = self._net.predict(X)
        correct += np.count_nonzero(Y.argmax(axis=1) == Y_pred.argmax(axis=1))
        acc = correct / X.shape[0]
        return acc

    def _decay_learning_rate(self):
        self._learning_rate *= self._lr_decay

    def _update_net_regularizer(self):
        gradients = self._regularizer.backward()
        layers = self._net.layers_with_internal_params()
        assert len(gradients) == len(layers)
        eta = self._learning_rate / self._batch_size
        for i in range(len(layers)):
            layers[i].weights += -(eta * gradients[i])

    @abc.abstractmethod
    def _update_net_parameters(self):
        pass
