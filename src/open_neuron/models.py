"""Сборка слоёв в сеть и цикл её обучения."""

import numpy as np

from open_neuron.activations import sigmoid
from open_neuron.layers import Layer
from open_neuron.losses import CE, MAE, MSE
from open_neuron.optimizers import SGD
from open_neuron.units import Unit


class NeuralNetwork:
    """Последовательно соединённая сеть с обучением мини-пакетами."""

    def __init__(self, layers, inputs_dim, loss="mse", optimizer=None):
        if not layers:
            raise ValueError("Сеть должна содержать хотя бы один слой")

        Unit.count = 0
        Layer.count = 0
        self.layers = list(layers)
        dense_layers = [layer for layer in self.layers if isinstance(layer, Layer)]
        if not dense_layers:
            raise ValueError("Сеть должна содержать хотя бы один полносвязный слой")

        self.last_layer = dense_layers[-1]
        self.inputs_dim = inputs_dim
        self.loss_function = self.init_loss_function(loss)
        self.optimizer = optimizer if optimizer is not None else SGD(momentum=0)
        self.batch_size = None
        self.final_batch_size = None
        self.epochs = None
        self.X_len = None
        self.initialize(inputs_dim)

    @staticmethod
    def init_loss_function(loss_name):
        """Создать одну из поддержанных функций потерь по её имени."""
        losses = {
            "mae": MAE,
            "mse": MSE,
            "categorical_crossentropy": CE,
        }
        try:
            return losses[loss_name]()
        except KeyError as error:
            raise ValueError("Unsupported loss function") from error

    def initialize(self, inputs_dim):
        """Инициализировать слои слева направо, передавая размер выхода."""
        input_size = inputs_dim
        for layer in self.layers:
            input_size = layer.initialize(input_size)

        output_activation = self.last_layer.activation
        is_softmax = getattr(output_activation, "__name__", None) == "softmax"
        self.type = (
            '"Classification"'
            if output_activation is sigmoid or is_softmax
            else '"Regression"'
        )

    def forward(self, X, training=True):
        """Передать вход через все слои и вернуть выход сети."""
        values = X
        for layer in self.layers:
            values = layer.forward(values, training=training)
        return values

    def backward(self, error):
        """Передать остаток через функцию потерь и слои в обратном порядке."""
        delta = self.loss_function.loss_derivative(error)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update(self):
        """Обновить параметры слоёв в обратном порядке сети."""
        for layer in reversed(self.layers):
            layer.update(self.optimizer)

    def fit(
        self,
        X,
        y,
        epochs,
        batch_size=1,
        final_batch_size=None,
        shuffle=True,
        validation_data=None,
    ):
        """Обучить сеть, показывая среднюю потерю по эпохам.

        Если ``validation_data`` не переданы, потери в конце эпохи считаются
        на обучающей выборке. ``batch_size=None`` означает один пакет на всю
        выборку; ``final_batch_size`` постепенно меняет размер пакета к концу
        обучения.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.epochs = epochs
        self.batch_size = batch_size
        self.final_batch_size = final_batch_size
        self.X_len = X.shape[0]

        if validation_data is not None:
            X_validation, y_validation = map(np.asarray, validation_data)
            validation_name = "Test Data"
        else:
            X_validation, y_validation = X, y
            validation_name = "Train Data"

        validation_loss = self.loss_function.evaluate_loss(
            y_validation, self.predict(X_validation)
        )
        print(
            f"Training started with Overall Loss {validation_loss:.4f} "
            f"on {validation_name}"
        )

        current_batch_size = batch_size
        report_interval = max(1, epochs // 10)

        for epoch in range(epochs):
            if batch_size is not None and final_batch_size is not None:
                current_batch_size = batch_size + int(
                    (final_batch_size - batch_size) * (epoch + 1) / epochs
                )

            if shuffle:
                permutation = np.random.permutation(self.X_len)
                X_epoch = X[permutation]
                y_epoch = y[permutation]
            else:
                X_epoch, y_epoch = X, y

            step_size = current_batch_size or self.X_len
            total_epoch_loss = 0.0

            for start in range(0, self.X_len, step_size):
                X_batch = X_epoch[start : start + step_size]
                y_batch = y_epoch[start : start + step_size]
                predictions = self.forward(X_batch)

                error = self.loss_function.evaluate_error(y_batch, predictions)
                batch_loss = self.loss_function.evaluate_loss(y_batch, predictions)
                # Учитываем размер батча, чтобы последний неполный батч
                # не влиял на среднюю потерю так же сильно, как полный.
                total_epoch_loss += batch_loss * len(X_batch)

                self.backward(error)
                self.update()

            epoch_loss = total_epoch_loss / self.X_len
            validation_loss = self.loss_function.evaluate_loss(
                y_validation, self.predict(X_validation)
            )
            should_report = (
                epochs < 10
                or epoch == 0
                or (epoch + 1) % report_interval == 0
                or epoch == epochs - 1
            )
            if should_report:
                end = "\n" if epochs < 10 or epoch == epochs - 1 else "\r"
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
                    f"Validation Loss: {validation_loss:.4f} on {validation_name}",
                    end=end,
                )

        print(f"Training completed with Overall Loss {validation_loss:.4f}")

    def predict(self, X):
        """Вычислить выход сети без применения обучающего поведения Dropout."""
        return self.forward(X, training=False)

    def summary(self):
        """Показать конфигурацию сети и состояние её полносвязных слоёв."""
        print(f"Info:\n{self}")
        print("Last forward:")
        for layer in self.layers:
            if not isinstance(layer, Layer):
                continue

            print(layer)
            neurons = layer.neurons
            if len(neurons) > 4:
                print(neurons[0])
                print(neurons[1])
                print("...")
                print(neurons[-2])
                print(neurons[-1])
            else:
                for neuron in neurons:
                    print(neuron)

    def __str__(self):
        batch_size = str(self.batch_size or self.X_len)
        if self.final_batch_size is not None:
            batch_size += f"-{self.final_batch_size}"

        return (
            f"Network type: {self.type}, layers: {self.last_layer.number}, "
            f"neurons: {self.last_layer.neurons[-1].number}, "
            f"optimizer: {self.optimizer}, epochs: {self.epochs}, "
            f"batch size: {batch_size}, loss function: {self.loss_function}"
        )
