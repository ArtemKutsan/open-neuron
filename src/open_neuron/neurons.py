"""Самостоятельный обучаемый нейрон вне матричного слоя."""

import numpy as np

from open_neuron.activations import linear
from open_neuron.initializers import HeUniform
from open_neuron.losses import MSE
from open_neuron.optimizers import SGD


# Исторический черновик Neuron из прежнего units.py. Сохранён целиком как комментарий.
# Ошибки исходного варианта: строковые значения весов и смещений не преобразовывались в инициализаторы;
# `self.bias or 0` неоднозначно для NumPy-массивов; встроенный в слой нейрон читал Z/A родительского слоя.
# В standalone-режиме Z и A вычислялись свойствами, а forward сохранял вход и возвращал A.
# Исправление для смещения: `self.bias if self.bias is not None else 0`.
# Старый вариант класса для нейронов
# Класс для нейрона
# class Neuron:
#     count = 0
#
#     def __init__(self, weights='he', bias='he', input_size=2, activation=linear):
#         Neuron.count += 1
#         self.number = Neuron.count  # порядковый номер объекта нейрон
#         self.name = self.__class__.__name__  # имя класса объекта
#         self.id = id(self)  # уникальный id объекта
#         self.layer = None  # ссылка на слой в котором находится нейрон (устанавливается слоем если нейрон находится в слое)
#         self.index = None  # индекс нейрона в слое
#         self.weights = weights
#         self.bias = bias
#         self.activation = activation or linear  # функция активации нейрона
#
#     @property
#     # Массив значений inputs (x) батча (X)
#     def Inputs(self):
#         return self.layer.Inputs if self.layer is not None else self.X
#
#     @Inputs.setter
#     def Inputs(self, X):
#         self.X = X
#
#     @property
#     # Значение inputs последнего прошедшего через нейрон объекта (в расчётах не используется, просто для вывода информации в методе __str__)
#     def inputs(self):
#         return self.Inputs[-1] if self.Inputs is not None else None
#
#     @property
#     # Массив значений z батча (z каждого x в батче X)
#     def Z(self):
#         if self.layer is not None:
#             return self.layer.Z.T[self.index]
#         return (np.dot(self.Inputs, self.weights) + (self.bias or 0)) if self.Inputs is not None else None
#
#     @property
#     # Значение z последнего прошедшего через нейрон объекта (в расчётах не используется, просто для вывода информации в методе __str__)
#     def z(self):
#         return self.Z.T[-1].astype(float) if self.Z is not None else None
#
#     @property
#     # Массив значений a батча (a каждого x в батче X)
#     def A(self):
#         if self.layer is not None:
#             return self.layer.A.T[self.index]
#         return self.activation(self.Z) if self.Z is not None else None
#
#     @property
#     # Значение a последнего прошедшего через нейрон объекта (в расчётах не используется, просто для вывода информации в методе __str__)
#     def a(self):
#         return self.A.T[-1].astype(float) if self.A is not None else None
#
#     def activation_derivative(self, Z):
#         return self.activation(Z, derivative=True)
#
#     # Метод Forward pass (прогон данных через нейрон)
#     def forward(self, X, training=True):
#         self.Inputs = X
#         # self.Z = np.dot(self.Inputs, self.weights) + (self.bias if self.bias is not None else 0)
#         # self.A = self.activation(self.Z)
#         return self.A
#
#     def call(self, X, training=False):
#         """ # Метод call вызывается после выполнения встроенного метода __call__.
#         Служит для дополнения/изменения работы встроенного метода __call__ не нарушая необходимый для правильной работы процесс
#         вычисления данных при вызове объекта. На вход получает входные данные (массив X). По умолчанию возвращает значение(я)
#         активированного выхода нейрона (массив A).
#         """
#         # Любой дополнительный код
#         return self.A
#
#     def __call__(self, X, training=False):
#         self.forward(X, training)
#         return self.call(X, training)
#
#     def __str__(self):
#         decimals = 4
#         return f'{self.name} {self.number}, inputs: {format(self.inputs, decimals=decimals)}, ' \
#                f'weights: {format(self.weights.flatten(), decimals=decimals)}, bias: {format(self.bias.item(), decimals=decimals)}, ' \
#                f'z: {format(self.z, decimals=decimals)}, activation: {self.activation.__name__}, a: {format(self.a, decimals=decimals)}'

class Neuron:
    """Один нейрон, который сам хранит параметры и выполняет обучение.

    В отличие от ``Unit``, связанного с ``Layer``, этот объект самостоятельно
    вычисляет выход, градиенты и обновляет собственные веса и смещение.
    """

    count = 0

    def __init__(
        self,
        weights=None,
        bias=None,
        input_size=2,
        output_size=1,
        activation=linear,
        loss=None,
        optimizer=None,
    ):
        Neuron.count += 1
        self.number = Neuron.count
        self.name = self.__class__.__name__
        self.id = id(self)
        self.Inputs = None
        self.Z = None
        self.A = None
        self.activation = activation if activation is not None else linear
        self.loss_function = loss if loss is not None else MSE()
        self.optimizer = optimizer if optimizer is not None else SGD(momentum=0)
        self.delta = None
        self.gradient = {"weights": None, "bias": None}
        self.epochs = None
        self.batch_size = None
        self.final_batch_size = None
        self.X_len = None

        weights = weights if weights is not None else HeUniform()
        bias = bias if bias is not None else HeUniform()
        self.weights = self._initialize_parameter(
            weights, (input_size, output_size), "weights"
        )
        self.bias = self._initialize_bias(bias, output_size)

    @staticmethod
    def _initialize_parameter(value, shape, name):
        """Вызвать инициализатор или привести вручную заданные веса к матрице."""
        if callable(value):
            return np.asarray(value(shape=shape), dtype=float)

        values = np.asarray(value, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape != shape:
            raise ValueError(f"Форма {name} должна быть {shape}, получено {values.shape}")
        return values.copy()

    @staticmethod
    def _initialize_bias(value, output_size):
        """Создать строку смещений из инициализатора или заданных значений."""
        if callable(value):
            return np.asarray(value(shape=(1, output_size)), dtype=float)

        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            return np.full((1, output_size), values.item(), dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape != (1, output_size):
            raise ValueError(
                f"Форма bias должна быть {(1, output_size)}, получено {values.shape}"
            )
        return values.copy()

    @property
    def inputs(self):
        """Последний входной пример, переданный нейрону."""
        return self.Inputs[-1] if self.Inputs is not None else None

    @property
    def z(self):
        """Последнее значение до функции активации."""
        return self._last_value(self.Z)

    @property
    def a(self):
        """Последний выход нейрона после функции активации."""
        return self._last_value(self.A)

    @staticmethod
    def _last_value(values):
        if values is None:
            return None
        result = np.asarray(values)[-1]
        return result.item() if result.size == 1 else result

    def activation_derivative(self, Z):
        """Вычислить производную активации по значениям ``Z``."""
        return self.activation(Z, derivative=True)

    def forward(self, X, training=True):
        """Вычислить ``Z = X @ W + b`` и активированный выход ``A``."""
        self.Inputs = np.asarray(X)
        self.Z = self.Inputs @ self.weights + self.bias
        self.A = self.activation(self.Z)
        return self.A

    def backward(self, error):
        """Вычислить градиенты весов и смещения по пакету примеров."""
        self.delta = self.loss_function.loss_derivative(error) * (
            self.activation_derivative(self.Z)
        )
        self.gradient["weights"] = self.Inputs.T @ self.delta
        self.gradient["bias"] = np.sum(self.delta, axis=0, keepdims=True)

    def update(self):
        """Поручить оптимизатору обновить параметры нейрона."""
        self.optimizer.update(self)

    def fit(
        self,
        X,
        y,
        epochs=10,
        batch_size=1,
        final_batch_size=None,
        shuffle=True,
        validation_data=None,
    ):
        """Обучить нейрон пакетами и показать потери по эпохам."""
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
        """Вычислить выход нейрона без обновления параметров."""
        return self.forward(X, training=False)

    def call(self, X, training=False):
        """Вернуть выход после прямого прохода, сохраняя интерфейс объекта."""
        return self.A

    def __call__(self, X, training=False):
        self.forward(X, training=training)
        return self.call(X, training=training)

    def __str__(self):
        return (
            f"{self.name} {self.number}, inputs: {self.inputs}, "
            f"weights: {np.array2string(self.weights.ravel(), precision=4)}, "
            f"bias: {np.array2string(self.bias.ravel(), precision=4)}, "
            f"z: {self.z}, activation: {self.activation.__name__}, a: {self.a}"
        )
