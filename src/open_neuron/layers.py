"""Полносвязный слой с матричным прямым проходом."""

from collections.abc import Callable

import numpy as np

from open_neuron.activations import linear
from open_neuron.initializers import GlorotUniform
from open_neuron.units import Unit


class Layer:
    """Слой, который считает выходы всех нейронов одной матричной операцией.

    ``Layer`` принимает число нейронов и сам создаёт параметры либо список
    ``Unit`` с заданными вручную весами и смещениями. Для пакета ``X`` формы
    ``(batch_size, input_size)`` его веса имеют форму
    ``(input_size, output_size)``, а смещения — ``(1, output_size)``.

    Прямой проход вычисляет ``Z = X @ W + b`` и затем ``A = activation(Z)``.
    Обратный проход получает градиент по ``A``, применяет производную активации
    и вычисляет градиенты по входу, весам и смещениям. Объекты ``Unit`` дают
    доступ к параметрам отдельных нейронов, но расчёт выполняет слой сразу
    для всей матрицы.
    """

    count = 0

    def __init__(self, neurons, activation=linear, kernel=GlorotUniform()):
        Layer.count += 1
        self.number = Layer.count
        self.name = self.__class__.__name__
        self.id = id(self)
        self.output_size = None
        self.Inputs = None
        self.kernel = kernel if kernel is not None else GlorotUniform()
        self.weights = None
        self.bias = None
        self.Z = None
        self.activation = activation if activation is not None else linear
        self.A = None
        self.delta = None
        self.gradient = {"weights": None, "bias": None}
        self.neurons = neurons
        self.num_neurons = None

    def initialize(self, input_size):
        """Создать матрицы параметров и связать с ними объекты ``Unit``."""
        self.input_size = input_size

        if isinstance(self.neurons, int):
            # Каждый столбец весов соответствует одному нейрону слоя.
            self.output_size = self.neurons
            self.weights = self.kernel(shape=(self.input_size, self.output_size))
            self.bias = self.kernel(shape=(1, self.output_size))
            self.neurons = [Unit() for _ in range(self.neurons)]
        else:
            self.output_size = len(self.neurons)
            self.weights = np.array([neuron.weights for neuron in self.neurons]).T
            self.bias = np.array([neuron.bias for neuron in self.neurons]).reshape(1, -1)

        self.num_neurons = self.output_size

        # Каждый Unit получает представление своего столбца, а слой продолжает
        # обновлять общие матрицы. При замене матриц эти представления устареют.
        # Столбцы весов и смещений остаются общими с массивами параметров слоя.
        for index, (neuron, weights, bias) in enumerate(
            zip(self.neurons, self.weights.T, self.bias.T)
        ):
            neuron.weights = weights
            neuron.bias = bias
            neuron.index = index
            neuron.layer = self
            # Softmax связывает выходы слоя между собой, поэтому Unit показывает logits.
            unit_activation = (
                linear
                if getattr(self.activation, "__name__", None) == "softmax"
                else self.activation
            )
            neuron.activation = unit_activation

            # Индивидуальные активации потребовали бы отдельной производной
            # для каждого Unit; здесь слой применяет одну функцию ко всей матрице.

        return self.output_size

    def forward(self, Inputs, training=True):
        """Вычислить выход слоя для всего пакета входных примеров."""
        self.Inputs = np.asarray(Inputs)

        # Смещение применяется ко всем строкам пакета; None трактуется как ноль.
        biases = np.where(self.bias == None, 0, self.bias).astype(float)
        self.Z = self.Inputs @ self.weights + biases
        self.A = self.activation(self.Z)

        return self.A

    def activation_derivative(self, Z):
        """Вычислить производную функции активации для значений ``Z``."""
        return self.activation(Z, derivative=True)

    def backward(self, delta):
        """Передать градиент через слой и сохранить градиенты параметров.

        ``delta`` имеет форму ``(batch_size, output_size)``. Для обычных
        поэлементных активаций это градиент по выходу ``A``. Для сочетания
        categorical cross-entropy и softmax это уже fused-градиент по ``Z``;
        softmax передаёт его без дополнительного якобиана. Слой вычисляет:

        - для поэлементной активации ``dL/dZ = delta * activation'(Z)``;
        - для fused softmax и categorical cross-entropy ``delta`` уже равен ``dL/dZ``;
        - ``dL/dW = X.T @ dL/dZ``;
        - ``dL/db = sum(dL/dZ, axis=0)``;
        - возвращает ``dL/dX = dL/dZ @ W.T``.

        Масштабирование по размеру пакета здесь не выполняется: входящий
        градиент уже задаёт масштаб, как и в исходной реализации.
        """
        derivatives = self.activation_derivative(self.Z)
        self.delta = delta * derivatives

        self.gradient["weights"] = self.Inputs.T @ self.delta
        self.gradient["bias"] = np.sum(self.delta, axis=0, keepdims=True)

        return self.delta @ self.weights.T

    def update(self, optimizer):
        """Поручить оптимизатору обновить параметры этого слоя."""
        optimizer.update(self)

    def __str__(self):
        return (
            f"{self.name} {self.number}, neurons: {len(self.neurons)}, "
            f"inputs: {self.Inputs[-1]}, activation: {self.activation.__name__}, "
            f"output (a): {self.A[-1]}"
        )


class Dropout:
    """Случайно отключать долю входных значений во время обучения.

    При обучении слой сохраняет маску и масштабирует оставшиеся значения,
    чтобы их ожидаемый масштаб не менялся. При предсказании вход проходит
    без изменений. У слоя нет обучаемых параметров.
    """

    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def initialize(self, input_size):
        """Вернуть размер входа без изменений."""
        return input_size

    def forward(self, Inputs, training=True):
        """Применить случайную маску при обучении или пропустить вход."""
        if training:
            self.mask = np.random.binomial(
                1, 1 - self.dropout_rate, size=Inputs.shape
            )
            return Inputs * self.mask / (1 - self.dropout_rate)
        return Inputs

    def backward(self, delta):
        """Передать градиент только через значения, оставленные маской."""
        return delta * self.mask / (1 - self.dropout_rate)

    def update(self, optimizer):
        """Ничего не обновлять: Dropout не содержит параметров."""
