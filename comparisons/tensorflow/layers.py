"""Пользовательский Dense-слой Keras для учебного сравнения."""

import numpy as np
import tensorflow as tf

from .units import Neuron


class CustomDense(tf.keras.layers.Layer):
    """Вычисляет ``activation(inputs @ kernel + bias)``.

    ``neurons`` можно задать числом, чтобы создать параметры слоя с
    инициализатором Keras, или последовательностью ``Neuron`` с начальными
    весами. Для последовательности смещения по умолчанию создаются только
    если хотя бы один нейрон получил смещение.
    """

    def __init__(self, neurons, activation=None, use_bias=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

        if isinstance(neurons, int):
            if neurons <= 0:
                raise ValueError("Число нейронов должно быть положительным")
            self.units = neurons
            self.neurons = None
            self.use_bias = True if use_bias is None else bool(use_bias)
        else:
            self.neurons = list(neurons)
            if not self.neurons:
                raise ValueError("Слой должен содержать хотя бы один нейрон")
            if not all(isinstance(neuron, Neuron) for neuron in self.neurons):
                raise TypeError("Список слоя должен содержать объекты Neuron")
            self.units = len(self.neurons)
            has_initial_bias = any(neuron.init_bias is not None for neuron in self.neurons)
            self.use_bias = (
                has_initial_bias if use_bias is None else bool(use_bias)
            )

        self.Inputs = None
        self.Z = None
        self.A = None

    def build(self, input_shape):
        input_size = input_shape[-1]
        if input_size is None:
            raise ValueError("Размер последней оси входа должен быть известен")

        self.kernel = self.add_weight(
            name="kernel",
            shape=(int(input_size), self.units),
            initializer="he_uniform",
            trainable=True,
        )
        self.bias = (
            self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )
            if self.use_bias
            else None
        )

        if self.neurons is None:
            self.neurons = [
                Neuron(weights=np.zeros(int(input_size), dtype=np.float32))
                for _ in range(self.units)
            ]
        else:
            initial_weights = np.column_stack(
                [neuron.init_weights for neuron in self.neurons]
            )
            if initial_weights.shape != (int(input_size), self.units):
                raise ValueError(
                    "Каждый нейрон должен иметь число начальных весов, "
                    "равное размеру входа слоя"
                )
            self.kernel.assign(initial_weights)

            if self.bias is not None:
                initial_bias = np.array(
                    [
                        0.0 if neuron.init_bias is None else neuron.init_bias.item()
                        for neuron in self.neurons
                    ],
                    dtype=np.float32,
                )
                self.bias.assign(initial_bias)

        for index, neuron in enumerate(self.neurons):
            neuron.index = index
            neuron.layer = self
            neuron.activation = (
                None
                if self.activation is tf.keras.activations.softmax
                else self.activation
            )

        super().build(input_shape)

    def call(self, inputs):
        self.Inputs = inputs
        self.Z = tf.linalg.matmul(inputs, self.kernel)
        if self.bias is not None:
            self.Z = self.Z + self.bias
        self.A = self.activation(self.Z)
        return self.A

    def __str__(self):
        activation_name = getattr(self.activation, "__name__", type(self.activation).__name__)
        return f"name: {self.name}, neurons: {self.units}, activation: {activation_name}"
