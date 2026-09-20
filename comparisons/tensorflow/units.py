"""Объект нейрона для просмотра параметров Keras-слоя."""

import numpy as np
import tensorflow as tf


class Neuron:
    """Представляет один выход Dense-слоя и его параметры.

    Пока объект не связан со слоем, он хранит начальные параметры. После
    добавления в ``CustomDense`` текущие веса и смещение принадлежат слою.
    Сам этот класс не обучается и не хранит копию обучаемых параметров.
    """

    def __init__(self, weights, bias=None, activation=None):
        self.index = None
        self.layer = None
        self._inputs = None
        self.init_weights = np.asarray(weights, dtype=np.float32).copy()
        self.init_bias = (
            None if bias is None else np.asarray(bias, dtype=np.float32).copy()
        )
        self.activation = activation

    @staticmethod
    def _as_numpy(value):
        if value is None:
            return None
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    @staticmethod
    def _last_sample(values):
        if values is None:
            return None
        values = np.asarray(values)
        return values[-1] if values.ndim > 1 else values

    @property
    def Inputs(self):
        if self.layer is not None:
            return self._as_numpy(self.layer.Inputs)
        return self._as_numpy(self._inputs)

    @Inputs.setter
    def Inputs(self, value):
        self._inputs = value

    @property
    def inputs(self):
        return self._last_sample(self.Inputs)

    @property
    def weights(self):
        if self.layer is not None:
            return self.layer.kernel[:, self.index]
        return self.init_weights

    @weights.setter
    def weights(self, value):
        values = np.asarray(value, dtype=np.float32)
        if self.layer is None:
            self.init_weights = values.copy()
        else:
            self.layer.kernel[:, self.index].assign(value)

    @property
    def bias(self):
        if self.layer is not None:
            if self.layer.bias is None:
                return None
            return self.layer.bias[self.index]
        return self.init_bias

    @bias.setter
    def bias(self, value):
        if self.layer is None:
            self.init_bias = (
                None if value is None else np.asarray(value, dtype=np.float32).copy()
            )
        elif self.layer.bias is not None:
            self.layer.bias[self.index].assign(value)

    @property
    def Z(self):
        if self.layer is not None:
            if self.layer.Z is None:
                return None
            return self._as_numpy(self.layer.Z[..., self.index])
        if self.Inputs is None:
            return None
        bias = 0 if self.init_bias is None else self.init_bias
        return np.matmul(self.Inputs, self.init_weights) + bias

    @property
    def z(self):
        return self._last_sample(self.Z)

    @property
    def A(self):
        if self.layer is not None:
            if self.layer.A is None:
                return None
            return self._as_numpy(self.layer.A[..., self.index])
        values = self.Z
        if values is None or self.activation is None:
            return values
        return self._as_numpy(self.activation(values))

    @property
    def a(self):
        return self._last_sample(self.A)

    def call(self, inputs):
        """Сохранить вход и вернуть выход соответствующего слоя."""
        self.Inputs = inputs
        if self.layer is not None:
            return self.layer(inputs)[..., self.index]
        values = self.Z
        if values is None or self.activation is None:
            return values
        return self.activation(values)

    def __call__(self, inputs):
        return self.call(inputs)

    def __str__(self):
        activation_name = getattr(self.activation, "__name__", "None")
        bias = self.bias
        bias_text = "None" if bias is None else np.array2string(self._as_numpy(bias))
        return (
            f"Neuron {self.index + 1 if self.index is not None else '?'}: "
            f"inputs={self.inputs}, "
            f"weights={np.array2string(self._as_numpy(self.weights).ravel(), precision=4)}, "
            f"bias={bias_text}, z={self.z}, activation={activation_name}, a={self.a}"
        )
