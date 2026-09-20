"""Оптимизаторы для обновления параметров полносвязного слоя."""

import numpy as np


class Optimizer:
  """Общий интерфейс оптимизатора параметров слоя."""

  def update(self, layer):
    raise NotImplementedError("Оптимизатор должен реализовать обновление слоя")


class SGD(Optimizer):
  """Стохастический градиентный спуск с импульсом.

  Скорости весов и смещений хранятся отдельно для каждого слоя, чтобы один
  экземпляр оптимизатора можно было использовать для всей сети.
  """

  def __init__(self, learning_rate=0.1, momentum=0):
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.m = {"weights": {}, "bias": {}}

  def update(self, layer):
    """Обновить веса и смещения слоя по сохранённым градиентам."""
    self.m["weights"].setdefault(layer.id, np.zeros_like(layer.weights))
    self.m["bias"].setdefault(layer.id, np.zeros_like(layer.bias))

    weight_update = (
        self.learning_rate * layer.gradient["weights"]
        + self.momentum * self.m["weights"][layer.id]
    )
    layer.weights -= weight_update
    self.m["weights"][layer.id] = weight_update

    bias_update = (
        self.learning_rate * layer.gradient["bias"]
        + self.momentum * self.m["bias"][layer.id]
    )
    has_bias = np.where(layer.bias != None)
    layer.bias[has_bias] -= bias_update[has_bias]
    self.m["bias"][layer.id] = bias_update

  def __str__(self):
    return '"SGD"'


class Adam(Optimizer):
  """Адаптивный моментный оптимизатор с поправкой начального смещения.

  Моменты и номер шага учитываются отдельно для каждого слоя. Поэтому при
  последовательном обновлении всех слоёв один пакет обучения не сдвигает
  глобальный счётчик несколько раз.
  """

  def __init__(
      self,
      learning_rate=0.1,
      beta1=0.9,
      beta2=0.999,
  ):
    self.m = {"weights": {}, "bias": {}}
    self.v = {"weights": {}, "bias": {}}
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = 1e-8
    self.t = {}
    self.learning_rate = learning_rate

  def update(self, layer):
    """Обновить параметры слоя, сохранив его собственные моменты Adam."""
    for state in (self.m, self.v):
      state["weights"].setdefault(layer.id, np.zeros_like(layer.weights))
      state["bias"].setdefault(layer.id, np.zeros_like(layer.bias))

    self.t[layer.id] = self.t.get(layer.id, 0) + 1
    timestep = self.t[layer.id]
    for parameter in ("weights", "bias"):
      gradient = layer.gradient[parameter]
      self.m[parameter][layer.id] = (
          self.beta1 * self.m[parameter][layer.id]
          + (1 - self.beta1) * gradient
      )
      self.v[parameter][layer.id] = (
          self.beta2 * self.v[parameter][layer.id]
          + (1 - self.beta2) * np.square(gradient)
      )

      corrected_moment = self.m[parameter][layer.id] / (
          1 - self.beta1**timestep
      )
      corrected_variance = self.v[parameter][layer.id] / (
          1 - self.beta2**timestep
      )
      update = self.learning_rate * corrected_moment / (
          np.sqrt(corrected_variance) + self.epsilon
      )

      values = getattr(layer, parameter)
      if parameter == "bias":
        has_bias = np.where(values != None)
        values[has_bias] -= update[has_bias]
      else:
        values -= update

  def __str__(self):
    return '"Adam"'
