"""Оптимизаторы для обновления параметров полносвязного слоя."""

import numpy as np


class Optimizer:
  """Общий интерфейс оптимизатора параметров слоя."""

  @staticmethod
  def _validate_clip_norm(clip_norm):
    if clip_norm is not None and (
        not isinstance(clip_norm, (int, float, np.integer, np.floating))
        or not np.isfinite(clip_norm)
        or clip_norm <= 0
    ):
      raise ValueError("clip_norm должен быть положительным конечным числом")
    return clip_norm

  @staticmethod
  def _clip_gradient(gradient, clip_norm):
    """Ограничить норму градиента, если задан порог."""
    if clip_norm is None:
      return gradient
    norm = np.linalg.norm(gradient)
    if norm > clip_norm:
      return gradient * (clip_norm / (norm + 1e-12))
    return gradient

  def update(self, layer):
    raise NotImplementedError("Оптимизатор должен реализовать обновление слоя")


class SGD(Optimizer):
  """Стохастический градиентный спуск с импульсом.

  Скорости весов и смещений хранятся отдельно для каждого слоя, чтобы один
  экземпляр оптимизатора можно было использовать для всей сети.
  """

  def __init__(self, learning_rate=0.1, momentum=0, clip_norm=None):
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.clip_norm = self._validate_clip_norm(clip_norm)
    self.m = {"weights": {}, "bias": {}}

  def update(self, layer):
    """Обновить веса и смещения слоя по сохранённым градиентам."""
    self.m["weights"].setdefault(layer.id, np.zeros_like(layer.weights))
    self.m["bias"].setdefault(layer.id, np.zeros_like(layer.bias))

    weight_gradient = self._clip_gradient(
        layer.gradients["weights"], self.clip_norm
    )
    weight_update = (
        self.learning_rate * weight_gradient
        + self.momentum * self.m["weights"][layer.id]
    )
    layer.weights -= weight_update
    self.m["weights"][layer.id] = weight_update

    bias_gradient = self._clip_gradient(
        layer.gradients["bias"], self.clip_norm
    )
    bias_update = (
        self.learning_rate * bias_gradient
        + self.momentum * self.m["bias"][layer.id]
    )
    has_bias = np.where(layer.bias != None)
    layer.bias[has_bias] -= bias_update[has_bias]
    self.m["bias"][layer.id] = bias_update

  def __str__(self):
    return '"SGD"'


class RMSprop(Optimizer):
  """Адаптивный оптимизатор со скользящим средним квадратов градиентов.

  Для каждого слоя и каждого параметра хранится среднее ``v``. Большие
  повторяющиеся градиенты увеличивают знаменатель шага, поэтому обновления
  становятся меньше. Это снижает чувствительность к масштабу градиентов, но
  не отменяет необходимость выбрать разумный ``learning_rate``.
  """

  def __init__(self, learning_rate=0.001, rho=0.9, clip_norm=None):
    self.learning_rate = learning_rate
    self.rho = rho
    self.clip_norm = self._validate_clip_norm(clip_norm)
    self.epsilon = 1e-8
    self.v = {"weights": {}, "bias": {}}

  def update(self, layer):
    """Обновить параметры слоя по адаптированным градиентам RMSprop."""
    for parameter in ("weights", "bias"):
      values = getattr(layer, parameter)
      gradient = self._clip_gradient(
          layer.gradients[parameter], self.clip_norm
      )
      self.v[parameter].setdefault(
          layer.id, np.zeros_like(values)
      )
      self.v[parameter][layer.id] = (
          self.rho * self.v[parameter][layer.id]
          + (1 - self.rho) * np.square(gradient)
      )
      update = self.learning_rate * gradient / (
          np.sqrt(self.v[parameter][layer.id]) + self.epsilon
      )

      if parameter == "bias":
        has_bias = np.where(values != None)
        values[has_bias] -= update[has_bias]
      else:
        values -= update

  def __str__(self):
    return '"RMSprop"'


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
      clip_norm=None,
  ):
    self.m = {"weights": {}, "bias": {}}
    self.v = {"weights": {}, "bias": {}}
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = 1e-8
    self.t = {}
    self.learning_rate = learning_rate
    self.clip_norm = self._validate_clip_norm(clip_norm)

  def update(self, layer):
    """Обновить параметры слоя, сохранив его собственные моменты Adam."""
    for state in (self.m, self.v):
      state["weights"].setdefault(layer.id, np.zeros_like(layer.weights))
      state["bias"].setdefault(layer.id, np.zeros_like(layer.bias))

    self.t[layer.id] = self.t.get(layer.id, 0) + 1
    timestep = self.t[layer.id]
    for parameter in ("weights", "bias"):
      gradient = self._clip_gradient(
          layer.gradients[parameter], self.clip_norm
      )
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
