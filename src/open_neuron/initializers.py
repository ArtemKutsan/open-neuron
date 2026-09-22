"""Создание начальных массивов весов и смещений нейронной сети.

Каждый инициализатор вызывается с формой массива. Для матрицы весов слоя
форма имеет вид ``(число входов, число нейронов)``.
"""

import numpy as np


def _sample_truncated_normal(shape, mean, stddev):
  values = np.random.normal(loc=mean, scale=stddev, size=shape)
  rejected = np.abs(values - mean) > 2 * stddev
  while np.any(rejected):
    values[rejected] = np.random.normal(
        loc=mean, scale=stddev, size=np.count_nonzero(rejected)
    )
    rejected = np.abs(values - mean) > 2 * stddev
  return values


class Initializer:
  """Общий интерфейс callable-инициализатора."""

  def __call__(self, shape):
    raise NotImplementedError("Initializer must implement this method")


class GlorotNormal(Initializer):
  """Инициализировать веса нормальным распределением Glorot."""

  def __call__(self, shape):
    fan_in, fan_out = shape
    stddev = np.sqrt(2 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * stddev


class GlorotUniform(Initializer):
  """Инициализировать веса равномерным распределением Glorot."""

  def __call__(self, shape):
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)


class HeNormal(Initializer):
  """Инициализировать веса нормальным распределением He."""

  def __call__(self, shape):
    fan_in, fan_out = shape
    stddev = np.sqrt(2 / fan_in)
    return np.random.randn(*shape) * stddev


class HeUniform(Initializer):
  """Инициализировать веса равномерным распределением He."""

  def __call__(self, shape):
    fan_in, fan_out = shape
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)


class LecunNormal(Initializer):
  """Инициализировать веса нормальным распределением LeCun."""

  def __call__(self, shape):
    fan_in, fan_out = shape
    stddev = np.sqrt(1 / fan_in)
    return np.random.randn(*shape) * stddev


class LecunUniform(Initializer):
  """Инициализировать веса равномерным распределением LeCun."""

  def __call__(self, shape):
    fan_in, _ = shape
    limit = np.sqrt(3 / fan_in)
    return np.random.uniform(-limit, limit, size=shape)


class RandomNormal(Initializer):
  """Создать массив из нормального распределения с заданными параметрами."""

  def __init__(self, mean=0.0, stddev=0.05):
    self.mean = mean
    self.stddev = stddev

  def __call__(self, shape):
    return np.random.normal(loc=self.mean, scale=self.stddev, size=shape)


class RandomUniform(Initializer):
  """Создать массив из равномерного распределения на заданном интервале."""

  def __init__(self, minval=-0.05, maxval=0.05):
    self.minval = minval
    self.maxval = maxval

  def __call__(self, shape):
    return np.random.uniform(self.minval, self.maxval, size=shape)


class TruncatedNormal(Initializer):
  """Создать нормальные значения, ограниченные двумя стандартными отклонениями."""

  def __init__(self, mean=0.0, stddev=0.05):
    self.mean = mean
    self.stddev = stddev

  def __call__(self, shape):
    return _sample_truncated_normal(shape, self.mean, self.stddev)


class Constant(Initializer):
  """Заполнить массив одним постоянным значением."""

  def __init__(self, value=0):
    self.value = value

  def __call__(self, shape):
    return np.full(shape, self.value)


class Identity(Initializer):
  """Создать масштабированную единичную матрицу."""

  def __init__(self, gain=1.0):
    self.gain = gain

  def __call__(self, shape):
    if len(shape) != 2 or shape[0] != shape[1]:
      raise ValueError(
          "Identity matrix initializer can only be used for 2D square matrices."
      )
    return self.gain * np.eye(shape[0])


class Orthogonal(Initializer):
  """Инициализировать матрицу ортогональными столбцами или строками."""

  def __init__(self, gain=1.0):
    self.gain = gain

  def __call__(self, shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return self.gain * q


class VarianceScaling(Initializer):
  """Масштабировать случайные значения по fan-in, fan-out или их среднему."""

  def __init__(self, scale=1.0, mode="fan_in", distribution="truncated_normal"):
    self.scale = scale
    self.mode = mode
    self.distribution = distribution

  def __call__(self, shape):
    if self.mode == "fan_in":
      fan = shape[0]
    elif self.mode == "fan_out":
      fan = shape[1]
    elif self.mode == "fan_avg":
      fan = (shape[0] + shape[1]) / 2
    else:
      raise ValueError("Invalid mode: " + self.mode)

    scale = self.scale / max(1.0, fan)
    if self.distribution == "truncated_normal":
      stddev = np.sqrt(scale) / 0.87962566103423978
      return _sample_truncated_normal(shape, 0.0, stddev)
    elif self.distribution == "normal":
      stddev = np.sqrt(scale)
      return np.random.normal(0, stddev, size=shape)
    elif self.distribution == "uniform":
      limit = np.sqrt(3.0 * scale)
      return np.random.uniform(-limit, limit, size=shape)
    else:
      raise ValueError("Invalid distribution: " + self.distribution)
