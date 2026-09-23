"""Метрики для оценки уже обученных моделей."""

import numpy as np


def mean_squared_error(y_true, y_pred):
  """Вернуть средний квадрат разности между ответами и предсказаниями."""
  return float(np.mean(np.square(np.asarray(y_pred) - np.asarray(y_true))))


def mean_absolute_error(y_true, y_pred):
  """Вернуть среднюю абсолютную разность между ответами и предсказаниями."""
  return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def root_mean_squared_error(y_true, y_pred):
  """Вернуть корень из MSE в исходных единицах целевого значения."""
  return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred):
  """Вернуть коэффициент детерминации относительно среднего ``y_true``.

  Значение 0 соответствует постоянному прогнозу средним целевым значением.
  Если все истинные значения одинаковы, коэффициент не определён.
  """
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  residual_sum = np.sum(np.square(y_true - y_pred))
  total_sum = np.sum(np.square(y_true - np.mean(y_true)))
  if total_sum == 0:
    raise ValueError("R² не определён при одинаковых целевых значениях")
  return float(1.0 - residual_sum / total_sum)
