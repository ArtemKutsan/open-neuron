"""Поэлементные функции активации и их производные.

Функции принимают число или массив ``values``. В полносвязном слое это обычно
результат линейного преобразования ``Z = X @ W + b``. Если передать
``derivative=True``, функция вернёт производную активации по ``Z`` для каждого
элемента. При обратном распространении её умножают на входящий градиент:
``dL/dZ = dL/dA * f'(Z)``.

Звёздочка в сигнатуре делает ``derivative`` именованным параметром: его нужно
передавать как ``sigmoid(values, derivative=True)``.
"""

import numpy as np


def sigmoid(values, *, derivative=False):
    """Вычислить сигмоиду или её поэлементную производную.

    Сигмоида: ``σ(z) = 1 / (1 + exp(-z))``.
    Производная: ``σ'(z) = σ(z) * (1 - σ(z))``.
    """
    values = np.asarray(values)

    if derivative:
        activated = sigmoid(values)
        return activated * (1.0 - activated)

    result = np.empty_like(values, dtype=np.result_type(values.dtype, np.float64))
    non_negative = values >= 0
    result[non_negative] = 1.0 / (1.0 + np.exp(-values[non_negative]))

    # Для отрицательных z используем равносильную форму: так exp(z) не переполняется.
    exp_values = np.exp(values[~non_negative])
    result[~non_negative] = exp_values / (1.0 + exp_values)
    return result


def tanh(values, *, derivative=False):
    """Вычислить гиперболический тангенс или его поэлементную производную.

    ``tanh'(z) = 1 - tanh(z)**2``.
    """
    values = np.asarray(values)
    activated = np.tanh(values)

    if derivative:
        return 1.0 - np.square(activated)

    return activated


def relu(values, *, derivative=False):
    """Вычислить ReLU или её производную.

    ``ReLU(z) = max(0, z)``; производная равна ``1`` при ``z > 0`` и ``0``
    при ``z <= 0``. В нуле производная математически не определена; здесь
    выбрано соглашение считать её равной нулю.
    """
    values = np.asarray(values)

    if derivative:
        return np.where(values > 0, 1.0, 0.0)

    return np.maximum(values, 0.0)


def linear(values, *, derivative=False):
    """Вернуть вход без изменений или массив единиц как производную.

    ``f(z) = z`` и ``f'(z) = 1``. Такая активация оставляет линейный выход
    слоя без ограничения, например для предсказания числового значения.
    """
    values = np.asarray(values)

    if derivative:
        return np.ones_like(values, dtype=float)

    return values
