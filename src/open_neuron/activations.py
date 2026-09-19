"""Функции активации для слоёв полносвязной сети.

``sigmoid``, ``tanh``, ``relu`` и ``linear`` применяются поэлементно. ``softmax``
нормализует значения вдоль последней оси, связывая выходы классов одного
примера. Для поэлементных активаций ``derivative=True`` возвращает производную
по ``Z``. Для softmax единицы обозначают только соглашение с уже объединённым
градиентом categorical cross-entropy; это не общая производная softmax.

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


def softmax(values, *, derivative=False):
    """Преобразовать logits в вероятности вдоль последней оси.

    Для пакета формы ``(batch_size, class_count)`` каждая строка независимо
    преобразуется в распределение вероятностей. Вычитание максимального logit
    сохраняет результат и предотвращает переполнение экспоненты.

    Если ``derivative=True``, возвращаются единицы только для совместимости с
    fused-градиентом categorical cross-entropy: ``dL/dZ = probabilities - y``.
    Это соглашение не является производной softmax для произвольного входящего
    градиента; общий градиент softmax требует якобиан-векторного произведения.
    """
    values = np.asarray(values)

    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("Softmax требует непустой оси классов")

    if derivative:
        return np.ones_like(values, dtype=np.result_type(values.dtype, np.float64))

    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


# Исторический вариант реестра активаций. Восстановлен как комментарий.
# В исходнике неизвестное имя возвращало строку; здесь показана корректная ошибка.
# Остальные имена оставлены как исторический перечень: часть функций не входит в текущий API.
#
# Словарь с функциями активации
# activation_functions = {
#     'sigmoid': sigmoid,
#     'tanh': tanh,
#     'relu': relu,
#     'leaky_relu': leaky_relu,
#     'prelu': prelu,
#     'gelu': gelu,
#     'elu': elu,
#     'selu': selu,
#     'swish': swish,
#     'softplus': softplus,
#     'softmax': softmax,
#     'linear': linear,
#     'heaviside': heaviside,
# }
#
# Функция для получения активационной функции по имени
# def get(activation_name):
#     if activation_name is None:
#         return linear
#     raise ValueError(f"Unknown activation function: {activation_name}")
