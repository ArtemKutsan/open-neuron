"""Функции активации для слоёв полносвязной сети.

Все активации, кроме ``softmax``, применяются поэлементно. ``softmax``
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


def leaky_relu(values, *, derivative=False, alpha=0.01):
    """Вычислить Leaky ReLU с заданным наклоном отрицательной ветви.

    ``f(z) = z`` при ``z > 0`` и ``alpha * z`` иначе. В точке излома
    ``z = 0`` производная не определена; здесь используется значение ``alpha``.
    """
    values = np.asarray(values)

    if derivative:
        return np.where(values > 0, 1.0, alpha)

    return np.where(values > 0, values, alpha * values)


def prelu(values, *, derivative=False, alpha=0.01):
    """Вычислить PReLU с фиксированным, задаваемым вызывающим кодом ``alpha``.

    ``alpha`` здесь не обучается: функция возвращает только значение и
    производную по входу. В точке ``z = 0`` используется наклон ``alpha``.
    """
    return leaky_relu(values, derivative=derivative, alpha=alpha)


def gelu(values, *, derivative=False):
    """Вычислить GELU tanh-аппроксимацией и её производную.

    Используется ``0.5 * z * (1 + tanh(sqrt(2/pi) *
    (z + 0.044715 * z**3)))``. Производная вычисляется для этой же
    аппроксимации. Ограничение аргумента tanh предотвращает переполнение
    кубического члена на очень больших конечных входах; за пределами диапазона
    ``[-10, 10]`` tanh уже численно насыщен.
    """
    values = np.asarray(values)
    values = values.astype(np.result_type(values.dtype, np.float64), copy=False)
    safe_values = np.clip(values, -10.0, 10.0)
    scale = np.sqrt(2.0 / np.pi)
    cubic_coefficient = 0.044715
    tanh_value = np.tanh(
        scale * (safe_values + cubic_coefficient * np.power(safe_values, 3))
    )

    if derivative:
        tanh_derivative = scale * (
            1.0 + 3.0 * cubic_coefficient * np.square(safe_values)
        )
        return (
            0.5 * (1.0 + tanh_value)
            + 0.5 * values * (1.0 - np.square(tanh_value)) * tanh_derivative
        )

    return 0.5 * values * (1.0 + tanh_value)


def elu(values, *, derivative=False):
    """Вычислить ELU с ``alpha = 1`` или её производную.

    На отрицательной ветви используется ``exp(z) - 1``; экспонента
    ограничивается сверху нулём, чтобы положительные входы не вызывали
    лишнее переполнение в невыбранной ветви ``numpy.where``.
    """
    values = np.asarray(values)
    negative_exponential = np.exp(np.minimum(values, 0.0))

    if derivative:
        return np.where(values > 0, 1.0, negative_exponential)

    return np.where(values > 0, values, np.expm1(np.minimum(values, 0.0)))


def selu(values, *, derivative=False):
    """Вычислить SELU с рекомендованными константами ``alpha`` и ``scale``.

    В нуле производная не определена; здесь выбирается производная
    отрицательной ветви. Экспонента ограничивается нулём сверху, чтобы
    положительные входы не вызывали переполнение в невыбранной ветви.
    """
    values = np.asarray(values)
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    negative_exponential = np.exp(np.minimum(values, 0.0))

    if derivative:
        return np.where(values > 0, scale, scale * alpha * negative_exponential)

    negative_values = alpha * np.expm1(np.minimum(values, 0.0))
    return scale * np.where(values > 0, values, negative_values)


def swish(values, *, derivative=False):
    """Вычислить Swish, ``z * sigmoid(z)``, или её производную."""
    values = np.asarray(values)
    sigmoid_values = sigmoid(values)

    if derivative:
        return sigmoid_values + values * sigmoid_values * (1.0 - sigmoid_values)

    return values * sigmoid_values


def softplus(values, *, derivative=False):
    """Вычислить устойчивую Softplus или её производную.

    ``log(1 + exp(z))`` вычисляется через ``numpy.logaddexp``, а производная
    равна сигмоиде.
    """
    values = np.asarray(values)

    if derivative:
        return sigmoid(values)

    return np.logaddexp(0.0, values)


def heaviside(values, *, derivative=False, threshold=0.5):
    """Вычислить ступенчатую функцию с указанным порогом.

    Производная равна нулю всюду, где она определена, поэтому эта активация
    не обучается обычным backprop. В точке порога скачок не дифференцируем; для
    обратного прохода выбрана производная ноль.
    """
    values = np.asarray(values)

    if derivative:
        return np.zeros_like(values, dtype=float)

    return np.where(values >= threshold, 1.0, 0.0)


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
