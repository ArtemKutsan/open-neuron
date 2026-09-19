"""Функции потерь и градиенты, передаваемые обратному проходу слоя."""

import numpy as np


class LossFunction:
    """Общий интерфейс функции потерь.

    ``evaluate_error`` возвращает остаток ``predictions - targets``.
    ``loss_derivative`` превращает его в градиент для обратного прохода.
    Масштабирование здесь должно совпадать с усреднением в ``evaluate_loss``.
    """

    def evaluate_error(self, y, predictions):
        raise NotImplementedError("Функция потерь должна вычислять остаток")

    def loss_derivative(self, error):
        raise NotImplementedError("Функция потерь должна вычислять градиент")

    def evaluate_loss(self, y, predictions):
        raise NotImplementedError("Функция потерь должна вычислять значение потерь")


class MAE(LossFunction):
    """Средняя абсолютная ошибка по всем элементам целевого массива."""

    def evaluate_error(self, y, predictions):
        return predictions - y

    def loss_derivative(self, error):
        """Вернуть dL/dA, усреднённый по примерам и выходным признакам."""
        return np.sign(error) / error.size

    def evaluate_loss(self, y, predictions):
        return np.mean(np.abs(predictions - y))

    def __str__(self):
        return '"Mean Absolute Error"'


class MSE(LossFunction):
    """Средняя квадратичная ошибка по всем элементам целевого массива."""

    def evaluate_error(self, y, predictions):
        return predictions - y

    def loss_derivative(self, error):
        """Вернуть dL/dA с учётом усреднения и производной квадрата."""
        return 2.0 * error / error.size

    def evaluate_loss(self, y, predictions):
        return np.mean(np.square(predictions - y))

    def __str__(self):
        return '"Mean Squared Error"'


class CE(LossFunction):
    """Категориальная кросс-энтропия для one-hot меток и softmax-выхода.

    Градиент ``predictions - y`` — это упрощённый градиент по logits для
    сочетания softmax и кросс-энтропии. Он не является производной кросс-
    энтропии по вероятностям отдельно. Класс усредняет этот градиент по пакету;
    слой softmax должен передать его дальше без дополнительного якобиана.
    """

    def evaluate_error(self, y, predictions):
        return predictions - y

    def loss_derivative(self, error):
        """Вернуть fused dL/dZ для softmax и кросс-энтропии, усреднённый по пакету."""
        return error / error.shape[0]

    def evaluate_loss(self, y, predictions):
        epsilon = 1e-8
        probabilities = np.clip(predictions, epsilon, 1.0)
        return -np.mean(np.sum(y * np.log(probabilities), axis=1))

    def __str__(self):
        return '"Categorical Crossentropy"'
