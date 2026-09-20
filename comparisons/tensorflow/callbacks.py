"""Callback для краткого вывода хода обучения Keras."""

import tensorflow as tf


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Показывает потери и доступные метрики примерно десять раз за обучение.

    Чтобы выводить только строки этого callback, передавайте ``verbose=0``
    в ``model.fit``.
    """

    _metric_names = ("loss", "mae", "mse", "val_loss", "val_mae", "val_mse")

    def __init__(self, report_count=10):
        super().__init__()
        if report_count <= 0:
            raise ValueError("report_count должен быть положительным")
        self.report_count = report_count
        self.epochs = 0
        self._last_logs = {}

    @staticmethod
    def _format_metric(value):
        if value is None:
            return None
        if hasattr(value, "numpy"):
            value = value.numpy()
        return f"{float(value):.4f}"

    def on_train_begin(self, logs=None):
        self.epochs = int(self.params.get("epochs", 0))
        print(f"Обучение: эпох {self.epochs}")

    def on_epoch_end(self, epoch, logs=None):
        self._last_logs = dict(logs or {})
        interval = max(1, self.epochs // self.report_count)
        is_report_epoch = (
            epoch == 0
            or (epoch + 1) % interval == 0
            or epoch + 1 == self.epochs
        )
        if not is_report_epoch:
            return

        metrics = [
            f"{name}={self._format_metric(self._last_logs[name])}"
            for name in self._metric_names
            if self._format_metric(self._last_logs.get(name)) is not None
        ]
        print(f"Эпоха {epoch + 1}/{self.epochs}: " + ", ".join(metrics))

    def on_train_end(self, logs=None):
        final_logs = dict(logs or self._last_logs)
        final_loss = self._format_metric(final_logs.get("val_loss"))
        if final_loss is None:
            final_loss = self._format_metric(final_logs.get("loss"))
        if final_loss is None:
            print("Обучение завершено")
        else:
            print(f"Обучение завершено; итоговая потеря: {final_loss}")
