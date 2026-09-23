"""Самостоятельный обучаемый нейрон вне матричного слоя."""

import numpy as np

from open_neuron.activations import linear, softmax
from open_neuron.initializers import HeUniform
from open_neuron.losses import CE, MSE
from open_neuron.optimizers import SGD


class Neuron:
  """Один нейрон, который сам хранит параметры и выполняет обучение.

  В отличие от ``Unit``, связанного с ``Layer``, этот объект самостоятельно
  вычисляет выход, градиенты и обновляет собственные веса и смещение.
  """

  count = 0

  def __init__(
      self,
      weights=None,
      bias=None,
      input_size=2,
      output_size=1,
      activation=linear,
      loss=None,
      optimizer=None,
  ):
    Neuron.count += 1
    self.number = Neuron.count
    self.name = self.__class__.__name__
    self.id = id(self)
    self.Inputs = None
    self.Z = None
    self.A = None
    self.activation = activation if activation is not None else linear
    self.loss_function = loss if loss is not None else MSE()
    uses_softmax = self.activation is softmax
    uses_categorical_crossentropy = isinstance(self.loss_function, CE)
    if uses_softmax != uses_categorical_crossentropy:
      raise ValueError(
          "Для softmax требуется categorical_crossentropy, и наоборот"
      )
    self.optimizer = optimizer if optimizer is not None else SGD(momentum=0)
    self.delta = None
    self.gradients = {"weights": None, "bias": None}
    self.epochs = None
    self.batch_size = None
    self.final_batch_size = None
    self.X_len = None

    weights = weights if weights is not None else HeUniform()
    bias = bias if bias is not None else HeUniform()
    self.weights = self._initialize_parameter(
        weights, (input_size, output_size), "weights"
    )
    self.bias = self._initialize_bias(bias, output_size)

  @staticmethod
  def _initialize_parameter(value, shape, name):
    """Вызвать инициализатор или привести вручную заданные веса к матрице."""
    if callable(value):
      return np.asarray(value(shape=shape), dtype=float)

    values = np.asarray(value, dtype=float)
    if values.ndim == 1:
      values = values.reshape(-1, 1)
    if values.shape != shape:
      raise ValueError(
          f"Форма {name} должна быть {shape}, получено {values.shape}")
    return values.copy()

  @staticmethod
  def _initialize_bias(value, output_size):
    """Создать строку смещений из инициализатора или заданных значений."""
    if callable(value):
      return np.asarray(value(shape=(1, output_size)), dtype=float)

    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
      return np.full((1, output_size), values.item(), dtype=float)
    if values.ndim == 1:
      values = values.reshape(1, -1)
    if values.shape != (1, output_size):
      raise ValueError(
          f"Форма bias должна быть {(1, output_size)}, получено {values.shape}"
      )
    return values.copy()

  @property
  def inputs(self):
    """Последний входной пример, переданный нейрону."""
    return self.Inputs[-1] if self.Inputs is not None else None

  @property
  def z(self):
    """Последнее значение до функции активации."""
    return self._last_value(self.Z)

  @property
  def a(self):
    """Последний выход нейрона после функции активации."""
    return self._last_value(self.A)

  @staticmethod
  def _last_value(values):
    if values is None:
      return None
    result = np.asarray(values)[-1]
    return result.item() if result.size == 1 else result

  def activation_derivative(self, Z):
    """Вычислить производную активации по значениям ``Z``."""
    return self.activation(Z, derivative=True)

  def forward(self, X, training=True):
    """Вычислить ``Z = X @ W + b`` и активированный выход ``A``."""
    # X имеет форму (число примеров, число входных признаков). Сохраняем
    # пакет: backward использует его, чтобы найти вклад входов в градиент весов.
    self.Inputs = np.asarray(X)
    # Для каждого примера считаем взвешенную сумму Z и добавляем смещение b.
    # Последний размер результата соответствует выходам этого нейрона.
    self.Z = self.Inputs @ self.weights + self.bias
    # Функция активации переводит Z в выход A; именно A сравнивается с целью.
    self.A = self.activation(self.Z)
    # Возвращаем весь пакет выходов. Z, A и X также остаются в состоянии объекта
    # до следующего прямого прохода и нужны обратному проходу.
    return self.A

  def backward(self, error):
    """По ошибке выхода вычислить производные потерь по параметрам.

    Метод сохраняет производные в ``self.gradients`` для последующего вызова
    ``update``. Сам параметры не меняет и значения не возвращает.
    """
    # error — разница между предсказанием A и правильным ответом y. Например,
    # если A = 0, а y = 1, то error = -1. Это показывает промах, но ещё не
    # говорит, насколько изменить вес: для этого берём производные функции потерь.
    # Здесь используется соглашение error = A - y. Оно связано с MSE и её
    # производной: L = (A - y)^2, поэтому dL/dA содержит множитель (A - y).
    # В ручном перцептроне из учебного ноутбука знак записан наоборот:
    # target - prediction, потому что там применяется правило weights += ... .
    # В этом нейроне производная применяется градиентным спуском через -= .
    # Оба варианта направляют параметры в одну сторону, если знак остатка и
    # знак операции обновления согласованы; это не два разных вида ошибки.
    #
    # Производная по выходу отвечает на вопрос: как изменится потеря L, если
    # немного изменить предсказание A? Для MSE на одном выходе L = (A - y)^2,
    # поэтому dL/dA = 2 * (A - y). При A = 0 и y = 1 получаем -2.
    # Минус показывает, что около этой точки увеличение A уменьшает L;
    # величина производной показывает крутизну изменения L. Это не размер
    # обновления параметра: его ещё умножит на скорость обучения оптимизатор.
    # Для пакета MSE делит производную на число элементов error, чтобы градиент
    # соответствовал среднему значению потери. MAE и CE задают свой масштаб.
    # loss_derivative превращает остаток в dL/dA (или в особый градиент dL/dZ
    # для сочетания CE и softmax).
    #
    # Для обычной поэлементной активации применяем правило цепочки: чтобы узнать,
    # как потеря зависит от Z, умножаем dL/dA на dA/dZ. Например, для линейной
    # активации dA/dZ = 1, поэтому значение производной не меняется.
    # Для CE вместе с softmax loss_derivative уже даёт объединённый dL/dZ;
    # производная softmax здесь по соглашению равна единицам и оставляет его
    # без изменения. Это упрощение действует только для этой пары.
    self.delta = self.loss_function.loss_derivative(error) * (
        self.activation_derivative(self.Z)
    )
    # Теперь ищем производную потери по каждому весу. Выход зависит от весов:
    # z_i = x_i1*w_1 + x_i2*w_2 + ... + b. При взятии производной по w_j
    # остальные веса, входы и b считаются постоянными: они не меняются вместе
    # с w_j. Производная z_i по w_j равна x_ij — коэффициенту перед этим весом.
    # Это не значит, что коэффициент обнуляется: например, d(3*w_j)/dw_j = 3,
    # а производная слагаемого, не зависящего от w_j, равна нулю.
    # По правилу цепочки dL/dw_j = сумма по примерам (dL/dz_i * x_ij).
    # В примере x = [1, 0], A = 0, y = 1, линейная активация даёт delta = -2:
    # dL/dw_1 = -2*1 = -2, а dL/dw_2 = -2*0 = 0. Первый вес влияет на выход,
    # второй при таком входе не влияет. Для пакета матричная запись этой суммы
    # — X.T @ delta; усреднение MSE уже учтено функцией loss_derivative.
    self.gradients["weights"] = self.Inputs.T @ self.delta
    # Смещение b прибавляется к каждому z с коэффициентом 1, значит его
    # производная по каждому примеру равна delta этого примера. Складываем
    # вклады пакета; keepdims сохраняет форму строки bias.
    self.gradients["bias"] = np.sum(self.delta, axis=0, keepdims=True)

  def update(self):
    """Поручить оптимизатору обновить параметры нейрона."""
    # Здесь производные превращаются в изменение параметров. Обычный SGD
    # выполняет параметр = параметр - скорость_обучения * производная.
    # Например, при dL/dw_1 = -2 и скорости 0.1 вес увеличивается на 0.2:
    # w_1 = w_1 - 0.1*(-2). Знак производной задаёт направление, её величина
    # влияет на шаг; скорость обучения тоже влияет на размер шага. Momentum,
    # если он включён, дополнительно учитывает предыдущее обновление.
    self.optimizer.update(self)

  def fit(
      self,
      X,
      y,
      epochs=10,
      batch_size=1,
      final_batch_size=None,
      shuffle=True,
      validation_data=None,
  ):
    """Обучить нейрон пакетами и показать потери по эпохам.

    ``X`` содержит признаки, ``y`` — соответствующие целевые значения.
    ``validation_data`` задаёт отдельную пару массивов для оценки после эпохи;
    если её нет, для этой оценки используется вся обучающая выборка.
    Обучение обновляет параметры на каждом пакете. Метод не возвращает модель:
    она обучается на месте, а последние параметры остаются в этом объекте.
    """
    # Переводим входы в массивы NumPy, чтобы индексировать выборку одинаково
    # при перемешивании и при разбиении на пакеты.
    X = np.asarray(X)
    y = np.asarray(y)
    for name, value in (
        ("batch_size", batch_size),
        ("final_batch_size", final_batch_size),
    ):
      if value is not None and (
          isinstance(value, (bool, np.bool_))
          or not isinstance(value, (int, np.integer))
          or value <= 0
      ):
        raise ValueError(
            f"{name} должен быть положительным целым числом или None"
        )
    self.epochs = epochs
    self.batch_size = batch_size
    self.final_batch_size = final_batch_size
    self.X_len = X.shape[0]

    if validation_data is not None:
      # Отложенные данные нужны только для оценки и не участвуют в обновлениях.
      X_validation, y_validation = map(np.asarray, validation_data)
      validation_name = "Validation Data"
    else:
      # Без отдельной пары оцениваем всю обучающую выборку. Это значение
      # не является метрикой на независимых тестовых данных.
      X_validation, y_validation = X, y
      validation_name = "Train Data"

    # Начальная оценка показывает потери до первого обновления параметров.
    # predict делает прямой проход, но не вызывает backward или update.
    validation_loss = self.loss_function.evaluate_loss(
        y_validation, self.predict(X_validation)
    )
    print(
        f"Training started with Overall Loss {validation_loss:.4f} "
        f"on {validation_name}"
    )

    current_batch_size = batch_size
    report_interval = max(1, epochs // 10)

    for epoch in range(epochs):
      # Если заданы оба размера, плавно меняем размер пакета от batch_size
      # к final_batch_size по мере прохождения эпох.
      if batch_size is not None and final_batch_size is not None:
        current_batch_size = batch_size + int(
            (final_batch_size - batch_size) * (epoch + 1) / epochs
        )

      if shuffle:
        # Одна и та же перестановка применяется к X и y, чтобы не разорвать
        # соответствие между признаками и правильными ответами.
        permutation = np.random.permutation(self.X_len)
        X_epoch = X[permutation]
        y_epoch = y[permutation]
      else:
        X_epoch, y_epoch = X, y

      # None означает один пакет со всей выборкой. Иначе range ниже проходит
      # по выборке с шагом, равным текущему размеру пакета.
      step_size = current_batch_size or self.X_len
      total_epoch_loss = 0.0

      for start in range(0, self.X_len, step_size):
        # Берём очередные признаки и соответствующие им цели. Последний пакет
        # может быть короче step_size, если размер выборки не делится на него.
        X_batch = X_epoch[start: start + step_size]
        y_batch = y_epoch[start: start + step_size]
        # Прямой проход сохраняет X_batch, Z и предсказания A внутри нейрона.
        # Эти значения backward использует для производных по параметрам.
        predictions = self.forward(X_batch)
        # evaluate_error возвращает разницу предсказания и ответа для backward.
        # evaluate_loss отдельно считает одно число для отчёта. backward берёт
        # производные из error, а не использует число batch_loss как градиент.
        error = self.loss_function.evaluate_error(y_batch, predictions)
        batch_loss = self.loss_function.evaluate_loss(y_batch, predictions)
        # batch_loss — среднее по элементам пакета. Умножаем на число примеров,
        # чтобы ниже корректно усреднить эпоху и при неполном последнем пакете.
        total_epoch_loss += batch_loss * len(X_batch)

        # backward записывает градиенты, затем update применяет их к весам и
        # смещениям. Следующий пакет уже проходит через обновлённые параметры.
        self.backward(error)
        self.update()

      # Это среднее потерь на пакетах во время обучения: каждый пакет оценён
      # до собственного обновления, а более короткий пакет имеет меньший вес.
      epoch_loss = total_epoch_loss / self.X_len
      # Отдельно после всех обновлений эпохи оцениваем текущую модель на всей
      # validation-выборке (или на всей обучающей, если validation_data нет).
      # Эта оценка не вызывает backward/update и потому не меняет параметры.
      validation_loss = self.loss_function.evaluate_loss(
          y_validation, self.predict(X_validation)
      )
      should_report = (
          epochs < 10
          or epoch == 0
          or (epoch + 1) % report_interval == 0
          or epoch == epochs - 1
      )
      if should_report:
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
            f"Validation Loss: {validation_loss:.4f} on {validation_name}"
        )

    print(f"Training completed with Overall Loss {validation_loss:.4f}")

  def predict(self, X):
    """Вычислить выход нейрона без обновления параметров."""
    # Предсказание выполняет только прямой проход. Параметры не обучаются,
    # хотя последние входы, Z и A сохраняются в объекте для отображения.
    return self.forward(X, training=False)

  def call(self, X, training=False):
    """Вернуть выход после прямого прохода, сохраняя интерфейс объекта."""
    return self.A

  def __call__(self, X, training=False):
    self.forward(X, training=training)
    return self.call(X, training=training)

  def __str__(self):
    return (
        f"{self.name} {self.number}, inputs: {self.inputs}, "
        f"weights: {np.array2string(self.weights.ravel(), precision=4)}, "
        f"bias: {np.array2string(self.bias.ravel(), precision=4)}, "
        f"z: {self.z}, activation: {self.activation.__name__}, a: {self.a}"
    )
