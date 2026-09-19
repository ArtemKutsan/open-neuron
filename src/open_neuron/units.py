"""Объект одного нейрона, принадлежащего полносвязному слою."""

class Unit:
    """Представляет параметры и значения одного нейрона внутри ``Layer``.

    ``Unit`` сам не выполняет матричные вычисления и не обучается отдельно.
    Слой связывает его веса и смещение со своими массивами параметров, а вход,
    ``Z`` и активацию ``A`` он читает у родительского слоя.
    """

    count = 0

    def __init__(self, weights="kernel", bias="kernel"):
        Unit.count += 1
        self.number = Unit.count
        self.name = self.__class__.__name__
        self.id = id(self)
        self.layer = None
        self.index = None
        self.weights = weights
        self.bias = bias
        self.activation = None

    @property
    def Inputs(self):
        """Пакет входов родительского слоя."""
        return self.layer.Inputs

    @property
    def inputs(self):
        """Последний пример пакета, для наглядного вывода состояния."""
        return self.Inputs[-1] if self.Inputs is not None else None

    @property
    def Z(self):
        """Значения до активации для этого нейрона во всём пакете."""
        return self.layer.Z.T[self.index] if self.layer.Z is not None else None

    @property
    def z(self):
        """Значение до активации для последнего примера."""
        return self.Z[-1].item() if self.Z is not None else None

    @property
    def A(self):
        """Значения после активации для этого нейрона во всём пакете."""
        return self.activation(self.Z) if self.Z is not None else None

    @property
    def a(self):
        """Активация последнего примера."""
        return self.A[-1].item() if self.A is not None else None

    @Inputs.setter
    def Inputs(self, value):
        raise NotImplementedError("Входами Unit управляет родительский слой")

    def __str__(self):
        return (
            f"{self.name} {self.index + 1}/{self.layer.num_neurons}, "
            f"inputs: {self.inputs}, weights: {self.weights}, bias: {self.bias}, "
            f"z: {self.z}, activation: {self.activation.__name__}, a: {self.a}"
        )
