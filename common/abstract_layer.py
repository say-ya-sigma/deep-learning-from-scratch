from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from numpy import floating

class AbstractLayer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: NDArray[floating]) -> NDArray[floating]:
        pass

    @abstractmethod
    def backward(self, dout: NDArray[floating]) -> NDArray[floating]:
        pass