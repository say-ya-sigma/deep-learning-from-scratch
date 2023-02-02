from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray

class AbstractOptimizer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> None:
        pass
