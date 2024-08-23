
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, TypeAlias, Callable


Function: TypeAlias = Callable[[Iterable[float]], float]
Gradient: TypeAlias = Callable[[Iterable[float]], Iterable[float]]
FuncGradTuple: TypeAlias = tuple[Function, Gradient] 


class Algorithm(ABC):
    @property
    @abstractmethod
    def func(self) -> Function | FuncGradTuple:
        pass
    
    @func.setter
    @abstractmethod
    def func(self, val: Function | FuncGradTuple) -> None:
        pass
    
    @abstractmethod
    def next_iteration(self) -> None:
        pass
    
    @abstractmethod
    def is_over(self) -> bool:
        pass
    
    @abstractmethod
    def result(self) -> Iterable[float]:
        pass
    
    
class GDAlgorithm(Algorithm, ABC):
    def __init__(self,
                 iterations: int,
                 eps1: int,
                 eps2: int,
                 x0: Iterable[float],
                 xbound: float,
                 ybound: float) -> GDAlgorithm:
        self._total_iters: int = iterations
        self._cur_iter: int = 0
        self._e1: float = eps1
        self._e2: float = eps2
        self._x: Iterable[float] = x0
        self._xbound = xbound
        self._ybound = ybound 
        self._func: Function = None
        self._grad: Gradient = None
        self._is_over = False
    
    @property
    def func(self) -> tuple[Function, Gradient]:
        return (self._func, self._grad)
    
    @func.setter
    def func(self, val: FuncGradTuple) -> None:
        self._func, self._grad = val
        
    def next_iteration(self) -> None:
        self.step1()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        self.step6()
        self.step7()
        self.step8()
        
    def is_over(self) -> bool:
        return self._is_over
    
    def result(self) -> Iterable[float]:
        return self._x
    
    @abstractmethod
    def step1(self) -> None:
        pass
    
    @abstractmethod
    def step2(self) -> None:
        pass
    
    @abstractmethod
    def step3(self) -> None:
        pass
    
    @abstractmethod
    def step4(self) -> None:
        pass
    
    @abstractmethod
    def step5(self) -> None:
        pass
    
    @abstractmethod
    def step6(self) -> None:
        pass
    
    @abstractmethod
    def step7(self) -> None:
        pass
    
    @abstractmethod
    def step8(self) -> None:
        pass


class GDConstStep(GDAlgorithm):
    pass
           