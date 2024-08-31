'''Источники: какая-то методичка и https://en.wikipedia.org/wiki/Test_functions_for_optimization'''

__all__ = [
    'example_3_1', 
    'Rosenbrock', 
    'sphere', 
    'Rastrigin', 
    'Himmelblau', 
    'Simionescu', 
    'Gomez_Levy'
]

from typing import Iterable
import numpy as np

def _example_3_1_func(x: Iterable[float]) -> float:
    x1, x2, *_ = x
    return 2 * x1 * x1 + x1 * x2 + x2 * x2

def _example_3_1_grad(x: Iterable[float]) -> Iterable[float]:
    x1, x2, *_ = x
    return [4 * x1 + x2, x1 + 2 * x2]

example_3_1 = (_example_3_1_func, _example_3_1_grad)

def Rosenbrock(x: Iterable[float]) -> float:
    x1, x2, *_ = x
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

def sphere(x: Iterable[float]) -> float:
    x1, x2, *_ = x
    return x1 ** 2 + x2 ** 2

def Rastrigin(x: Iterable[float]) -> float:
    x1, x2, *_ = x
    return 20 + (x1**2 - 10*np.cos(2*np.pi*x1)) + (x2**2 - 10*np.cos(2*np.pi*x2))

def Himmelblau(x: Iterable[float]) -> float:
    x1, x2, *_ = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def Simionescu(x: Iterable[float]) -> float:
    x1, x2, *_ = x
    return 0.1 * x1 * x2

def Gomez_Levy(x: Iterable[float]) -> float:
    x1, x2, *_ = x
    return 4*x1**2 - 2.1*x**4 + 1/3*x**6 + x1*x2 - 4*x2**2 + 4*x2**4
