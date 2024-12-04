import numpy as np

def _example_3_1_func(x1: float, x2: float) -> float:
    return 2 * x1 * x1 + x1 * x2 + x2 * x2

def _example_3_1_grad(x1: float, x2: float) -> tuple[float, float]:
    return [4 * x1 + x2, x1 + 2 * x2]

example_3_1 = (_example_3_1_func, _example_3_1_grad)

def Rosenbrock(x1: float, x2: float) -> float:
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

def Rastrigin(x1: float, x2: float) -> float:
    return 20 + (x1**2 - 10*np.cos(2*np.pi*x1)) + (x2**2 - 10*np.cos(2*np.pi*x2))

def Himmelblau(x1: float, x2: float) -> float:
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
