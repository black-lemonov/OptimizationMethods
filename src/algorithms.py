from typing import TypeAlias, Callable
import random as rnd
import math as m

import numpy as np 


Function: TypeAlias = Callable[[float, float], float]
Gradient: TypeAlias = Callable[[float, float], tuple[float, float]]

    
class GDAlgorithm:
    _func: Function 
    _grad: Gradient
    _xbound: float
    _ybound: float 
    _total_iters: int
    _cur_iter: int
    _is_over: bool
    _e1: float
    _e2: float
    _step: float
    _x: tuple[float, float]
    _is_over: bool
    
    def __init__(self,
                 func: Function,
                 grad: Gradient,
                 xbound: float,
                 ybound: float,
                 iterations: int,
                 eps1: float,
                 eps2: float,
                 x0: tuple[float, float],
                 step: float):
        self._func = func
        self._xbound = xbound
        self._ybound = ybound
        self._total_iters = iterations
        self._cur_iter: int = 0
        self._is_over: bool = False
        self._func: Function = func
        self._grad: Gradient = grad
        self._e1: float = eps1
        self._e2: float = eps2
        self._step = step
        self._x = x0
        self._is_over = False
    
    def next_iteration(self) -> None:
        self._calc_grad()
        
        if self._check_grad():
            self._is_over = True
            return
        
        if self._check_iters():
            self._is_over = True
            return
        
        self._calc_new_x()

        if self._check_new_x():
            self._is_over = True
            self._x = self._new_x
            return

        self._cur_iter += 1
        self._x = self._new_x
        
    def _calc_grad(self) -> None:
        self._grad_x = self._grad(*self._x)
    
    def _check_grad(self) -> bool:
        return np.linalg.norm(self._grad_x) < self._e1
    
    def _check_iters(self) -> bool:
        return self._cur_iter >= self._total_iters
    
    def _calc_new_x(self) -> None:
        self._new_x = (self._x[0] - self._step * self._grad_x[0], self._x[1] - self._step * self._grad_x[1])  # шаг 7
        
        while self._func(*self._new_x) - self._func(*self._x) >= 0:
            self._step /= 2
        
        self._new_x = (self._x[0] - self._step * self._grad_x[0], self._x[1] - self._step * self._grad_x[1])
    
    def _check_new_x(self) -> bool:
        cond1 = np.linalg.norm((self._new_x[0] - self._x[0], self._new_x[1] - self._x[1])) < self._e1
        cond2 = np.abs(self._func(*self._new_x) - self._func(*self._x)) < self._e2
        return cond1 and cond2
    
    @property
    def result(self) -> tuple[float, float, float]:
        return self._x + (self._func(*self._x),)

    @property
    def func(self) -> Function:
        return self._func
    
    @property
    def xbound(self) -> float:
        return self._xbound
    
    @property
    def ybound(self) -> float:
        return self._ybound
    
    @property
    def is_over(self) -> bool:
        return self._is_over
    


class GeneticAlgorithm:
    _func: Function
    _xbound: float
    _ybound: float
    _iterations: int
    _p_mut: float
    _p_surv: float
    _pop_size: int
    _population: list[list[float]]
    _total_iters: int
    _cur_iter: int
    _is_over: bool
    
    def __init__(self,
                 func: Function,
                 xbound: float,
                 ybound: float,
                 iterations: int,
                 p_mutation: float,
                 p_survival: float,
                 population_size: int):
        self._func = func
        self._xbound = xbound
        self._ybound = ybound
        self._p_mut = p_mutation
        self._p_surv = p_survival
        self._pop_size = population_size
        self._population: list[list[float]] = self._make_start_pop()
        self._total_iters = iterations
        self._cur_iter: int = 0
        self._is_over: bool = False
        
    def _make_start_pop(self) -> list[list[float]]:
        return [
            [
                x:=rnd.uniform(-self._xbound, self._xbound),
                y:=rnd.uniform(-self._ybound, self._ybound),
                self._func(x, y)
            ]
            for _ in range(self._pop_size)
        ]

    def _do_selection(self) -> None:
        self._population.sort(key=lambda x: x[2], reverse=True)
        
        children_count = m.floor(self._pop_size * (1 - self._p_surv))
        parents = self._population[self._pop_size - 2 * children_count:]

        for one in self.population[:children_count]:
            if rnd.random() > 0.5:
                one[0], one[1], one[2] = (x:=parents.pop()[0]), (y:=parents.pop()[1]), self._func(x, y)
            else:
                one[1], one[0], one[2] = (y:=parents.pop()[1]), (x:=parents.pop()[0]), self._func(x, y)

    def _do_mutation(self) -> None:
        for one in self._population:
            if rnd.random() < self._p_mut:
                one[0] += rnd.randint(-1, 1) * 0.1 * one[0]
            if rnd.random() < self._p_mut:
                one[1] += rnd.randint(-1, 1) * 0.1 * one[1]
            one[2] = self._func(one[0], one[1])
    
    def next_iteration(self) -> None:
        if self._cur_iter >= self._total_iters:
            self._is_over = True
            return
        
        self._do_selection()
        self._do_mutation()
        
        self._cur_iter += 1
    
    @property
    def result(self) -> list[float]:
        return min(self._population, key=lambda x: x[2])
    
    @property
    def population(self) -> list[list[float]]:
        return self._population
    
    @property
    def func(self) -> Function:
        return self._func
    
    @property
    def xbound(self) -> float:
        return self._xbound
    
    @property
    def ybound(self) -> float:
        return self._ybound
    
    @property
    def is_over(self) -> bool:
        return self._is_over
    