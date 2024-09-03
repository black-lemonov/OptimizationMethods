from __future__ import annotations

__all__ = [
    'Function',
    'Gradient',
    'Algorithm',
    'GD_Algorithm',
    'GeneticAlgorithm',
    'PSO_Algorithm',
    'BeeAlgorithm'
]

from typing import TypeAlias, Callable, Iterable
from abc import ABC, abstractmethod
import random as rnd
import math as m

import numpy as np 


Function: TypeAlias = Callable[[float, float], float]
Gradient: TypeAlias = Callable[[float, float], Iterable[float]]


class Algorithm(ABC):
    '''Алгоритм оптимизации функций'''
    def __init__(self,
                 func: Function,
                 xbound: float,
                 ybound: float,
                 iterations: int) -> Algorithm:
        self._func = func
        self._xbound = xbound
        self._ybound = ybound
        self._total_iters = iterations
        self._cur_iter: int = 0
        self._is_over: bool = False
    
    @property
    def func(self) -> Function:
        return self._func
    
    @property
    def xbound(self) -> float:
        return self._xbound
    
    @property
    def ybound(self) -> float:
        return self._ybound
    
    @abstractmethod
    def next_iteration(self) -> None:
        '''Выполнение одной итерации алгоритма'''
        pass
    
    @property
    def is_over(self) -> bool:
        return self._is_over
    
    @property
    @abstractmethod
    def result(self) -> Iterable[float]:
        pass
    
    
class GD_Algorithm(Algorithm):
    '''Метод градиентного спуска с постоянным шагом.'''
    def __init__(self,
                 func: Function,
                 grad: Gradient,
                 xbound: float,
                 ybound: float,
                 iterations: int,
                 eps1: float,
                 eps2: float,
                 x0: tuple[float, float],
                 step: float) -> GD_Algorithm:
        super().__init__(func, xbound, ybound, iterations)
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
            return # шаг 4
        
        if self._check_iters():
            self._is_over = True
            return # шаг 5
        
        self._calc_step()
        
        self._calc_new_x()

        if self._check_new_x():
            self._is_over = True
            self._x = self._new_x
            return

        self._cur_iter += 1
        self._x = self._new_x
        
    def _calc_grad(self) -> None:
        '''Вычисление градиента'''
        self._grad_x = self._grad(*self._x)  # шаг 3
    
    def _check_grad(self) -> bool:
        '''Проверка градиента'''
        return np.linalg.norm(self._grad_x) < self._e1
    
    def _check_iters(self) -> bool:
        '''Проверка кол-ва итераций'''
        return self._cur_iter >= self._total_iters
    
    def _calc_step(self) -> None:
        '''Вычисление шага'''
        pass
    
    def _calc_new_x(self) -> None:
        '''Вычисление нового x'''
        self._new_x = (self._x[0] - self._step * self._grad_x[0], self._x[1] - self._step * self._grad_x[1])  # шаг 7
        
        while self._func(*self._new_x) - self._func(*self._x) >= 0:
            self._step /= 2
        
        self._new_x = (self._x[0] - self._step * self._grad_x[0], self._x[1] - self._step * self._grad_x[1])
    
    def _check_new_x(self) -> bool:
        '''Проверка нового x'''
        cond1 = np.linalg.norm((self._new_x[0] - self._x[0], self._new_x[1] - self._x[1])) < self._e1
        cond2 = np.abs(self._func(*self._new_x) - self._func(*self._x)) < self._e2
        return cond1 and cond2
    
    @property
    def result(self) -> tuple[float, float, float]:
        return self._x + (self._func(*self._x),)


class GeneticAlgorithm(Algorithm):
    '''Генетический алгоритм'''
    def __init__(self,
                 func: Function,
                 xbound: float,
                 ybound: float,
                 iterations: int,
                 p_mutation: float,
                 p_survival: float,
                 population_size: int) -> GeneticAlgorithm:
        super().__init__(func, xbound, ybound, iterations)
        
        self._p_mut = p_mutation
        self._p_surv = p_survival
        self._pop_size = population_size
        
        self._population: list[list[float]] = self._make_start_pop()
        
    def _make_start_pop(self) -> list[list[float]]:
        '''
        Генерирует начальную популяцию особей вида [x, y, fitness(x,y)]\n
        на отрезках [-x_bound, +x_bound] и [-y_bound, +y_bound].
        '''
        return [
            [
                x:=rnd.uniform(-self._xbound, self._xbound),
                y:=rnd.uniform(-self._ybound, self._ybound),
                self._func(x, y)
            ]
            for _ in range(self._pop_size)
        ]

    def _do_selection(self) -> None:
        '''
        Упорядочивает особей в популяции по фитнес-функции\n
        и выбирает родителей для следующего поколения
        '''
        self._population.sort(key=lambda x: x[2], reverse=True)    # ранжирование

        # Кроссинговер - случайным образом выбираются 2 родителя и создаются 2 ребенка путем обмена их генами.
        children_count = m.floor(self._pop_size * (1 - self._p_surv))
        parents = self._population[self._pop_size - 2 * children_count:]

        for one in self.population[:children_count]:
            if rnd.random() > 0.5:
                one[0], one[1], one[2] = (x:=parents.pop()[0]), (y:=parents.pop()[1]), self._func(x, y)
            else:
                one[1], one[0], one[2] = (y:=parents.pop()[1]), (x:=parents.pop()[0]), self._func(x, y)

    def _do_mutation(self) -> None:
        '''
        Вносятся случайные изменения в гены с определенной вероятностью,\n
        что помогает исследовать новые области пространства решений.\n
        '''
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


class PSO_Algorithm(Algorithm):
    '''Роевой алгоритм'''
    def __init__(self,
                 func: Function,
                 xbound: float,
                 ybound: float,
                 iterations: int,
                 particles_number: int,
                 fi_p: float,
                 fi_g: float) -> PSO_Algorithm:
        super().__init__(func, xbound, ybound, iterations)

        if particles_number <= 0:
            raise ValueError('Неправильное значение кол-ва частиц. Кол-во частиц должно быть > 0 .')
        self._particles_number = particles_number
        
        # Проверяем, что fi_p + fi_g > 4,
        # иначе срабатывает исключение.
        if fi_p + fi_g <= 4: 
            raise ValueError("Неправильное значение коэффициентов fi_p и fi_g . Сумма коэффициентов должна быть > 4 .")
        self._fi_p = fi_p
        self._fi_g = fi_g

        # Вычисляем параметр xi,
        # который используется при обновлении скорости частиц по формуле
        self._Xi = 2 / (np.abs(2 - (fi_p + fi_g) - np.sqrt((fi_p + fi_g) ** 2 - 4 * (fi_p + fi_g))))

        # Инициализируется стартовая популяция частиц particles,
        # каждая из которых представлена как список [x, y, fitness],
        # где x и y - начальные координаты частиц,
        # а fitness - значение функции fitness в этих координатах
        self._particles = [
            [
                x := rnd.uniform(-self._xbound, self._xbound),
                y := rnd.uniform(-self._ybound, self._ybound), 
                self._func(x, y)
            ]
            for _ in range(self._particles_number)
        ]

        # Создается копия популяции nostalgia,
        # использующаяся для хранения лучших позиций частиц
        self._nostalgia = [
            p.copy()
            for p in self._particles
        ]

        # Инициализируется список velocity,
        # который представляет скорость каждой частицы
        # (изначально все скорости установлены в 0)
        self._velocity = [
            [0.0] * 2
            for _ in range(self._particles_number)
        ]

        # Находится начальное лучшее решение generation_best,
        # выбирая частицу с минимальным значением fitness
        # из текущей популяции.
        self._generation_best = min(self._particles, key=lambda x: x[2])

    def _update_velocity(
            self,
            velocity  : list[float],
            particle  : list[float], 
            point_best: list[float]) -> list[float]:
        '''
        Обновление скорости частиц по формуле ☠
        ''' 
        v_x = self._Xi * (velocity[0] + self._fi_p * rnd.random() * (point_best[0] - particle[0]) + self._fi_g * rnd.random() * (self._generation_best[0] - particle[0]))
        v_y = self._Xi * (velocity[1] + self._fi_p * rnd.random() * (point_best[1] - particle[1]) + self._fi_g * rnd.random() * (self._generation_best[1] - particle[1]))
        return [v_x, v_y]    

    def _update_position(
            self,
            velocity  : list[float],
            particle  : list[float]) -> list[float]:
        '''
        Обновление позиции частицы
        '''
        x = particle[0] + velocity[0]
        y = particle[1] + velocity[1]

        return [x, y, self._func(x, y)]
    
    def next_iteration(self) -> None:
        if self._cur_iter >= self._total_iters:
            self._is_over = True
            return
        
        # обновляет скорость и позицию каждой частицы, 
            # а также находит новую лучшую частицу в поколении
            
        for j in range(self._particles_number):

            if self._nostalgia[j][2] < self._particles[j][2]:
                point_best = self._nostalgia[j]
            else:
                self._nostalgia[j] = self._particles[j]
                point_best = self._particles[j]

            self._velocity[j] = self._update_velocity(self._velocity[j], self._particles[j], point_best)
            self._particles[j] = self._update_position(self._velocity[j], self._particles[j])

        self._cur_iter += 1
                
    @property
    def result(self) -> list[float]:
        return min(self._particles, key=lambda x: x[2])
    
    @property
    def particles(self) -> list[list[float]]:
        '''частицы'''
        return self._particles
    

class BeeAlgorithm(Algorithm):
    '''
    Реализация пчелиного алг-ма
    для минимизации ф-ии 
    '''
    def __init__(self,
                 func: Function,
                 xbound: float,
                 ybound: float,
                 iterations: int,
                 scouts_number: int,
                 elite_areas_number: int,
                 persp_areas_number: int,
                 bees_to_elite: int,
                 bees_to_persp: int,
                 radius: float) -> BeeAlgorithm:
        super().__init__(func, xbound, ybound, iterations)
        
        self._scouts_number = scouts_number
        self._elite_areas_number = elite_areas_number
        self._persp_areas_number = persp_areas_number
        self._bees_to_elite = bees_to_elite
        self._bees_to_persp = bees_to_persp
        self._radius = radius
        
        # Списки разведчиков и рабочих: 
        self._scouts: list[list[float]] = []
        self._workers: list[list[float]] = []
        self._bees = self._scouts + self._workers
        
        # Центры элитных и перспективных участков
        self._selected: list[list[float]] = []
        
    
    def _send_scouts(self) -> None:
        '''
        Инициализирует список разведчиков
        '''
        # Случайным образом генерируются новые позиции для скаутов
        # и оценивается их пригодность с помощью фитнес-функции
        self._scouts = [
            [
                x := rnd.uniform(-self._xbound, self._xbound),
                y := rnd.uniform(-self._ybound, self._ybound),
                self._func(x, y)
            ]
            for _ in range(self._scouts_number)
        ]

    def _select_areas(self) -> None:
        '''
        Инициализирует участки
        '''
        # Объединяет разведчиков и рабочих пчел,
        # сортирует их по пригодности
        self._bees = self._scouts + self._workers
        self._bees = sorted(self._bees, key=lambda x: x[2])
        self._selected = self._bees[0 : self._elite_areas_number + self._persp_areas_number]
            
    def _send_workers(
            self,
            bee_part: list[list[float]],
            sector: list[float],
            radius: float) -> None:
        '''
        Инициализация рабочих в областях участков
        '''
        for bee in bee_part:
            bee[0] = rnd.uniform(sector[0] - radius, sector[0] + radius)
            bee[1] = rnd.uniform(sector[1] - radius, sector[1] + radius)
            bee[2] = self._func(bee[0], bee[1])

    def _selected_search(self, param: float) -> None:
        '''
        Распределение пчел по участкам\n
        param : коэф. ограничивающий размер участка
        '''
        for i in range(self._elite_areas_number):
            _from = i * self._bees_to_elite
            _to = i * self._bees_to_elite + self._bees_to_elite
            self._send_workers(
                self._workers[_from : _to],
                self._selected[i],
                self._radius * param
            )

        for i in range(self._persp_areas_number):
            _from = self._elite_areas_number * self._bees_to_elite + i * self._bees_to_persp
            _to = _from + self._bees_to_persp
            self._send_workers(
                self._workers[_from : _to],
                self._selected[self._elite_areas_number + i],
                self._radius * param
            )   
    
    def next_iteration(self) -> None:
        if self._cur_iter >= self._total_iters:
            self._is_over = True
            return
        
        self._send_scouts()
        self._selected_search(1 / (self._cur_iter + 1))
        self._select_areas()
        
        self._cur_iter += 1
    
    @property
    def result(self) -> list[float]:
        return min(self._bees, key=lambda x: x[2])
    
    @property
    def radius(self) -> float:
        '''радиус'''
        return self._radius
    
    @property
    def scouts(self) -> list[list[float]]:
        '''пчелы-разведчики'''
        return self._scouts
    
    @property
    def areas(self) -> list[list[float]]:
        '''центры участков'''
        return self._selected
    
    @property
    def workers(self) -> list[list[float]]:
        '''рабочие пчелы'''
        return self._workers
