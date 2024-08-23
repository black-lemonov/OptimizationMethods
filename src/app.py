
from __future__ import annotations
from typing import TypeAlias, Iterable, Any
from abc import ABC, abstractmethod

from algorithms import Algorithm, Function, Gradient, FuncGradTuple


FuncDict: TypeAlias = dict[str, Function]
FuncGradDict: TypeAlias = dict[str, tuple[Function, Gradient]] 

    
class Context:
    def __init__(self,
                 available_interactors: dict[str, Interactor]) -> None:
        self._interactors_dict: dict[str, Interactor] = available_interactors
        self._interactor: Interactor = None
        
    @property
    def available_interactors(self) -> Iterable[str]:
        return self._interactors_dict.keys()
        
    def set_interactor(self, interactor_name: str) -> None:
        self._interactor = self._interactors_dict[interactor_name]
        self._interactor.init_window()
        
    def run_interactor(self,
                       func: str,
                       params: dict[str, Any],
                       color_params: dict[str, str]) -> None:
        self._interactor.run_algorithm(func, params, color_params)
    

class Interactor:
    def __init__(self,
                 available_funcs: FuncDict | FuncGradDict,
                 alg_params_dict: dict[str, type],
                 title: str,
                 color_params: Iterable[str],
                 out: UI) -> Interactor:
        self._funcs_dict: FuncDict | FuncGradDict = available_funcs
        self._alg_params_dict = alg_params_dict
        self._color_params = color_params
        self._title = title
        self._out: UI = out
        
    @abstractmethod
    def __str__(self) -> str:
        pass
        
    def init_window(self) -> None:
        self._out.enable_all()
        self._out.set_title(self._title)
        self._out.set_combo_boxes(self._funcs_dict.keys())
        self._out.set_color_boxes(self._color_params) 
        self._out.set_text_fields(self._alg_params_dict.keys())
    
    def run_algorithm(self,
                      func: str,
                      params: dict[str, Any],
                      color_params: dict[str, str]) -> None:
        alg: Algorithm = self.create_algorithm(
            func = self._funcs_dict[func],
            params = {
                k: v(params[k])
                for k, v in self._alg_params_dict.items()
            }
        )
        self._out.disable_all()
        while alg.is_over():
            alg.next_iteration()
            self.draw_iteration(color_params)
    
    @abstractmethod    
    def draw_iteration(self, color_params: dict[str, str]) -> None:
        pass
    
    @abstractmethod
    def create_algorithm(self,
                         func: Function | FuncGradTuple,
                         params: dict[str, Any]) -> Algorithm:
        pass
    

class UI(ABC):
    @abstractmethod
    def disable_all(self) -> None:
        pass
    
    @abstractmethod
    def enable_all(self) -> None:
        pass
    
    @abstractmethod
    def set_combo_boxes(self, options: Iterable[str]) -> None:
        pass
    
    @abstractmethod
    def set_color_boxes(self, options: Iterable[str]) -> None:
        pass
    
    @abstractmethod
    def set_title(self, title: str) -> None:
        pass
    
    @abstractmethod
    def set_text_fields(self, fields_names: Iterable[str]) -> None:
        pass
    
    @abstractmethod
    def draw_plot(self, func: Function, xbound: float, ybound: float) -> None:
        pass
    
    @abstractmethod
    def draw_point(self, point: Iterable[float], color: str) -> None:
        pass
    
    @abstractmethod
    def draw_square_area(self,
                         point: Iterable[float],
                         width: float,
                         border_color: str) -> None:
        pass