from __future__ import annotations
from abc import ABC, abstractmethod
import tkinter as tk
from tkinter import ttk

import ttkthemes

import app
import funcs


class AppBuilder(ABC):
    '''Строитель приложений по компонентам'''
    def __init__(self) -> AppBuilder:
        self._plt = None
        self._txt = None
        
    @abstractmethod
    def set_algorithm_widgets(self) -> None:
        '''Установка виджетов с алгоритмами'''
        pass
    
    @abstractmethod
    def set_plot_widget(self) -> None:
        '''Установка виджета с графиком'''
        pass
    
    @abstractmethod
    def set_text_widget(self) -> None:
        '''Установка текстового виджета'''
        pass
    
    @property
    @abstractmethod
    def app(self) -> app.App:
        '''созданное приложение'''
        pass
    
    
class TkAppBuilder(AppBuilder):
    '''Построение приложения из компонентов tkinter'''
    def __init__(self) -> TkAppBuilder:
        super().__init__()
        self._set_root()
        self._right_frame = ttk.Frame(self._main)
        self._right_frame.pack(expand=True, side='right', fill='both')
        self._notebook = ttk.Notebook(self._right_frame)
        self._notebook.pack(expand=True, fill='both', side='top')
        
    def _set_root(self) -> None:
        '''Установка корневого элемента для tkinter'''
        self._main = tk.Tk()
        
    def set_algorithm_widgets(self) -> None:
        gd_functions = {
        'пример 3.1': funcs.example_3_1,
        }
        functions = {
            'ф-я сферы': funcs.sphere,
            'ф-я Растригина': funcs.Rastrigin,
            'ф-я Розенброка': funcs.Rosenbrock,
            'ф-я Химмельблау': funcs.Himmelblau
        }
        self._algorithms = (
            app.GD_AlgorithmTkFrame(self._notebook, gd_functions, self._plt, self._txt),
            app.GeneticAlgorithmFrame(self._notebook, functions, self._plt, self._txt),
            app.PSO_AlgorithmFrame(self._notebook, functions, self._plt, self._txt),
            app.BeeAlgorithmFrame(self._notebook, functions, self._plt, self._txt)
        )
    
    def set_plot_widget(self) -> None:
        self._plt = app.PlotTkFrame(self._main)
    
    def set_text_widget(self) -> None:
        self._txt = app.TextTkFrame(self._right_frame)
    
    @property
    def app(self) -> app.TkApp:
        return app.TkApp(
            self._main,
            self._notebook,
            self._algorithms,
            self._plt,
            self._txt
        )


class ThemedTkAppBuilder(TkAppBuilder):
    '''Строитель с стилизованнными компонентами из ttkthemes'''
    def _set_root(self) -> None:
        self._main = ttkthemes.ThemedTk(theme='plastik')


class AppBuilderDirector:
    '''Создание приложений разной конфигурации при помощи строителей'''
    def no_plt_app(self, builder: AppBuilder) -> app.App:
        builder.set_text_widget()
        builder.set_algorithm_widgets()
        return builder.app
    
    def no_txt_app(self, builder: AppBuilder) -> app.App:
        builder.set_plot_widget()
        builder.set_algorithm_widgets()
        return builder.app
    
    def minimum_app(self, builder: AppBuilder) -> app.App:
        builder.set_algorithm_widgets()
        return builder.app
    
    def maximum_app(self, builder: AppBuilder) -> app.App:
        builder.set_plot_widget()
        builder.set_text_widget()
        builder.set_algorithm_widgets()
        return builder.app
    

