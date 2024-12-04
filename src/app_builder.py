import tkinter as tk
from tkinter import ttk
import ttkthemes

from app import PlotFrame, TextFrame, AlgorithmFrame, GDAlgorithmFrame, GeneticAlgorithmFrame, App
from funcs import example_3_1, Rastrigin, Rosenbrock, Himmelblau
    
    
class Builder:
    '''Построение приложения из компонентов tkinter'''
    _plt: PlotFrame | None
    _txt: TextFrame | None
    _right_frame: ttk.Frame
    _notebook: ttk.Notebook
    _main: tk.Tk
    _algorithms: tuple[AlgorithmFrame, ...]
        
    def __init__(self):
        self._plt = None
        self._txt = None
        self._set_root()
        self._right_frame = ttk.Frame(self._main)
        self._right_frame.pack(expand=True, side='right', fill='both')
        self._notebook = ttk.Notebook(self._right_frame)
        self._notebook.pack(expand=True, fill='both', side='top')
        
    def _set_root(self) -> None:
        '''Установка корневого элемента для tkinter'''
        self._main = ttkthemes.ThemedTk(theme='plastik')
        
    def set_algorithm_widgets(self) -> None:
        gd_functions = {
        'пример 3.1': example_3_1,
        }
        functions = {
            'ф-я Растригина': Rastrigin,
            'ф-я Розенброка': Rosenbrock,
            'ф-я Химмельблау': Himmelblau
        }
        self._algorithms = (
            GDAlgorithmFrame(self._notebook, gd_functions, self._plt, self._txt),
            GeneticAlgorithmFrame(self._notebook, functions, self._plt, self._txt),
        )
    
    def set_plot_widget(self) -> None:
        self._plt = PlotFrame(self._main)
    
    def set_text_widget(self) -> None:
        self._txt = TextFrame(self._right_frame)
    
    @property
    def app(self) -> App:
        return App(
            self._main,
            self._notebook,
            self._algorithms,
            self._plt,
            self._txt,
            ('Градиентный спуск', 'Генетический алгоритм')
        )


class Director:
    '''Создание приложений разной конфигурации при помощи строителей'''
    def no_plt_app(self, builder: Builder) -> App:
        builder.set_text_widget()
        builder.set_algorithm_widgets()
        return builder.app
    
    def no_txt_app(self, builder: Builder) -> App:
        builder.set_plot_widget()
        builder.set_algorithm_widgets()
        return builder.app
    
    def minimum_app(self, builder: Builder) -> App:
        builder.set_algorithm_widgets()
        return builder.app
    
    def maximum_app(self, builder: Builder) -> App:
        builder.set_plot_widget()
        builder.set_text_widget()
        builder.set_algorithm_widgets()
        return builder.app
    