from __future__ import annotations
from typing import Iterable, Callable, Any
from abc import ABC, abstractmethod
import sys
import time

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import ttkthemes

from algorithms import Function, Gradient, GDAlgorithm, GeneticAlgorithm


class App:
    '''Приложение для размещения виджетов с алгоритмами, графиком и текстовым полем.'''
    
    _window: ttkthemes.ThemedTk # окно
    _notebook: ttk.Notebook # виджет для алгоритмов
    _names: tuple[str, ...] # названия алгоритмов
    _algorithms: tuple[AlgorithmFrame, ...] # виджеты с алгоритмами                
    _plt: PlotFrame # виджет с графиком
    _txt: TextFrame # виджет для вывода текста
    
    def __init__(self,
                 window: ttkthemes.ThemedTk,
                 notebook: ttk.Notebook,
                 algorithms: tuple[AlgorithmFrame, ...],
                 plt_widget: PlotFrame,
                 txt_widget: TextFrame,
                 notes_names: tuple[str, ...]):
        '''Создание всех виджетов приложения и их размещение.'''
        self._window = window
        self._notebook = notebook
        self._names = notes_names
        self._algorithms = algorithms
        self._plt = plt_widget
        self._txt = txt_widget
        
        self._set_root()
        self._set_plot()
        self._set_algorithms()
        self._set_text()
   
    def _set_root(self) -> None:
        '''Создание корневого виджета, на к-ом все будет расположено'''
        self._window.title('Методы поисковой оптимизации')
        self._window.protocol('WM_DELETE_WINDOW', self._exit)
        
    def _set_plot(self) -> None:
        '''Размещение виджета с графиком'''
        if self._plt is not None:
            self._plt.get_widget().pack(expand=True, side='left', fill='both')
                
    def _set_algorithms(self) -> None: 
        '''Размещение виджетов с алгоритмами'''
        if self._names is None: 
            for alg_widget in self._algorithms:
                self._notebook.add(alg_widget.get_widget()) 
        else:
            for alg_widget, name in zip(self._algorithms, self._names):
                self._notebook.add(alg_widget.get_widget(), text=name) 
    
    def _set_text(self) -> None:
        '''Размещение виджета с текстом'''
        if self._txt is not None:
            self._txt.get_widget().pack(expand=True, side='bottom', fill='x')   
    
    def run(self) -> None:
        '''Запуск приложения'''
        self._window.mainloop()
        
    def _exit(self) -> None:
        '''Закрытие приложения'''
        sys.exit()
    
    
class TextFrame:
    '''Виджет для текстового вывода'''
    
    _master: ttk.Frame # родительский компонент
    _root: ttk.Frame
    _txt: ScrolledText
    
    def __init__(self, master: ttk.Frame):
        self._master = master
        self._set_root()
        self._set_title()
        self._set_txt()
    
    def _set_root(self) -> ttk.Frame | Any:
        '''Создание корневого элемента'''
        self._root = ttk.Frame(self._master)
    
    def _set_title(self) -> None:
        '''Заголовок виджета'''
        ttk.Label(self._root, text='Выполнение алгоритма').pack(expand=True)
    
    def _set_txt(self) -> None:
        '''Виджет, куда будет выводиться текст'''
        self._txt = ScrolledText(self._root, state='disabled')
        self._txt.pack(expand=True, fill='both')
        
    def _enable_txt(self) -> None:
        self._txt.config(state='normal')
    
    def _disable_txt(self) -> None:
        self._txt.config(state='disabled')
    
    @staticmethod
    def _block(f: Callable) -> Callable:
        '''Разблокирует и затем блокирует текстовое поле.'''
        def _f(self: TextFrame, *args) -> Any:
            self._enable_txt()
            f(self, *args)
            self._disable_txt()
        return _f
    
    @_block
    def print_point(self, point: Iterable[float], iter: int) -> None:
        '''Выводит точку в текстовый виджет'''
        print_format: str = '№{i:3d} ({x:6.3f}; {y:6.3f}) = {f:6.3f}\n'
        self._txt.insert(
            'insert',
            print_format.format(
                i=iter, x=point[0], y=point[1], f=point[2]
            )
        )
    
    @_block
    def clear_text(self) -> None:
        '''Очищает текстовый виджет'''
        self._txt.delete(1.0, 'end')
    
    @_block
    def print_msg(self, msg: str) -> None:
        '''Выводит в текстовый виджет сообщение'''
        self._txt.insert('insert', msg)
    
    def get_widget(self) -> ttk.Frame:
        '''Виджет на котором всё расположено'''
        return self._root
    

class PlotFrame:
    '''Компонент с графиком через классы tkinter'''
    
    _master: ttkthemes.ThemedTk # родительский компонент
    _root: ttk.Frame
    _fig: Figure
    _axes: Axes
    _canvas: FigureCanvasTkAgg
        
    def __init__(self, master: ttkthemes.ThemedTk):
        self._master = master
        self._set_root()
        self._set_plot()
        
    @staticmethod
    def _make_plot_data(func: Function,
                        x_bnd: float, y_bnd: float) -> tuple[Iterable[float], Iterable[float], Iterable[float]]:
        '''Формирование данных для графика ф-ии'''
        x = np.linspace(-x_bnd, x_bnd, 100)
        y = np.linspace(-y_bnd, y_bnd, 100)
        
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = func(x_grid, y_grid)
        
        return x_grid, y_grid, z_grid
    
    def _set_root(self) -> None:
        self._root = ttk.Frame(self._master)
    
    def _set_plot(self) -> None:
        '''Виджет, где будет график'''
        self._fig = plt.figure(figsize=(10, 10))
        self._axes = self._fig.add_subplot(projection='3d')
        self._canvas = FigureCanvasTkAgg(self._fig, master=self._root)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(expand=True, fill='both')
    
    def draw_plot(self, func: Callable[[Iterable[float]], float], x_bnd: float, y_bnd: float) -> None:
        '''Рисует график'''
        self._fig.clear()
        self._axes = self._fig.add_subplot(projection='3d')
        x, y, z = self._make_plot_data(func, x_bnd, y_bnd)
        self._axes.plot_surface(x, y, z, rstride=5, cstride=5, alpha=0.4)
    
    def draw_point(self, point: Iterable[float], color: str) -> None:
        '''Добавляет на график точку'''
        self._axes.scatter(*point, c=color, s=24)
    
    def draw_square_area(self, center: Iterable[float], rad: float, func: Callable[[Iterable[float]], float]) -> None:
        '''Добавляет на график квадратную область'''
        x, y, *_ = center
        rx = [x - rad, x - rad, x + rad, x + rad]  # x
        ry = [y - rad, y + rad, y + rad, y - rad]  # y
        rz = [func([x, y]) for x, y in zip(rx, ry)]  # z

        rx.append(rx[0])
        ry.append(ry[0])
        rz.append(rz[0])

        self._axes.plot(rx, ry, rz, label='parametric curve')
   
    def update(self) -> None:
        self._canvas.draw()
        
    def get_widget(self) -> ttk.Frame:
        return self._root
    
    
class AlgorithmFrame(ABC):
    '''Окно алг-ма с заголовком, полями для ввода, списком ф-ий и кнопками для управления'''
    
    _master: ttk.Notebook
    _start_btn: ttk.Button | None
    _funcs_dict: dict[str, Function] | dict[str, tuple[Function, Gradient]] # словарь с функциями
    _plt: PlotFrame # виджет с графиком
    _txt: TextFrame # текстовый виджет
        
    def __init__(self,
                 master: ttk.Notebook,
                 functions: dict[str, Function] | dict[str, tuple[Function, Gradient]],
                 plt_widget: PlotFrame,
                 txt_widget: TextFrame):
        self._master = master
        self._start_btn: ttk.Button | None = None
        self._funcs_dict = functions
        self._plt = plt_widget
        self._txt = txt_widget
        self._set_root()
        self._set_title()
        self._set_input_fields()
        self._set_funcs_box()
        self._set_control_btns()
    
    @abstractmethod
    def _set_root(self) -> None:
        '''Создание корневого элемента'''
        pass
    
    @abstractmethod
    def get_widget(self) -> ttk.Frame:
        '''Виджет на котором всё расположено'''
        pass        
    
    @property
    def text_widget(self) -> TextFrame:
        '''текстовый виджет'''
        return self._txt
    
    @text_widget.setter
    def text_widget(self, new_txt: TextFrame) -> None:
        self._txt = new_txt
    
    @property
    def plot_widget(self) -> PlotFrame:
        '''виджет с графиком'''
        return self._plt
    
    @plot_widget.setter
    def plot_widget(self, new_plt: PlotFrame) -> None:
        self._plt = new_plt
    
    @abstractmethod
    def _set_title(self) -> None:
        '''Заголовок окна'''
        pass
    
    @abstractmethod
    def _set_input_fields(self) -> None:
        '''Поля для ввода параметров'''
        pass
    
    @abstractmethod
    def _set_control_btns(self) -> None:
        '''Кнопки для управления'''
        pass
    
    def _run_algorithm(self) -> None:
        '''Запуск алгоритма.'''
        self._create_algorithm()
        self._iter_algorithm()
        self._end_notify()
        
    @abstractmethod
    def _create_algorithm(self) -> None:
        '''Метод для инициализации алг-ма значениями парам-ов из виджетов.'''
        # инициализация алгоритма...
        pass
        
    @abstractmethod
    def _iter_algorithm(self) -> None:
        '''Итерация алгоритма с отрисовкой графика и текстовым выводом'''
        pass
    
    def _set_funcs_box(self) -> None:
        '''Combobox с функциями. Можно переопределить, но зачем'''
        funcs: tuple[str, ...] = tuple(self._funcs_dict.keys())
        try:
            self._func_var = tk.StringVar(value=funcs[0])  
        except IndexError:
            messagebox.showerror(title='Ошибка', message='Словарь с функциями functions не может быть пустым.')
        else:
            func_box = ttk.Combobox(self.get_widget(), values=funcs, textvariable=self._func_var, state='readonly')
            func_box.pack(expand=True)
            func_box.bind(
                '<<ComboboxSelected>>',
                self._set_func
            )
    
    def _set_func(self, e) -> None:
        '''Обработчик для списка с функциями. Можно переопределить'''
        self._func = self._funcs_dict[self._func_var.get()]
        self._start_btn.config(state='normal')
        
    def _end_notify(self) -> None:
        '''Уведомление о завершении итерации алгоритма. Можно переопределить'''
        if self._txt is not None: self._txt.print_msg('работа завершена.')
        messagebox.showinfo(
            title='Расчет завершен',
            message='Программа успешно завершила свою работу!'
        )
        

class GDAlgorithmFrame(AlgorithmFrame):
    _iters_var: tk.IntVar
    _x_bnd_var: tk.DoubleVar
    _y_bnd_var: tk.DoubleVar
    _eps1_var: tk.DoubleVar
    _eps2_var: tk.DoubleVar
    _x0_var: tk.DoubleVar
    _y0_var: tk.DoubleVar
    _step_var: tk.DoubleVar
    _delay_var: tk.DoubleVar
    _start_btn: ttk.Button
    _stop_btn: ttk.Button
    _alg: GDAlgorithm
        
    def _set_root(self) -> None:
        self._root = ttk.Frame(self._master)
        
    def _set_title(self) -> None:
        ttk.Label(self._root, text='Метод градиентного спуска с постоянным шагом').pack(expand=True)
    
    def _set_input_fields(self) -> None:
        entries_frame = ttk.Frame(self._root)
        
        for i in range(8):
            entries_frame.rowconfigure(index=i, weight=1)
        entries_frame.columnconfigure(index=0, weight=1)
        entries_frame.columnconfigure(index=1, weight=1)
        
        self._iters_var = tk.IntVar(value=10)
        self._x_bnd_var = tk.DoubleVar(value=5)
        self._y_bnd_var = tk.DoubleVar(value=5)
        self._eps1_var = tk.DoubleVar(value=0.1)
        self._eps2_var = tk.DoubleVar(value=0.15)
        self._x0_var = tk.DoubleVar(value=0.5)
        self._y0_var = tk.DoubleVar(value=1)
        self._step_var = tk.DoubleVar(value=0.1)
        self._delay_var = tk.DoubleVar(value=0.5)
        
        ttk.Label(entries_frame, text='итераций').grid(row=0, column=0)
        ttk.Entry(entries_frame, textvariable=self._iters_var).grid(row=0, column=1)
        
        ttk.Label(entries_frame, text='ограничение по X (+-)').grid(row=1, column=0)
        ttk.Entry(entries_frame, textvariable=self._x_bnd_var).grid(row=1, column=1)
        
        ttk.Label(entries_frame, text='ограничение по Y (+-)').grid(row=2, column=0)
        ttk.Entry(entries_frame, textvariable=self._y_bnd_var).grid(row=2, column=1)
        
        ttk.Label(entries_frame, text='эпсилон1').grid(row=3, column=0)
        ttk.Entry(entries_frame, textvariable=self._eps1_var).grid(row=3, column=1)
        
        ttk.Label(entries_frame, text='эпсилон2').grid(row=4, column=0)
        ttk.Entry(entries_frame, textvariable=self._eps2_var).grid(row=4, column=1)
        
        ttk.Label(entries_frame, text='начальная точка').grid(row=5, column=0)
        
        point_frame = ttk.Frame(entries_frame)
        point_frame.rowconfigure(index=0, weight=1)
        for i in range(4):
            point_frame.columnconfigure(index=i, weight=1)
            
        ttk.Label(point_frame, text='X').grid(row=0, column=0)
        ttk.Entry(point_frame, textvariable=self._x0_var).grid(row=0, column=1)
        ttk.Label(point_frame, text='Y').grid(row=0, column=2)
        ttk.Entry(point_frame, textvariable=self._y0_var).grid(row=0, column=3)
        
        point_frame.grid(row=5, column=1)
        
        ttk.Label(entries_frame, text='скорость (итераций в сек.)').grid(row=6, column=0)
        ttk.Entry(entries_frame, textvariable=self._delay_var).grid(row=6, column=1)
        
        ttk.Label(entries_frame, text='шаг').grid(row=7, column=0)
        ttk.Entry(entries_frame, textvariable=self._step_var).grid(row=7, column=1)
        
        entries_frame.pack(expand=True)   
        
    def _set_func(self, e) -> None:
        '''Обработчик для списка с функциями. Можно переопределить'''
        self._func, self._grad = self._funcs_dict[self._func_var.get()]
        self._start_btn.config(state='normal')
        
    def _set_control_btns(self) -> None:
        btns_frame = ttk.Frame(self._root)
        btns_frame.rowconfigure(index=0, weight=1)
        btns_frame.columnconfigure(index=0, weight=1)
        btns_frame.columnconfigure(index=1, weight=1)
        self._start_btn = ttk.Button(btns_frame, text='Запустить', command=self._run_algorithm, state='disabled')
        self._start_btn.grid(row=0, column=0)
        self._stop_btn = ttk.Button(btns_frame, text='Остановить', state='disabled')
        self._stop_btn.grid(row=0, column=1)
        btns_frame.pack(expand=True) 
    
    def _create_algorithm(self) -> None:
        try:
            self._alg = GDAlgorithm(
                self._func,
                self._grad,
                self._x_bnd_var.get(),
                self._y_bnd_var.get(),
                self._iters_var.get(),
                self._eps1_var.get(),
                self._eps2_var.get(),
                (self._x0_var.get(), self._y0_var.get()),
                self._step_var.get()
            )
        except AttributeError:
            messagebox.showwarning(title='Внимание', message='Перед запуском алгоритма необходимо явно задать значение функции!')
        
    def _iter_algorithm(self) -> None:
        if self._txt is not None: 
            self._txt.clear_text()
            i: int = 1
        while not self._alg.is_over:
            self._alg.next_iteration()    
            if self._plt is not None:
                self._plt.draw_plot(self._alg.func, self._alg.xbound, self._alg.ybound)
                self._plt.draw_point(self._alg.result, 'red')
                self._plt.update()
            if self._txt is not None:
                self._txt.print_point(self._alg.result, i)
                i+=1
            if self._txt is not None or self._plt is not None:
                self._root.update()
                time.sleep(self._delay_var.get())
        
    def get_widget(self) -> ttk.Frame | Any:
        return self._root


class GeneticAlgorithmFrame(AlgorithmFrame): 
    _iters_var = tk.IntVar
    _x_bnd_var = tk.DoubleVar
    _y_bnd_var = tk.DoubleVar
    _p_mut_var = tk.DoubleVar
    _p_surv_var = tk.DoubleVar
    _pop_size_var = tk.IntVar
    _delay_var = tk.DoubleVar
    _delay_var: tk.DoubleVar
    _start_btn: ttk.Button
    _stop_btn: ttk.Button
    _alg: GeneticAlgorithm
    
    def _set_root(self) -> None:
        self._root = ttk.Frame(self._master)
    
    def _set_title(self) -> None:
        ttk.Label(self._root, text='Генетический алгоритм').pack(expand=True)
    
    def _set_input_fields(self) -> None:
        entries_frame = ttk.Frame(self._root)
        
        for i in range(7):
            entries_frame.rowconfigure(index=i, weight=1)
        entries_frame.columnconfigure(index=0, weight=1)
        entries_frame.columnconfigure(index=1, weight=1)
        
        self._iters_var = tk.IntVar(value=50)
        self._x_bnd_var = tk.DoubleVar(value=5)
        self._y_bnd_var = tk.DoubleVar(value=5)
        self._p_mut_var = tk.DoubleVar(value=0.8)
        self._p_surv_var = tk.DoubleVar(value=0.8)
        self._pop_size_var = tk.IntVar(value=100)
        self._delay_var = tk.DoubleVar(value=0.5)
        
        ttk.Label(entries_frame, text='итераций').grid(row=0, column=0)
        ttk.Entry(entries_frame, textvariable=self._iters_var).grid(row=0, column=1)
        
        ttk.Label(entries_frame, text='ограничение по X (+-)').grid(row=1, column=0)
        ttk.Entry(entries_frame, textvariable=self._x_bnd_var).grid(row=1, column=1)
        
        ttk.Label(entries_frame, text='ограничение по Y (+-)').grid(row=2, column=0)
        ttk.Entry(entries_frame, textvariable=self._y_bnd_var).grid(row=2, column=1)
        
        ttk.Label(entries_frame, text='шанс мутации').grid(row=3, column=0)
        ttk.Entry(entries_frame, textvariable=self._p_mut_var).grid(row=3, column=1)
        
        ttk.Label(entries_frame, text='шанс выживания').grid(row=4, column=0)
        ttk.Entry(entries_frame, textvariable=self._p_surv_var).grid(row=4, column=1)
        
        ttk.Label(entries_frame, text='размер популяции').grid(row=5, column=0)
        ttk.Entry(entries_frame, textvariable=self._pop_size_var).grid(row=5, column=1)
        
        ttk.Label(entries_frame, text='скорость (итераций в сек.)').grid(row=6, column=0)
        ttk.Entry(entries_frame, textvariable=self._delay_var).grid(row=6, column=1)
        
        entries_frame.pack(expand=True)
    
    def _set_control_btns(self) -> None:
        btns_frame = ttk.Frame(self._root)
        btns_frame.rowconfigure(index=0, weight=1)
        btns_frame.columnconfigure(index=0, weight=1)
        btns_frame.columnconfigure(index=1, weight=1)
        self._start_btn = ttk.Button(btns_frame, text='Запустить', command=self._run_algorithm, state='disabled')
        self._start_btn.grid(row=0, column=0)
        self._stop_btn = ttk.Button(btns_frame, text='Остановить', state='disabled')
        self._stop_btn.grid(row=0, column=1)
        btns_frame.pack(expand=True)
    
    def _create_algorithm(self) -> None:
        try:
            self._alg = GeneticAlgorithm(
                self._func,
                self._x_bnd_var.get(),
                self._y_bnd_var.get(),
                self._iters_var.get(),
                self._p_mut_var.get(),
                self._p_surv_var.get(),
                self._pop_size_var.get()
            )
        except AttributeError:
            messagebox.showwarning(title='Внимание', message='Перед запуском алгоритма необходимо явно задать значение функции!')
            
    def _iter_algorithm(self) -> None:
        if self._txt is not None:
            self._txt.clear_text()
            i: int = 1
        while True:
            self._alg.next_iteration()
            if self._alg.is_over: break
            if self._plt is not None:
                self._plt.draw_plot(self._alg.func, self._alg.xbound, self._alg.ybound)
                for p in self._alg.population: self._plt.draw_point(p, 'blue')
                self._plt.draw_point(self._alg.result, 'red')
                self._plt.update()
            if self._txt is not None:
                self._txt.print_point(self._alg.result, i)
                i += 1
            if self._txt is not None or self._plt is not None:
                self._root.update()
                time.sleep(self._delay_var.get())
        
    def get_widget(self) -> ttk.Frame | Any:
        return self._root
