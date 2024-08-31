from __future__ import annotations

__all__ = [
    'App',
    'TextFrame',
    'TextWidget',
    'PlotFrame',
    'PlotWidget',
    'GD_AlgorithmFrame',
    'GeneticAlgorithmFrame',
    'PSO_AlgorithmFrame',
    'BeeAlgorithmFrame',
    'Algorithm_Widget'
]

from typing import Iterable, Callable, Any
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
import sys
import time
from abc import ABC, abstractmethod

import numpy as np
import ttkthemes

from algorithms import *


class App(ABC):
    def __init__(self,
                 root: tk.Tk | ttkthemes.ThemedTk,
                 algorithms: dict[str, Algorithm_Widget],
                 text_widget: TextWidget | None = None,
                 plot_widget: PlotWidget | None = None) -> None:
        root.title('Методы поисковой оптимизации')
                
        plot_widget.pack(root, expand=True, fill='both', side='left')
        
        right_frame = ttk.Frame(root)
        right_frame.pack(expand=True, fill='both', side='right')
            
        alg_notebook = ttk.Notebook(right_frame)
        alg_notebook.pack(expand=True, fill='both', side='top')
        
        for title_str, alg_widget in algorithms.items():
            alg_widget.pack(alg_notebook)
            alg_notebook.add(alg_widget.get_tk_widget(), text=title_str)             
        
        text_widget.pack(right_frame, expand=True, fill='x')
        
        root.protocol('WM_DELETE_WINDOW', self._exit)
        root.mainloop()
        
    def _exit(self) -> None:
        sys.exit(1)
    

class Widget(ABC):
    '''Виджет, для размещения к-го нужно указывать родителя'''
    def pack(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs: Any) -> None:
        '''Создание виджета и его размещение в родителе. Нужно расширить для конкретного виджета.'''
        self._set_root(master, **kwargs)
    
    @abstractmethod
    def _set_root(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs: Any) -> None:
        '''Создание основного виджета, где все будет располагаться'''
        pass
    
    @abstractmethod
    def get_tk_widget(self) -> ttk.Frame | Any:
        '''Виджет, на к-ом всё размещено'''
        pass

    
class TextWidget(Widget, ABC):
    '''Виджет для текстового вывода'''
    def pack(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs) -> None:
        super().pack(master, **kwargs)
        self._set_title()
        self._set_txt()
    
    @abstractmethod
    def _set_title(self) -> None:
        '''Заголовок виджета'''
        pass
    
    @abstractmethod
    def _set_txt(self) -> None:
        '''Виджет, куда будет выводиться текст'''
        pass
        
    @abstractmethod
    def _enable(self) -> None:
        '''Сделать виджет с текстом активным'''
        pass
    
    @abstractmethod
    def _disable(self) -> None:
        '''Отключить виджет с текстом'''
        pass
    
    @staticmethod
    def _block(f: Callable) -> Callable:
        '''Разблокирует и затем блокирует текстовое поле.'''
        print('Декоратор работает!')
        def _f(self: TextWidget, *args) -> Any:
            self._enable()
            f(self, *args)
            self._disable()
        return _f
    
    @abstractmethod
    def print_point(self, point: Iterable[float], iter: int) -> None:
        '''Выводит точку в текстовый виджет'''
        pass
    
    @abstractmethod
    def clear_text(self) -> None:
        '''Очищает текстовый виджет'''
        pass
    
    @abstractmethod
    def print_msg(self, msg: str) -> None:
        '''Выводит в текстовый виджет сообщение'''
        pass
    

class TextFrame(TextWidget):
    def _set_root(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs) -> ttk.Frame | Any:
        self._root = ttk.Frame(master)
        self._root.pack(**kwargs)
    
    def _set_title(self) -> None:
        ttk.Label(self._root, text='Выполнение алгоритма').pack(expand=True)
    
    def _set_txt(self) -> None:
        self._txt = ScrolledText(self._root, state='disabled')
        self._txt.pack(expand=True, fill='both')
    
    def _enable(self) -> None:
        self._txt.config(state='normal')
    
    def _disable(self) -> None:
        self._txt.config(state='disabled')
    
    @TextWidget._block
    def print_point(self, point: Iterable[float], iter: int) -> None:
        print_format: str = '№{i:3d} ({x:6.3f}; {y:6.3f}) = {f:6.3f}\n'
        self._txt.insert(
            'insert',
            print_format.format(
                i=iter, x=point[0], y=point[1], f=point[2]
            )
        )
    
    @TextWidget._block
    def clear_text(self) -> None:
        self._txt.delete(1.0, 'end')
    
    @TextWidget._block
    def print_msg(self, msg: str) -> None:
        '''Выводит в текстовый виджет сообщение'''
        self._txt.insert('insert', msg)
    
    def get_tk_widget(self) -> ttk.Frame | Any:
        return self._root
        

class PlotWidget(Widget, ABC):
    '''Виджет для графика'''
    def pack(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs) -> None:
        super().pack(master, **kwargs)
        self._set_plot()
    
    @abstractmethod
    def _set_plot(self) -> None:
        '''Виджет, где будет график'''
        pass
    
    @abstractmethod
    def add_plot(self, func: Function, x_bnd: float, y_bnd: float) -> None:
        '''Рисует график'''
        pass
    
    @abstractmethod
    def add_point(self, point: Iterable[float], color: str) -> None:
        '''Добавляет на график точку'''
        pass
        
    @abstractmethod
    def add_square_area(self, center: Iterable[float], rad: float, func: Function) -> None:
        '''Добавляет на график квадратную область'''
        pass
    
    @abstractmethod
    def update(self) -> None:
        '''Обновляет виджет с графиком'''
        pass
    
    @staticmethod
    def _make_plot_data(func: Function,
                        x_bnd: float, y_bnd: float) -> tuple[Iterable[float], Iterable[float], Iterable[float]]:
        '''Формирование данных для графика ф-ии'''
        x = np.linspace(-x_bnd, x_bnd, 100)
        y = np.linspace(-y_bnd, y_bnd, 100)
        
        x_grid, y_grid = np.meshgrid(x, y)
        z_grid = func(np.array([x_grid, y_grid]))
        
        return x_grid, y_grid, z_grid


class PlotFrame(PlotWidget):
    def _set_root(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs) -> None:
        self._root = ttk.Frame(master)
        self._root.pack(**kwargs)
    
    def _set_plot(self) -> None:
        self._fig = plt.figure(figsize=(10, 10))
        self._axes = self._fig.add_subplot(projection='3d')
        self._canvas = FigureCanvasTkAgg(self._fig, master=self._root)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(expand=True, fill='both')
    
    def add_plot(self, func: Callable[[Iterable[float]], float], x_bnd: float, y_bnd: float) -> None:
        self._fig.clear()
        self._axes = self._fig.add_subplot(projection='3d')
        x, y, z = self._make_plot_data(func, x_bnd, y_bnd)
        self._axes.plot_surface(x, y, z, rstride=5, cstride=5, alpha=0.4)
    
    def add_point(self, point: Iterable[float], color: str) -> None:
        self._axes.scatter(*point, c=color)
    
    def add_square_area(self, center: Iterable[float], rad: float, func: Callable[[Iterable[float]], float]) -> None:
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
        
    def get_tk_widget(self) -> ttk.Frame | Any:
        return self._root
    

class Algorithm_Widget(Widget, ABC):
    '''Окно алг-ма с заголовком, полями для ввода, списком ф-ий и кнопками для управления'''
    def __init__(self,
                 functions: dict[str, Function] | dict[str, tuple[Function, Gradient]],
                 txt_widget: TextWidget | None = None,
                 plt_widget: PlotWidget | None = None) -> Algorithm_Widget:
        
        self._funcs_dict = functions
        '''словарь с функциями'''
        self._txt = txt_widget
        '''текстовый вывод'''
        self._plt = plt_widget
        '''вывод на график'''
    
    def pack(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs) -> None:
        super().pack(master, **kwargs)
        self._set_title()
        self._set_input_fields()
        self._set_funcs_box()
        self._set_control_btns()
        
    @abstractmethod
    def _set_title(self) -> None:
        '''Заголовок окна'''
        pass
    
    @abstractmethod
    def _set_input_fields(self) -> None:
        '''Поля для ввода параметров'''
        pass
    
    def _set_funcs_box(self) -> None:
        '''Combobox с функциями. Можно переопределить, но зачем'''
        funcs: tuple[str, ...] = tuple(self._funcs_dict.keys())
        try:
            self._func_var = tk.StringVar(value=funcs[0])  
        except IndexError:
            messagebox.showerror(title='Ошибка', message='Словарь с функциями functions не может быть пустым.')
        else:
            func_box = ttk.Combobox(self.get_tk_widget(), values=funcs, textvariable=self._func_var, state='readonly')
            func_box.pack(expand=True)
            func_box.bind(
                '<<ComboboxSelected>>',
                self._set_func
            )
    
    def _set_func(self, e) -> None:
        '''Обработчик для списка с функциями. Можно переопределить'''
        self._func = self._funcs_dict[self._func_var.get()]
    
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
    def _iter_algorithm(self) -> None:
        '''Итерация алгоритма с отрисовкой графика и текстовым выводом'''
        pass
    
    def _end_notify(self) -> None:
        '''Уведомление о завершении итерации алгоритма. Можно переопределить'''
        self._txt.print_msg('работа завершена.')
        messagebox.showinfo(
            title='Расчет завершен',
            message='Программа успешно завершила свою работу!'
        )
        
    @abstractmethod
    def _create_algorithm(self) -> None:
        '''Метод для инициализации алг-ма значениями парам-ов из виджетов.'''
        # инициализация алгоритма...
        pass
    

class GD_AlgorithmFrame(Algorithm_Widget):
    def _set_root(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs: Any) -> None:
        self._root = ttk.Frame(master)
    
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
        self._func, self._grad = self._funcs_dict[self._func_var.get()]
    
    def _set_control_btns(self) -> None:
        btns_frame = ttk.Frame(self._root)
        btns_frame.rowconfigure(index=0, weight=1)
        btns_frame.columnconfigure(index=0, weight=1)
        btns_frame.columnconfigure(index=1, weight=1)
        ttk.Button(btns_frame, text='Запустить', command=self._run_algorithm).grid(row=0, column=0)
        ttk.Button(btns_frame, text='Остановить', state='disabled').grid(row=0, column=1)
        btns_frame.pack(expand=True) 
    
    def _create_algorithm(self) -> None:
        try:
            self._alg = GD_Algorithm(
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
        self._txt.clear_text()
        i: int = 1
        while not self._alg.is_over:
            self._alg.next_iteration()    
            self._plt.add_plot(self._alg.func, self._alg.xbound, self._alg.ybound)
            self._plt.add_point(self._alg.result, 'red')
            self._plt.update()
            self._txt.print_point(self._alg.result, i)
            self._root.update()
            time.sleep(self._delay_var.get())
            i+=1
        
    def get_tk_widget(self) -> ttk.Frame | Any:
        return self._root


class GeneticAlgorithmFrame(Algorithm_Widget): 
    def _set_root(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs: Any) -> None:
        self._root = ttk.Frame(master)
    
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
        ttk.Button(btns_frame, text='Запустить', command=self._run_algorithm).grid(row=0, column=0)
        ttk.Button(btns_frame, text='Остановить', state='disabled').grid(row=0, column=1)
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
        self._txt.clear_text()
        i: int = 1
        while True:
            self._alg.next_iteration()
            if self._alg.is_over: break
            self._plt.add_plot(self._alg.func, self._alg.xbound, self._alg.ybound)
            for p in self._alg.population: self._plt.add_point(p, 'blue')
            self._plt.add_point(self._alg.result, 'red')
            self._plt.update()
            self._txt.print_point(self._alg.result, i)
            self._root.update()
            time.sleep(self._delay_var.get())
            i += 1
        
    def get_tk_widget(self) -> ttk.Frame | Any:
        return self._root
        

class PSO_AlgorithmFrame(Algorithm_Widget):
    def _set_root(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs: Any) -> None:
        self._root = ttk.Frame(master)
    
    def _set_title(self) -> None:
        ttk.Label(self._root, text='Алгоритм роя частиц').pack(expand=True)
    
    def _set_input_fields(self) -> None:
        entries_frame = ttk.Frame(self._root)
        entries_frame.pack(expand=True)
        
        for i in range(7):
            entries_frame.rowconfigure(index=i, weight=1)
        entries_frame.columnconfigure(index=0, weight=1)
        entries_frame.columnconfigure(index=1, weight=1)
        
        self._iters_var = tk.IntVar(value=50)
        self._x_bnd_var = tk.DoubleVar(value=5)
        self._y_bnd_var = tk.DoubleVar(value=5)
        self._part_n_var = tk.IntVar(value=50)
        self._fi_p_var = tk.DoubleVar(value=2)
        self._fi_g_var = tk.DoubleVar(value=3)
        self._delay_var = tk.DoubleVar(value=0.5)
        
        ttk.Label(entries_frame, text='итераций').grid(row=0, column=0)
        ttk.Entry(entries_frame, textvariable=self._iters_var).grid(row=0, column=1)
        
        ttk.Label(entries_frame, text='ограничение по X (+-)').grid(row=1, column=0)
        ttk.Entry(entries_frame, textvariable=self._x_bnd_var).grid(row=1, column=1)
        
        ttk.Label(entries_frame, text='ограничение по Y (+-)').grid(row=2, column=0)
        ttk.Entry(entries_frame, textvariable=self._y_bnd_var).grid(row=2, column=1)
        
        ttk.Label(entries_frame, text='кол-во частиц').grid(row=3, column=0)
        ttk.Entry(entries_frame, textvariable=self._part_n_var).grid(row=3, column=1)
        
        ttk.Label(entries_frame, text='коэффициент фи p').grid(row=4, column=0)
        ttk.Entry(entries_frame, textvariable=self._fi_p_var).grid(row=4, column=1)
        
        ttk.Label(entries_frame, text='коэффициент фи g').grid(row=5, column=0)
        ttk.Entry(entries_frame, textvariable=self._fi_g_var).grid(row=5, column=1)
        
        ttk.Label(entries_frame, text='скорость (итераций в сек.)').grid(row=6, column=0)
        ttk.Entry(entries_frame, textvariable=self._delay_var).grid(row=6, column=1)
    
    def _set_control_btns(self) -> None:
        btns_frame = ttk.Frame(self._root)
        btns_frame.rowconfigure(index=0, weight=1)
        btns_frame.columnconfigure(index=0, weight=1)
        btns_frame.columnconfigure(index=1, weight=1)
        ttk.Button(btns_frame, text='Запустить', command=self._run_algorithm).grid(row=0, column=0)
        ttk.Button(btns_frame, text='Остановить', state='disabled').grid(row=0, column=1)
        btns_frame.pack(expand=True)
        
    def _create_algorithm(self) -> None:
        try:
            self._alg = PSO_Algorithm(
                self._func,
                self._x_bnd_var.get(),
                self._y_bnd_var.get(),
                self._iters_var.get(),
                self._part_n_var.get(),
                self._fi_p_var.get(),
                self._fi_g_var.get()
            )
        except AttributeError:
            messagebox.showwarning(title='Внимание', message='Перед запуском алгоритма необходимо явно задать значение функции!')
        
    def _iter_algorithm(self) -> None:
        self._txt.clear_text()
        i: int = 1
        while True:
            self._alg.next_iteration()
            if not self._alg.is_over: break
            self._plt.add_plot(self._alg.func, self._alg.xbound, self._alg.ybound)
            for p in self._alg.population: self._plt.add_point(p, 'orange')
            self._plt.add_point(self._alg.result, 'red')
            self._plt.update()
            self._txt.print_point(self._alg.result, i)
            self._root.update()
            time.sleep(self._delay_var.get())
            i += 1
    
    def get_tk_widget(self) -> ttk.Frame | Any:
        return self._root
    
        
class BeeAlgorithmFrame(Algorithm_Widget):
    def _set_root(self, master: tk.Misc | ttk.Frame | ttk.Notebook | Any, **kwargs: Any) -> None:
        self._root = ttk.Frame(master)
        
    def _set_title(self) -> None:
        ttk.Label(self._root, text='Пчелинный алгоритм').pack(expand=True)

    def _set_input_fields(self) -> None:
        entries_frame = ttk.Frame(self._root)
        entries_frame.pack(expand=True)    
        
        for i in range(9):
            entries_frame.rowconfigure(index=i, weight=1)
        entries_frame.columnconfigure(index=0, weight=1)
        entries_frame.columnconfigure(index=1, weight=1)
        
        self._iters_var = tk.IntVar(value=50)
        self._x_bnd_var = tk.DoubleVar(value=5)
        self._y_bnd_var = tk.DoubleVar(value=5)
        self._scouts_var = tk.IntVar(value=10)
        self._elite_areas_var = tk.IntVar(value=1)
        self._persp_areas_var = tk.IntVar(value=2)
        self._b_to_elite_var = tk.IntVar(value=10)
        self._b_to_persp_var = tk.IntVar(value=10)
        self._rad_var = tk.DoubleVar(value=3)
        self._delay_var = tk.DoubleVar(value=0.5)
        
        ttk.Label(entries_frame, text='итераций').grid(row=0, column=0)
        ttk.Entry(entries_frame, textvariable=self._iters_var).grid(row=0, column=1)
        
        ttk.Label(entries_frame, text='ограничение по X (+-)').grid(row=1, column=0)
        ttk.Entry(entries_frame, textvariable=self._x_bnd_var).grid(row=1, column=1)
        
        ttk.Label(entries_frame, text='ограничение по Y (+-)').grid(row=2, column=0)
        ttk.Entry(entries_frame, textvariable=self._y_bnd_var).grid(row=2, column=1)
        
        ttk.Label(entries_frame, text='кол-во разведчиков').grid(row=3, column=0)
        ttk.Entry(entries_frame, textvariable=self._scouts_var).grid(row=3, column=1)
        
        ttk.Label(entries_frame, text='кол-во элитных участков').grid(row=4, column=0)
        ttk.Entry(entries_frame, textvariable=self._elite_areas_var).grid(row=4, column=1)
        
        ttk.Label(entries_frame, text='кол-во перспективных участков').grid(row=5, column=0)
        ttk.Entry(entries_frame, textvariable=self._persp_areas_var).grid(row=5, column=1)
        
        ttk.Label(entries_frame, text='кол-во пчел на элитных участках').grid(row=6, column=0)
        ttk.Entry(entries_frame, textvariable=self._b_to_elite_var).grid(row=6, column=1)
        
        ttk.Label(entries_frame, text='кол-во пчел на перспективных участках').grid(row=7, column=0)
        ttk.Entry(entries_frame, textvariable=self._b_to_persp_var).grid(row=7, column=1)
        
        ttk.Label(entries_frame, text='радиус участков').grid(row=8, column=0)
        ttk.Entry(entries_frame, textvariable=self._rad_var).grid(row=8, column=1)
        
        ttk.Label(entries_frame, text='скорость (итераций в сек.)').grid(row=9, column=0)
        ttk.Entry(entries_frame, textvariable=self._delay_var).grid(row=9, column=1)
    
    def _set_control_btns(self) -> None:
        btns_frame = ttk.Frame(self._root)
        btns_frame.rowconfigure(index=0, weight=1)
        btns_frame.columnconfigure(index=0, weight=1)
        btns_frame.columnconfigure(index=1, weight=1)
        ttk.Button(btns_frame, text='Запустить', command=self._run_algorithm).grid(row=0, column=0)
        ttk.Button(btns_frame, text='Остановить', state='disabled').grid(row=0, column=1)
        btns_frame.pack(expand=True)
        
    def _create_algorithm(self) -> None:
        try:
            self._alg = BeeAlgorithm(
                self._func,
                self._x_bnd_var.get(),
                self._y_bnd_var.get(),
                self._iters_var.get(),
                self._scouts_var.get(),
                self._elite_areas_var.get(),
                self._persp_areas_var.get(),
                self._b_to_elite_var.get(),
                self._b_to_persp_var.get(),
                self._rad_var.get()
            )
        except AttributeError:
            messagebox.showwarning(title='Внимание', message='Перед запуском алгоритма необходимо явно задать значение функции!')
        
    def _iter_algorithm(self) -> None:
        self._txt.clear_text()
        i: int = 1
        while True:
            self._alg.next_iteration()
            if self._alg.is_over: break 
            self._plt.add_plot(self._alg.func, self._alg.xbound, self._alg.ybound)
            for s in self._alg.population[0]: self._plt.add_point(s, 'blue')
            for c in self._alg.population[1]: self._plt.add_square_area(c, self._rad_var.get(), self._alg.func)
            for w in self._alg.population[2]: self._plt.add_point(w, 'black')
            self._plt.add_point(self._alg.result, 'red')
            self._plt.update()
            self._txt.print_point(self._alg.result, i)
            self._root.update()
            time.sleep(self._delay_var.get())
            i += 1
        
    def get_tk_widget(self) -> ttk.Frame | Any:
        return self._root
        