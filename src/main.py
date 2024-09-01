from tkinter import Tk

from ttkthemes import ThemedTk

from app import *
from funcs import *


if __name__ == '__main__':
    txt: TextWidget = TextFrame()
    plt: PlotWidget = PlotFrame()
    
    gd_functions = {
        'пример 3.1': example_3_1,
    }
    
    functions = {
        'ф-я сферы': sphere,
        'ф-я Растригина': Rastrigin,
        'ф-я Розенброка': Rosenbrock,
        'ф-я Химмельблау': Himmelblau
    }
    
    ThemedTkApp(
        {
            'ЛР1': GD_AlgorithmFrame(gd_functions, txt, plt),
            'ЛР3': GeneticAlgorithmFrame(functions, txt, plt),
            'ЛР4': PSO_AlgorithmFrame(functions, txt, plt),
            'ЛР5': BeeAlgorithmFrame(functions, txt, plt)
        },
        plt,
        txt
    ).run()

