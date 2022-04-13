from .multiobjective.constrained import Srinivas, Tanaka, Psp31, Psp85, Psp89, Psp98, Psp225
from .multiobjective.dtlz import *
from .multiobjective.lz09 import *
from .multiobjective.unconstrained import Kursawe, Fonseca, Schaffer, Viennet2
from .multiobjective.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from .singleobjective.unconstrained import OneMax, Sphere

__all__ = [
    'Psp31', 'Psp85', "Psp89", "Psp98", "Psp225",
    'Srinivas', 'Tanaka',
    'Kursawe', 'Fonseca', 'Schaffer', 'Viennet2',
    'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6',
    'LZ09_F1', 'LZ09_F2', 'LZ09_F3', 'LZ09_F4', 'LZ09_F5', 'LZ09_F6', 'LZ09_F7', 'LZ09_F8', 'LZ09_F9',
    'DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7',
    'OneMax', 'Sphere',
]
