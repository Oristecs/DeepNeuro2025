# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:35:28 2025

@author: User
"""
#импорт функции, задающей псевдослучайные числа
from random import randint
#создание списка и заполнение его случйными числами
lstX = []
minX = int(input("Введите нижнюю границу значений генерируемых чисел: "))
maxX = int(input("Введите верхнюю границу значений генерируемых чисел: "))
quanX = int(input("Введите количество элементов списка x: "))
for i in range(quanX):
    lstX.append(randint(minX, maxX))
#суммирование всех чётных элементов списка
sumEvenX = 0
for i in lstX:
    sumEvenX += i * (i%2 == 0)
#вывод суммы в консоль
print("Сумма элементов списка: "+ str(sumEvenX))
