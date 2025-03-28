# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 20:11:25 2025

@author: User
"""

import torch
from random import randint

#рандом от 1 до 10, тензора из одного значения, его преобразование к типу int32
tenzor = torch.randint(1,10, (1,), dtype=torch.int64) 
print("Начальный тензор: ", tenzor.item())

tenzor = tenzor.to(dtype=torch.float32)
tenzor.requires_grad = True #начало отслеживания изменений тензора

n=2
tenzorN = tenzor ** n
print("Тензор в степени n=2: ", tenzorN)

randValue = randint(1,10)
tenzorV = tenzorN * randValue
print("Тензор умноженный на случайное значение randValue = ", randValue, ": ", tenzorV)

tenzorE = torch.exp(tenzorV)
print("Взятие экспоненты от тензоора: ", tenzorE)

tenzorE.backward()
print("Продифференцированный тензор: ",tenzor.grad)
print("Значение без e","{:f}".format(tenzor.grad.item()))
