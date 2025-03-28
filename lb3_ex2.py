# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 21:32:22 2025

@author: User
"""
import pandas as pd 
import numpy as np
import torch 
import torch.nn as nn

# Считываем данные
df = pd.read_csv('data.csv')

# Отделение значений
answers = df.iloc[:, 4:].values
answers = np.where(answers == "Iris-setosa", 1, -1)
signs = df.iloc[:, :4 ].values

#занесение значений в тензоры
X = torch.Tensor(signs)
y = torch.Tensor(answers)
# создадим 4 сумматора без функци активации - 4 признака, 1 - т.к. 1 ответ

linear = nn.Linear(4, 1)

# при создании веса и смещения инициализируются автоматически
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# выберем вид функции ошибки и оптимизатор
# фунция ошибки показывает как сильно ошибается наш алгоритм в своих прогнозах
lossFn = nn.MSELoss() # MSE - среднеквадратичная ошибка, вычисляется как sqrt(sum(y^2 - yp^2))


# создадим оптимизатор - алгоритм, который корректирует веса наших сумматоров (нейронов)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.001) # lr - скорость обучения

# прямой проход (предсказание) выглядит так:
yp = linear(X)

# имея предсказание можно вычислить ошибку
loss = lossFn(yp, y)
print('Ошибка: ', loss.item())

# и сделать обратный проход, который вычислит градиенты (по ним скорректируем веса)
loss.backward()

# градиенты по параметрам
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# далее можем сделать шаг оптимизации, который изменит веса 
# на сколько изменится каждый вес зависит от градиентов и скорости обучения lr
optimizer.step()

# итерационно повторяем шаги
# в цикле (фактически это и есть алгоритм обучения):
stopWork = True
i = 1;
while stopWork == True:
    pred = linear(X)
    loss = lossFn(pred, y)
    print('Ошибка на ' + str(i) + ' итерации: ', loss.item())
    loss.backward()
    optimizer.step()
    i += 1
    if (loss.item() < 1): stopWork = False

maxPred = torch.max(pred)
maxP = float(maxPred)
minPred = torch.min(pred)
minP = float(minPred)
rift = (maxP+minP)/2


print("Введите четыре параметра цветка, чтобы определить его вид.")
stopWork = True
while stopWork == True:
    p1 = float(input("Введите признак 1: "))
    p2 = float(input("Введите признак 2: "))
    p3 = float(input("Введите признак 3: "))
    p4 = float(input("Введите признак 4: "))
    z = torch.Tensor([p1,p2,p3,p4],)
    indexZ = linear(z)
    print("\nВид цветка: ", end='')
    if (indexZ >= rift): print("Iris-setosa")
    else: print("Iris-versicolor")
    print()
    wish = input("Завершить работу? Введите y\n: ")
    if (wish == "Yes" or wish == "y"): stopWork = False
