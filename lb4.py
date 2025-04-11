import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd

class NNet(nn.Module):
    # для инициализации сети на вход нужно подать размеры (количество нейронов) входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет слои и позволяет запускать их одновременно
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.ReLU(),                       # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred



df = pd.read_csv('dataset_simple.csv')
X = df.iloc[0:100, 0:2].values
X = torch.Tensor(X)
y = df.iloc[0:100, 2].values
y = torch.Tensor(y)



inputSize = X.shape[1] # количество признаков задачи (автоматически)

hiddenSizes = 3 #  число нейронов скрытого слоя 

outputSize = 1


# Создаем экземпляр нашей сети
net = NNet(inputSize,hiddenSizes,outputSize)

for param in net.parameters():
    print(param)

# Можно вывести их с названиями
for name, param in net.named_parameters():
    print(name, param)


# Посчитаем ошибку нашего не обученного алгоритма
# градиенты нужны только для обучения, тут их можно отключить, 
# это немного ускорит вычисления
with torch.no_grad():
    pred = net.forward(X)

# Так как наша сеть предсказывает числа от -1 до 1, то ее ответы нужно привести 
# к значениям меток
maxPred = torch.max(pred)
maxP = float(maxPred)
minPred = torch.min(pred)
minP = float(minPred)
rift = (maxP+minP)/2

pred = torch.Tensor(np.where(pred >= rift, 1, 0).reshape(-1,1))

# Считаем количество ошибочно классифицированных примеров
err = sum(abs(y-pred))
print(err) # до обучения сеть работает случайно, как бросание монетки

# Для обучения нам понадобится выбрать функцию вычисления ошибки
lossFn = nn.L1Loss()

# и алгоритм оптимизации весов
# при создании оптимизатора в него передаем настраиваемые параметры сети (веса)
# и скорость обучения
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

# В цикле обучения "прогоняем" обучающую выборку
# X - признаки
# y - правильные ответы
# epohs - количество итераций обучения

epohs = 10
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    
    
    loss = lossFn(pred, y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%1==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

maxPred = torch.max(pred)
maxP = float(maxPred)
minPred = torch.min(pred)
minP = float(minPred)
rift = (maxP+minP)/2

pred = torch.Tensor(np.where(pred >= rift, 1, 0).reshape(-1,1))
err = sum(abs(y-pred))
print('\nОшибка (количество несовпавших ответов): ')
print(err) # обучение работает, не делает ошибок или делает их достаточно мало
