import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Данные
df = pd.read_csv('dataset_simple.csv')
X = torch.FloatTensor(StandardScaler().fit_transform(df.iloc[:, :2].values))
y = torch.FloatTensor(df.iloc[:, 2].values).reshape(-1, 1)  # метки 0 и 1

# Архитектура сети
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()  # выход в [0, 1]
        )
    
    def forward(self, X):
        return self.layers(X)

# Параметры
inputSize = X.shape[1]
hiddenSizes = 100
outputSize = 1  # один выход (бинарная классификация)

# Сеть, функция потерь, оптимизатор
net = NNet(inputSize, hiddenSizes, outputSize)
lossFn = nn.BCELoss()  # Binary Cross-Entropy
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Adam работает лучше SGD
pred = net(X)
# Обучение
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = net(X)
    loss = lossFn(pred, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Оценка
with torch.no_grad():
    pred = net(X)
    pred_labels = (pred > 0.5).float()  # порог 0.5
    accuracy = (pred_labels == y).float().mean()
    print(f"\nТочность: {accuracy.item() * 100:.2f}%")
    
maxPred = torch.max(pred)
maxP = float(maxPred)
minPred = torch.min(pred)
minP = float(minPred)
rift = (maxP+minP)/2

pred = torch.Tensor(np.where(pred >= rift, 1, 0).reshape(-1,1))
err = sum(abs(y-pred))/2
print('\nОшибка (количество несовпавших ответов): ')
print(err)
