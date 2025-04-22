# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225] )
    ])


#  Создание датасетов
train_dataset = torchvision.datasets.ImageFolder(root='./data5lb/train',
                                                 transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(root='./data5lb/test',
                                             transform=data_transforms)


batch_size = 10

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True,  num_workers=2)


test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                    shuffle=True, num_workers=2) 



train_dataset.classes

# Названия классов
class_names = train_dataset.classes

# Загрузка порции данных.
inputs, classes = next(iter(train_loader))
inputs.shape 

# Сеть
class CnNet(nn.Module):
    def __init__(self, num_classes=10):
        nn.Module.__init__(self)
        self.layer1 = nn.Sequential(
        # первый сверточный слой с ReLU активацией и maxpooling-ом
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=2), # 3 канала, 16 фильтров, размер ядра 7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # второй сверточный слой 
        # количество каналов второго слоя равно количеству фильтров предыдущего слоя
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # третий сверточный слой 
        # ядро фильтра от слоя к слою уменьшается
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # классификационный слой имеет нейронов: количество фильтров * размеры карты признаков
        self.fc = nn.Linear(8*8*64, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1) # флаттеринг
        out = self.fc(out)
        return out



# Количество классов
num_classes = 3

# создаем экземпляр сети
net = CnNet(num_classes).to(device)

net = torchvision.models.alexnet(pretrained=True)

for param in net.parameters():
    param.requires_grad = False

new_classifier = net.classifier[:-1]
new_classifier.add_module('fc',nn.Linear(4096,num_classes))
net.classifier = new_classifier 

net = net.to(device)


# Обучение
num_epochs = 2
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

t = time.time()
save_loss = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # прямой проход
        outputs = net(images)
        # вычисление значения функции потерь
        loss = lossFn(outputs, labels)
         # Обратный проход (вычисляем градиенты)
        optimizer.zero_grad()
        loss.backward()
        # делаем шаг оптимизации весов
        optimizer.step()
        save_loss.append(loss.item())
        # выводим немного диагностической информации
        if i%100==0:
            print('Эпоха ' + str(epoch) + ' из ' + str(num_epochs) + ' Шаг ' +
                  str(i) + ' Ошибка: ', loss.item())

print(time.time() - t)


inputs, classes = next(iter(test_loader))

inputs.shape


pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)

for i,j in zip(inputs, pred_class):
    img = i.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(class_names[j])
    plt.pause(2)
