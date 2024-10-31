# Model Build

Date : June 18 <br>
Writer : 양경식

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torchinfo import summary
import pandas as pd
import os
from collections import Counter

data_path = "./data/spectrograms" # "./train_mfcc"

dataset = datasets.ImageFolder(
    root = data_path,
    transform = transforms.Compose([transforms.Resize((201,81)),
                                  transforms.ToTensor()
                                  ])
)

# dataset

class_map = dataset.class_to_idx
# class_map

rate = 0.8 # train / test

size = len(dataset)
train_size = int(rate * size)
test_size = size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print("Training size:", len(train_dataset))
print("Testing size:",len(test_dataset))

train_classes = [label for _, label in train_dataset]
Counter(train_classes)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = 15,
    num_workers = 2,
    shuffle = True
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = 15,
    num_workers = 2,
    shuffle = True
)

td = train_dataloader.dataset[0][0][0][0]
# print(td)

if (torch.cuda.is_available()) : device = "cuda"
else : device = "cpu"
print(device)

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x,dim=1)  

model = CNNet().to(device)

cost = torch.nn.CrossEntropyLoss()

# Setting learning rate
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def train(dataloader, model, loss, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
            
            
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1)==Y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    # print(f'\nTest Error:\nacc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')

    
epochs = 10 # 크게 하면, over-fitting

for t in range(epochs):
    print(f"Epoch {t+1}")
    print("============")
    train(train_dataloader, model, cost, optimizer)
    test(test_dataloader, model)
print("Done!")

# summary(model)

print("Start!")
test(test_dataloader, model)
print("Done!")
```