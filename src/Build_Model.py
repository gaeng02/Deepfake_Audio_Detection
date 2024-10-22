'''
This script represents the process of building model using the data gathered so far.

PS. If you'd like to use this code, fell free to refer to the example code for guidance.
I will indicate where revisions can be made.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter

from Split import Dataset, Split, Dataloader 


def check_device () :
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class CNNet (nn.Module) :
    
    def __init__ (self) :
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, 2)


    def forward (self, x) :
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        #x = x.view(x.size(0), -1)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        
        return F.log_softmax(x,dim=1)  


def Train (dataloader, model, loss, optimizer) :

    model.train()
    
    for batch, (X, Y) in enumerate(dataloader) :
        
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        # This section prints the progress.
        ''' 
        if ((batch % 100) == 0) :
            print(f"[{batch * len(X):>5d}/{len(dataloader.dataset):>5d}] :: loss : {loss.item():>7f}")
        '''

    
def Test (dataloader, model) :
    ''' Purpose of "Test" function is to evaluate the fit on the data and to prevent overfitting. '''

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad() :
        for batch, (X, Y) in enumerate(dataloader) :
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1)==Y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(f"Test Accuracy :: {(100*correct):>0.1f}%")
    print(f"Test Average loss :: {test_loss:>6f}")


def Save (model, path = "../data/model.pth") :
    torch.save(model.state_dict(), path)


if (__name__ == "__main__") :

    # Refer to the "Split.py" for guidance.
    dataset = Dataset()
    train_dataset, test_dataset = Split(dataset)
    train_dataloader = Dataloader(train_dataset)
    test_dataloader = Dataloader(test_dataset)

    
    device = check_device() # using cuda or cpu
    model = CNNet().to(device)
    
    loss = torch.nn.CrossEntropyLoss()

    learning_rate = 0.0001 # Can modify.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 5 # Can modify.

    for e in range (epochs) :
        print(f"Epoch :: {e+1}")

        Train(train_datalader, model, cost, optimizer)
        Test(test_dataloader, model)

    print("Build Model")

    # You can summary model
    # summary(model)
    # model.eval()

    Save (model)
