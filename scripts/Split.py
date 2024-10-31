'''
This script splits data into train and test dataset.
'''

import torch
import torchaudio
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter


def Dataset (path = "../data/preprocessed/spectrogram") : 

    data_path = path

    dataset = datasets.ImageFolder(
        root = data_path,
        transform = transforms.Compose([transforms.Resize((201,81)),
                transforms.ToTensor()
                ])
    )

    return dataset


def Dataloader (dataset, batch_size = 15, num_workers = 2, shuffle = True) :

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle
    )

    return dataloader


def Split (dataset, rate = 0.8) :

    size = len(dataset)

    train_size = int(rate * size)
    test_size = size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def Split_Loader () :
    
    dataset = Dataset()
    train_dataset, test_dataset = Split(dataset)

    train_dataloader = Dataloader(train_dataset)
    test_dataloader = Dataloader(test_dataset)
    
    return train_dataloader, test_dataloader


if (__name__ == "__main__") :

    dataset = Dataset()
    # print(dataset)
    ''' results
    Dataset ImageFolder
        Number of datapoints: {the number of data}
        Root location: ./data/spectrograms
        StandardTransform
    Transform: Compose(
                   Resize(size=(201, 81), interpolation=bilinear, max_size=None, antialias=True)
                   ToTensor()
               )
    '''
    
    class_map = dataset.class_to_idx
    # print(class_map)
    ''' results
    {'false': 0, 'true': 1}
    '''

    train_dataset, test_dataset = Split(dataset)
    # print(f"Train size = {len(train_dataset)}, Test size = {len(test_dataset)})

    train_classes = [label for _, label in train_dataset]
    # print(Counter(train_classes))
    ''' results
    Counter({0: {false number}, 1: {true number}})
    '''

    train_dataloader = Dataloader(train_dataset)
    test_dataset = Dataloader(test_dataset)
