import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os
import CNN_v1
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def load_split_train_valid_test(datadir, valid_proportion=0.10, test_proportion=0.2, bs=32, random_state=9999):
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor(),
                                           ])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          ])
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                                transforms.ToTensor(),
                                                ])
    train_data = datasets.ImageFolder(datadir,
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    valid_data = datasets.ImageFolder(datadir,
                    transform=valid_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    test_size = int(np.floor(test_proportion * num_train))
    valid_size = int(np.floor(valid_proportion * num_train))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    train_idx, test_idx, valid_idx = indices[test_size+valid_size:], indices[:test_size], indices[test_size:test_size+valid_size]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=bs)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=bs)
    validloader = torch.utils.data.DataLoader(valid_data,
                   sampler=valid_sampler, batch_size=bs)
    return trainloader, testloader, validloader