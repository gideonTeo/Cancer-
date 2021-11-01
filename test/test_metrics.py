from numpy.random.mtrand import shuffle
import torch
from torch import optim,nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import os
import numpy as np
import pandas as pd
import time
import matplotlib
import wandb
import sklearn
from torch.utils.data.sampler import SubsetRandomSampler

test_transforms = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()])
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
testloader = torch.utils.data.DataLoader(test_data, batch_size=bs,shuffle=True)
