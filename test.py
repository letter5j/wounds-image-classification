import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import visdom

from get_data import get_data_loaders

import numpy as np
import time
import copy
import os

from model.model_128_all_512_5 import build_model

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    dataloaders, dataset_sizes, class_names = get_data_loaders()
    inputs, labels = next(iter(dataloaders['test']))

    inputs = inputs.to(device)
    labels = labels.to(device)

    model, ftr = build_model('senet154')

    y = model(inputs)
    print(y)

    print(y.size())
    print(ftr)





