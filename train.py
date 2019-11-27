import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse
import imp

ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

ap.add_argument('data_dir', action="store")
ap.add_argument('--gpu',action='store_true')
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs


image_datasets,dataloaders = imp.load_data(where)

model, optimizer, criterion = imp.nn_setup(structure,dropout,hidden_layer1,lr,power)

imp.train_network(model, optimizer, criterion,dataloaders,epochs, power)

imp.save_checkpoint(model,image_datasets,path,structure,hidden_layer1,dropout,lr,epochs)

print("The Model is trained") 