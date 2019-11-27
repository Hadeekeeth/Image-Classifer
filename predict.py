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

ap = argparse.ArgumentParser(description='predict-file')
ap.add_argument('input_img',action="store", type = str)
ap.add_argument('checkpoint', action="store",type = str)
ap.add_argument('--top_k', dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint

image_datasets,dataloaders = imp.load_data()
model=imp.load_checkpoint(path)
with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
    

top_probs,classes = imp.predict(path_image, model,number_of_outputs, power)
class_names = [cat_to_name[item]for item in classes]

i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(class_names[i], top_probs[i]))
    i += 1

