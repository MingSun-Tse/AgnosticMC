from __future__ import print_function
import sys
import os
pjoin = os.path.join
import shutil
import time
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import scipy.io as sio
import math
# torch
import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
# my libs
from model import AlexNet_Encoder

# Passed-in params
parser = argparse.ArgumentParser(description="")
parser.add_argument('-m', '--mode', type=str)
parser.add_argument('--model', type=str)
opt = parser.parse_args()

prune_ratio = {
"conv1": 0.5,
"conv2": 0.5,
"conv3": 0.5,
"conv4": 0.5,
"conv5": 0.5,
"fc6": 0.5,
"fc7": 0.5,
"fc8": 0.5,
}

# Load model
model = AlexNet_Encoder(opt.model)

dict_param = model.state_dict()
for tensor_name in dict_param:
  if "bias" in tensor_name: 
    continue # do not prune biases
  weight = dict_param[tensor_name].data.numpy()
  w_flat = weight.flatten()
  w_abs = np.abs(w_flat)
  print("\n" + "*"*20, tensor_name)
  print(np.sort(w_abs[:20]))
  # Get the order
  layer_name = tensor_name.split(".weight")[0]
  num_weight_to_prune = int(prune_ratio[layer_name] * len(w_flat))
  order = np.argsort(w_abs)[:num_weight_to_prune]
  # Prune
  w_flat[order] = 0
  # Save
  weight = w_flat.reshape(weight.shape)
  dict_param[tensor_name].data.copy_(torch.from_numpy(weight))
torch.save(dict_param, "pruned_model.pth")



  