import torchvision.models as models
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
pjoin = os.path.join

# Use the LeNet model as https://github.com/iRapha/replayed_distillation/blob/master/models/lenet.py
class LeNet5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(LeNet5, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d( 1,  6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d( 6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.fc3 = nn.Linear(400, 120)
    self.fc4 = nn.Linear(120,  84)
    self.fc5 = nn.Linear( 84,  10)
    self.relu = nn.ReLU(inplace=True)
    
    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
      
  def forward(self, y):          # input: 1x32x32
    y = self.relu(self.conv1(y)) # 6x28x28
    y = self.pool1(y)            # 6x14x14
    y = self.relu(self.conv2(y)) # 16x10x10
    y = self.pool2(y)            # 16x5x5
    y = y.view(y.size(0), -1)    # 400
    y = self.relu(self.fc3(y))   # 120
    y = self.relu(self.fc4(y))   # 84
    y = self.fc5(y)              # 10
    return y
  
  def forward_branch(self, y):
    y = self.relu(self.conv1(y)); out1 = y
    y = self.pool1(y)
    y = self.relu(self.conv2(y)); out2 = y
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y)); out3 = y
    y = self.relu(self.fc4(y)); out4 = y
    y = self.fc5(y)
    return out1, out2, out3, out4, y
    
class DLeNet5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(DLeNet5, self).__init__()
    self.fixed = fixed
    
    self.fc5 = nn.Linear( 10,  84)
    self.fc4 = nn.Linear( 84, 120)
    self.fc3 = nn.Linear(120, 400)
    self.conv2 = nn.Conv2d(16, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)) # to maintain the spatial size, so padding=2
    self.conv1 = nn.Conv2d( 6, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((2,2,2,2))
    
    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
      
  def forward(self, y):          # input: 10
    y = self.relu(self.fc5(y))   # 84
    y = self.relu(self.fc4(y))   # 120
    y = self.relu(self.fc3(y))   # 400
    y = y.view(-1, 16, 5, 5)     # 16x5x5
    y = self.unpool(y)           # 16x10x10
    y = self.pad(y)              # 16x14x14
    y = self.relu(self.conv2(y)) # 6x14x14
    y = self.unpool(y)           # 6x28x28
    y = self.pad(y)              # 6x32x32
    y = self.relu(self.conv1(y)) # 1x32x32
    return y
    
class SmallLeNet5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallLeNet5, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d( 1,  3, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d( 3,  8, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.fc3 = nn.Linear(200, 120)
    self.fc4 = nn.Linear(120,  84)
    self.fc5 = nn.Linear( 84,  10)
    self.relu = nn.ReLU(inplace=True)
    
    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
      
  def forward(self, y):
    y = self.relu(self.conv1(y))
    y = self.pool1(y)
    y = self.relu(self.conv2(y))
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y))
    y = self.relu(self.fc4(y))
    y = self.fc5(y)
    return y
  
  def forward_branch(self, y):
    y = self.relu(self.conv1(y)); out1 = y
    y = self.pool1(y)
    y = self.relu(self.conv2(y)); out2 = y
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y)); out3 = y
    y = self.relu(self.fc4(y)); out4 = y
    y = self.fc5(y)
    return out1, out2, out3, out4, y

# ---------------------------------------------------
# AutoEncoder part
Encoder = LeNet5
Decoder = DLeNet5
SmallEncoder = SmallLeNet5

class AutoEncoder_BD(nn.Module):
  def __init__(self, e1=None, d=None, e2=None):
    super(AutoEncoder_BD, self).__init__()
    self.enc = Encoder(e1, fixed=True).eval() # note to use the 'eval' mode to keep dropout layer fixed if there is one
    self.dec = Decoder(d, fixed=False)
  
  def forward(self, code):
    img_rec1 = self.dec(code)
    feats1   = self.enc.forward_branch(img_rec1)
    img_rec2 = self.dec(feats1[-1])
    feats2   = self.enc.forward_branch(img_rec2)
    return img_rec1, feats1, img_rec2, feats2
    
class AutoEncoder_SE(nn.Module):
  def __init__(self, e1=None, d=None, e2=None):
    super(AutoEncoder_SE, self).__init__()
    self.enc = Encoder(e1, fixed=True).eval()
    self.dec = Decoder(d,  fixed=True).eval()
    self.small_enc = SmallEncoder(e2, fixed=False)
  
  def forward(self, code):
    img_rec1 = self.dec(code)
    feats1 = self.enc.forward_branch(img_rec1)
    small_feats1 = self.small_enc.forward_branch(img_rec1)
    feats2 = self.enc.forward_branch(self.dec(small_feats1[-1]))
    return feats1, small_feats1, feats2
    
AutoEncoders = {
"BD": AutoEncoder_BD,
"SE": AutoEncoder_SE,
}
  