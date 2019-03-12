import torchvision.models as models
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
from torch.distributions.one_hot_categorical import OneHotCategorical
import torch.nn.functional as F
import math
pjoin = os.path.join

# Exponential Moving Average
class EMA():
  def __init__(self, mu):
    self.mu = mu
    self.shadow = {}
  def register(self, name, val):
    self.shadow[name] = val.clone()
  def __call__(self, name, x):
    assert name in self.shadow
    new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
    self.shadow[name] = new_average.clone()
    return new_average

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

class LeNet5_drop(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(LeNet5_drop, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d( 1,  6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d( 6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.fc3 = nn.Linear(400, 120); self.drop3 = nn.Dropout(p=0.5)
    self.fc4 = nn.Linear(120,  84); self.drop4 = nn.Dropout(p=0.5)
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
    y = self.relu(self.drop3(self.fc3(y)))   # 120
    y = self.relu(self.drop4(self.fc4(y)))   # 84
    y = self.fc5(y)              # 10
    return y
  
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
    
class DLeNet5_drop(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(DLeNet5_drop, self).__init__()
    self.fixed = fixed
    
    self.fc5 = nn.Linear( 10,  84); self.drop5 = nn.Dropout(p=0.1)
    self.fc4 = nn.Linear( 84, 120); self.drop4 = nn.Dropout(p=0.1)
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
    y = self.drop5(y)
    y = self.relu(self.fc4(y))   # 120
    y = self.drop4(y)
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

class LearnedTransform(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(LearnedTransform, self).__init__()
    self.fixed = fixed
    
    self.conv11 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv12 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv13 = nn.Conv2d(8, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    self.conv21 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv22 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv23 = nn.Conv2d(8, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    self.conv31 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv32 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv33 = nn.Conv2d(8, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    self.conv41 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv42 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv43 = nn.Conv2d(8, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    self.relu = nn.ReLU(inplace=True)
    self.trans = Transform8()
    
    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
      
  def forward(self, x):
    y = self.relu(self.conv11(x))
    y = self.relu(self.conv12(y))
    x = (self.relu(self.conv13(y)) + x) / 2
    y = self.relu(self.conv21(x))
    y = self.relu(self.conv22(y))
    x = (self.relu(self.conv23(y)) + x) / 2
    y = self.relu(self.conv31(x))
    y = self.relu(self.conv32(y))
    x = (self.relu(self.conv33(y)) + x) / 2
    y = self.relu(self.conv41(x))
    y = self.relu(self.conv42(y))
    x = (self.relu(self.conv43(y)) + x) / 2
    return x
    
# ---------------------------------------------------
class Transform1(nn.Module):
  def __init__(self):
    super(Transform1, self).__init__()
    kernel = [[[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]]
    kernel = torch.from_numpy(np.array(kernel)).float()
    self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.conv1.weight = nn.Parameter(kernel)
    self.drop = nn.Dropout(p=0.08)
                                     
    for param in self.parameters():
      param.requires_grad = False
  
  def forward(self, x):
    y = self.conv1(x)
    y = (x + self.drop(y)) / 2
    return y
    
class Transform2(nn.Module): # drop out
  def __init__(self):
    super(Transform2, self).__init__()
    self.drop = nn.Dropout(p=0.08)
  def forward(self, x):
    return self.drop(x)
    
class Transform3(nn.Module): # 8-direction translation
  def __init__(self):
    super(Transform3, self).__init__()
    kernel_left = [[[[0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]]]
    kernel_left = torch.from_numpy(np.array(kernel_left)).float()
    self.conv_left = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv_left.weight = nn.Parameter(kernel_left)
    
    kernel_right = [[[[0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0], 
                      [1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]]]
    kernel_right = torch.from_numpy(np.array(kernel_right)).float()
    self.conv_right = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv_right.weight = nn.Parameter(kernel_right)
    
    kernel_up = [[[[0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0]]]]
    kernel_up = torch.from_numpy(np.array(kernel_up)).float()
    self.conv_up = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv_up.weight = nn.Parameter(kernel_up)
           
    kernel_down = [[[[0, 0, 1, 0, 0], 
                     [0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]]]
    kernel_down = torch.from_numpy(np.array(kernel_down)).float()
    self.conv_down = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv_down.weight = nn.Parameter(kernel_down)  
    
    kernel5 = [[[[1, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]]]
    kernel5 = torch.from_numpy(np.array(kernel5)).float()
    self.conv5 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv5.weight = nn.Parameter(kernel5)
    
    kernel6 = [[[[0, 0, 0, 0, 1], 
                 [0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]]]
    kernel6 = torch.from_numpy(np.array(kernel6)).float()
    self.conv6 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv6.weight = nn.Parameter(kernel6)
    
    kernel7 = [[[[0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0]]]]
    kernel7 = torch.from_numpy(np.array(kernel7)).float()
    self.conv7 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv7.weight = nn.Parameter(kernel7)
    
    kernel8 = [[[[0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1]]]]
    kernel8 = torch.from_numpy(np.array(kernel8)).float()
    self.conv8 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv8.weight = nn.Parameter(kernel8)
    self.one_hot1 = OneHotCategorical(torch.Tensor([1./8] * 8))
    
    for param in self.parameters():
      param.requires_grad = False
  
  def forward(self, x):
    switch = self.one_hot1.sample().cuda()
    y = self.conv_left(x) * switch[0] + self.conv_right(x) * switch[1] + \
        self.conv_up(x)   * switch[2] + self.conv_down(x)  * switch[3] + \
        self.conv5(x)     * switch[4] + self.conv6(x)      * switch[5] + \
        self.conv7(x)     * switch[6] + self.conv8(x)      * switch[7]
    return y
    
class Transform4(nn.Module): # rand translation
  def __init__(self):
    super(Transform4, self).__init__()
    self.conv_trans  = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True)
    self.one_hot2 = OneHotCategorical(torch.Tensor([1/24., 1/24., 1/24., 1/24., 1/24.,
                                                    1/24., 1/24., 1/24., 1/24., 1/24.,
                                                    1/24., 1/24., 0.000, 1/24., 1/24.,
                                                    1/24., 1/24., 1/24., 1/24., 1/24.,
                                                    1/24., 1/24., 1/24., 1/24., 1/24.]))
  def forward(self, x):
    self.conv_trans.weight = nn.Parameter(self.one_hot2.sample().cuda().view(1,1,5,5))
    y = self.conv_trans(x)
    self.conv_trans.requires_grad = False
    return y
    
class Transform5(nn.Module): # combine
  def __init__(self):
    super(Transform5, self).__init__()
    kernel = [[[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]]
    kernel = torch.from_numpy(np.array(kernel)).float()
    self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.conv1.weight = nn.Parameter(kernel)
    
    self.conv_trans1 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv_trans2 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv_trans3 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    self.conv_trans4 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
    
    self.conv_smooth = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.conv_smooth.weight = nn.Parameter(torch.ones(9).cuda().view(1,1,3,3) * 1/9.)
    self.drop = nn.Dropout(p=0.05)
    self.relu = nn.ReLU(inplace=True)
    
    self.one_hot1 = OneHotCategorical(torch.Tensor([0.6, 0.4]))
    self.one_hot2 = OneHotCategorical(torch.Tensor([1/24., 1/24., 1/24., 1/24., 1/24.,
                                                    1/24., 1/24., 1/24., 1/24., 1/24.,
                                                    1/24., 1/24., 0.000, 1/24., 1/24.,
                                                    1/24., 1/24., 1/24., 1/24., 1/24.,
                                                    1/24., 1/24., 1/24., 1/24., 1/24.])) 
    
    for param in self.parameters():
      param.requires_grad = False
  
  def forward(self, x):
    # random translation 
    self.conv_trans1.weight = nn.Parameter(self.one_hot2.sample().cuda().view(1,1,5,5))
    self.conv_trans2.weight = nn.Parameter(self.one_hot2.sample().cuda().view(1,1,5,5))
    y1 = self.conv_trans2(self.conv_trans1(x)) # equivalent to random crop
    
    # smooth
    # y2 = self.conv_smooth(x)
    
    # data dropout
    y3 = (self.drop(self.conv1(x)) + x) / 2.
    
    # gaussian noise
    # y4 = self.relu(torch.randn_like(x).cuda() * torch.mean(x) * 0.01)
    
    switch = self.one_hot1.sample().cuda()
    y = y1 * switch[0] + y3 * switch[1]
    
    for param in self.parameters():
      param.requires_grad = False
    return y
     
class Transform6(nn.Module): # resize or scale
  def __init__(self):
    super(Transform6, self).__init__()
    
  def forward(self, x):
    rand_scale = np.random.rand() * 0.05 + 1.03125
    y = F.interpolate(x, scale_factor=rand_scale)
    new_width = int(rand_scale*32)
    w = np.random.randint(new_width-32); h = np.random.randint(new_width-32)
    rand_crop = y[:, :, w:w+32, h:h+32]
    return rand_crop
     
class Transform7(nn.Module): # rotate
  def __init__(self):
    super(Transform7, self).__init__()
    
  def forward(self, x):
    theta = []
    for _ in range(x.shape[0]):
      angle = np.random.randint(-5, 6) / 180.0 * math.pi
      # trans = np.arange(-2, 3) / 32. # 32: the width/height of the MNIST image is 32x32
      # trans1 = trans[np.random.randint(len(trans))]
      # trans2 = trans[np.random.randint(len(trans))]
      theta.append([[math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle),  math.cos(angle), 0]])
    theta = torch.from_numpy(np.array(theta)).float().cuda()
    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid)
    return x
    
class Transform9(nn.Module): # sharpen
  def __init__(self):
    super(Transform9, self).__init__()
    kernel = [[[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]]
    kernel = torch.from_numpy(np.array(kernel)).float()
    self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.conv1.weight = nn.Parameter(kernel)
    self.conv1.requires_grad = False
  
  def forward(self, x):
    return self.conv1(x)
    
class Transform10(nn.Module): # smooth
  def __init__(self):
    super(Transform10, self).__init__()
    kernel = [[[[1, 2, 1], [2, 4, 1], [1, 2, 1]]]] # Gaussian smoothing
    kernel = torch.from_numpy(np.array(kernel)).float() * 0.0625
    self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.conv1.weight = nn.Parameter(kernel)
    self.conv1.requires_grad = False
  
  def forward(self, x):
    return self.conv1(x)
    
class Transform11(nn.Module): # Gaussian smoothing
  def __init__(self):
    super(Transform11, self).__init__()
    kernel = [[[[1, 2, 1], [2, 4, 1], [1, 2, 1]]]] 
    kernel = torch.from_numpy(np.array(kernel)).float() * 0.0625
    self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.conv1.weight = nn.Parameter(kernel)
    self.conv1.requires_grad = False
  
  def forward(self, x):
    return self.conv1(x)
    
class Transform8(nn.Module): # random transform combination
  def __init__(self):
    super(Transform8, self).__init__()
    self.T2 = Transform2()
    self.T4 = Transform4()
    self.T6 = Transform6()
    self.T7 = Transform7()
    self.T9 = Transform9()
    self.T10 = Transform10()
    self.transforms = [self.T2, self.T4, self.T6, self.T7, self.T9, self.T10]
    # for name, value in vars(self).items():
      # print(name)
      # if name[0] == "T" and name[1:].isdigit():
        # self.transforms.append(eval("self.%s" % name))
    self.transforms = np.array(self.transforms)
    print(self.transforms)
    
  def forward(self, y):
    rand = np.random.permutation(len(self.transforms))
    Ts = self.transforms[rand]
    for T in Ts:
      if np.random.rand() >= 0.5:
        y = T(y)
    return y    

# ---------------------------------------------------
# AutoEncoder part
Encoder = LeNet5
Decoder = DLeNet5
SmallEncoder = SmallLeNet5
AdvEncoder = LeNet5_drop

class AutoEncoder_BD(nn.Module):
  def __init__(self, e1=None, d=None, e2=None):
    super(AutoEncoder_BD, self).__init__()
    self.enc = Encoder(e1, fixed=True).eval() # note to use the 'eval' mode to keep dropout layer fixed if there is one
    self.dec = Decoder(d, fixed=False)
  def forward(self, code):
    img_rec1 = self.dec(code);       feats1 = self.enc.forward_branch(img_rec1)
    img_rec2 = self.dec(feats1[-1]); feats2 = self.enc.forward_branch(img_rec2)
    return img_rec1, feats1, img_rec2, feats2
    
class AutoEncoder_SE(nn.Module):
  def __init__(self, e1=None, d=None, e2=None):
    super(AutoEncoder_SE, self).__init__()
    self.enc = Encoder(e1, fixed=True).eval()
    self.dec = Decoder(d,  fixed=False).eval()
    self.small_enc = SmallEncoder(e2, fixed=False)
  def forward(self, code):
    img_rec1 = self.dec(code)
    feats1   = self.enc.forward_branch(img_rec1); small_feats1 = self.small_enc.forward_branch(img_rec1)
    img_rec2 = self.dec(small_feats1[-1])
    feats2   = self.enc.forward_branch(img_rec2)
    return img_rec1, feats1, small_feats1, img_rec2, feats2
    
class AutoEncoder_BDSE(nn.Module):
  def __init__(self, e1=None, d=None, e2=None):
    super(AutoEncoder_BDSE, self).__init__()
    self.enc = Encoder(e1, fixed=True).eval()
    self.dec = Decoder(d,  fixed=False) # decoder is also trainable
    self.small_enc = SmallEncoder(e2, fixed=False)  
  def forward(self, code):
    img_rec1 = self.dec(code)
    feats1   = self.enc.forward_branch(img_rec1); small_feats1 = self.small_enc.forward_branch(img_rec1)
    img_rec2 = self.dec(feats1[-1])
    feats2   = self.enc.forward_branch(img_rec2)
    return img_rec1, feats1, small_feats1, img_rec2, feats2
    
class AutoEncoder_BDSE_Trans(nn.Module):
  def __init__(self, e1=None, d=None, e2=None):
    super(AutoEncoder_BDSE_Trans, self).__init__()
    self.enc = Encoder(e1, fixed=True).eval()
    self.dec = Decoder(d,  fixed=False) # decoder is also trainable
    self.small_enc = SmallEncoder(e2, fixed=False)
    self.transform = Transform8()
  
    # -----Spatial Transformer Network --------------
    # Spatial transformer localization-network
    self.localization = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=7),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU(True),
        nn.Conv2d(8, 10, kernel_size=5),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU(True)
    )
    # Regressor for the 3 * 2 affine matrix
    self.fc_loc = nn.Sequential(
        nn.Linear(10 * 4 * 4, 32),
        nn.ReLU(True),
        nn.Linear(32, 3 * 2)
    )
    # Initialize the weights/bias with identity transformation
    self.fc_loc[2].weight.data.zero_()
    self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    # -----------------------------------------------
    
  # Spatial transformer network forward function
  def stn(self, x):
    xs = self.localization(x) # shape: batch x 10 x 4 x 4
    xs = xs.view(-1, 10 * 4 * 4)
    theta = self.fc_loc(xs) # batch x 6
    theta = theta.view(-1, 2, 3)
    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid)
    return x
  
  def forward(self, code):
    img_rec1     = self.dec(code);                               img_rec1_trans = self.transform(img_rec1)
    feats1       = self.enc.forward_branch(img_rec1);             logits1_trans = self.enc(img_rec1_trans)
    small_feats1 = self.small_enc.forward_branch(img_rec1); small_logits1_trans = self.small_enc(img_rec1_trans.data) # note that gradients should not be passed through img_rec1
    img_rec2 = self.dec(feats1[-1])
    feats2   = self.enc.forward_branch(img_rec2)
    return img_rec1, feats1, logits1_trans, small_feats1, small_logits1_trans, img_rec2, feats2
    
class AutoEncoder_BDSE_GAN(nn.Module):
  def __init__(self, e1=None, d=None, e2=None):
    super(AutoEncoder_BDSE_GAN, self).__init__()
    self.enc = Encoder(e1, fixed=True).eval()
    self.dec = Decoder(d,  fixed=False) # decoder is also trainable
    self.advbe = Encoder(None, fixed=False) # adversarial encoder
    self.small_enc = SmallEncoder(e2, fixed=False)
    self.transform = Transform8()

# Spatial Transformer Network
class STN(nn.Module):
  def __init__(self):
    super(STN, self).__init__()
    # Spatial transformer localization-network
    self.localization = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=7),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU(True),
        nn.Conv2d(8, 10, kernel_size=5),
        nn.MaxPool2d(2, stride=2),
        nn.ReLU(True)
    )
    # Regressor for the 3 * 2 affine matrix
    self.fc_loc = nn.Sequential(
        nn.Linear(10 * 4 * 4, 32),
        nn.ReLU(True),
        nn.Linear(32, 3 * 2)
    )
    # Initialize the weights/bias with identity transformation
    self.fc_loc[2].weight.data.zero_()
    self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
  
  # Spatial transformer network forward function
  def forward(self, x):
    xs = self.localization(x) # shape: batch x 10 x 4 x 4
    xs = xs.view(-1, 10 * 4 * 4)
    theta = self.fc_loc(xs) # batch x 6
    theta = theta.view(-1, 2, 3)
    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid)
    return x


class AutoEncoder_BDSE_GAN2(nn.Module):
  def __init__(self, e1=None, d=None, e2=None, trans_model=None):
    super(AutoEncoder_BDSE_GAN2, self).__init__()
    self.enc = Encoder(e1, fixed=True).eval()
    self.dec = Decoder(d,  fixed=False)
    self.small_enc = SmallEncoder(e2, fixed=False)
    self.defined_trans = Transform8()
    self.advbe  = AdvEncoder(None, fixed=False); self.learned_trans = STN() # LearnedTransform(trans_model, fixed=False)
    self.advbe2 = AdvEncoder(None, fixed=False); self.learned_trans2 = LearnedTransform(trans_model, fixed=False)
    
AutoEncoders = {
"BD": AutoEncoder_BD,
"SE": AutoEncoder_SE,
"BDSE": AutoEncoder_BDSE_Trans,
"BDSE_GAN": AutoEncoder_BDSE_GAN2,
}
  