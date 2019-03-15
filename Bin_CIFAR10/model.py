import torchvision.models as models
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
from torch.distributions.one_hot_categorical import OneHotCategorical
import torch.nn.functional as F
import math
import vgg

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

# ---------------------------------------------------
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'SE': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
    'Dec': ["Up", 512, 512, "Up", 512, 512, "Up", 256, 256, "Up", 128, 128, "Up", 64, 3],
} # "M": maxpooling

def make_layers(cfg, batch_norm=False):
  layers = []
  in_channels = 3
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)
  
def make_layers_dec(cfg, batch_norm=False):
  layers = []
  in_channels = 512
  for v in cfg:
    if v == 'Up':
      layers += [nn.UpsamplingNearest2d(scale_factor=2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)
  
class VGG19(nn.Module):
  def __init__(self, model=None, fixed=None):
    super(VGG19, self).__init__()
    self.features = make_layers(cfg["E"])
    self.features = torch.nn.DataParallel(self.features) # model wrapper that enables parallel GPU utilization
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(512, 512),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(512, 512),
      nn.ReLU(True),
      nn.Linear(512, 10),
    )
    if model:
     checkpoint = torch.load(model)
     self.load_state_dict(checkpoint["state_dict"])
    else:
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
          m.bias.data.zero_()
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

class SmallVGG19(nn.Module):
  def __init__(self, model=None, fixed=None):
    super(SmallVGG19, self).__init__()
    self.features = make_layers(cfg["SE"])
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256, 512),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(512, 512),
      nn.ReLU(True),
      nn.Linear(512, 10),
    )
    if model:
     checkpoint = torch.load(model)
     self.load_state_dict(checkpoint)
    else:
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
          m.bias.data.zero_()
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

class DVGG19(nn.Module):
  def __init__(self, model=None, fixed=None):
    super(DVGG19, self).__init__()
    self.classifier = nn.Sequential(
      nn.Linear(10, 512),
      nn.ReLU(True),
      nn.Linear(512, 512),
      nn.ReLU(True),
      nn.Linear(512, 512),
      nn.ReLU(True),
    )
    self.features = make_layers_dec(cfg["Dec"])
    if model:
     checkpoint = torch.load(model)
     self.load_state_dict(checkpoint)
    else:
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
          m.bias.data.zero_()
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
          
  def forward(self, x):
    x = self.classifier(x)
    x = x.view(x.size(0), 512, 1, 1)
    x = self.features(x)
    return x
# ---------------------------------------------------
class Transform2(nn.Module): # drop out
  def __init__(self):
    super(Transform2, self).__init__()
    self.drop = nn.Dropout(p=0.08)
  def forward(self, x):
    return self.drop(x)
    
class Transform4(nn.Module): # rand translation
  def __init__(self):
    super(Transform4, self).__init__()
    self.conv_trans = nn.Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False, groups=3)
    self.one_hot2 = OneHotCategorical(torch.Tensor([1/24., 1/24., 1/24., 1/24., 1/24.,
                                                    1/24., 1/24., 1/24., 1/24., 1/24.,
                                                    1/24., 1/24., 0.000, 1/24., 1/24.,
                                                    1/24., 1/24., 1/24., 1/24., 1/24.,
                                                    1/24., 1/24., 1/24., 1/24., 1/24.]))
  def forward(self, x):
    kernel = self.one_hot2.sample().view(1,5,5) # 1x5x5
    kernel = torch.stack([kernel] * 3).cuda() # 3x1x5x5
    self.conv_trans.weight = nn.Parameter(kernel)
    y = self.conv_trans(x)
    self.conv_trans.requires_grad = False
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
      # trans = np.arange(-2, 3) / 32. # 32: the width/height of the MNIST/CIFAR10 image is 32x32
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
    self.conv1 = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=3)
    kernel = [[-1, -1, -1], 
              [-1,  9, -1], 
              [-1, -1, -1]]
    kernel = torch.from_numpy(np.array(kernel)).float().view(1,3,3)
    kernel = torch.stack([kernel] * 3).cuda()
    self.conv1.weight = nn.Parameter(kernel)
    self.conv1.requires_grad = False
  
  def forward(self, x):
    return self.conv1(x)
    
class Transform10(nn.Module): # smooth
  def __init__(self):
    super(Transform10, self).__init__()
    kernel = [[1, 2, 1],
              [2, 4, 1],
              [1, 2, 1]] # Gaussian smoothing
    kernel = torch.from_numpy(np.array(kernel)).float().view(1,3,3) * 0.0625
    kernel = torch.stack([kernel] * 3).cuda()
    self.conv1 = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=3)
    self.conv1.weight = nn.Parameter(kernel)
    self.conv1.requires_grad = False
  
  def forward(self, x):
    return self.conv1(x)
    
class Transform8(nn.Module): # random transform combination
  def __init__(self):
    super(Transform8, self).__init__()
    self.T2  = Transform2()
    self.T4  = Transform4()
    self.T6  = Transform6()
    self.T7  = Transform7()
    self.T9  = Transform9()
    self.T10 = Transform10()
    self.transforms = []
    for name in dir(self):
      if name[0] == "T" and name[1:].isdigit():
        self.transforms.append(eval("self.%s" % name))
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
BE  = VGG19 # Big Encoder
Dec = DVGG19 # Decoder
SE  = SmallVGG19 # Small Encoder

class AutoEncoder_GAN4(nn.Module):
  def __init__(self, args):
    super(AutoEncoder_GAN4, self).__init__()
    self.be = BE(args.e1, fixed=True).eval()
    self.defined_trans = Transform8()
    for di in range(1, args.num_dec+1):
      pretrained_model = None
      if args.pretrained_dir:
        assert(args.pretrained_timeid != None)
        pretrained_model = [x for x in os.listdir(args.pretrained_dir) if "_d%s_" % di in x and args.pretrained_timeid in x] # the number of pretrained decoder should be like "SERVER218-20190313-1233_d3_E0S0.pth"
        assert(len(pretrained_model) == 1)
        pretrained_model = pretrained_model[0]
      self.__setattr__("d" + str(di), Dec(pretrained_model, fixed=False))
    for sei in range(1, args.num_se+1):
      self.__setattr__("se" + str(sei), SE(None, fixed=False))
      
AutoEncoders = {
"GAN4": AutoEncoder_GAN4,
}
  