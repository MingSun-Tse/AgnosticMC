import torchvision.models as models
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
from torch.distributions.one_hot_categorical import OneHotCategorical
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
    
class DLeNet5_drop(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(DLeNet5_drop, self).__init__()
    self.fixed = fixed
    
    self.fc5 = nn.Linear( 10,  84); self.drop5 = nn.Dropout(p=0.5)
    self.fc4 = nn.Linear( 84, 120); self.drop4 = nn.Dropout(p=0.5)
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
 
class Transform(nn.Module):
  def __init__(self):
    super(Transform, self).__init__()
    kernel = [[[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]]
    kernel = torch.from_numpy(np.array(kernel)).float()
    self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv1.weight = nn.Parameter(kernel)
    self.drop = nn.Dropout(p=0.05)
                                     
    for param in self.parameters():
      param.requires_grad = False
  
  def forward(self, x):
    y = self.conv1(x)
    y = (x + self.drop(y)) / 2
    return y
    
class Transform2(nn.Module): # sharpen
  def __init__(self):
    super(Transform2, self).__init__()
    kernel = [[[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]]]
    kernel = torch.from_numpy(np.array(kernel)).float()
    self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv1.weight = nn.Parameter(kernel)
    self.drop = nn.Dropout(p=0.05)
                                     
    for param in self.parameters():
      param.requires_grad = False
  
  def forward(self, x):
    y = self.conv1(x)
    y = (x + self.drop(y)) / 2
    return y
    
class Transform3(nn.Module): # translation
  def __init__(self):
    super(Transform3, self).__init__()
    kernel_left = [[[[0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]]]
    kernel_left = torch.from_numpy(np.array(kernel_left)).float()
    self.conv_left = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv_left.weight = nn.Parameter(kernel_left)
    
    kernel_right = [[[[0, 0, 0, 0, 0], 
                      [0, 0, 0, 0, 0], 
                      [1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]]]]
    kernel_right = torch.from_numpy(np.array(kernel_right)).float()
    self.conv_right = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv_right.weight = nn.Parameter(kernel_right)
    
    kernel_up = [[[[0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0], 
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0]]]]
    kernel_up = torch.from_numpy(np.array(kernel_up)).float()
    self.conv_up = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv_up.weight = nn.Parameter(kernel_up)
           
    kernel_down = [[[[0, 0, 1, 0, 0], 
                     [0, 0, 0, 0, 0], 
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]]]
    kernel_down = torch.from_numpy(np.array(kernel_down)).float()
    self.conv_down = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv_down.weight = nn.Parameter(kernel_down)  
    
    kernel5 = [[[[1, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]]]
    kernel5 = torch.from_numpy(np.array(kernel5)).float()
    self.conv5 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv5.weight = nn.Parameter(kernel5)
    
    kernel6 = [[[[0, 0, 0, 0, 1], 
                 [0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]]]
    kernel6 = torch.from_numpy(np.array(kernel6)).float()
    self.conv6 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv6.weight = nn.Parameter(kernel6)
    
    kernel7 = [[[[0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0]]]]
    kernel7 = torch.from_numpy(np.array(kernel7)).float()
    self.conv7 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv7.weight = nn.Parameter(kernel7)
    
    kernel8 = [[[[0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0], 
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1]]]]
    kernel8 = torch.from_numpy(np.array(kernel8)).float()
    self.conv8 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv8.weight = nn.Parameter(kernel8)
    self.one_hot1 = OneHotCategorical(torch.Tensor([1./25] * 25))
    
    for param in self.parameters():
      param.requires_grad = False
  
  def forward(self, x):
    switch = self.one_hot1.sample().cuda()
    y = self.conv_left(x) * switch[0] + self.conv_right(x) * switch[1] + \
        self.conv_up(x)   * switch[2] + self.conv_down(x)  * switch[3]
    return y
    
class Transform4(nn.Module): # rand translation
  def __init__(self):
    super(Transform4, self).__init__()
    kernel = [[[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]]
    kernel = torch.from_numpy(np.array(kernel)).float()
    self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv1.weight = nn.Parameter(kernel)
    self.conv_trans  = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.conv_smooth = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv_smooth.weight = nn.Parameter(torch.ones(9).cuda().view(1,1,3,3) * 1/9.)
    self.drop = nn.Dropout(p=0.05)
    
    self.one_hot1 = OneHotCategorical(torch.Tensor([0.4, 0.2, 0.4]))
    self.one_hot2 = OneHotCategorical(torch.Tensor([0.0625, 0.0250, 0.0625, 0.0250, 0.0625,
                                                    0.0250, 0.0375, 0.0375, 0.0375, 0.0250,
                                                    0.0625, 0.0375, 0.0000, 0.0375, 0.0625,
                                                    0.0250, 0.0375, 0.0375, 0.0375, 0.0250,
                                                    0.0625, 0.0250, 0.0625, 0.0250, 0.0625])) 
    
    for param in self.parameters():
      param.requires_grad = False
  
  def forward(self, x):
    # random translation 
    self.conv_trans.weight = nn.Parameter(self.one_hot2.sample().cuda().view(1,1,5,5))
    y1 = self.conv_trans(x)
    
    # smooth
    y2 = self.conv_smooth(x)
    
    # data dropout
    y3 = (self.drop(self.conv1(x)) + x) / 2.
    
    switch = self.one_hot1.sample().cuda()
    y = y1 * switch[0] + y2 * switch[1] + y3 * switch[2]
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
    self.transform = Transform4()
  def forward(self, code):
    img_rec1     = self.dec(code);                               img_rec1_trans = self.transform(img_rec1)
    feats1       = self.enc.forward_branch(img_rec1);             logits1_trans = self.enc(img_rec1_trans)
    small_feats1 = self.small_enc.forward_branch(img_rec1); small_logits1_trans = self.small_enc(self.transform(img_rec1))
    img_rec2 = self.dec(feats1[-1])
    feats2   = self.enc.forward_branch(img_rec2)
    return img_rec1, feats1, logits1_trans, small_feats1, small_logits1_trans, img_rec2, feats2
    
AutoEncoders = {
"BD": AutoEncoder_BD,
"SE": AutoEncoder_SE,
"BDSE": AutoEncoder_BDSE_Trans,
}
  