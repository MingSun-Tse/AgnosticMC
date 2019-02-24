import torchvision.models as models
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
pjoin = os.path.join

class AlexNet_Encoder(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(AlexNet_Encoder, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d(  3,  64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    self.conv2 = nn.Conv2d( 64, 192, kernel_size=( 5,  5), stride=(1, 1), padding=(2, 2))
    self.conv3 = nn.Conv2d(192, 384, kernel_size=( 3,  3), stride=(1, 1), padding=(1, 1))
    self.conv4 = nn.Conv2d(384, 256, kernel_size=( 3,  3), stride=(1, 1), padding=(1, 1))
    self.conv5 = nn.Conv2d(256, 256, kernel_size=( 3,  3), stride=(1, 1), padding=(1, 1))
    
    self.drop6 = nn.Dropout(p=0.5); self.fc6 = nn.Linear(9216, 4096)
    self.drop7 = nn.Dropout(p=0.5); self.fc7 = nn.Linear(4096, 4096)
    self.fc8 = nn.Linear(4096, 1000)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
      
  def forward(self, y):
    y = self.relu(self.conv1(y)); #print(y.shape)
    y = self.pool(y); #print(y.shape)
    y = self.relu(self.conv2(y)); #print(y.shape)
    y = self.pool(y); #print(y.shape)
    y = self.relu(self.conv3(y)); #print(y.shape)
    y = self.relu(self.conv4(y)); #print(y.shape)
    y = self.relu(self.conv5(y)); #print(y.shape)
    y = self.pool(y); #print(y.shape)
    y = y.view(y.size(0), -1); #print(y.shape)
    y = self.relu(self.fc6(self.drop6(y))); #print(y.shape)
    y = self.relu(self.fc7(self.drop7(y))); #print(y.shape)
    y = self.fc8(y); #print(y.shape)
    return y
  
  
  def forward_branch(self, y):
    y = self.pool(self.relu(self.conv1(y))); out1 = y
    y = self.pool(self.relu(self.conv2(y))); out2 = y
    y = self.relu(self.conv3(y)); out3 = y
    y = self.relu(self.conv4(y)); out4 = y
    y = self.pool(self.relu(self.conv5(y))); out5 = y
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc6(self.drop6(y))); out6 = y
    y = self.relu(self.fc7(self.drop7(y))); out7 = y
    y = self.fc8(y)
    return out1, out2, out3, out4, out5, out6, out7, y
    
class AlexNet_SmallEncoder(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(AlexNet_SmallEncoder, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d(  3,  32, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    self.conv2 = nn.Conv2d( 32,  96, kernel_size=( 5,  5), stride=(1, 1), padding=(2, 2))
    self.conv3 = nn.Conv2d( 96, 192, kernel_size=( 3,  3), stride=(1, 1), padding=(1, 1))
    self.conv4 = nn.Conv2d(192, 128, kernel_size=( 3,  3), stride=(1, 1), padding=(1, 1))
    self.conv5 = nn.Conv2d(128, 128, kernel_size=( 3,  3), stride=(1, 1), padding=(1, 1))
    
    self.drop6 = nn.Dropout(p=0.5); self.fc6 = nn.Linear(4608, 2048)
    self.drop7 = nn.Dropout(p=0.5); self.fc7 = nn.Linear(2048, 2048)
    self.fc8 = nn.Linear(2048, 1000)
    
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
      
  def forward(self, y):
    y = self.relu(self.conv1(y))
    y = self.pool(y)
    y = self.relu(self.conv2(y))
    y = self.pool(y)
    y = self.relu(self.conv3(y))
    y = self.relu(self.conv4(y))
    y = self.relu(self.conv5(y))
    y = self.pool(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc6(self.drop6(y)))
    y = self.relu(self.fc7(self.drop7(y)))
    y = self.fc8(y)
    return y
  
  
  def forward_branch(self, y):
    y = self.pool(self.relu(self.conv1(y))); out1 = y
    y = self.pool(self.relu(self.conv2(y))); out2 = y
    y = self.relu(self.conv3(y)); out3 = y
    y = self.relu(self.conv4(y)); out4 = y
    y = self.pool(self.relu(self.conv5(y))); out5 = y
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc6(self.drop6(y))); out6 = y
    y = self.relu(self.fc7(self.drop7(y))); out7 = y
    y = self.fc8(y)
    return out1, out2, out3, out4, out5, out6, out7, y
    
    
class AlexNet_Decoder(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(AlexNet_Decoder, self).__init__()
    self.fixed = fixed
    self.fc8 = nn.Linear(1000, 4096)
    self.fc7 = nn.Linear(4096, 4096)
    self.fc6 = nn.Linear(4096, 9216)
    
    # The stride below has to be 1, since larger than 1 can only lead to decreased spatial size. Keep the kernel size unchanged.
    self.conv5 = nn.Conv2d(256, 256, kernel_size=( 3,  3), stride=(1, 1), padding=(0, 0)) # will be padded with (1,1) to make the spatial size not to change
    self.conv4 = nn.Conv2d(256, 384, kernel_size=( 3,  3), stride=(1, 1), padding=(0, 0)) # will be padded with (1,1) to make the spatial size not to change
    self.conv3 = nn.Conv2d(384, 192, kernel_size=( 3,  3), stride=(1, 1), padding=(0, 0)) # will be padded with (1,1) to make the spatial size not to change
    self.conv2 = nn.Conv2d(192,  64, kernel_size=( 5,  5), stride=(1, 1), padding=(0, 0)) # will be padded with (2,2) to make the spatial size not to change
    self.conv1 = nn.Conv2d( 64,   3, kernel_size=(11, 11), stride=(1, 1), padding=(0, 0)) # will be padded with (2,2) to make the spatial size not to change
    
    self.relu = nn.ReLU(inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad1 = nn.ReflectionPad2d((1,1,1,1))
    self.pad2 = nn.ReflectionPad2d((2,2,2,2))
    self.pad5 = nn.ReflectionPad2d((5,5,5,5))
    
    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
      
  def forward(self, y):
    y = self.relu(self.fc8(y))
    y = self.relu(self.fc7(y))
    y = self.relu(self.fc6(y))
    y = y.view(-1, 256, 6, 6)
    y = self.unpool(y)                      # (1, 256, 12, 12)
    y = self.pad1(y)                        # (1, 256, 14, 14)
    y = self.relu(self.conv5(self.pad1(y))) # (1, 256, 14, 14)
    y = self.unpool(y)                      # (1, 256, 28, 28)
    y = self.relu(self.conv4(self.pad1(y))) # (1, 384, 28, 28)
    y = self.unpool(y)                      # (1, 256, 56, 56)
    y = self.relu(self.conv3(self.pad1(y))) # (1, 192, 56, 56)
    y = self.unpool(y)                      # (1, 192,112,112)
    y = self.relu(self.conv2(self.pad2(y))) # (1,  64,112,112)
    y = self.unpool(y)                      # (1,  64,224,224)
    y = self.relu(self.conv1(self.pad5(y))) # (1,   3,224,224)
    return y

    
class AutoEncoder_BD(nn.Module):
  def __init__(self, e1=None, d=None, e2=None):
    super(AutoEncoder_BD, self).__init__()
    self.enc = AlexNet_Encoder(e1, fixed=True).eval() # note to use the 'eval' mode to keep dropout fixed
    self.dec = AlexNet_Decoder(d, fixed=False)
  def forward(self, code):
    feats1 = self.enc.forward_branch(self.dec(code))
    feats2 = self.enc.forward_branch(self.dec(feats1[-1]))
    return feats1, feats2
    
class AutoEncoder_SE(nn.Module):
  def __init__(self, e1=None, d=None, e2=None):
    super(AutoEncoder_SE, self).__init__()
    self.enc = AlexNet_Encoder(e1, fixed=True).eval()
    self.dec = AlexNet_Decoder(d,  fixed=True).eval()
    self.small_enc = AlexNet_SmallEncoder(e2, fixed=False)
  def forward(self, code):
    img1 = self.dec(code)
    feats1 = self.enc.forward_branch(img1)
    small_feats1 = self.small_enc.forward_branch(img1)
    feats2 = self.enc.forward_branch(self.dec(small_feats1[-1]))
    return feats1, small_feats1, feats2
    
AutoEncoders = {
"BD": AutoEncoder_BD,
"SE": AutoEncoder_SE,
}
  