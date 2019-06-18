import torchvision.models as models
import torchvision.models as models
import numpy as np
import os
import copy
import torch.nn as nn
import torch
from torch.utils.serialization import load_lua
from torch.distributions.one_hot_categorical import OneHotCategorical
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
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

################# CIFAR10 #################
def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((512, 512))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var
    
def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im
# ---------------------------------------------------
cifar10_vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'SE_backup': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
    'SE': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'Dec':       ["Up", 512, 512, "Up", 512, 512, "Up", 256, 256, "Up", 128, 128, "Up", 64, 3],
    'Dec_s':     ["Up", 128, 128, "Up", 128, 128, "Up",  64,  64, "Up",  32,  32, "Up", 16, 3],
    'Dec_s_aug': ["Up", 128, 128, "Up", 128, 128, "Up",  64,  64, "Up", "64-2", "32x-4", "Up", "16x-x", "3x-x"],
    'Dec_gray':  ["Up", 512, 512, "Up", 512, 512, "Up", 256, 256, "Up", 128, 128, "Up", 64, 1],
}

def make_layers(cifar10_vgg_cfg, batch_norm=False):
  layers = []
  in_channels = 3
  for v in cifar10_vgg_cfg:
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
  
def make_layers_dec(cifar10_vgg_cfg, batch_norm=False):
  layers = []
  in_channels = 512
  for v in cifar10_vgg_cfg:
    if v == 'Up':
      layers += [nn.UpsamplingNearest2d(scale_factor=2)]
    else: # conv layer
      if str(v).isdigit():
        v = v
        g = 1
      else:
        g = int(v.split("g")[1])
        v = int(v.split("-")[0])
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, groups=g)
      if batch_norm:
        if v == cifar10_vgg_cfg[-1]:
          layers += [conv2d, nn.BatchNorm2d(v), nn.Sigmoid()] # normalize output image to [0, 1]
        else: 
          layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        if v == cifar10_vgg_cfg[-1]:
          layers += [conv2d, nn.Sigmoid()] # normalize output image to [0, 1]
        else: 
          layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)
 
def make_layers_augdec(cifar10_vgg_cfg, batch_norm=False, num_divbranch=1):
  layers = []
  in_channels = 512
  for v in cifar10_vgg_cfg:
    if v == 'Up':
      layers += [nn.UpsamplingNearest2d(scale_factor=2)]
    else: # conv layer
      if str(v).isdigit():
        group = 1
      else:
        num_filter, group = v.split("-")
        v = int(num_filter) if num_filter.isdigit() else int(num_filter.split("x")[0]) * num_divbranch
        group = int(group) if group.isdigit() else num_divbranch
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, groups=group)
      if batch_norm:
        if v == cifar10_vgg_cfg[-1]:
          layers += [conv2d, nn.BatchNorm2d(v), nn.Sigmoid()] # normalize output image to [0, 1]
        else: 
          layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        if v == cifar10_vgg_cfg[-1]:
          layers += [conv2d, nn.Sigmoid()] # normalize output image to [0, 1]
        else: 
          layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)
  
class VGG19(nn.Module):
  def __init__(self, model=None, fixed=None):
    super(VGG19, self).__init__()
    self.features = make_layers(cifar10_vgg_cfg["E"])
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
    # get layers for forward_branch
    self.branch_layer = ["f0"] # Convx_1. The first layer, i.e., Conv1_1 is included in default.
    self.features_num_module = len(self.features.module)
    
    for i in range(1, self.features_num_module):
      m = self.features.module[i-1]
      if isinstance(m, nn.MaxPool2d):
        self.branch_layer.append("f" + str(i))
      if i == self.features_num_module - 2: # for Huawei's idea
        self.branch_layer.append("f" + str(i))
    
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
  
  def forward_branch(self, x):
    y = []
    for i in range(self.features_num_module):
      m = self.features.module[i]
      x = m(x)
      if "f" + str(i) in self.branch_layer:
        y.append(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    y.append(x)
    return y

# ref: https://github.com/polo5/ZeroShotKnowledgeTransfer/blob/master/models/wresnet.py
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)
        
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
        
class WideResNet_16_2(nn.Module):
    def __init__(self, model=None, fixed=None):
        super(WideResNet_16_2, self).__init__()
        # -----------------------------
        depth = 16
        num_classes = 10
        widen_factor = 2
        dropRate = 0
        # -----------------------------
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        if model:
          checkpoint = torch.load(model)
          self.load_state_dict(checkpoint["state_dict"])
        if fixed:
          for param in self.parameters():
              param.requires_grad = False

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    def forward_branch(self, x):
        out = self.conv1(x)
        out = self.block1(out); activation1 = out
        out = self.block2(out); activation2 = out
        out = self.block3(out); activation3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return activation3, self.fc(out)
        
class WideResNet_16_1(nn.Module):
    def __init__(self, model=None, fixed=None):
        super(WideResNet_16_1, self).__init__()
        # -----------------------------
        depth = 16
        num_classes = 10
        widen_factor = 1
        dropRate = 0
        # -----------------------------
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        if model:
          checkpoint = torch.load(model)
          self.load_state_dict(checkpoint["state_dict"])
        if fixed:
          for param in self.parameters():
              param.requires_grad = False

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        activation1 = out
        out = self.block2(out)
        activation2 = out
        out = self.block3(out)
        activation3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    def forward_branch(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        activation1 = out
        out = self.block2(out)
        activation2 = out
        out = self.block3(out)
        activation3 = out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return activation3, self.fc(out)
        
class SmallVGG19(nn.Module):
  def __init__(self, model=None, fixed=None):
    super(SmallVGG19, self).__init__()
    self.features = make_layers(cifar10_vgg_cfg["SE"])
    self.classifier = nn.Sequential(
      nn.Linear(256, 512),
      nn.ReLU(True),
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

class Normalize_CIFAR10(nn.Module):
  def __init__(self):
    super(Normalize_CIFAR10, self).__init__()
    self.normalize = nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1,1), bias=True, groups=3)
    self.normalize.weight = nn.Parameter(torch.from_numpy(np.array(
                                    [[[[1/0.229]]],
                                     [[[1/0.224]]],
                                     [[[1/0.225]]]])).float()) # 3x1x1x1
    self.normalize.bias = nn.Parameter(torch.from_numpy(np.array(
                                  [-0.485/0.229, -0.456/0.224, -0.406/0.225])).float())
    self.normalize.requires_grad = False
  def forward(self, x):
    return self.normalize(x)
    
class DVGG19(nn.Module):
  def __init__(self, input_dim, model=None, fixed=None, gray=False, num_divbranch=1, dropout=0):
    super(DVGG19, self).__init__()
    self.classifier = nn.Sequential(
      nn.Linear(input_dim, 512),
      nn.ReLU(True),
      nn.Linear(512, 512),
      nn.ReLU(True),
      nn.Linear(512, 512),
      nn.ReLU(True),
    )
    self.gray = gray
    self.features = make_layers_dec(cifar10_vgg_cfg["Dec_gray"]) if gray else make_layers_dec(cifar10_vgg_cfg["Dec_s"], batch_norm=True)

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
    x = torch.stack([x]*3, dim=1).squeeze(2) if self.gray else x
    return x

# mimic the net architecture of MNIST deconv
class DVGG19_deconv(nn.Module):
  def __init__(self, input_dim, model=None, fixed=False, gray=False, num_divbranch=1, dropout=0):
    super(DVGG19_deconv, self).__init__()
    img_size = 32
    num_channel = 3
    self.init_size = img_size // 4
    self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))
    self.conv_blocks = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.BatchNorm2d(128, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, 3, stride=1, padding=1),
        nn.BatchNorm2d(64, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, num_channel, 3, stride=1, padding=1),
        nn.BatchNorm2d(num_channel, 0.8), # Ref: Huawei's paper. They add a BN layer at the end of the generator.
        nn.Tanh(),
    )
  def forward(self, z):
      out = self.l1(z)
      out = out.view(out.shape[0], 128, self.init_size, self.init_size)
      img = self.conv_blocks(out)
      return img
    
class DVGG19_aug(nn.Module): # augmented DVGG19
  def __init__(self, input_dim, model=None, fixed=None, gray=False, num_divbranch=1, dropout=0):
    super(DVGG19_aug, self).__init__()
    self.classifier = nn.Sequential(
      nn.Linear(input_dim, 512),
      nn.ReLU(True),
      nn.Linear(512, 512),
      nn.ReLU(True),
      nn.Linear(512, 512),
      nn.ReLU(True),
    )
    self.gray = gray
    self.features = make_layers_augdec(cifar10_vgg_cfg["Dec_s_aug"], True, num_divbranch)
    self.classifier_num_module = len(self.classifier)
    self.features_num_module = len(self.features)
    self.branch_layer = ["c5", "f3", "f10", "f17", "f24", "f31"]
    
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
    x = torch.stack([x] * 3, dim=1).squeeze(2) if self.gray else x
    return x
    
  def forward_branch(self, x):
    y = []
    for ci in range(self.classifier_num_module):
      m = self.classifier[ci]
      x = m(x)
      if "c" + str(ci) in self.branch_layer:
        y.append(x)
    x = x.view(x.size(0), 512, 1, 1)
    for fi in range(self.features_num_module):
      m = self.features[fi]
      x = m(x)
      if "f" + str(fi) in self.branch_layer:
        y.append(x)
    y.append(x)
    return y

# ZSKT's generator, ref: https://github.com/MingSun-Tse/ZeroShotKnowledgeTransfer/blob/master/models/generator.py
class View(nn.Module):
  def __init__(self, size):
      super(View, self).__init__()
      self.size = size
  def forward(self, tensor):
      return tensor.view(self.size)
class Generator(nn.Module):
  def __init__(self, input_dim, model=None, fixed=False, gray=False, num_divbranch=1, dropout=0):
    super(Generator, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(input_dim, 128 * 10**2),
      View((-1, 128, 10, 10)),
      nn.BatchNorm2d(128),

      nn.Upsample(scale_factor=1.6),
      nn.Conv2d(128, 128, 3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, 3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(64, 3, 3, stride=1, padding=1),
      nn.BatchNorm2d(3, affine=False), # This is optional
  )

  def forward(self, z):
    return self.layers(z)
  
  def forward_branch(self, z):
    pass

class Generator_Random(nn.Module):
  def __init__(self, input_dim, model=None, fixed=False, gray=False, num_divbranch=1, dropout=0):
    super(Generator_Random, self).__init__()
    self.layers1 = nn.Sequential(
      nn.Linear(input_dim, 128 * 10**2),
      View((-1, 128, 10, 10)),
      nn.BatchNorm2d(128),

      nn.Upsample(scale_factor=1.6),
      nn.Conv2d(128, 128, 3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),
      
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 96, 3, stride=1, padding=1), # 64 -> 96
      nn.BatchNorm2d(96),
      nn.LeakyReLU(0.2, inplace=True),
    )
    self.layers2 = nn.Sequential(
        nn.Conv2d(64, 3, 3, stride=1, padding=1), # 3 -> 32
        nn.BatchNorm2d(3, affine=False),
    )
  def forward(self, y):
      y = self.layers1(y); c1 = torch.randperm(96)[:64]; y = y[:,c1,:,:]
      y = self.layers2(y); # c2 = torch.randperm(50)[: 3]; y = y[:,c2,:,:]
      return y
      
################# MNIST #################
# ref: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
class DLeNet5_deconv(nn.Module):
  def __init__(self, input_dim, model=None, fixed=False, gray=False, num_divbranch=1, dropout=0):
    super(DLeNet5_deconv, self).__init__()
    img_size = 32
    num_channel = 1
    self.init_size = img_size // 4
    self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))
    self.conv_blocks = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.BatchNorm2d(128, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, 3, stride=1, padding=1),
        nn.BatchNorm2d(64, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(64, num_channel, 3, stride=1, padding=1),
        nn.BatchNorm2d(num_channel, 0.8), # Ref: DFL. They add a BN layer at the end of the generator.
        nn.Tanh() # I added this, because without this normalization, the norm will explode (still the reason is unknown)
    )
    
    # Rewrite conv_blocks above, for forward_branch
    self.conv1 = nn.Sequential(
         nn.BatchNorm2d(128),
         nn.Upsample(scale_factor=2),
         nn.Conv2d(128, 128, 3, stride=1, padding=1),
         nn.BatchNorm2d(128, 0.8),
         nn.LeakyReLU(0.2, inplace=True),
    )
    self.conv2 = nn.Sequential(
        nn.Upsample(scale_factor=2), 
        nn.Conv2d(128, 64, 3, stride=1, padding=1),
        nn.BatchNorm2d(64, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
    )
    self.conv3 = nn.Sequential(
        nn.Conv2d(64, num_channel, 3, stride=1, padding=1),
        nn.BatchNorm2d(num_channel, 0.8),
        nn.Tanh(),
    )

    self.drop = nn.Dropout(p=dropout)
    self.dropout = dropout
  
  def forward(self, z):
    y = self.l1(z)
    if self.dropout:
      y = self.drop(y)
    y = y.view(y.shape[0], 128, self.init_size, self.init_size)
    y = self.conv_blocks(y)
    return y
  
  def forward_branch(self, z):
    y = self.l1(z)
    if self.dropout:
      y = self.drop(y)
    y = y.view(y.shape[0], 128, self.init_size, self.init_size)
    out1 = self.conv1(y)
    out2 = self.conv2(out1)
    out3 = self.conv3(out2)
    return out1, out2, out3

# Imitate the generator above for CIFAR10
class Generator_MNIST(nn.Module):
  def __init__(self, input_dim, model=None, fixed=False, gray=False, num_divbranch=1, dropout=0):
    super(Generator_MNIST, self).__init__()
    self.layers = nn.Sequential(
      nn.Linear(input_dim, 128 * 10**2),
      View((-1, 128, 10, 10)),
      nn.BatchNorm2d(128),

      nn.Upsample(scale_factor=1.6),
      nn.Conv2d(128, 128, 3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, 3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(64, 1, 3, stride=1, padding=1),
      nn.BatchNorm2d(1, affine=False)
  )

  def forward(self, z):
    return self.layers(z)
    
class DLeNet5_deconv_Random(nn.Module):
  def __init__(self, input_dim, model=None, fixed=False, gray=False, num_divbranch=1, dropout=0):
    super(DLeNet5_deconv_Random, self).__init__()
    img_size = 32
    num_channel = 1
    self.init_size = img_size // 4
    self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))
    self.conv_blocks1 = nn.Sequential(
        nn.BatchNorm2d(128),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.BatchNorm2d(128, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 70, 3, stride=1, padding=1), # 64 -> 96
        nn.BatchNorm2d(70, 0.8),
        nn.LeakyReLU(0.2, inplace=True),
    )
    self.conv_blocks2 = nn.Sequential(
        nn.Conv2d(64, 1, 3, stride=1, padding=1), # 1 -> 10
        nn.BatchNorm2d(1, 0.8),
        nn.Tanh(),
    )
  def forward(self, z):
    y = self.l1(z)
    y = y.view(y.shape[0], 128, self.init_size, self.init_size)
    y = self.conv_blocks1(y); c1 = torch.randperm(70)[:64]; y = y[:,c1,:,:]
    y = self.conv_blocks2(y); # c2 = torch.randperm( 3)[: 1]; y = y[:,c2,:,:]
    return y
    
class DLeNet5_upsample(nn.Module):
  def __init__(self, input_dim, model=None, fixed=False, gray=False, num_divbranch=1):
    super(DLeNet5_upsample, self).__init__()
    self.fixed = fixed
    
    self.fc5 = nn.Linear(input_dim, 84)
    self.fc4 = nn.Linear( 84, 120)
    self.fc3 = nn.Linear(120, 400)
    self.conv2 = nn.Conv2d(16, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)) # to maintain the spatial size, so padding=2
    self.conv1 = nn.Conv2d( 6, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    self.bn2 = nn.BatchNorm2d(6, 0.8)
    self.bn1 = nn.BatchNorm2d(1, 0.8)
    
    self.relu = nn.ReLU(inplace=True)
    self.sigm = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.relu5 = nn.LeakyReLU(0.2, inplace=True)
    self.relu4 = nn.LeakyReLU(0.2, inplace=True)
    self.relu3 = nn.LeakyReLU(0.2, inplace=True)
    self.relu2 = nn.LeakyReLU(0.2, inplace=True)
    self.relu2 = nn.LeakyReLU(0.2, inplace=True)
    self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
    self.pad = nn.ReflectionPad2d((2,2,2,2))

    if model:
      self.load_state_dict(torch.load(model))
    if fixed:
      for param in self.parameters():
          param.requires_grad = False
      
  def forward(self, y):          # input: 10
    y = self.relu5(self.fc5(y))   # 84
    y = self.relu4(self.fc4(y))   # 120
    y = self.relu3(self.fc3(y))   # 400
    y = y.view(-1, 16, 5, 5)     # 16x5x5
    y = self.unpool(y)           # 16x10x10
    y = self.pad(y)              # 16x14x14
    y = self.relu2(self.bn2(self.conv2(y))) # 6x14x14
    y = self.unpool(y)           # 6x28x28
    y = self.pad(y)              # 6x32x32
    y = self.tanh(self.bn1(self.conv1(y))) # 1x32x32
    return y
    
# ref: https://github.com/iRapha/replayed_distillation/blob/master/models/lenet.py
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
    y = self.conv1(y); out1 = y
    y = self.relu(y)
    y = self.pool1(y)
    y = self.relu(self.conv2(y)); out2 = y
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y)); out3 = y
    y = self.fc4(y); out4 = y
    y = self.relu(y)
    y = self.fc5(y)
    return out1, out4, out2, y
   
# The LeNet5 model that has only two neurons in the last FC hidden layer, easy for feature visualization.
# Take the idea from 2016 ECCV center loss: https://kpzhang93.github.io/papers/eccv2016.pdf
class LeNet5_2neurons(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(LeNet5_2neurons, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d( 1,  6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d( 6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)); self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.fc3 = nn.Linear(400, 120)
    self.fc4 = nn.Linear(120,  84)
    self.fc5 = nn.Linear( 84,   2)
    self.fc6 = nn.Linear(  2,  10)
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
    y = self.relu(self.fc5(y))   # 2
    y = self.fc6(y)              # 10
    return y
  
  def forward_branch(self, y):
    y = self.relu(self.conv1(y)); out1 = y
    y = self.pool1(y)
    y = self.relu(self.conv2(y)); out2 = y
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y)); out3 = y
    y = self.relu(self.fc4(y)); out4 = y
    y = self.relu(self.fc5(y)); out5 = y
    y = self.fc6(y)
    return out1, out2, out3, out4, out5, out2, y
   
  def forward_2neurons(self, y):
    y = self.relu(self.conv1(y))
    y = self.pool1(y)
    y = self.relu(self.conv2(y))
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y))
    y = self.relu(self.fc4(y))
    y = self.fc5(y)
    return y
   
class LeNet5_deep(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(LeNet5_deep, self).__init__()
    self.fixed = fixed
    
    self.conv1  = nn.Conv2d( 1,  6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv11 = nn.Conv2d( 6,  8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv12 = nn.Conv2d( 8, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv13 = nn.Conv2d(10, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv14 = nn.Conv2d(12, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv15 = nn.Conv2d(14, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv16 = nn.Conv2d(14, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv17 = nn.Conv2d(14, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv18 = nn.Conv2d(14, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv19 = nn.Conv2d(14, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv110 = nn.Conv2d(14, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv111 = nn.Conv2d(14, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv112 = nn.Conv2d(14, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv113 = nn.Conv2d(14, 14, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv2  = nn.Conv2d(14, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
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
    y = self.relu(self.conv11(y))
    y = self.relu(self.conv12(y))
    y = self.relu(self.conv13(y))
    y = self.relu(self.conv14(y))
    y = self.relu(self.conv15(y))
    y = self.relu(self.conv16(y))
    y = self.relu(self.conv17(y))
    y = self.relu(self.conv18(y))
    y = self.relu(self.conv19(y))
    # y = self.relu(self.conv110(y))
    # y = self.relu(self.conv111(y))
    # y = self.relu(self.conv112(y))
    # y = self.relu(self.conv113(y))
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
    y = self.relu(self.conv11(y))
    y = self.relu(self.conv12(y))
    y = self.relu(self.conv13(y))
    y = self.relu(self.conv14(y))
    y = self.relu(self.conv15(y))
    y = self.relu(self.conv16(y))
    y = self.relu(self.conv17(y))
    y = self.relu(self.conv18(y))
    y = self.relu(self.conv19(y))
    # y = self.relu(self.conv110(y))
    # y = self.relu(self.conv111(y))
    # y = self.relu(self.conv112(y))
    # y = self.relu(self.conv113(y))
    y = self.relu(self.conv2(y)); out2 = y
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y)); out3 = y
    y = self.relu(self.fc4(y)); out4 = y
    y = self.fc5(y)
    return out2, y
 
class SmallLeNet5(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallLeNet5, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d( 1,  3, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d( 3,  8, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
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

class SmallLeNet5_2neurons(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallLeNet5_2neurons, self).__init__()
    self.fixed = fixed
    
    self.conv1 = nn.Conv2d( 1,  3, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv2d( 3,  8, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.fc3 = nn.Linear(200, 120)
    self.fc4 = nn.Linear(120,  84)
    self.fc5 = nn.Linear( 84,   2)
    self.fc6 = nn.Linear(  2,  10)
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
    y = self.relu(self.fc5(y))
    y = self.fc6(y)
    return y
  
  def forward_2neurons(self, y):
    y = self.relu(self.conv1(y))
    y = self.pool1(y)
    y = self.relu(self.conv2(y))
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y))
    y = self.relu(self.fc4(y))
    y = self.fc5(y)
    return y
    
class SmallLeNet5_deep(nn.Module):
  def __init__(self, model=None, fixed=False):
    super(SmallLeNet5_deep, self).__init__()
    self.fixed = fixed
    self.conv1  = nn.Conv2d(1, 3, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.conv11 = nn.Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv12 = nn.Conv2d(4, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv13 = nn.Conv2d(5, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv14 = nn.Conv2d(6, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv15 = nn.Conv2d(7, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv16 = nn.Conv2d(7, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv17 = nn.Conv2d(7, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv18 = nn.Conv2d(7, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv19 = nn.Conv2d(7, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv110 = nn.Conv2d(7, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv111 = nn.Conv2d(7, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv112 = nn.Conv2d(7, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv113 = nn.Conv2d(7, 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    self.conv2  = nn.Conv2d(7, 8, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
    self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.fc3 = nn.Linear(200, 120)
    self.fc4 = nn.Linear(120,  84)
    self.fc5 = nn.Linear( 84,  10)
    self.relu = nn.ReLU(inplace=True)
    
  def forward(self, y):
    y = self.relu(self.conv1(y))
    y = self.pool1(y)
    y = self.relu(self.conv11(y))
    y = self.relu(self.conv12(y))
    y = self.relu(self.conv13(y))
    y = self.relu(self.conv14(y))
    y = self.relu(self.conv15(y))
    y = self.relu(self.conv16(y))
    y = self.relu(self.conv17(y))
    y = self.relu(self.conv18(y))
    y = self.relu(self.conv19(y))
    # y = self.relu(self.conv110(y))
    # y = self.relu(self.conv111(y))
    # y = self.relu(self.conv112(y))
    # y = self.relu(self.conv113(y))
    y = self.relu(self.conv2(y))
    y = self.pool2(y)
    y = y.view(y.size(0), -1)
    y = self.relu(self.fc3(y))
    y = self.relu(self.fc4(y))
    y = self.fc5(y)
    return y

class Normalize_MNIST(nn.Module):
  def __init__(self):
    super(Normalize_MNIST, self).__init__()
    self.normalize = nn.Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), bias=True, groups=1)
    self.normalize.weight = nn.Parameter(torch.from_numpy(np.array([[[[1/0.3081]]]])).float())
    self.normalize.bias   = nn.Parameter(torch.from_numpy(np.array([-0.1307/0.3081])).float())
    self.normalize.requires_grad = False
  def forward(self, x):
    return self.normalize(x)
    
################# Transform #################
class Transform2(nn.Module): # drop out
  def __init__(self):
    super(Transform2, self).__init__()
    self.drop = nn.Dropout(p=0.08)
  def forward(self, x):
    return self.drop(x)
    
class Transform4(nn.Module): # rand translation
  def __init__(self):
    super(Transform4, self).__init__()
    self.conv1_c1 = nn.Conv2d(1, 1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False, groups=1)
    self.conv1_c3 = nn.Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False, groups=3)
    self.one_hot = OneHotCategorical(torch.Tensor([1/24., 1/24., 1/24., 1/24., 1/24.,
                                                   1/24., 1/24., 1/24., 1/24., 1/24.,
                                                   1/24., 1/24., 0.000, 1/24., 1/24.,
                                                   1/24., 1/24., 1/24., 1/24., 1/24.,
                                                   1/24., 1/24., 1/24., 1/24., 1/24.]))
  def forward(self, x):
    kernel = self.one_hot.sample().view(1,5,5) # 1x5x5
    if x.size(1) == 1:
      kernel_c1 = kernel.view(1,1,5,5).cuda()
      self.conv1_c1.weight = nn.Parameter(kernel_c1)
      y = self.conv1_c1(x)
      self.conv1_c1.requires_grad = False
      return y
    else:
      kernel_c3 = torch.stack([kernel] * 3).cuda() # 3x1x5x5
      self.conv1_c3.weight = nn.Parameter(kernel_c3)
      y = self.conv1_c3(x)
      self.conv1_c3.requires_grad = False
      return y
    
class Transform6(nn.Module): # resize or scale
  """
    Note: the image size is 32x32. Otherwise, this func may need redesign.
  """
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
    kernel = [[-1, -1, -1], 
              [-1,  9, -1], 
              [-1, -1, -1]]
    kernel = torch.from_numpy(np.array(kernel)).float().view(1,3,3)
    
    self.conv1_c1 = nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=1)
    self.conv1_c3 = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=3)
    kernel_c1 = kernel.view(1,1,3,3).cuda()
    kernel_c3 = torch.stack([kernel] * 3).cuda()
    self.conv1_c1.weight = nn.Parameter(kernel_c1)
    self.conv1_c3.weight = nn.Parameter(kernel_c3)
    self.conv1_c1.requires_grad = False
    self.conv1_c3.requires_grad = False
  
  def forward(self, x):
    if x.size(1) == 1:
      return self.conv1_c1(x)
    else:
      return self.conv1_c3(x)
    
class Transform10(nn.Module): # smooth
  def __init__(self):
    super(Transform10, self).__init__()
    kernel = [[1, 2, 1],
              [2, 4, 1],
              [1, 2, 1]] # Gaussian smoothing
    kernel = torch.from_numpy(np.array(kernel)).float().view(1,3,3) * 0.0625
    
    kernel_c1 = kernel.view(1,1,3,3).cuda()
    kernel_c3 = torch.stack([kernel] * 3).cuda()
    self.conv1_c1 = nn.Conv2d(3, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=1)
    self.conv1_c3 = nn.Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False, groups=3)
    self.conv1_c1.weight = nn.Parameter(kernel_c1)
    self.conv1_c3.weight = nn.Parameter(kernel_c3)
    self.conv1_c1.requires_grad = False
    self.conv1_c3.requires_grad = False
  
  def forward(self, x):
    if x.size(1) == 1:
      return self.conv1_c1(x)
    else:
      return self.conv1_c3(x)
    
class Transform(nn.Module): # random transform combination
  def __init__(self):
    super(Transform, self).__init__()
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
    # print(self.transforms)
    
  def forward(self, y):
    rand = np.random.permutation(len(self.transforms))
    Ts = self.transforms[rand]
    for T in Ts:
      if np.random.rand() >= 0.5:
        y = T(y)
    return y    

################# AutoEncoder #################
class AutoEncoder_GAN4(nn.Module):
  def __init__(self, args):
    super(AutoEncoder_GAN4, self).__init__()
    if "CIFAR" in args.dataset:
      BE = WideResNet_16_2; Dec = eval("Generator" + "_Random" * args.random_dec); SE = WideResNet_16_1 # converge!
      Embed = WideResNet_16_2 # TODO
      # BE = VGG19; Dec = Generator; SE = WideResNet_SE # converge! 
      # BE = WideResNet; Dec = Generator; SE = SmallVGG19 # TODO-@mingsuntse-20190528: this cannot converge. Still don't know why.
    elif args.dataset == "MNIST":
      Dec = eval("DLeNet5_deconv" + "_Random" * args.random_dec) # Generator_MNIST works best.
      BE  = eval("LeNet5" + args.which_lenet)
      SE  = eval("SmallLeNet5" + int(args.deep_lenet5[1]) * "_deep")
      Embed = LeNet5_2neurons
      
    self.be = BE(args.e1, fixed=True)
    self.em = Embed(args.embed_net).eval()
    self.defined_trans = Transform()
    self.upscale = nn.UpsamplingNearest2d(scale_factor=2)
    
    for di in range(1, args.num_dec + 1):
      pretrained_model = None
      if args.pretrained_dir:
        assert(args.pretrained_timeid != None)
        pretrained_model = [x for x in os.listdir(args.pretrained_dir) if "_d%s_" % di in x and args.pretrained_timeid in x] # the number of pretrained decoder should be like "SERVER218-20190313-1233_d3_E0S0.pth"
        assert(len(pretrained_model) == 1)
        pretrained_model = pretrained_model[0]
      input_dim = args.num_z + args.num_class if args.use_condition else args.num_z
      self.__setattr__("d" + str(di), Dec(input_dim, pretrained_model, fixed=False, gray=args.gray, num_divbranch=args.num_divbranch, dropout=args.dec_dropout))
    
    for sei in range(1, args.num_se + 1):
      self.__setattr__("se" + str(sei), SE(args.e2, fixed=False))
      
AutoEncoders = {
"GAN4": AutoEncoder_GAN4,
}
  
