from __future__ import print_function
import sys
import os
pjoin = os.path.join
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1] # The args MUST has an option "--gpu".
import shutil
import time
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
# torch
import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
import torch.utils.data as Data
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.models as models
# my libs
from model import AlexNet_Encoder, AlexNet_Decoder

# Passed-in params
parser = argparse.ArgumentParser(description="")
parser.add_argument('--img', type=str, default="./cat.jpg")
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--gpu', type=int)
parser.add_argument('--e1', type=str, default="models/my_alexnet.pth")
parser.add_argument('--d', type=str, default="../Experiments/only_closs1/weights/*E0S2000*.pth")
parser.add_argument('--e2', type=str)
parser.add_argument('--closs_weight', type=float, default=10)
parser.add_argument('--use_pseudo_code', action="store_true")
parser.add_argument('--code_generator', type=str, default="alexnet")
opt = parser.parse_args()
opt.d = glob.glob(opt.d)[0]

# Prepare model
code_generator = eval("models.%s(pretrained=True).cuda().eval()" % opt.code_generator)
encoder = AlexNet_Encoder(opt.e1).cuda().eval()
decoder = AlexNet_Decoder(opt.d).cuda()

# Get the code
if opt.use_pseudo_code:
  if not os.path.exists(".pseudo_code.npy"):
    print("==> pseudo_code does not exist: Generate one and save.")
    randperm = torch.randperm(1000)[:1] # to get one_hot
    code_gt = torch.randn([1, 1000]) + torch.eye(1000)[randperm] * 10  # logits
    code_gt = code_gt.cuda()
    np.save(".pseudo_code.npy", code_gt.data.cpu().numpy())
  else:
    print("==> pseudo_code has already existed: Use it.")
    code_gt = torch.from_numpy(np.load(".pseudo_code.npy")).cuda()
else:
  # Prepare input image
  img = Image.open(opt.img).convert("RGB")
  img = img.resize([opt.img_size, opt.img_size])
  img = transforms.ToTensor()(img).cuda().unsqueeze(0)
  print(img)
  code_gt = code_generator(img) # code GT
  print("==> The predict class: {}".format(code_gt.argmax()))

# Code reconstruction
prob_gt = nn.functional.softmax(code_gt, dim=1) # prob GT
code_rec1 = encoder(decoder(code_gt)) # code reconstruction 1
code_rec2 = encoder(decoder(code_rec1)) # code reconstruction 2

# Get loss
logprob1 = nn.functional.log_softmax(code_rec1, dim=1) 
logprob2 = nn.functional.log_softmax(code_rec2, dim=1) 
closs1 = nn.KLDivLoss()(logprob1, prob_gt.data) * opt.closs_weight
closs2 = nn.KLDivLoss()(logprob2, prob_gt.data) * opt.closs_weight
print("==> The KLDivLoss between reconstructed code and gt code: {:.6f} {:.6f}".format(closs1.data.cpu().numpy(), closs2.data.cpu().numpy()))





