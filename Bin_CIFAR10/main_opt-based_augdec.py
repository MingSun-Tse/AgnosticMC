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
import math
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.serialization import load_lua
import torch.utils.data as Data
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.distributions.one_hot_categorical import OneHotCategorical
import torchvision.datasets as datasets
from torch.autograd import Variable
# my libs
from model_augdec import AutoEncoders, EMA, Normalize, preprocess_image, recreate_image


def logprint(some_str):
  print(time.strftime("[%Y/%m/%d-%H:%M] ") + str(some_str), file=args.log, flush=True)
  
def path_check(x):
  if x:
    complete_path = glob.glob(x)
    assert(len(complete_path) == 1)
    x = complete_path[0]
  return x

# Passed-in params
parser = argparse.ArgumentParser(description="Knowledge Transfer")
parser.add_argument('--e1',  type=str,   default="models/model_best.pth.tar")
parser.add_argument('--e2',  type=str,   default=None)
parser.add_argument('--pretrained_dir',   type=str, default=None, help="the directory of pretrained decoder models")
parser.add_argument('--pretrained_timeid',type=str, default=None, help="the timeid of the pretrained models.")
parser.add_argument('--num_dec', type=int, default=1)
parser.add_argument('--num_se', type=int, default=1)
parser.add_argument('--num_divbranch', type=int, default=5)
parser.add_argument('--t',   type=str,   default=None)
parser.add_argument('--gpu', type=int,   default=0)
parser.add_argument('--lr',  type=float, default=1e-3)
parser.add_argument('--b1',  type=float, default=5e-4, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2',  type=float, default=5e-4, help='adam: decay of second order momentum of gradient')
# ----------------------------------------------------------------
# various losses
parser.add_argument('--lw_perc', type=float, default=1, help="perceptual loss")
parser.add_argument('--lw_soft', type=float, default=10) # According to the paper KD, the soft target loss weight should be considarably larger than that of hard target loss.
parser.add_argument('--lw_hard', type=float, default=1)
parser.add_argument('--lw_tv',   type=float, default=1e-6)
parser.add_argument('--lw_norm', type=float, default=1e-4)
parser.add_argument('--lw_DA',   type=float, default=10)
parser.add_argument('--lw_adv',  type=float, default=0.5)
parser.add_argument('--lw_actimax',  type=float, default=10)
parser.add_argument('--lw_msgan',  type=float, default=1)
# ----------------------------------------------------------------
parser.add_argument('-b', '--batch_size', type=int, default=256)
parser.add_argument('-p', '--project_name', type=str, default="test")
parser.add_argument('-r', '--resume', action='store_true')
parser.add_argument('-m', '--mode', type=str, default="GAN4", help='the training mode name.')
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--use_pseudo_code', action="store_false")
parser.add_argument('--begin', type=float, default=25)
parser.add_argument('--end',   type=float, default=20)
parser.add_argument('--temp',  type=float, default=1, help="the tempature in KD")
parser.add_argument('--adv_train', type=int, default=0)
parser.add_argument('--ema_factor', type=float, default=0.9, help="exponential moving average") 
parser.add_argument('--show_interval', type=int, default=10, help="the interval to print logs")
parser.add_argument('--save_interval', type=int, default=100, help="the interval to save sample images")
parser.add_argument('--test_interval', type=int, default=1000, help="the interval to test and save models")
parser.add_argument('--classloss_update_interval', type=int, default=1)
parser.add_argument('--gray', action="store_true")
parser.add_argument('--acc_thre_reset_dec', type=float, default=0)
parser.add_argument('--history_acc_weight', type=float, default=0.25)
parser.add_argument('--num_z', type=int, default=100, help="the dimension of hidden z")
parser.add_argument('--msgan_option', type=str, default="pixel")
parser.add_argument('--noise_magnitude', type=float, default=0.1)
parser.add_argument('--CodeID', type=str)
args = parser.parse_args()

# Update and check args
assert(args.num_se == 1)
assert(args.mode in AutoEncoders.keys())
args.e1 = path_check(args.e1)
args.e2 = path_check(args.e2)
args.pretrained_dir = path_check(args.pretrained_dir)
args.adv_train = int(args.mode[-1])
args.pid = os.getpid()

# Set up directories and logs, etc.
TimeID = time.strftime("%Y%m%d-%H%M")
ExpID = "SERVER" + os.environ["SERVER"] + "-" + TimeID
project_path = pjoin("../Experiments", ExpID + "_" + args.project_name)
rec_img_path = pjoin(project_path, "reconstructed_images")
weights_path = pjoin(project_path, "weights") # to save torch model
if not os.path.exists(project_path):
  os.makedirs(project_path)
else:
  if not args.resume:
    shutil.rmtree(project_path)
    os.makedirs(project_path)
if not os.path.exists(rec_img_path):
  os.makedirs(rec_img_path)
if not os.path.exists(weights_path):
  os.makedirs(weights_path)

log_path = pjoin(weights_path, "log_" + ExpID + ".txt")
args.log = open(log_path, "w+") if args.CodeID else sys.stdout # Given CodeID, it means this is a formal experiment, i.e., not debugging
  
if __name__ == "__main__":
  # Set up model
  AE = AutoEncoders[args.mode]
  ae = AE(args).cuda()
  
  # Set up exponential moving average
  history_acc_se_all = []
  history_acc_dec_all = []
  ema_dec = []; ema_se = []
  for di in range(1, args.num_dec+1):
    ema_dec.append(EMA(args.ema_factor))
    dec = eval("ae.d%s"  % di)
    for name, param in dec.named_parameters():
      if param.requires_grad:
        ema_dec[-1].register(name, param.data)
    for _ in range(args.num_divbranch):
      history_acc_se_all.append(0)
      history_acc_dec_all.append(0)
  for sei in range(1, args.num_se+1):
    ema_se.append(EMA(args.ema_factor))
    se = eval("ae.se%s" % sei)
    for name, param in se.named_parameters():
      if param.requires_grad:
        ema_se[-1].register(name, param.data)
        
  # Prepare data
  # ref: https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py
  tensor_normalize = Normalize().cuda()
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
  data_train = datasets.CIFAR10('./CIFAR10_data', train=True, download=True,
                              transform=transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                                transforms.ToTensor(),
                                normalize,
                              ]))
  if args.gray:
    data_test = datasets.CIFAR10('./CIFAR10_data', train=False, download=True,
                              transform=transforms.Compose([
                                transforms.Grayscale(num_output_channels=3), # test with gray image
                                transforms.ToTensor(),
                                normalize,
                              ]))
  else:
    data_test = datasets.CIFAR10('./CIFAR10_data', train=False, download=True,
                              transform=transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                              ]))
  kwargs = {'num_workers': 4, 'pin_memory': True}
  train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
  
  # Prepare transform and one hot generator
  one_hot = OneHotCategorical(torch.Tensor([1./args.num_class] * args.num_class))
  
  # Print setting for later check
  logprint(args._get_kwargs())
  
  # Optimizer
  optimizer_se  = [] 
  optimizer_dec = []
  for di in range(1, args.num_dec+1):
    dec = eval("ae.d"+str(di))
    optimizer_dec.append(torch.optim.Adam(dec.parameters(),  lr=args.lr, betas=(args.b1, args.b2)))
  for sei in range(1, args.num_se+1):
    se = eval("ae.se" + str(sei))
    optimizer_se.append(torch.optim.Adam(se.parameters(),  lr=args.lr, betas=(args.b1, args.b2)))
      
  # Resume previous step
  previous_epoch = previous_step = 0
  if args.e2 and args.resume:
    for clip in os.path.basename(args.e2).split("_"):
      if clip[0] == "E" and "S" in clip:
        num1 = clip.split("E")[1].split("S")[0]
        num2 = clip.split("S")[1]
        if num1.isdigit() and num2.isdigit():
          previous_epoch = int(num1)
          previous_step  = int(num2)
    
  # Optimization
  t1 = time.time()
  for epoch in range(previous_epoch, args.num_epoch):
    for step, (img, label) in enumerate(train_loader):
      ae.train()
      # Generate codes randomly
      random_z1 = torch.cuda.FloatTensor(args.batch_size, args.num_z); random_z1.copy_(torch.randn(args.batch_size, args.num_z))
      random_z2 = torch.cuda.FloatTensor(args.batch_size, args.num_z); random_z2.copy_(torch.randn(args.batch_size, args.num_z))
      z_concat = torch.cat([random_z1, random_z2], dim=0)
      onehot_label = one_hot.sample_n(args.batch_size).view([args.batch_size, args.num_class]).cuda()
      label_concat = torch.cat([onehot_label, onehot_label], dim=0)
      x = torch.cat([z_concat, label_concat], dim=1).detach() # input to the Generator network
      label = label_concat.data.cpu().numpy().argmax(axis=1)
      label = torch.from_numpy(label).long().detach().cuda()
        
      # update decoder
      imgrec_all = []; imgrec_DT_all = []; hardloss_dec_all = []; trainacc_dec_all = []
      for di in range(1, args.num_dec + 1):
        dec = eval("ae.d" + str(di)); optimizer = optimizer_dec[di-1]; ema = ema_dec[di-1]
        imgrecs = torch.split(dec(x), 3, dim=1) # 3 channels
        total_loss = 0
        imgrec_inner = []
        for imgrec1 in imgrecs:
          feats1 = ae.be.forward_branch(tensor_normalize(imgrec1)); logits1 = feats1[-1]
          imgrec1_DT = ae.defined_trans(imgrec1); logits1_DT = ae.be(tensor_normalize(imgrec1_DT)) # DT: defined transform
          imgrec_all.append(imgrec1); imgrec_DT_all.append(imgrec1_DT) # for SE
          
          tvloss1 = args.lw_tv * (torch.sum(torch.abs(imgrec1[:, :, :, :-1] - imgrec1[:, :, :, 1:])) + 
                                  torch.sum(torch.abs(imgrec1[:, :, :-1, :] - imgrec1[:, :, 1:, :])))
          imgnorm1 = torch.pow(torch.norm(imgrec1, p=6), 6) * args.lw_norm
          
          ## Classification loss, the bottomline loss
          # logprob1 = F.log_softmax(logits1/args.temp, dim=1)
          # logprob2 = F.log_softmax(logits2/args.temp, dim=1)
          # softloss1 = nn.KLDivLoss()(logprob1, prob_gt.data) * (args.temp*args.temp) * args.lw_soft
          # softloss2 = nn.KLDivLoss()(logprob2, prob_gt.data) * (args.temp*args.temp) * args.lw_soft
          hardloss1 = nn.CrossEntropyLoss()(logits1, label) * args.lw_hard
          hardloss1_DT = nn.CrossEntropyLoss()(logits1_DT, label) * args.lw_DA
          
          pred = logits1.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().cpu().data.numpy() / label.size(0)
          hardloss_dec_all.append(hardloss1.data.cpu().numpy()); trainacc_dec_all.append(trainacc)
          index = len(imgrec_all) - 1
          history_acc_dec_all[index] = history_acc_dec_all[index] * args.history_acc_weight + trainacc * (1 - args.history_acc_weight)
          
          ## Adversarial loss
          advloss = 0
          for sei in range(1, args.num_se+1):
            se = eval("ae.se" + str(sei))
            logits_dse = se(imgrec1)
            advloss += args.lw_adv / nn.CrossEntropyLoss()(logits_dse, label)
          
          ## Diversity encouraging loss 1: MSGAN
          # ref: 2019 CVPR Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis
          if args.msgan_option == "pixel":
            imgrec1_1, imgrec1_2 = torch.split(imgrec1, args.batch_size, dim=0)
            lz = torch.mean(torch.abs(imgrec1_1 - imgrec1_2)) / torch.mean(torch.abs(random_z1 - random_z2))
          elif args.msgan_option == "pixelgray":
            imgrec1_1, imgrec1_2 = torch.split(imgrec1, args.batch_size, dim=0)
            imgrec1_1 = imgrec1_1[:,0,:,:] * 0.299 + imgrec1_1[:,1,:,:] * 0.587 + imgrec1_1[:,2,:,:] * 0.114
            imgrec1_2 = imgrec1_2[:,0,:,:] * 0.299 + imgrec1_2[:,1,:,:] * 0.587 + imgrec1_2[:,2,:,:] * 0.114
            lz = torch.mean(torch.abs(imgrec1_1 - imgrec1_2)) / torch.mean(torch.abs(random_z1 - random_z2))
          elif args.msgan_option == "feature":
            lz = 0
            for i in range(len(feats1) - 1):
              feats1_1, feats1_2 = torch.split(feats1[i], args.batch_size, dim=0)
              lz += torch.mean(torch.abs(feats1_1 - feats1_2)) / torch.mean(torch.abs(random_z1 - random_z2))
            lz /= len(feats1) - 1
          elif args.msgan_option == "feature+pixel":
            lz = 0
            for i in range(len(feats1) - 1):
              feats1_1, feats1_2 = torch.split(feats1[i], args.batch_size, dim=0)
              lz += torch.mean(torch.abs(feats1_1 - feats1_2)) / torch.mean(torch.abs(random_z1 - random_z2))
            lz /= len(feats1) - 1
            imgrec1_1, imgrec1_2 = torch.split(imgrec1, args.batch_size, dim=0)
            lz += torch.mean(torch.abs(imgrec1_1 - imgrec1_2)) / torch.mean(torch.abs(random_z1 - random_z2))
          loss_diversity = args.lw_msgan / lz
          
          ## Diversity encouraging loss 2
          # ref: 2017 CVPR Diversified Texture Synthesis with Feed-forward Networks
          
          ## Activation maximization loss
          args.lw_actimax = max(args.lw_actimax - args.lw_actimax / 10.0 * epoch, 0)
          rand_loss_weight = torch.rand_like(logits1) * args.noise_magnitude
          activmax_loss = 0
          for i in range(logits1.size(0)):
            rand_loss_weight[i, label[i]] = 1
          activmax_loss = -torch.dot(logits1.flatten(), rand_loss_weight.flatten()) / logits1.size(0) * args.lw_actimax
          
          
          ## Total loss
          loss = hardloss1 + hardloss1_DT + \
                 advloss + \
                 tvloss1 + imgnorm1 + \
                 activmax_loss + \
                 loss_diversity
          total_loss += loss
        
        dec.zero_grad()
        total_loss.backward()
              
      # Update params
      for di in range(1, args.num_dec + 1):
        dec = eval("ae.d" + str(di)); optimizer = optimizer_dec[di-1]; ema = ema_dec[di-1]
        optimizer.step()
        for name, param in dec.named_parameters():
          if param.requires_grad:
            param.data = ema(name, param.data)

      # Update SE
      hardloss_se_all = []; trainacc_se_all = []
      for sei in range(1, args.num_se+1):
        se = eval("ae.se" + str(sei)); optimizer = optimizer_se[sei-1]; ema = ema_se[sei-1]
        se.zero_grad()
        loss_se = 0
        
        for i in range(len(imgrec_all)):
          logits = se(tensor_normalize(imgrec_all[i].detach()))
          logits_DT = se(tensor_normalize(imgrec_DT_all[i].detach()))
          hardloss = nn.CrossEntropyLoss()(logits, label) * args.lw_hard
          hardloss_DT = nn.CrossEntropyLoss()(logits_DT, label) * args.lw_DA
          loss_se += hardloss + hardloss_DT
          pred = logits.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().cpu().data.numpy() / label.size(0)
          hardloss_se_all.append(hardloss.data.cpu().numpy()); trainacc_se_all.append(trainacc)
          history_acc_se_all[i] = history_acc_se_all[i] * args.history_acc_weight + trainacc * (1 - args.history_acc_weight)
        
        loss_se.backward()
        optimizer.step()
        for name, param in se.named_parameters():
          if param.requires_grad:
            param.data = ema(name, param.data)
      
      # Save sample images
      if step % args.save_interval == 0:
        ae.dec = ae.d1
        ae.small_enc = ae.se1
        ae.enc = ae.be
        ae.eval()
        # save some test images
        logprint("E{}S{} | Saving image samples".format(epoch, step))
        onehot_label = torch.eye(args.num_class)
        test_codes = torch.cat([torch.randn([args.num_class, args.num_z]), onehot_label], dim=1)
        test_labels = onehot_label.numpy().argmax(axis=1)        
        for i in range(len(test_codes)):
          x = test_codes[i].cuda()
          x = x.unsqueeze(0)
          for di in range(1, args.num_dec+1):
            dec = eval("ae.d%s" % di)
            imgs = torch.split(dec(x), 3, dim=1)
            for j in range(len(imgs)):
              img1 = imgs[j]
              out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_imgrec%s_label=%s_d%s_%s.jpg" % (ExpID, epoch, step, i, test_labels[i], di, j))
              vutils.save_image(img1.data.cpu().float(), out_img1_path)
            
      # Test and save models
      if step % args.test_interval == 0:
        ae.eval()
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, **kwargs)
        test_acc = 0
        for i, (img, label) in enumerate(test_loader):
          label = label.cuda()
          pred = ae.se1(img.cuda()).detach().max(1)[1]
          test_acc += pred.eq(label.view_as(pred)).sum().cpu().data.numpy()
        test_acc /= float(len(data_test))
        format_str = "E{}S{} | =======> Test accuracy on SE: {:.4f} (ExpID: {})"
        logprint(format_str.format(epoch, step, test_acc, ExpID))
        torch.save(ae.se1.state_dict(), pjoin(weights_path, "%s_se_E%sS%s_testacc=%.4f.pth" % (ExpID, epoch, step, test_acc)))
        torch.save(ae.d1.state_dict(), pjoin(weights_path, "%s_d1_E%sS%s.pth" % (ExpID, epoch, step)))
        for di in range(2, args.num_dec+1):
          dec = eval("ae.d" + str(di))
          torch.save(dec.state_dict(), pjoin(weights_path, "%s_d%s_E%sS%s.pth" % (ExpID, di, epoch, step)))

      # Print training loss
      if step % args.show_interval == 0:
        format_str1 = "E{}S{}"
        format_str2 = " | dec:" + " {:.3f}({:.3f}-{:.3f})" * args.num_dec * args.num_divbranch
        format_str3 = " | se:" + " {:.3f}({:.3f}-{:.3f})" * args.num_dec * args.num_divbranch 
        format_str4 = " | tv: {:.3f} norm: {:.3f} diversity: {:.3f}"
        format_str5 = " ({:.3f}s/step)"
        format_str = "".join([format_str1, format_str2, format_str3, format_str4, format_str5])
        strvalue2 = []; strvalue3 = []
        for i in range(args.num_dec * args.num_divbranch):
          strvalue2.append(hardloss_dec_all[i]); strvalue2.append(trainacc_dec_all[i]); strvalue2.append(history_acc_dec_all[i])
          strvalue3.append(hardloss_se_all[i]); strvalue3.append(trainacc_se_all[i]); strvalue3.append(history_acc_se_all[i])
        logprint(format_str.format(
            epoch, step,
            *strvalue2,
            *strvalue3,
            tvloss1.data.cpu().numpy(), imgnorm1.data.cpu().numpy(), loss_diversity.data.cpu().numpy(),
            (time.time()-t1)/args.show_interval))

        t1 = time.time()