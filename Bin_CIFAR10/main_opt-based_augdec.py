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
parser.add_argument('--num_divbranch', type=int, default=1)
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
parser.add_argument('--lw_masknorm', type=float, default=1e-5)
parser.add_argument('--lw_DT',   type=float, default=10)
parser.add_argument('--lw_adv',  type=float, default=0)
parser.add_argument('--lw_actimax',  type=float, default=0)
parser.add_argument('--lw_msgan',  type=float, default=100)
parser.add_argument('--lw_maskdiversity',  type=float, default=100)
parser.add_argument('--lw_feat_L1_norm', type=float, default=0.1)
parser.add_argument('--lw_class_balance', type=float, default=5)
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
parser.add_argument('--show_interval_gradient', type=int, default=0, help="the interval to print gradient")
parser.add_argument('--save_interval', type=int, default=100, help="the interval to save sample images")
parser.add_argument('--test_interval', type=int, default=1000, help="the interval to test and save models")
parser.add_argument('--classloss_update_interval', type=int, default=1)
parser.add_argument('--gray', action="store_true")
parser.add_argument('--acc_thre_reset_dec', type=float, default=0)
parser.add_argument('--history_acc_weight', type=float, default=0.25)
parser.add_argument('--num_z', type=int, default=100, help="the dimension of hidden z")
parser.add_argument('--msgan_option', type=str, default="pixel")
parser.add_argument('--noise_magnitude', type=float, default=0)
parser.add_argument('--CodeID', type=str)
parser.add_argument('--clip_actimax', action="store_true")
args = parser.parse_args()

# Update and check args
assert(args.num_se == 1)
assert(args.num_dec == 1)
assert(args.mode in AutoEncoders.keys())
assert(args.msgan_option in ["pixel", "pixelgray"])
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
  ema_dec = []; ema_se = []; ema_mask = []; ema_meta = []
  for di in range(1, args.num_dec+1):
    ema_dec.append(EMA(args.ema_factor))
    ema_mask.append(EMA(args.ema_factor))
    ema_meta.append(EMA(args.ema_factor))
    dec = eval("ae.d%s"  % di)
    masknet = ae.mask
    metanet = ae.meta
    for name, param in dec.named_parameters():
      if param.requires_grad:
        ema_dec[-1].register(name, param.data)
    for name, param in masknet.named_parameters():
      if param.requires_grad:
        ema_mask[-1].register(name, param.data)
    for name, param in metanet.named_parameters():
      if param.requires_grad:
        ema_meta[-1].register(name, param.data)
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
  optimizer_se   = []
  optimizer_dec  = []
  optimizer_mask = []
  optimizer_meta = []
  for di in range(1, args.num_dec + 1):
    dec = eval("ae.d" + str(di))
    masknet = ae.mask
    optimizer_dec.append(torch.optim.Adam(dec.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
    optimizer_mask.append(torch.optim.Adam(masknet.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
    optimizer_meta.append(torch.optim.Adam(metanet.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
  for sei in range(1, args.num_se + 1):
    se = eval("ae.se" + str(sei))
    optimizer_se.append(torch.optim.Adam(se.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
      
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
  num_digit_show_step  = len(str(int(len(data_train) / args.batch_size)))
  num_digit_show_epoch = len(str(args.num_epoch))
  t1 = time.time()
  for epoch in range(previous_epoch, args.num_epoch):
    for step, (img, label) in enumerate(train_loader):
      ae.train()
      # Generate codes randomly
      random_z1 = torch.cuda.FloatTensor(args.batch_size, args.num_z); random_z1.copy_(torch.randn(args.batch_size, args.num_z))
      random_z2 = torch.cuda.FloatTensor(args.batch_size, args.num_z); random_z2.copy_(torch.randn(args.batch_size, args.num_z))
      x = torch.cat([random_z1, random_z2], dim=0)
              
      # Update decoder
      imgrec_all = []; imgrec_DT_all = []; hardloss_dec_all = []; trainacc_dec_all = []
      for di in range(1, args.num_dec + 1):
        # Set up model and ema
        dec = eval("ae.d" + str(di)); optimizer_d = optimizer_dec[di-1]; ema_d = ema_dec[di-1]
        total_loss_dec = 0

        # Forward
        imgrecs = dec(x)
        
        ## Diversity encouraging loss: MSGAN
        # ref: 2019 CVPR Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis
        if args.lw_msgan:
          if args.msgan_option == "pixel":
            imgrecs_1, imgrecs_2 = torch.split(imgrecs, args.batch_size, dim=0)
            lz_pixel = torch.mean(torch.abs(imgrecs_1 - imgrecs_2)) / torch.mean(torch.abs(random_z1 - random_z2))
          elif args.msgan_option == "pixelgray": # deprecated
            imgrecs_1, imgrecs_2 = torch.split(imgrecs, args.batch_size, dim=0)
            imgrecs_1 = imgrecs_1[:,0,:,:] * 0.299 + imgrecs_1[:,1,:,:] * 0.587 + imgrecs_1[:,2,:,:] * 0.114 # the Y channel (Luminance) of an image
            imgrecs_2 = imgrecs_2[:,0,:,:] * 0.299 + imgrecs_2[:,1,:,:] * 0.587 + imgrecs_2[:,2,:,:] * 0.114
            lz_pixel = torch.mean(torch.abs(imgrecs_1 - imgrecs_2)) / torch.mean(torch.abs(random_z1 - random_z2))
          loss_diversity_pixel = -args.lw_msgan * lz_pixel
          # total_loss_dec += loss_diversity_pixel
        

        imgrecs_split = torch.split(imgrecs, 3, dim=1) # 3 channels
        actimax_loss_print = []
        for imgrec in imgrecs_split:
          # forward
          imgrec_all.append(imgrec) # for SE
          feats = ae.be.forward_branch(tensor_normalize(imgrec))
          logits = feats[-1]; last_feature = feats[-2]
          label = logits.argmax(dim=1)
          
          ## Low-level natural image prior: tv + image norm
          # ref: 2015 CVPR Understanding Deep Image Representations by Inverting Them
          if args.lw_tv:
            tvloss = args.lw_tv * (torch.sum(torch.abs(imgrec[:, :, :, :-1] - imgrec[:, :, :, 1:])) + 
                                   torch.sum(torch.abs(imgrec[:, :, :-1, :] - imgrec[:, :, 1:, :])))
            total_loss_dec += tvloss
          if args.lw_norm:            
            imgnorm = torch.pow(torch.norm(imgrec, p=6), 6) * args.lw_norm
            total_loss_dec += imgnorm
          
          ## Classification loss, the bottomline loss
          hardloss = nn.CrossEntropyLoss()(logits, label) * args.lw_hard
          total_loss_dec += hardloss
          if args.lw_DT:
            imgrec_DT = ae.defined_trans(imgrec) # DT: defined transform
            imgrec_DT_all.append(imgrec_DT) # for SE
            logits_DT = ae.be(tensor_normalize(imgrec_DT))
            total_loss_dec += nn.CrossEntropyLoss()(logits_DT, label) * args.lw_DT
          # for accuracy print
          pred = logits.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().item() / label.size(0)
          hardloss_dec_all.append(hardloss.item()); trainacc_dec_all.append(trainacc)
          index = len(imgrec_all) - 1
          history_acc_dec_all[index] = history_acc_dec_all[index] * args.history_acc_weight + trainacc * (1 - args.history_acc_weight)
          
          ## Adversarial loss, combat with SE
          if args.lw_adv:
            for sei in range(1, args.num_se + 1):
              se = eval("ae.se" + str(sei))
              logits_dse = se(imgrec)
              total_loss_dec += args.lw_adv / nn.CrossEntropyLoss()(logits_dse, label)
          
          ## Activation maximization loss
          # ref: 2016 IJCV Visualizing Deep Convolutional Neural Networks Using Natural Pre-images
          if args.clip_actimax and epoch >= 7:
            args.lw_actimax = 0
          if args.lw_actimax:
            rand_loss_weight = torch.rand_like(logits) * args.noise_magnitude
            for i in range(logits.size(0)):
              rand_loss_weight[i, label[i]] = 1
            actimax_loss = -args.lw_actimax * (torch.dot(logits.flatten(), rand_loss_weight.flatten()) / logits.size(0))
            actimax_loss_print.append(actimax_loss.item())
            total_loss_dec += actimax_loss
          
          ## Huawei's idea
          if args.lw_feat_L1_norm:
            L_alpha = -torch.norm(last_feature, p=1) / last_feature.size(0)
            total_loss_dec += L_alpha * args.lw_feat_L1_norm
          if args.lw_class_balance:
            pred_prob = logits.softmax(dim=1).mean(dim=0)
            L_ie = -torch.dot(pred_prob, torch.log(pred_prob)) / args.num_class
            total_loss_dec += L_ie * args.lw_class_balance
          
        dec.zero_grad()
        total_loss_dec.backward()
        optimizer_d.step()
        for name, param in dec.named_parameters():
          if param.requires_grad:
            param.data = ema_d(name, param.data)        
        # Gradient checking
        if args.show_interval_gradient and step % args.show_interval_gradient == 0:
          ave_grad = []
          for p in dec.named_parameters():
            layer_name = p[0]
            if "bias" in layer_name: continue
            if p[1].grad is not None:
              ave_grad.append([layer_name, np.average(p[1].grad.abs()) * args.lr, np.average(p[1].data.abs())])
          ave_grad = ["{:<20} {:.6f}  /  {:.6f}  ({:.10f})\n".format(x[0], x[1], x[2], x[1]/x[2]) for x in ave_grad]
          ave_grad = "".join(ave_grad)
          logprint(("E{:0>%s}S{:0>%s} (grad x lr) / weight:\n{}" % (num_digit_show_epoch, num_digit_show_step)).format(epoch, step, ave_grad))

     # Update SE
      hardloss_se_all = []; trainacc_se_all = []
      for sei in range(1, args.num_se + 1):
        se = eval("ae.se" + str(sei)); optimizer = optimizer_se[sei-1]; ema = ema_se[sei-1]
        loss_se = 0
        for i in range(len(imgrec_all)):
          logits = se(tensor_normalize(imgrec_all[i].detach()))
          hardloss = nn.CrossEntropyLoss()(logits, label) * args.lw_hard
          loss_se += hardloss
          if args.lw_DT:
            logits_DT = se(tensor_normalize(imgrec_DT_all[i].detach()))
            hardloss_DT = nn.CrossEntropyLoss()(logits_DT, label) * args.lw_DT
            loss_se += hardloss_DT
          pred = logits.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().item() / label.size(0)
          hardloss_se_all.append(hardloss.item()); trainacc_se_all.append(trainacc)
          history_acc_se_all[i] = history_acc_se_all[i] * args.history_acc_weight + trainacc * (1 - args.history_acc_weight)
        se.zero_grad()
        loss_se.backward()
        optimizer.step()
        for name, param in se.named_parameters():
          if param.requires_grad:
            param.data = ema(name, param.data)
      
      # Save sample images
      if step % args.save_interval == 0:
        ae.eval()
        # save some test images
        logprint(("E{:0>%s}S{:0>%s} | Saving image samples" % (num_digit_show_epoch, num_digit_show_step)).format(epoch, step))
        onehot_label = torch.eye(args.num_class)
        test_codes = torch.randn([args.num_class, args.num_z])
        
        for i in range(len(test_codes)):
          x = test_codes[i].cuda().unsqueeze(0)
          for di in range(1, args.num_dec + 1):
            dec = eval("ae.d%s" % di)
            imgrecs = dec(x)
            imgs = torch.split(imgrecs, 3, dim=1)
            for bi in range(len(imgs)):
              img1 = imgs[bi]
              logits = ae.be(img1)[0]
              test_label = logits.argmax().item()
              out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_d%s_b%s_imgrec%s_label%s.jpg" % (ExpID, epoch, step, di, bi, i, test_label))
              vutils.save_image(img1.data.cpu().float(), out_img1_path)

      # Test and save models
      if step % args.test_interval == 0:
        ae.eval()
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, **kwargs)
        test_acc = 0
        for i, (img, label) in enumerate(test_loader):
          label = label.cuda()
          pred = ae.se1(img.cuda()).detach().max(1)[1]
          test_acc += pred.eq(label.view_as(pred)).sum().item()
        test_acc /= float(len(data_test))
        format_str = "E{:0>%s}S{:0>%s} | " % (num_digit_show_epoch, num_digit_show_step) + "=" * (int(TimeID[-1]) + 1) + "> Test accuracy on SE: {:.4f} (ExpID: {})"
        logprint(format_str.format(epoch, step, test_acc, ExpID))
        # torch.save(ae.se1.state_dict(), pjoin(weights_path, "%s_se_E%sS%s_testacc=%.4f.pth" % (ExpID, epoch, step, test_acc)))
        # torch.save(ae.d1.state_dict(), pjoin(weights_path, "%s_d1_E%sS%s.pth" % (ExpID, epoch, step)))
        # for di in range(2, args.num_dec+1):
          # dec = eval("ae.d" + str(di))
          # torch.save(dec.state_dict(), pjoin(weights_path, "%s_d%s_E%sS%s.pth" % (ExpID, di, epoch, step)))

      # Print training loss
      if step % args.show_interval == 0:
        format_str1 = "E{:0>%s}S{:0>%s}" % (num_digit_show_epoch, num_digit_show_step)
        format_str2 = " | dec:" + " {:.3f}({:.3f}-{:.3f})" * args.num_dec * args.num_divbranch
        format_str3 = " | se:" + " {:.3f}({:.3f}-{:.3f})" * args.num_dec * args.num_divbranch 
        format_str4 = " | tv: {:.3f} norm: {:.3f} diversity: {:.3f} {:.3f} actimax: {:.3f} mask_diversity: {:.3f} mask_norm: {:.3f}"
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
            tvloss.item(), imgnorm.item(), 0, loss_diversity_pixel.item(), np.average(actimax_loss_print), 0, 0,
            (time.time()-t1)/args.show_interval)) # loss_mask_diversity

        t1 = time.time()