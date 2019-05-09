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
from model import AutoEncoders, EMA, preprocess_image, recreate_image
from data import set_up_data
from util import check_path, get_previous_step, LogPrint, set_up_dir


# Passed-in params
parser = argparse.ArgumentParser(description="Knowledge Transfer")
parser.add_argument('--e1',  type=str,   default=None)
parser.add_argument('--e2',  type=str,   default=None)
parser.add_argument('--pretrained_dir',   type=str, default=None, help="the directory of pretrained decoder models")
parser.add_argument('--pretrained_timeid',type=str, default=None, help="the timeid of the pretrained models.")
parser.add_argument('--num_dec', type=int, default=1)
parser.add_argument('--num_se', type=int, default=1)
parser.add_argument('--num_divbranch', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=300)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--num_z', type=int, default=100, help="the dimension of hidden z")
parser.add_argument('--gpu', type=int,   default=0)
parser.add_argument('--lr',  type=float, default=2e-2)
parser.add_argument('--b1',  type=float, default=5e-4, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2',  type=float, default=0.999, help='adam: decay of second order momentum of gradient')
# ----------------------------------------------------------------
# various losses
parser.add_argument('--lw_perc', type=float, default=1, help="perceptual loss")
parser.add_argument('--lw_soft', type=float, default=10) # According to the paper KD, the soft target loss weight should be considarably larger than that of hard target loss.
parser.add_argument('--lw_hard_dec', type=float, default=1)
parser.add_argument('--lw_hard_se', type=float, default=1)
parser.add_argument('--lw_tv',   type=float, default=0) #1e-6)
parser.add_argument('--lw_norm', type=float, default=0) #1e-4)
parser.add_argument('--lw_masknorm', type=float, default=0) # 1e-5)
parser.add_argument('--lw_DT',   type=float, default=0, help="DT means 'defined transformation', ie, the common data augmentation operations like random translation, rotation, etc.") # 10)
parser.add_argument('--lw_adv',  type=float, default=0)
parser.add_argument('--lw_actimax',  type=float, default=0)
parser.add_argument('--lw_msgan',  type=float, default=1e-30) # 100)
parser.add_argument('--lw_maskdiversity',  type=float, default=0) # 100)
parser.add_argument('--lw_feat_L1_norm', type=float, default=0, help="ref to DFL paper")
parser.add_argument('--lw_class_balance', type=float, default=1000, help="ref to DFL paper")
# ----------------------------------------------------------------
parser.add_argument('-b', '--batch_size', type=int, default=600) # 256)
parser.add_argument('-p', '--project_name', type=str, default="test")
parser.add_argument('-r', '--resume', action='store_true')
parser.add_argument('-m', '--mode', type=str, default="GAN4", help='the training mode name.')
parser.add_argument('--use_pseudo_code', action="store_false")
parser.add_argument('--use_random_input', action="store_true")
parser.add_argument('--begin', type=float, default=25)
parser.add_argument('--end',   type=float, default=20)
parser.add_argument('--temp',  type=float, default=1, help="the tempature in KD")
parser.add_argument('--adv_train', type=int, default=0)
parser.add_argument('--ema_factor', type=float, default=0.9, help="exponential moving average") 
parser.add_argument('--show_interval', type=int, default=10, help="the interval to print logs")
parser.add_argument('--show_interval_gradient', type=int, default=0, help="the interval to print gradient")
parser.add_argument('--save_interval', type=int, default=100, help="the interval to save sample images")
parser.add_argument('--test_interval', type=int, default=1000, help="the interval to test and save models")
parser.add_argument('--gray', action="store_true")
parser.add_argument('--history_acc_weight', type=float, default=0.25)
parser.add_argument('--msgan_option', type=str, default="pixel")
parser.add_argument('--noise_magnitude', type=float, default=0)
parser.add_argument('--CodeID', type=str)
parser.add_argument('--clip_actimax', action="store_true")
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--use_condition', action="store_true")
parser.add_argument('--deep_lenet5', type=str, default="00", help="'1' means using deep model and '0' means \
not, such as '11': deep teacher + deep student, '10': deep teacher + shallow student")
args = parser.parse_args()

# Update and check args
pretrained_be_path = {
"MNIST": "train_baseline_lenet5/trained_weights2/w*/*E17S0*.pth",
"MNIST_deep": "train_baseline_lenet5/trained_weights_verydeep/weights/*E16S0*.pth", #"train_baseline_lenet5/trained_weights_deep/w*/*E19S0*.pth",
"CIFAR10": "models/model_best.pth.tar",
}

assert(args.num_se == 1)
assert(args.num_dec == 1)
assert(args.mode in AutoEncoders.keys())
assert(args.msgan_option in ["pixel", "pixelgray"])
assert(args.dataset in ["MNIST", "CIFAR10"])
if args.e1 == None:
  if args.dataset == "CIFAR10":
    args.e1 = pretrained_be_path[args.dataset]
  else:
    key = "MNIST" + "_deep" * int(args.deep_lenet5[0])
    args.e1 = pretrained_be_path[key]
args.e1 = check_path(args.e1)
args.e2 = check_path(args.e2)
args.pretrained_dir = check_path(args.pretrained_dir)
args.adv_train = int(args.mode[-1])
num_channel = 1 if args.dataset == "MNIST" else 3

# Set up directories and logs, etc.
TimeID, ExpID, rec_img_path, weights_path, log = set_up_dir(args.project_name, args.resume, args.CodeID)
logprint = LogPrint(log)
args.ExpID = ExpID

if __name__ == "__main__":
  # Set up model
  AE = AutoEncoders[args.mode]
  ae = AE(args).cuda()
  
  # Set up exponential moving average
  ema_dec = []; ema_se = []; ema_mask = []; ema_meta = []; ema_codemap = []
  for di in range(1, args.num_dec + 1):
    ema_dec.append(EMA(args.ema_factor))
    ema_mask.append(EMA(args.ema_factor))
    ema_meta.append(EMA(args.ema_factor))
    ema_codemap.append(EMA(args.ema_factor))
    dec = eval("ae.d%s"  % di)
    masknet = ae.mask
    metanet = ae.meta
    codemap = ae.codemap
    for name, param in dec.named_parameters():
      if param.requires_grad:
        ema_dec[-1].register(name, param.data)
    for name, param in masknet.named_parameters():
      if param.requires_grad:
        ema_mask[-1].register(name, param.data)
    for name, param in metanet.named_parameters():
      if param.requires_grad:
        ema_meta[-1].register(name, param.data)
    for name, param in codemap.named_parameters():
      if param.requires_grad:
        ema_codemap[-1].register(name, param.data)

  for sei in range(1, args.num_se + 1):
    ema_se.append(EMA(args.ema_factor))
    se = eval("ae.se%s" % sei)
    for name, param in se.named_parameters():
      if param.requires_grad:
        ema_se[-1].register(name, param.data)

  # Prepare data
  train_loader, num_train, test_loader, num_test = set_up_data(args.dataset, args.batch_size)
  
  # Print settings after the model and data are set up normally
  logprint(args._get_kwargs())
  
  # Optimizer
  optimizer_se   = []
  optimizer_dec  = []
  optimizer_mask = []
  optimizer_meta = []
  optimizer_codemap = []
  for di in range(1, args.num_dec + 1):
    dec = eval("ae.d" + str(di))
    optimizer_dec.append(torch.optim.Adam(dec.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
    optimizer_mask.append(torch.optim.Adam(masknet.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
    optimizer_meta.append(torch.optim.Adam(metanet.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
    optimizer_codemap.append(torch.optim.Adam(codemap.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
  for sei in range(1, args.num_se + 1):
    se = eval("ae.se" + str(sei))
    optimizer_se.append(torch.optim.Adam(se.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
      
  # Resume previous step
  previous_epoch, previous_step = get_previous_step(args.e2, args.resume)
  
  # Optimization
  one_hot = OneHotCategorical(torch.Tensor([1. / args.num_class] * args.num_class))
  num_digit_show_step  = len(str(int(num_train / args.batch_size)))
  num_digit_show_epoch = len(str(args.num_epoch))
  t1 = time.time()
  for epoch in range(previous_epoch, args.num_epoch):
    for step, (img, label) in enumerate(train_loader):
      ae.train()
      imgrec_all = []; logits_all = []; imgrec_DT_all = []; hardloss_dec_all = []; trainacc_dec_all = []
      actimax_loss_print = []

      if not args.use_random_input:
        # Generate codes randomly
        if args.lw_msgan:
          half_bs = int(args.batch_size / 2)
          random_z1 = torch.cuda.FloatTensor(half_bs, args.num_z); random_z1.copy_(torch.randn(half_bs, args.num_z))
          random_z2 = torch.cuda.FloatTensor(half_bs, args.num_z); random_z2.copy_(torch.randn(half_bs, args.num_z))
          x = torch.cat([random_z1, random_z2], dim=0)
          if args.use_condition:
            onehot_label = one_hot.sample_n(half_bs).view([half_bs, args.num_class]).cuda()
            label_concat = torch.cat([onehot_label, onehot_label], dim=0)
            label = label_concat.argmax(dim=1).detach()
            x = torch.cat([x, label_concat], dim=1).detach()
            # label_noise = torch.randn(args.batch_size, args.num_class).cuda() * args.begin
            # label = label_noise.argmax(dim=1).detach()
            # for i in range(args.batch_size):
              # label_noise[i, label[i]] += 5
            # x = torch.cat([x, label_noise], dim=1).detach()
        else:
          x = torch.cuda.FloatTensor(args.batch_size, args.num_z); x.copy_(torch.randn(args.batch_size, args.num_z))
          if args.use_condition:
            onehot_label = one_hot.sample_n(args.batch_size).view([args.batch_size, args.num_class]).cuda()
            label = onehot_label.argmax(dim=1).detach()
            x = torch.cat([x, onehot_label], dim=1).detach()
        
        # Update decoder
        for di in range(1, args.num_dec + 1):
          # Set up model and ema
          dec = eval("ae.d" + str(di)); optimizer_d = optimizer_dec[di - 1]; ema_d = ema_dec[di - 1]
          codemap = ae.codemap; optimizer_c = optimizer_codemap[di - 1]; ema_c = ema_codemap[di - 1]
          total_loss_dec = 0

          # Forward
          # x = codemap(x) ## TODO: new idea to impl.
          imgrecs = dec(x)
          
          ## Diversity encouraging loss: MSGAN
          # ref: 2019 CVPR Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis
          if args.lw_msgan:
            if args.msgan_option == "pixel":
              imgrecs_1, imgrecs_2 = torch.split(imgrecs, half_bs, dim=0)
              lz_pixel = torch.mean(torch.abs(imgrecs_1 - imgrecs_2)) / torch.mean(torch.abs(random_z1 - random_z2))
            elif args.msgan_option == "pixelgray": # deprecated
              imgrecs_1, imgrecs_2 = torch.split(imgrecs, half_bs, dim=0)
              imgrecs_1 = imgrecs_1[:,0,:,:] * 0.299 + imgrecs_1[:,1,:,:] * 0.587 + imgrecs_1[:,2,:,:] * 0.114 # the Y channel (Luminance) of an image
              imgrecs_2 = imgrecs_2[:,0,:,:] * 0.299 + imgrecs_2[:,1,:,:] * 0.587 + imgrecs_2[:,2,:,:] * 0.114
              lz_pixel = torch.mean(torch.abs(imgrecs_1 - imgrecs_2)) / torch.mean(torch.abs(random_z1 - random_z2))
            total_loss_dec += -args.lw_msgan * lz_pixel
          
          imgrecs_split = torch.split(imgrecs, num_channel, dim=1)
          for imgrec in imgrecs_split:
            # forward
            imgrec_all.append(imgrec.detach()) # for SE
            feats = ae.be.forward_branch(imgrec)
            logits = feats[-1]; last_feature = feats[-2]
            logits_all.append(logits.detach())
            if not args.use_condition:
              label = logits.argmax(dim=1).detach()
            
            ## Low-level natural image prior: tv + image norm
            # ref: 2015 CVPR Understanding Deep Image Representations by Inverting Them
            tvloss = (torch.sum(torch.abs(imgrec[:, :, :, :-1] - imgrec[:, :, :, 1:])) + 
                      torch.sum(torch.abs(imgrec[:, :, :-1, :] - imgrec[:, :, 1:, :])))
            if args.lw_tv: total_loss_dec += tvloss * args.lw_tv
            imgnorm = torch.pow(torch.norm(imgrec, p=6), 6)
            if args.lw_norm: total_loss_dec += imgnorm * args.lw_norm
            
            ## Classification loss, or hard-target loss in KD
            # pred = logits.detach().max(1)[1] ## TODO: new idea to impl.
            # pred.eq(label.view_as(pred))
            hardloss = nn.CrossEntropyLoss()(logits, label)
            hardloss_dec_all.append(hardloss.item())
            if args.lw_hard_dec: total_loss_dec += hardloss * args.lw_hard_dec
            # for accuracy print
            pred = logits.detach().max(1)[1]
            trainacc = pred.eq(label.view_as(pred)).sum().item() / label.size(0)
            trainacc_dec_all.append(trainacc)
            
            ## Data augmentation loss
            if args.lw_DT:
              imgrec_DT = ae.defined_trans(imgrec) # DT: defined transform
              imgrec_DT_all.append(imgrec_DT) # for SE
              logits_DT = ae.be(imgrec_DT)
              total_loss_dec += nn.CrossEntropyLoss()(logits_DT, label) * args.lw_DT
            
            ## Adversarial loss, combat with SE
            if args.lw_adv:
              for sei in range(1, args.num_se + 1):
                se = eval("ae.se" + str(sei))
                logits_dse = se(imgrec)
                total_loss_dec += args.lw_adv / nn.CrossEntropyLoss()(logits_dse, label)
            
            ## Activation maximization loss
            # ref: 2016 IJCV Visualizing Deep Convolutional Neural Networks Using Natural Pre-images
            actimax_loss = torch.zeros(1)
            if args.clip_actimax and epoch >= 7:
              args.lw_actimax = 0
            if args.lw_actimax:
              rand_loss_weight = torch.rand_like(logits) * args.noise_magnitude
              for i in range(logits.size(0)):
                rand_loss_weight[i, label[i]] = 1
              actimax_loss = -torch.dot(logits.flatten(), rand_loss_weight.flatten()) / logits.size(0)
              actimax_loss_print.append(actimax_loss.item())
              total_loss_dec += actimax_loss * args.lw_actimax
            
            ## DFL
            # ref: 2019.04 arxiv Data-Free Learning of Student Networks (https://arxiv.org/abs/1904.01186)
            L_alpha = -torch.norm(last_feature, p=1) / last_feature.size(0)
            if args.lw_feat_L1_norm: total_loss_dec += L_alpha * args.lw_feat_L1_norm
            
            prob = logits.softmax(dim=1)
            ave_prob = prob.mean(dim=0)
            L_ie = torch.dot(ave_prob, torch.log(ave_prob)) / args.num_class
            if args.lw_class_balance: total_loss_dec += L_ie * args.lw_class_balance
            
            ## My diversity loss
            # pred_label = logits.argmax(dim=1)
            # true_prob = torch.zeros_like(prob); true_prob.copy_(prob)
            # for i in range(logits.size(0)):
              # true_prob[i, label[i]] = prob[i, pred_label[i]]
              # true_prob[i, pred_label[i]] = prob[i, label[i]]
            # loss_KL = nn.KLDivLoss()(F.log_softmax(logits, dim=1), true_prob.detach())
            # if args.lw_my_diversity: total_loss_dec += loss_KL * args.lw_my_diversity
          
          dec.zero_grad()
          total_loss_dec.backward(retain_graph=True)
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
            ave_grad = ["{:<30} {:.6f}  /  {:.6f}  ({:.10f})\n".format(x[0], x[1], x[2], x[1]/x[2]) for x in ave_grad]
            ave_grad = "".join(ave_grad)
            logprint(("E{:0>%s}S{:0>%s} (grad x lr) / weight:\n{}" % (num_digit_show_epoch, num_digit_show_step)).format(epoch, step, ave_grad))
          
          codemap.zero_grad()
          loss_codemap = hardloss * 100
          loss_codemap.backward()
          optimizer_c.step()
          for name, param in codemap.named_parameters():
            if param.requires_grad:
              param.data = ema_c(name, param.data)
          
      else:
        imgrec = torch.randn_like(img).cuda()
        imgrec_all.append(imgrec)
        feats = ae.be.forward_branch(imgrec)
        logits = feats[-1]; last_feature = feats[-2]
        logits_all.append(logits.detach())
        label = logits.argmax(dim=1).detach()
        hardloss = nn.CrossEntropyLoss()(logits, label)
        hardloss_dec_all.append(hardloss.item())
        # for accuracy print
        pred = logits.detach().max(1)[1]
        trainacc = pred.eq(label.view_as(pred)).sum().item() / label.size(0)
        trainacc_dec_all.append(trainacc)
        
        ## To maintain the normal log print
        tvloss = (torch.sum(torch.abs(imgrec[:, :, :, :-1] - imgrec[:, :, :, 1:])) + 
                  torch.sum(torch.abs(imgrec[:, :, :-1, :] - imgrec[:, :, 1:, :])))
        imgnorm = torch.pow(torch.norm(imgrec, p=6), 6)
        L_alpha = -torch.norm(last_feature, p=1) / last_feature.size(0)
        pred_prob = logits.softmax(dim=1).mean(dim=0)
        L_ie = torch.dot(pred_prob, torch.log(pred_prob)) / args.num_class

      # Update SE
      hardloss_se_all = []; trainacc_se_all = []; softloss_se_all = []
      for sei in range(1, args.num_se + 1):
        se = eval("ae.se" + str(sei)); optimizer = optimizer_se[sei - 1]; ema = ema_se[sei - 1]
        loss_se = 0
        for i in range(len(imgrec_all)):
          logits = se(imgrec_all[i])
          hardloss = nn.CrossEntropyLoss()(logits, label)
          if args.lw_hard_se: loss_se += hardloss * args.lw_hard_se # Huawei's paper does not mention using this hard loss for SE
          hardloss_se_all.append(hardloss.item())
          # for accuracy print
          pred = logits.detach().max(1)[1]
          trainacc = pred.eq(label.view_as(pred)).sum().item() / label.size(0)
          trainacc_se_all.append(trainacc)
          
          # knowledge distillation loss
          # ref: https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
          softloss = nn.KLDivLoss()(F.log_softmax(logits/args.temp, dim=1),
                          F.softmax(logits_all[i]/args.temp, dim=1)) * (args.temp * args.temp)
          if args.lw_soft: loss_se += softloss * args.lw_soft
          softloss_se_all.append(softloss.item())

          if args.lw_DT:
            logits_DT = se(imgrec_DT_all[i].detach())
            loss_se += nn.CrossEntropyLoss()(logits_DT, label) * args.lw_DT

        se.zero_grad()
        loss_se.backward()
        optimizer.step()
        for name, param in se.named_parameters():
          if param.requires_grad:
            param.data = ema(name, param.data)
      
      # Save sample images
      if (not args.use_random_input) and step % args.save_interval == 0:
        ae.eval()
        # save some test images
        logprint(("E{:0>%s}S{:0>%s} | Saving image samples" % (num_digit_show_epoch, num_digit_show_step)).format(epoch, step))
        if args.use_condition:
          test_codes = torch.randn([args.num_class, args.num_z])
          label_noise = torch.randn([args.num_class, args.num_class])
          test_codes = torch.cat([test_codes, label_noise], dim=1)
          label = label_noise.argmax(dim=1)
          for i in range(len(test_codes)):
            x = test_codes[i].cuda().unsqueeze(0)
            test_label = label[i].item()
            for di in range(1, args.num_dec + 1):
              dec = eval("ae.d%s" % di)
              imgrecs = dec(x)
              imgs = torch.split(imgrecs, num_channel, dim=1)
              for bi in range(len(imgs)):
                img1 = imgs[bi]
                out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_d%s_b%s_imgrec%s_label%s.jpg" % (ExpID, epoch, step, di, bi, i, test_label))
                vutils.save_image(img1.data.cpu().float(), out_img1_path)
        else:
          sign = torch.zeros(args.num_class)
          while sign.sum() != args.num_class:
            x = torch.rand(args.num_class * 5, args.num_z).cuda()
            imgs = ae.d1(x) # TODO: not use multi-decoders and multi-branches for now. Will add these features in the future
            logits = ae.be(imgs)
            test_lables = logits.argmax(dim=1)
            for i in range(len(imgs)):
              img1 = imgs[i]
              test_label = test_lables[i].item()
              # if epoch < 10 and i < args.num_class:
              if i < args.num_class:
                out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_d1_b1_imgrec%s_label%s.jpg" % (ExpID, epoch, step, i, test_label))
                vutils.save_image(img1.data.cpu().float(), out_img1_path)
                sign[i] = 1
              # elif sign[test_label] == 0:
                # out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_d1_b1_label%s.jpg" % (ExpID, epoch, step, test_label))
                # vutils.save_image(img1.data.cpu().float(), out_img1_path)
                # sign[test_label] = 1
                
      # Test and save models
      if step % args.test_interval == 0:
        ae.eval()
        test_acc = 0
        for i, (img, label) in enumerate(test_loader):
          label = label.cuda()
          pred = ae.se1(img.cuda()).detach().max(1)[1]
          test_acc += pred.eq(label.view_as(pred)).sum().item()
        test_acc /= float(num_test)
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
        format_str2 = " | dec:" + " {:.3f}({:.3f})" * args.num_dec * args.num_divbranch
        format_str3 = " | se:" + " {:.3f}({:.3f}) {:.3f}" * args.num_dec * args.num_divbranch 
        format_str4 = " | tv: {:.3f} norm: {:.3f} L_alpha: {:.3f} L_ie: {:.3f} actimax: {:.3f}"
        format_str5 = " ({:.3f}s/step)"
        format_str = "".join([format_str1, format_str2, format_str3, format_str4, format_str5])
        strvalue2 = []; strvalue3 = []
        for i in range(args.num_dec * args.num_divbranch):
          strvalue2.append(hardloss_dec_all[i]); strvalue2.append(trainacc_dec_all[i])
          strvalue3.append(hardloss_se_all[i]); strvalue3.append(trainacc_se_all[i]); strvalue3.append(softloss_se_all[i])
        logprint(format_str.format(
            epoch, step,
            *strvalue2,
            *strvalue3,
            tvloss.item(), imgnorm.item(), L_alpha.item(), L_ie.item(), np.average(actimax_loss_print),
            (time.time() - t1) / args.show_interval))

        t1 = time.time()