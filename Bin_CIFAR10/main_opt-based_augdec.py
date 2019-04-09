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
parser.add_argument('--lw_masknorm', type=float, default=1e-5)
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
  ema_dec = []; ema_se = []; ema_mask = []
  for di in range(1, args.num_dec+1):
    ema_dec.append(EMA(args.ema_factor))
    ema_mask.append(EMA(args.ema_factor))
    dec = eval("ae.d%s"  % di)
    masknet = ae.mask
    for name, param in dec.named_parameters():
      if param.requires_grad:
        ema_dec[-1].register(name, param.data)
    for name, param in masknet.named_parameters():
      if param.requires_grad:
        ema_mask[-1].register(name, param.data)
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
  optimizer_se   = [] 
  optimizer_dec  = []
  optimizer_mask = []
  for di in range(1, args.num_dec + 1):
    dec = eval("ae.d" + str(di))
    masknet = ae.mask
    optimizer_dec.append(torch.optim.Adam(dec.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
    optimizer_mask.append(torch.optim.Adam(masknet.parameters(), lr=args.lr, betas=(args.b1, args.b2)))
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
      z_concat = torch.cat([random_z1, random_z2], dim=0)
      onehot_label = one_hot.sample_n(args.batch_size).view([args.batch_size, args.num_class]).cuda()
      label_concat = torch.cat([onehot_label, onehot_label], dim=0)
      x = torch.cat([z_concat, label_concat], dim=1).detach() # input to the Generator network
      label = label_concat.data.cpu().numpy().argmax(axis=1)
      label = torch.from_numpy(label).long().detach().cuda()
        
      # Update decoder
      imgrec_all = []; imgrec_DT_all = []; hardloss_dec_all = []; trainacc_dec_all = []
      for di in range(1, args.num_dec + 1):
        # Set up model and ema
        dec = eval("ae.d" + str(di)); optimizer_d = optimizer_dec[di-1]; ema_d = ema_dec[di-1]
        masknet = ae.mask; optimizer_m = optimizer_mask[di-1]; ema_m = ema_mask[di-1]
        
        # Forward
        decfeats_imgrecs = dec.forward_branch(x)
        decfeats, imgrecs = decfeats_imgrecs[:-1], decfeats_imgrecs[-1]
        mask = masknet(x)
        imgrecs_masked = torch.zeros_like(imgrecs).cuda()
        for i in range(imgrecs.size(1)):
          imgrecs_masked[:,i,:,:] = mask.squeeze(1).mul(imgrecs[:,i,:,:])
          
        # Update masknet
        total_loss_mask = 0
        loss_mask_norm = torch.norm(mask, p=1) * args.lw_masknorm # for sparsity
        mask_1, mask_2 = torch.split(mask, args.batch_size, dim=0)
        loss_mask_diversity = -torch.mean(torch.abs(mask_1 - mask_2)) / torch.mean(torch.abs(random_z1 - random_z2)) * 100
        total_loss_mask += loss_mask_diversity + loss_mask_norm
        imgrecs_masked_split = torch.split(imgrecs_masked, 3, dim=1)
        for imgrec_masked in imgrecs_masked_split:
          logits = ae.be(tensor_normalize(imgrec_masked))
          total_loss_mask += nn.CrossEntropyLoss()(logits, label) * args.lw_hard
        masknet.zero_grad()
        total_loss_mask.backward(retain_graph=True)
        optimizer_m.step()
        for name, param in masknet.named_parameters():
          if param.requires_grad:
            param.data = ema_m(name, param.data)
        
        ## Diversity encouraging loss: MSGAN
        # ref: 2019 CVPR Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis
        total_loss_dec = 0
        lz_feat = 0; lz_pixel = 0
        for decfeat in decfeats:
          decfeat_1, decfeat_2 = torch.split(decfeat, args.batch_size, dim=0)
          lz_feat += torch.mean(torch.abs(decfeat_1 - decfeat_2)) / torch.mean(torch.abs(random_z1 - random_z2)) * 0.1
        lz_feat /= len(decfeats)
        if args.msgan_option == "pixel":
          imgrecs_1, imgrecs_2 = torch.split(imgrecs_masked, args.batch_size, dim=0)
          lz_pixel += torch.mean(torch.abs(imgrecs_1 - imgrecs_2)) / torch.mean(torch.abs(random_z1 - random_z2))
        elif args.msgan_option == "pixelgray": # deprecated
          imgrecs_1, imgrecs_2 = torch.split(imgrecs_masked, args.batch_size, dim=0)
          imgrecs_1 = imgrecs_1[:,0,:,:] * 0.299 + imgrecs_1[:,1,:,:] * 0.587 + imgrecs_1[:,2,:,:] * 0.114 # the Y channel (Luminance) of a image
          imgrecs_2 = imgrecs_2[:,0,:,:] * 0.299 + imgrecs_2[:,1,:,:] * 0.587 + imgrecs_2[:,2,:,:] * 0.114
          lz_pixel += torch.mean(torch.abs(imgrecs_1 - imgrecs_2)) / torch.mean(torch.abs(random_z1 - random_z2))
        loss_diversity_feat, loss_diversity_pixel = -args.lw_msgan * lz_feat, -args.lw_msgan * lz_pixel
        loss_diversity = loss_diversity_feat + loss_diversity_pixel
        total_loss_dec += loss_diversity
        
        imgrecs_split = torch.split(imgrecs, 3, dim=1) # 3 channels
        actimax_loss_print = []
        for imgrec1 in imgrecs_split:
          # forward
          feats1 = ae.be.forward_branch(tensor_normalize(imgrec1)); logits1 = feats1[-1]
          imgrec1_DT = ae.defined_trans(imgrec1); logits1_DT = ae.be(tensor_normalize(imgrec1_DT)) # DT: defined transform
          imgrec_all.append(imgrec1); imgrec_DT_all.append(imgrec1_DT) # for SE
          
          ## Low-level natural image prior: tv + image norm
          # ref: 2015 CVPR Understanding Deep Image Representations by Inverting Them
          tvloss = args.lw_tv * (torch.sum(torch.abs(imgrec1[:, :, :, :-1] - imgrec1[:, :, :, 1:])) + 
                                 torch.sum(torch.abs(imgrec1[:, :, :-1, :] - imgrec1[:, :, 1:, :])))
          imgnorm = torch.pow(torch.norm(imgrec1, p=6), 6) * args.lw_norm
          total_loss_dec += tvloss + imgnorm 
          
          ## Classification loss, the bottomline loss
          # logprob1 = F.log_softmax(logits1/args.temp, dim=1)
          # logprob2 = F.log_softmax(logits2/args.temp, dim=1)
          # softloss1 = nn.KLDivLoss()(logprob1, prob_gt.data) * (args.temp*args.temp) * args.lw_soft
          # softloss2 = nn.KLDivLoss()(logprob2, prob_gt.data) * (args.temp*args.temp) * args.lw_soft
          hardloss = nn.CrossEntropyLoss()(logits1, label) * args.lw_hard
          hardloss_DT = nn.CrossEntropyLoss()(logits1_DT, label) * args.lw_DA
          total_loss_dec += hardloss + hardloss_DT
          # for accuracy print
          pred = logits1.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().item() / label.size(0)
          hardloss_dec_all.append(hardloss.item()); trainacc_dec_all.append(trainacc)
          index = len(imgrec_all) - 1
          history_acc_dec_all[index] = history_acc_dec_all[index] * args.history_acc_weight + trainacc * (1 - args.history_acc_weight)
          
          ## Adversarial loss, combat with SE
          if args.lw_adv:
            for sei in range(1, args.num_se+1):
              se = eval("ae.se" + str(sei))
              logits_dse = se(imgrec1)
              total_loss_dec += args.lw_adv / nn.CrossEntropyLoss()(logits_dse, label)
          
          ## Activation maximization loss
          # ref: 2016 IJCV Visualizing Deep Convolutional Neural Networks Using Natural Pre-images
          if args.clip_actimax and epoch >= 7: args.lw_actimax = 0
          rand_loss_weight = torch.rand_like(logits1) * args.noise_magnitude
          for i in range(logits1.size(0)):
            rand_loss_weight[i, label[i]] = 1
          actimax_loss = -args.lw_actimax * (torch.dot(logits1.flatten(), rand_loss_weight.flatten()) / logits1.size(0))
          actimax_loss_print.append(actimax_loss.item())
          if args.lw_actimax:
            loss_actimax_diversity_attraction = \
              (actimax_loss - loss_diversity.data - 30) * (actimax_loss - loss_diversity.data - 30) if actimax_loss > loss_diversity + 30 else 0
          else:
            loss_actimax_diversity_attraction = 0
          total_loss_dec += actimax_loss + loss_actimax_diversity_attraction
        
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
          logprint(("E{:0<%s}S{:0<%s} (grad x lr) / weight:\n{}" % (num_digit_show_epoch, num_digit_show_step)).format(epoch, step, ave_grad))

     # Update SE
      hardloss_se_all = []; trainacc_se_all = []
      for sei in range(1, args.num_se + 1):
        se = eval("ae.se" + str(sei)); optimizer = optimizer_se[sei-1]; ema = ema_se[sei-1]
        
        loss_se = 0
        for i in range(len(imgrec_all)):
          logits = se(tensor_normalize(imgrec_all[i].detach()))
          logits_DT = se(tensor_normalize(imgrec_DT_all[i].detach()))
          hardloss = nn.CrossEntropyLoss()(logits, label) * args.lw_hard
          hardloss_DT = nn.CrossEntropyLoss()(logits_DT, label) * args.lw_DA
          loss_se += hardloss + hardloss_DT
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
        logprint(("E{:0<%s}S{:0<%s} | Saving image samples" % (num_digit_show_epoch, num_digit_show_step)).format(epoch, step))
        onehot_label = torch.eye(args.num_class)
        test_codes = torch.cat([torch.randn([args.num_class, args.num_z]), onehot_label], dim=1)
        test_labels = onehot_label.numpy().argmax(axis=1)        
        for i in range(len(test_codes)):
          x = test_codes[i].cuda().unsqueeze(0)
          for di in range(1, args.num_dec + 1):
            dec = eval("ae.d%s" % di); masknet = ae.mask
            imgrecs = dec(x); mask = masknet(x)
            imgrecs_masked = torch.zeros_like(imgrecs).cuda()
            for k in range(imgrecs.size(1)):
              imgrecs_masked[:,k,:,:] = mask.squeeze(1).mul(imgrecs[:,k,:,:])
            
            imgs = torch.split(imgrecs, 3, dim=1)
            imgs_masked = torch.split(imgrecs_masked, 3, dim=1)
            for bi in range(len(imgs)):
              img1, img2 = imgs[bi], imgs_masked[bi]
              out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_d%s_b%s_label%s.jpg"        % (ExpID, epoch, step, di, bi, test_labels[i]))
              out_img2_path = pjoin(rec_img_path, "%s_E%sS%s_d%s_b%s_masked_label%s.jpg" % (ExpID, epoch, step, di, bi, test_labels[i]))
              vutils.save_image(img1.data.cpu().float(), out_img1_path)
              vutils.save_image(img2.data.cpu().float(), out_img2_path)
            
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
        format_str = "E{:0<%s}S{:0<%s} | " % (num_digit_show_epoch, num_digit_show_step) + "=" * (int(TimeID[-1]) + 1) + "> Test accuracy on SE: {:.4f} (ExpID: {})"
        logprint(format_str.format(epoch, step, test_acc, ExpID))
        # torch.save(ae.se1.state_dict(), pjoin(weights_path, "%s_se_E%sS%s_testacc=%.4f.pth" % (ExpID, epoch, step, test_acc)))
        # torch.save(ae.d1.state_dict(), pjoin(weights_path, "%s_d1_E%sS%s.pth" % (ExpID, epoch, step)))
        # for di in range(2, args.num_dec+1):
          # dec = eval("ae.d" + str(di))
          # torch.save(dec.state_dict(), pjoin(weights_path, "%s_d%s_E%sS%s.pth" % (ExpID, di, epoch, step)))

      # Print training loss
      if step % args.show_interval == 0:
        format_str1 = "E{:0<%s}S{:0<%s}" % (num_digit_show_epoch, num_digit_show_step)
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
            tvloss.item(), imgnorm.item(), loss_diversity_feat.item(), loss_diversity_pixel.item(), np.average(actimax_loss_print),
            loss_mask_diversity.item(), loss_mask_norm.item(),
            (time.time()-t1)/args.show_interval))

        t1 = time.time()