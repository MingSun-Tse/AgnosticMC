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
import torch.nn.functional as F
from torch.utils.serialization import load_lua
import torch.utils.data as Data
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.distributions.one_hot_categorical import OneHotCategorical
import torchvision.datasets as datasets
# my libs
from model import AutoEncoders, EMA


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
parser.add_argument('--e1',  type=str,   default="train*/*2/w*/*E17S0*.pth")
parser.add_argument('--e2',  type=str,   default=None)
parser.add_argument('--pretrained_dir',   type=str, default=None, help="the directory of pretrained decoder models")
parser.add_argument('--pretrained_timeid',type=str, default=None, help="the timeid of the pretrained models.")
parser.add_argument('--num_dec', type=int, default=9)
parser.add_argument('--num_se', type=int, default=1)
parser.add_argument('--t',   type=str,   default=None)
parser.add_argument('--gpu', type=int,   default=0)
parser.add_argument('--lr',  type=float, default=1e-3)
parser.add_argument('--b1',  type=float, default=5e-4, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2',  type=float, default=5e-4, help='adam: decay of second order momentum of gradient')
# ----------------------------------------------------------------
# various losses
parser.add_argument('--floss_weight',    type=float, default=1)
parser.add_argument('--ploss_weight',    type=float, default=2)
parser.add_argument('--softloss_weight', type=float, default=10) # According to the paper KD, the soft target loss weight should be considarably larger than that of hard target loss.
parser.add_argument('--hardloss_weight', type=float, default=1)
parser.add_argument('--tvloss_weight',   type=float, default=1e-6)
parser.add_argument('--normloss_weight', type=float, default=1e-4)
parser.add_argument('--daloss_weight',   type=float, default=10)
parser.add_argument('--advloss_weight',  type=float, default=20)
parser.add_argument('--lw_adv',  type=float, default=0.5)
parser.add_argument('--lw_class',  type=float, default=10)
parser.add_argument('--floss_lw', type=str, default="1-1-1-1-1-1-1")
parser.add_argument('--ploss_lw', type=str, default="1-1-1-1-1-1-1")
# ----------------------------------------------------------------
parser.add_argument('-b', '--batch_size', type=int, default=100)
parser.add_argument('-p', '--project_name', type=str, default="test")
parser.add_argument('-r', '--resume', action='store_true')
parser.add_argument('-m', '--mode', type=str, help='the training mode name.')
parser.add_argument('--num_epoch', type=int, default=96)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--use_pseudo_code', action="store_false")
parser.add_argument('--begin', type=float, default=25)
parser.add_argument('--end',   type=float, default=20)
parser.add_argument('--Temp',  type=float, default=1, help="the Tempature in KD")
parser.add_argument('--adv_train', type=int, default=0)
parser.add_argument('--alpha', type=float, default=1, help="a factor to balance the GAN-style loss")
parser.add_argument('--beta', type=float, default=1e-6, help="a factor to balance the GAN-style loss")
parser.add_argument('--G_update_interval', type=int, default=1)
parser.add_argument('--ema_factor', type=float, default=0.9, help="Exponential Moving Average") 
parser.add_argument('--show_interval', type=int, default=50, help="the interval to print logs")
parser.add_argument('--save_interval', type=int, default=1000, help="the interval to save models")
args = parser.parse_args()

# Update and check args
assert(args.num_dec >= 1)
assert(args.mode in AutoEncoders.keys())
args.e1 = path_check(args.e1)
args.e2 = path_check(args.e2)
args.pretrained_dir = path_check(args.pretrained_dir)
args.adv_train = int(args.mode[-1])
assert(args.adv_train in [3,4])
args.pid = os.getpid()

# Set up directories and logs, etc.
TIME_ID = time.strftime("%Y%m%d-%H%M")
project_path = pjoin("../Experiments", TIME_ID + "_" + args.project_name)
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
TIME_ID = "SERVER" + os.environ["SERVER"] + "-" + TIME_ID
log_path = pjoin(weights_path, "log_" + TIME_ID + ".txt")
args.log = sys.stdout if args.debug else open(log_path, "w+")
  
if __name__ == "__main__":
  # Set up model
  AE = AutoEncoders[args.mode]
  if args.adv_train in [0, 1]:
    ae = AE(args.e1, args.d, args.e2)
  elif args.adv_train == 2:
    ae = AE(args.e1, args.d, args.e2, args.t)
  elif args.adv_train in [3, 4]:
    ae = AE(args)
  ae.cuda()
  
  # Set up exponential moving average
  if args.adv_train == 3:
    ema_dec = []
    for di in range(1, args.num_dec+1):
      ema_dec.append(EMA(args.ema_factor))
      dec = eval("ae.d%s"  % di)
      for name, param in dec.named_parameters():
        if param.requires_grad:
          ema_dec[-1].register(name, param.data)
    ema_se = EMA(args.ema_factor)
    for name, param in ae.se.named_parameters():
      if param.requires_grad:
        ema_se.register(name, param.data)
        
  elif args.adv_train == 4:
    ema_dec = []; ema_se = []
    for di in range(1, args.num_dec+1):
      ema_dec.append(EMA(args.ema_factor))
      dec = eval("ae.d%s"  % di)
      for name, param in dec.named_parameters():
        if param.requires_grad:
          ema_dec[-1].register(name, param.data)
    for sei in range(1, args.num_se+1):
      ema_se.append(EMA(args.ema_factor))
      se = eval("ae.se%s" % sei)
      for name, param in se.named_parameters():
        if param.requires_grad:
          ema_se[-1].register(name, param.data)
        
  # Prepare data
  data_train = datasets.MNIST('./MNIST_data', train=True, download=True,
                              transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]))
  data_test = datasets.MNIST('./MNIST_data', train=False, download=True,
                              transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]))
  kwargs = {'num_workers': 4, 'pin_memory': True}
  train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
  
  # Prepare transform and one hot generator
  one_hot = OneHotCategorical(torch.Tensor([1./args.num_class] * args.num_class))
  
  # Prepare test code
  onehot_label = torch.eye(args.num_class)
  test_codes = torch.randn([args.num_class, args.num_class]) * 5.0 + onehot_label * args.begin
  test_labels = onehot_label.data.numpy().argmax(axis=1)
  np.save(pjoin(rec_img_path, "test_codes.npy"), test_codes.data.cpu().numpy())
  
  # Print setting for later check
  logprint(args._get_kwargs())
  
  # Parse to get stage loss weight
  floss_lw = [float(x) for x in args.floss_lw.split("-")]
  ploss_lw = [float(x) for x in args.ploss_lw.split("-")]
  
  # Optimization
  if args.adv_train == 3:
    optimizer_se  = torch.optim.Adam(ae.se.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_dec = []
    for di in range(1, args.num_dec+1):
      dec = eval("ae.d"+str(di))
      optimizer_dec.append(torch.optim.Adam(dec.parameters(),  lr=args.lr, betas=(args.b1, args.b2)))
  elif args.adv_train == 4:
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
      if args.use_pseudo_code:
        onehot_label = one_hot.sample_n(args.batch_size)
        x = torch.randn([args.batch_size, args.num_class]) * (np.random.rand() * 5.0 + 2.0) + onehot_label * np.random.randint(args.end, args.begin) # logits
        x = x.cuda() / args.Temp
        label = onehot_label.data.numpy().argmax(axis=1)
        label = torch.from_numpy(label).long()
      else:
        x = ae.be(img.cuda()) / args.Temp
      prob_gt = F.softmax(x, dim=1) # prob, ground truth
      label = label.cuda()
      
      if args.adv_train == 3:
        # update decoder
        imgrec = []; imgrec_DT = []; hardloss_dec = []; trainacc_dec = []; ave_imgrec = 0
        for di in range(1, args.num_dec+1):
          dec = eval("ae.d" + str(di)); optimizer = optimizer_dec[di-1]; ema = ema_dec[di-1]
          dec.zero_grad()
          imgrec1 = dec(x);       feats1 = ae.be.forward_branch(imgrec1); logits1 = feats1[-1]
          imgrec2 = dec(logits1); feats2 = ae.be.forward_branch(imgrec2); logits2 = feats2[-1]
          imgrec1_DT = ae.defined_trans(imgrec1); logits1_DT = ae.be(imgrec1_DT) # DT: defined transform
          imgrec.append(imgrec1); imgrec_DT.append(imgrec1_DT) # for SE
          ave_imgrec += imgrec1 # to get average img
          
          tvloss1 = args.tvloss_weight * (torch.sum(torch.abs(imgrec1[:, :, :, :-1] - imgrec1[:, :, :, 1:])) + 
                                          torch.sum(torch.abs(imgrec1[:, :, :-1, :] - imgrec1[:, :, 1:, :])))
          tvloss2 = args.tvloss_weight * (torch.sum(torch.abs(imgrec2[:, :, :, :-1] - imgrec2[:, :, :, 1:])) + 
                                          torch.sum(torch.abs(imgrec2[:, :, :-1, :] - imgrec2[:, :, 1:, :])))
          imgnorm1 = torch.pow(torch.norm(imgrec1, p=6), 6) * args.normloss_weight
          imgnorm2 = torch.pow(torch.norm(imgrec2, p=6), 6) * args.normloss_weight

          ploss1 = nn.MSELoss()(feats2[0], feats1[0].data) * args.ploss_weight * ploss_lw[0]
          ploss2 = nn.MSELoss()(feats2[1], feats1[1].data) * args.ploss_weight * ploss_lw[1]
          ploss3 = nn.MSELoss()(feats2[2], feats1[2].data) * args.ploss_weight * ploss_lw[2]
          ploss4 = nn.MSELoss()(feats2[3], feats1[3].data) * args.ploss_weight * ploss_lw[3]
          
          logprob1 = F.log_softmax(logits1/args.Temp, dim=1)
          logprob2 = F.log_softmax(logits2/args.Temp, dim=1)
          softloss1 = nn.KLDivLoss()(logprob1, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
          softloss2 = nn.KLDivLoss()(logprob2, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
          hardloss1 = nn.CrossEntropyLoss()(logits1, label.data) * args.hardloss_weight
          hardloss2 = nn.CrossEntropyLoss()(logits2, label.data) * args.hardloss_weight
          hardloss1_DT = nn.CrossEntropyLoss()(logits1_DT, label.data) * args.daloss_weight
          pred = logits1.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().cpu().data.numpy() / float(args.batch_size)
          hardloss_dec.append(hardloss1.data.cpu().numpy()); trainacc_dec.append(trainacc)
          
          logits_dse = ae.se(imgrec1)
          hardloss_dse = nn.CrossEntropyLoss()(logits_dse, label.data) * args.hardloss_weight
          
          # total loss
          loss = tvloss1 + imgnorm1 + tvloss2 + imgnorm2 + \
                  ploss1 + ploss2 + ploss3 + ploss4 + \
                  softloss1 + softloss2 + hardloss1 + hardloss1_DT + hardloss2 \
                  + args.lw_adv / hardloss_dse
          loss.backward()
          optimizer.step()
          for name, param in dec.named_parameters():
            if param.requires_grad:
              param.data = ema(name, param.data)
        ave_imgrec /= args.num_dec
        
        ## update SE
        ae.se.zero_grad()
        loss_se = 0
        hardloss_se = []; trainacc_se = []
        for di in range(args.num_dec):
          logits = ae.se(imgrec[di].detach())
          logits_DT = ae.se(imgrec_DT[di].detach())
          hardloss = nn.CrossEntropyLoss()(logits, label.data) * args.hardloss_weight
          hardloss_DT = nn.CrossEntropyLoss()(logits_DT, label.data) * args.daloss_weight
          loss_se += hardloss + hardloss_DT
          pred = logits.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().cpu().data.numpy() / float(args.batch_size)
          hardloss_se.append(hardloss.data.cpu().numpy()); trainacc_se.append(trainacc)
        
        # loss_se += nn.CrossEntropyLoss()(ae.se(ave_imgrec), label.data) * args.hardloss_weight # average img loss
        loss_se.backward()
        optimizer_se.step()
        for name, param in ae.se.named_parameters():
          if param.requires_grad:
            param.data = ema_se(name, param.data)
        
      if args.adv_train == 4:
        # update decoder
        imgrec = []; imgrec_DT = []; hardloss_dec = []; trainacc_dec = []; ave_imgrec = 0
        for di in range(1, args.num_dec+1):
          dec = eval("ae.d" + str(di)); optimizer = optimizer_dec[di-1]; ema = ema_dec[di-1]
          dec.zero_grad()
          imgrec1 = dec(x);       feats1 = ae.be.forward_branch(imgrec1); logits1 = feats1[-1]
          imgrec2 = dec(logits1); feats2 = ae.be.forward_branch(imgrec2); logits2 = feats2[-1]
          imgrec1_DT = ae.defined_trans(imgrec1); logits1_DT = ae.be(imgrec1_DT) # DT: defined transform
          imgrec.append(imgrec1); imgrec_DT.append(imgrec1_DT) # for SE
          ave_imgrec += imgrec1 # to get average img
          
          tvloss1 = args.tvloss_weight * (torch.sum(torch.abs(imgrec1[:, :, :, :-1] - imgrec1[:, :, :, 1:])) + 
                                          torch.sum(torch.abs(imgrec1[:, :, :-1, :] - imgrec1[:, :, 1:, :])))
          tvloss2 = args.tvloss_weight * (torch.sum(torch.abs(imgrec2[:, :, :, :-1] - imgrec2[:, :, :, 1:])) + 
                                          torch.sum(torch.abs(imgrec2[:, :, :-1, :] - imgrec2[:, :, 1:, :])))
          imgnorm1 = torch.pow(torch.norm(imgrec1, p=6), 6) * args.normloss_weight
          imgnorm2 = torch.pow(torch.norm(imgrec2, p=6), 6) * args.normloss_weight

          ploss1 = nn.MSELoss()(feats2[0], feats1[0].data) * args.ploss_weight * ploss_lw[0]
          ploss2 = nn.MSELoss()(feats2[1], feats1[1].data) * args.ploss_weight * ploss_lw[1]
          ploss3 = nn.MSELoss()(feats2[2], feats1[2].data) * args.ploss_weight * ploss_lw[2]
          ploss4 = nn.MSELoss()(feats2[3], feats1[3].data) * args.ploss_weight * ploss_lw[3]
          
          logprob1 = F.log_softmax(logits1/args.Temp, dim=1)
          logprob2 = F.log_softmax(logits2/args.Temp, dim=1)
          softloss1 = nn.KLDivLoss()(logprob1, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
          softloss2 = nn.KLDivLoss()(logprob2, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
          hardloss1 = nn.CrossEntropyLoss()(logits1, label.data) * args.hardloss_weight
          hardloss2 = nn.CrossEntropyLoss()(logits2, label.data) * args.hardloss_weight
          hardloss1_DT = nn.CrossEntropyLoss()(logits1_DT, label.data) * args.daloss_weight
          pred = logits1.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().cpu().data.numpy() / float(args.batch_size)
          hardloss_dec.append(hardloss1.data.cpu().numpy()); trainacc_dec.append(trainacc)
          
          advloss = 0
          for sei in range(1, args.num_se+1):
            se = eval("ae.se" + str(sei))
            logits_dse = se(imgrec1)
            advloss += args.lw_adv / nn.CrossEntropyLoss()(logits_dse, label.data) * args.hardloss_weight
          
          class_loss = 0
          for i in range(args.batch_size):
            class_loss += -logits1[i, label[i]] * args.lw_class
          
          ## total loss
          loss = tvloss1 + imgnorm1 + tvloss2 + imgnorm2 + \
                  ploss1 + ploss2 + ploss3 + ploss4 + \
                  softloss1 + softloss2 + hardloss1 + hardloss1_DT + hardloss2 + \
                  advloss + \
                  class_loss
          loss.backward()
          optimizer.step()
          for name, param in dec.named_parameters():
            if param.requires_grad:
              param.data = ema(name, param.data)
        ave_imgrec /= args.num_dec
        
        # update SE
        hardloss_se = []; trainacc_se = []
        for sei in range(1, args.num_se+1):
          se = eval("ae.se" + str(sei)); optimizer = optimizer_se[sei-1]; ema = ema_se[sei-1]
          se.zero_grad()
          loss_se = 0
          for di in range(args.num_dec):
            logits = se(imgrec[di].detach())
            logits_DT = se(imgrec_DT[di].detach())
            hardloss = nn.CrossEntropyLoss()(logits, label.data) * args.hardloss_weight
            hardloss_DT = nn.CrossEntropyLoss()(logits_DT, label.data) * args.daloss_weight
            loss_se += hardloss + hardloss_DT
            pred = logits.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().cpu().data.numpy() / float(args.batch_size)
            hardloss_se.append(hardloss.data.cpu().numpy()); trainacc_se.append(trainacc)
          loss_se.backward()
          optimizer.step()
          for name, param in se.named_parameters():
            if param.requires_grad:
              param.data = ema(name, param.data)
        
      # Print and check the gradient
      # if step % 2000 == 0:
        # ave_grad = []
        # model = ae.dec if args.mode =="BD" else ae.small_enc
        # for p in model.named_parameters(): # get the params in each layer
          # layer_name = p[0].split(".")[0]
          # layer_name = "  "+layer_name if "fc" in layer_name else layer_name
          # if p[1].grad is not None:
            # ave_grad.append([layer_name, np.average(p[1].grad.abs()) * args.lr, np.average(p[1].data.abs())])
        # ave_grad = ["{}: {:.6f} / {:.6f} ({:.10f})\n".format(x[0], x[1], x[2], x[1]/x[2]) for x in ave_grad]
        # ave_grad = "".join(ave_grad)
        # logprint("E{}S{} grad x lr:\n{}".format(epoch, step, ave_grad))
      
      # Check decoder: If decoder does not converge, reinitialize it
      # TODO
      
      # Test and save models
      if step % args.save_interval == 0:
        if args.adv_train in [3, 4]:
          ae.dec = ae.d1; ae.learned_trans = ae.defined_trans
          ae.small_enc = ae.se if args.adv_train == 3 else ae.se1
          ae.enc = ae.be
        ae.eval()
        # save some test images
        for i in range(len(test_codes)):
          x = test_codes[i].cuda()
          if args.adv_train in [3, 4]:
            for di in range(1, args.num_dec+1):
              dec = eval("ae.d%s" % di)
              img1 = dec(x)
              out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_imgrec%s_label=%s_d%s.jpg" % (TIME_ID, epoch, step, i, test_labels[i], di))
              vutils.save_image(img1.data.cpu().float(), out_img1_path)
          else:
            img1 = ae.dec(x)
            img1_DA = ae.learned_trans(img1)
            out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_img%s-rec_label=%s.jpg" % (TIME_ID, epoch, step, i, test_labels[i]))
            out_img1_DA_path = pjoin(rec_img_path, "%s_E%sS%s_img%s-rec_DA_label=%s.jpg" % (TIME_ID, epoch, step, i, test_labels[i]))
            vutils.save_image(img1.data.cpu().float(), out_img1_path) # save some samples to check
            vutils.save_image(img1_DA.data.cpu().float(), out_img1_DA_path) # save some samples to check
        
        # test with the real codes generated from test set
        test_loader = torch.utils.data.DataLoader(data_test,  batch_size=100, shuffle=False, **kwargs)
        softloss1_test = Ssoftloss1_test = test_acc1 = Stest_acc = test_acc = test_acc_advbe = cnt = 0
        for i, (img, label) in enumerate(test_loader):
          x = ae.enc(img.cuda())
          label = label.cuda()
          prob_gt = F.softmax(x, dim=1)
          
          # forward
          img_rec1 = ae.dec(x); logits1 = ae.enc(img_rec1); Slogits = ae.small_enc(img_rec1)
          logprob1  = F.log_softmax(logits1, dim=1)
          Slogprob1 = F.log_softmax(Slogits, dim=1)
          
          # code reconstruction loss
          softloss1_  = nn.KLDivLoss()(logprob1,  prob_gt.data) * args.softloss_weight
          Ssoftloss1_ = nn.KLDivLoss()(Slogprob1, prob_gt.data) * args.softloss_weight
          
          softloss1_test  +=  softloss1_.data.cpu().numpy()
          Ssoftloss1_test += Ssoftloss1_.data.cpu().numpy()
          
          # test cls accuracy
          pred1 = logits1.detach().max(1)[1]; test_acc1 += pred1.eq(label.view_as(pred1)).sum().cpu().data.numpy()
          Spred = Slogits.detach().max(1)[1]; Stest_acc += Spred.eq(label.view_as(Spred)).sum().cpu().data.numpy()
          cnt += 1
           
          # test acc for small enc
          pred = ae.small_enc(img.cuda()).detach().max(1)[1]
          test_acc += pred.eq(label.view_as(pred)).sum().cpu().data.numpy()
          if args.adv_train == 2:
            pred_advbe = ae.advbe(img.cuda()).detach().max(1)[1]
            test_acc_advbe += pred_advbe.eq(label.view_as(pred_advbe)).sum().cpu().data.numpy()
        
        softloss1_test  /= cnt; test_acc1 /= float(len(data_test))
        Ssoftloss1_test /= cnt; Stest_acc /= float(len(data_test))
        test_acc /= float(len(data_test))
        test_acc_advbe /= float(len(data_test))
        
        format_str = "E{}S{} | =======> Test softloss with real logits: test accuracy on SE: {:.4f}"
        logprint(format_str.format(epoch, step, test_acc))
        if args.adv_train in [3, 4]:
          ae.se = ae.se if args.adv_train == 3 else ae.se1
          torch.save(ae.se.state_dict(), pjoin(weights_path, "%s_se_E%sS%s_testacc=%.4f.pth" % (TIME_ID, epoch, step, test_acc)))
          torch.save(ae.d1.state_dict(), pjoin(weights_path, "%s_d1_E%sS%s_testacc1=%.4f.pth" % (TIME_ID, epoch, step, test_acc1)))
          for di in range(2, args.num_dec+1):
            dec = eval("ae.d" + str(di))
            torch.save(dec.state_dict(), pjoin(weights_path, "%s_d%s_E%sS%s.pth" % (TIME_ID, di, epoch, step)))
            
      # Print training loss
      if step % args.show_interval == 0:
        if args.adv_train in [3, 4]:
          format_str1 = "E{}S{} | dec:"
          format_str2 = " {:.4f}({:.3f})" * args.num_dec
          format_str3 = " | se:"
          format_str4 = " | tv: {:.4f} norm: {:.4f} p: {:.4f} {:.4f} {:.4f} {:.4f} ({:.3f}s/step)"
          format_str = "".join([format_str1, format_str2, format_str3, format_str2, format_str4])
          tmp1 = []; tmp2 = []
          for i in range(args.num_dec):
            tmp1.append(hardloss_dec[i])
            tmp1.append(trainacc_dec[i])
            tmp2.append(hardloss_se[i])
            tmp2.append(trainacc_se[i])
          logprint(format_str.format(epoch, step,
              *tmp1, *tmp2,
              tvloss1.data.cpu().numpy(), imgnorm1.data.cpu().numpy(), ploss1.data.cpu().numpy(), ploss2.data.cpu().numpy(), ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(),
              (time.time()-t1)/args.show_interval))
        t1 = time.time()
      
      
  log.close()
