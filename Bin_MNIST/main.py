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


def logprint(some_str, f=sys.stdout):
  print(time.strftime("[%s" % os.getpid() + "-%Y/%m/%d-%H:%M] ") + str(some_str), file=f, flush=True)

  
if __name__ == "__main__":
  # Some settings
  SHOW_INTERVAL = 100
  SAVE_INTERVAL = 2000

  # Passed-in params
  parser = argparse.ArgumentParser(description="Knowledge Transfer")
  parser.add_argument('--e1',  type=str,   default="train*/*2/w*/*E17S0*.pth")
  parser.add_argument('--e2',  type=str,   default=None)
  parser.add_argument('--d',   type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d1',  type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d2',  type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d3',  type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d4',  type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d5',  type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d6',  type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d7',  type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d8',  type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d9',  type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d10', type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d11', type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d12', type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d13', type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d14', type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
  parser.add_argument('--d15', type=str,   default=None)# "../Ex*/*81/w*/*BD*E76S0*.pth")  # "../Ex*/test/w*/*1559_BD_E508S0*.pth"
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
  parser.add_argument('--floss_lw', type=str, default="1-1-1-1-1-1-1")
  parser.add_argument('--ploss_lw', type=str, default="1-1-1-1-1-1-1")
  # ----------------------------------------------------------------
  parser.add_argument('-b', '--batch_size', type=int, default=100)
  parser.add_argument('--test_batch_size',  type=int, default=5)
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
  parser.add_argument('--EMA', type=float, default=0.9, help="Exponential Moving Average") 
  args = parser.parse_args()
  
  # Get path
  args.e1 = glob.glob(args.e1)[0] if args.e1 != None else None
  args.e2 = glob.glob(args.e2)[0] if args.e2 != None else None
  args.d  = glob.glob(args.d )[0] if args.d  != None else None
  args.t  = glob.glob(args.t )[0] if args.t  != None else None
  args.d1 = glob.glob(args.d1)[0] if args.d1 != None else None
  args.d2 = glob.glob(args.d2)[0] if args.d2 != None else None
  args.d3 = glob.glob(args.d3)[0] if args.d3 != None else None
  args.d4 = glob.glob(args.d4)[0] if args.d4 != None else None
  args.d5 = glob.glob(args.d5)[0] if args.d5 != None else None
  args.d6 = glob.glob(args.d6)[0] if args.d6 != None else None
  args.d7 = glob.glob(args.d7)[0] if args.d7 != None else None
  args.d8 = glob.glob(args.d8)[0] if args.d8 != None else None
  args.d9 = glob.glob(args.d9)[0] if args.d9 != None else None
  args.d10 = glob.glob(args.d10)[0] if args.d10 != None else None
  args.d11 = glob.glob(args.d11)[0] if args.d11 != None else None
  args.d12 = glob.glob(args.d12)[0] if args.d12 != None else None
  args.d13 = glob.glob(args.d13)[0] if args.d13 != None else None
  args.d14 = glob.glob(args.d14)[0] if args.d14 != None else None
  args.d15 = glob.glob(args.d15)[0] if args.d15 != None else None
  NUM_DEC = 15
  
  # Check mode
  assert(args.mode in AutoEncoders.keys())
  
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
  log = sys.stdout if args.debug else open(log_path, "w+")
  
  # Set up model
  AE = AutoEncoders[args.mode]
  if args.adv_train in [0, 1]:
    ae = AE(args.e1, args.d, args.e2)
  elif args.adv_train == 2:
    ae = AE(args.e1, args.d, args.e2, args.t)
  elif args.adv_train == 3:
    d = (args.d1, args.d2, args.d3, args.d4, args.d5, args.d6, args.d7, args.d8, args.d9,
         args.d10, args.d11, args.d12, args.d13, args.d14, args.d15)
    ae = AE(args.e1, d, args.e2)
  ae.cuda()
  
  # Set up exponential moving average
  if args.adv_train == 0:
    ema = EMA(args.EMA)
    for name, param in ae.named_parameters():
      if param.requires_grad:
        ema.register(name, param.data)
  elif args.adv_train == 1:
    ema_BD = EMA(args.EMA)
    ema_SE = EMA(args.EMA)
    ema_AdvBE = EMA(args.EMA)
    for name, param in ae.dec.named_parameters():
      if param.requires_grad:
        ema_BD.register(name, param.data)
    for name, param in ae.small_enc.named_parameters():
      if param.requires_grad:
        ema_SE.register(name, param.data)
    for name, param in ae.advbe.named_parameters():
      if param.requires_grad:
        ema_AdvBE.register(name, param.data)
  elif args.adv_train == 2:
    ema_BD    = EMA(args.EMA)
    ema_SE    = EMA(args.EMA)
    ema_AdvBE = EMA(args.EMA); ema_AdvBE2 = EMA(args.EMA) 
    ema_trans = EMA(args.EMA); ema_trans2 = EMA(args.EMA)
    for name, param in ae.dec.named_parameters():
      if param.requires_grad:
        ema_BD.register(name, param.data)
    for name, param in ae.small_enc.named_parameters():
      if param.requires_grad:
        ema_SE.register(name, param.data)
    for name, param in ae.advbe.named_parameters():
      if param.requires_grad:
        ema_AdvBE.register(name, param.data)
    for name, param in ae.learned_trans.named_parameters():
      if param.requires_grad:
        ema_trans.register(name, param.data)
    for name, param in ae.advbe2.named_parameters():
      if param.requires_grad:
        ema_AdvBE2.register(name, param.data)
    for name, param in ae.learned_trans2.named_parameters():
      if param.requires_grad:
        ema_trans2.register(name, param.data)
  elif args.adv_train == 3:
    ema_d1  = EMA(args.EMA); ema_d2  = EMA(args.EMA); ema_d3  = EMA(args.EMA); ema_d4  = EMA(args.EMA); ema_d5  = EMA(args.EMA)
    ema_d6  = EMA(args.EMA); ema_d7  = EMA(args.EMA); ema_d8  = EMA(args.EMA); ema_d9  = EMA(args.EMA); ema_d10 = EMA(args.EMA)
    ema_d11 = EMA(args.EMA); ema_d12 = EMA(args.EMA); ema_d13 = EMA(args.EMA); ema_d14 = EMA(args.EMA); ema_d15 = EMA(args.EMA)
    for di in range(NUM_DEC):
      dec = eval("ae.d%s"  % (di+1))
      ema = eval("ema_d%s" % (di+1))
      for name, param in dec.named_parameters():
        if param.requires_grad:
          ema.register(name, param.data)
    ema_se = EMA(args.EMA)
    for name, param in ae.se.named_parameters():
      if param.requires_grad:
        ema_se.register(name, param.data)
        
  # Prepare data
  data_train = datasets.MNIST('./MNIST_data',
                              train=True,
                              download=True,
                              transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
                             )
  data_test = datasets.MNIST('./MNIST_data',
                              train=False,
                              download=True,
                              transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
                             )
  kwargs = {'num_workers': 4, 'pin_memory': True}
  train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
  
  # Prepare transform and one hot generator
  one_hot = OneHotCategorical(torch.Tensor([1./args.num_class] * args.num_class))
  
  # Prepare test code
  onehot_label = torch.eye(10)
  test_codes = torch.randn([args.num_class, args.num_class]) * 5.0 + onehot_label * args.begin
  test_labels = onehot_label.data.numpy().argmax(axis=1)
  np.save(pjoin(rec_img_path, "test_codes.npy"), test_codes.data.cpu().numpy())
  
  # Print setting for later check
  logprint(str(args._get_kwargs()), log)
  
  # Parse to get stage loss weight
  floss_lw = [float(x) for x in args.floss_lw.split("-")]
  ploss_lw = [float(x) for x in args.ploss_lw.split("-")]
  
  # Optimization
  if args.adv_train == 0:
    optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, betas=(args.b1, args.b2))
  elif args.adv_train == 1:
    optimizer_SE    = torch.optim.Adam(ae.small_enc.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_BD    = torch.optim.Adam(ae.dec.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_AdvBE = torch.optim.Adam(ae.advbe.parameters(), lr=args.lr, betas=(args.b1, args.b2))
  elif args.adv_train == 2:
    optimizer_SE    = torch.optim.Adam(ae.small_enc.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_BD    = torch.optim.Adam(ae.dec.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_AdvBE = torch.optim.Adam(ae.advbe.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_trans = torch.optim.Adam(ae.learned_trans.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_AdvBE2 = torch.optim.Adam(ae.advbe2.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_trans2 = torch.optim.Adam(ae.learned_trans2.parameters(), lr=args.lr, betas=(args.b1, args.b2))
  elif args.adv_train == 3:
    optimizer_se  = torch.optim.Adam(ae.se.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d1  = torch.optim.Adam(ae.d1.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d2  = torch.optim.Adam(ae.d2.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d3  = torch.optim.Adam(ae.d3.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d4  = torch.optim.Adam(ae.d4.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d5  = torch.optim.Adam(ae.d5.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d6  = torch.optim.Adam(ae.d6.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d7  = torch.optim.Adam(ae.d7.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d8  = torch.optim.Adam(ae.d8.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d9  = torch.optim.Adam(ae.d9.parameters(),  lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d10 = torch.optim.Adam(ae.d10.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d11 = torch.optim.Adam(ae.d11.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d12 = torch.optim.Adam(ae.d12.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d13 = torch.optim.Adam(ae.d13.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d14 = torch.optim.Adam(ae.d14.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_d15 = torch.optim.Adam(ae.d15.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    
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
        x = ae.enc(img.cuda()) / args.Temp
      prob_gt = F.softmax(x, dim=1) # prob, ground truth
      label = label.cuda()
      
      if args.adv_train == 1: # adversarial loss: Decoder should generate the images that can be "recognized" by the Encoder but "unrecognized" by the SE.
        img_rec1 = ae.dec(x); img_rec1_DA = ae.defined_trans(img_rec1)
        
        ## update AdvBE
        ae.advbe.zero_grad()
        logits_AdvBE = ae.advbe(img_rec1.detach()); hardloss_AdvBE = nn.CrossEntropyLoss()(logits_AdvBE, label.data) * args.hardloss_weight
        pred_AdvBE = logits_AdvBE.detach().max(1)[1]; trainacc_AdvBE = pred_AdvBE.eq(label.view_as(pred_AdvBE)).sum().cpu().data.numpy() / float(args.batch_size)
        loss_AdvBE = hardloss_AdvBE
        loss_AdvBE.backward()
        optimizer_AdvBE.step()
        
        # apply EMA, after updating params
        for name, param in ae.advbe.named_parameters():
          if param.requires_grad:
            param.data = ema_AdvBE(name, param.data)
        
        ## update BD
        ae.dec.zero_grad()
        # (1) loss from AdvBE
        hardloss1_AdvBE_ = nn.CrossEntropyLoss()(ae.advbe(img_rec1), label.data) * args.hardloss_weight # not detach
        
        # (2) loss from BD itself
        feats1 = ae.enc.forward_branch(img_rec1); logits1 = feats1[-1]; img_rec2 = ae.dec(logits1)
        feats2 = ae.enc.forward_branch(img_rec2); logits2 = feats2[-1]
        
        # SE KLDivLoss: Align SE to BE
        Slogits = ae.small_enc(img_rec1); Slogits_DA = ae.small_enc(img_rec1_DA)
        Slogprob = F.log_softmax(Slogits/args.Temp, dim=1); prob_BE = F.softmax(logits1/args.Temp, dim=1)
        Ssoftloss = nn.KLDivLoss()(Slogprob, prob_BE.data) * (args.Temp*args.Temp) * args.softloss_weight
        Shardloss = nn.CrossEntropyLoss()(Slogits, label.data) * args.hardloss_weight
        Shardloss_DA = nn.CrossEntropyLoss()(Slogits_DA, label.data) * args.daloss_weight
        Spred = Slogits.detach().max(1)[1]; trainacc_SE = Spred.eq(label.view_as(Spred)).sum().cpu().data.numpy() / float(args.batch_size)
        Spred_DA = Slogits_DA.detach().max(1)[1]; trainacc_DA_SE = Spred_DA.eq(label.view_as(Spred_DA)).sum().cpu().data.numpy() / float(args.batch_size)
        
        # total variation loss and image norm loss, from "2015-CVPR-Understanding Deep Image Representations by Inverting Them"
        tvloss1 = args.tvloss_weight * (torch.sum(torch.abs(img_rec1[:, :, :, :-1] - img_rec1[:, :, :, 1:])) + 
                                        torch.sum(torch.abs(img_rec1[:, :, :-1, :] - img_rec1[:, :, 1:, :])))
        tvloss2 = args.tvloss_weight * (torch.sum(torch.abs(img_rec2[:, :, :, :-1] - img_rec2[:, :, :, 1:])) + 
                                        torch.sum(torch.abs(img_rec2[:, :, :-1, :] - img_rec2[:, :, 1:, :])))
        img_norm1 = torch.pow(torch.norm(img_rec1, p=6), 6) * args.normloss_weight
        img_norm2 = torch.pow(torch.norm(img_rec2, p=6), 6) * args.normloss_weight
        
        # perceptual loss: train the big decoder
        ploss1 = nn.MSELoss()(feats2[0], feats1[0].data) * args.ploss_weight * ploss_lw[0]
        ploss2 = nn.MSELoss()(feats2[1], feats1[1].data) * args.ploss_weight * ploss_lw[1]
        ploss3 = nn.MSELoss()(feats2[2], feats1[2].data) * args.ploss_weight * ploss_lw[2]
        ploss4 = nn.MSELoss()(feats2[3], feats1[3].data) * args.ploss_weight * ploss_lw[3]
        
        # code reconstruction loss (KL Divergence)
        logprob1 = F.log_softmax(logits1/args.Temp, dim=1)
        logprob2 = F.log_softmax(logits2/args.Temp, dim=1)
        softloss1 = nn.KLDivLoss()(logprob1, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        softloss2 = nn.KLDivLoss()(logprob2, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        
        # hard target loss
        logits1_DA = ae.enc(img_rec1_DA)
        hardloss1 = nn.CrossEntropyLoss()(logits1, label.data) * args.hardloss_weight
        hardloss1_DA = nn.CrossEntropyLoss()(logits1_DA, label.data) * args.daloss_weight
        hardloss2 = nn.CrossEntropyLoss()(logits2, label.data) * args.hardloss_weight
        pred_BE = logits1.detach().max(1)[1]; trainacc_BE = pred_BE.eq(label.view_as(pred_BE)).sum().cpu().data.numpy() / float(args.batch_size)
        pred_BE_DA = logits1_DA.detach().max(1)[1]; trainacc_BE_DA = pred_BE_DA.eq(label.view_as(pred_BE_DA)).sum().cpu().data.numpy() / float(args.batch_size)
        
        # total loss
        loss_BDSE = (hardloss1 + hardloss1_DA + softloss1 + softloss2) + (tvloss1 + img_norm1) + (Ssoftloss + Shardloss + Shardloss_DA) \
                  + (ploss1 + ploss2 + ploss3 + ploss4) \
                  + tvloss2 + img_norm2 \
                  - (hardloss1_AdvBE_) * args.alpha \
                  - softloss1 / Ssoftloss.data * args.advloss_weight
                  
        if step % args.G_update_interval == 0:
          loss_BDSE.backward()
          optimizer_SE.step()
          optimizer_BD.step()
          # apply EMA, after updating params
          for name, param in ae.dec.named_parameters():
            if param.requires_grad:
              param.data = ema_BD(name, param.data)
          for name, param in ae.small_enc.named_parameters():
            if param.requires_grad:
              param.data = ema_SE(name, param.data)
              
      if args.adv_train == 0:
        # forward
        img_rec1, feats1, logits1_DA, Sfeats1, Slogits1_DA, img_rec2, feats2 = ae(x)
        
        # total variation loss and image norm loss, from "2015-CVPR-Understanding Deep Image Representations by Inverting Them"
        tvloss1 = args.tvloss_weight * (torch.sum(torch.abs(img_rec1[:, :, :, :-1] - img_rec1[:, :, :, 1:])) + 
                                        torch.sum(torch.abs(img_rec1[:, :, :-1, :] - img_rec1[:, :, 1:, :])))
        tvloss2 = args.tvloss_weight * (torch.sum(torch.abs(img_rec2[:, :, :, :-1] - img_rec2[:, :, :, 1:])) + 
                                        torch.sum(torch.abs(img_rec2[:, :, :-1, :] - img_rec2[:, :, 1:, :])))
        img_norm1 = torch.pow(torch.norm(img_rec1, p=6), 6) * args.normloss_weight
        img_norm2 = torch.pow(torch.norm(img_rec2, p=6), 6) * args.normloss_weight
        
        # code reconstruction loss (KL Divergence): train both
        logits1  =  feats1[-1];  logprob1 = F.log_softmax( logits1/args.Temp, dim=1)
        logits2  =  feats2[-1];  logprob2 = F.log_softmax( logits2/args.Temp, dim=1)
        Slogits1 = Sfeats1[-1]; Slogprob1 = F.log_softmax(Slogits1/args.Temp, dim=1)
        softloss1  = nn.KLDivLoss()( logprob1, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        softloss2  = nn.KLDivLoss()( logprob2, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        Ssoftloss1 = nn.KLDivLoss()(Slogprob1, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        
        # feature reconstruction loss: train the small encoder
        floss3 = nn.MSELoss()(Sfeats1[2], feats1[2].data) * args.floss_weight * floss_lw[2]
        floss4 = nn.MSELoss()(Sfeats1[3], feats1[3].data) * args.floss_weight * floss_lw[3]
        
        # perceptual loss: train the big decoder
        ploss1 = nn.MSELoss()(feats2[0], feats1[0].data) * args.ploss_weight * ploss_lw[0]
        ploss2 = nn.MSELoss()(feats2[1], feats1[1].data) * args.ploss_weight * ploss_lw[1]
        ploss3 = nn.MSELoss()(feats2[2], feats1[2].data) * args.ploss_weight * ploss_lw[2]
        ploss4 = nn.MSELoss()(feats2[3], feats1[3].data) * args.ploss_weight * ploss_lw[3]
        
        # hard target loss: train both 
        hardloss1  = nn.CrossEntropyLoss()( logits1, label.data) * args.hardloss_weight
        hardloss2  = nn.CrossEntropyLoss()( logits2, label.data) * args.hardloss_weight
        Shardloss1 = nn.CrossEntropyLoss()(Slogits1, label.data) * args.hardloss_weight
        
        # semantic consistency loss
        hardloss1_DA  = nn.CrossEntropyLoss()( logits1_DA, label.data) * args.daloss_weight
        Shardloss1_DA = nn.CrossEntropyLoss()(Slogits1_DA, label.data) * args.daloss_weight
        
        # train cls accuracy
        pred1 =    logits1.detach().max(1)[1]; trainacc1 = pred1.eq(label.view_as(pred1)).sum().cpu().data.numpy() / float(args.batch_size)
        pred2 = logits1_DA.detach().max(1)[1]; trainacc2 = pred2.eq(label.view_as(pred2)).sum().cpu().data.numpy() / float(args.batch_size)
      
        # Total loss settings ----------------------------------------------
        # (1.1) basic setting: BD fixed, train SE 
        # loss = Ssoftloss1 + Shardloss1 + floss3 + floss4 
        # (1.2) train SE, add DA loss
        # loss = Ssoftloss1 + Shardloss1 + floss3 + floss4 + Shardloss1_DA
               
        # (2) joint-training: both BD and SE are trainable
        loss = softloss1 + hardloss1 + softloss2 + hardloss2 + ploss1 + ploss2 + ploss3 + ploss4 + tvloss1 + tvloss2 + img_norm1 + img_norm2 + hardloss1_DA + \
               Ssoftloss1 + Shardloss1 + floss3 + floss4 + Shardloss1_DA + \
               softloss1 / Ssoftloss1.data * args.advloss_weight
        # ------------------------------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # apply EMA, after updating params
        for name, param in ae.named_parameters():
          if param.requires_grad:
            param.data = ema(name, param.data)
      
      if args.adv_train == 2:
        img_rec1 = ae.dec(x); img_rec1_definedDA = ae.defined_trans(img_rec1)
        ## update BD
        ae.dec.zero_grad()
        feats1 = ae.enc.forward_branch(img_rec1); logits1 = feats1[-1]; img_rec2 = ae.dec(logits1)
        feats2 = ae.enc.forward_branch(img_rec2); logits2 = feats2[-1]
        logits1_definedDA = ae.enc(img_rec1_definedDA)
        
        # total variation loss and image norm loss, from "2015-CVPR-Understanding Deep Image Representations by Inverting Them"
        tvloss1 = args.tvloss_weight * (torch.sum(torch.abs(img_rec1[:, :, :, :-1] - img_rec1[:, :, :, 1:])) + 
                                        torch.sum(torch.abs(img_rec1[:, :, :-1, :] - img_rec1[:, :, 1:, :])))
        tvloss2 = args.tvloss_weight * (torch.sum(torch.abs(img_rec2[:, :, :, :-1] - img_rec2[:, :, :, 1:])) + 
                                        torch.sum(torch.abs(img_rec2[:, :, :-1, :] - img_rec2[:, :, 1:, :])))
        img_norm1 = torch.pow(torch.norm(img_rec1, p=6), 6) * args.normloss_weight
        img_norm2 = torch.pow(torch.norm(img_rec2, p=6), 6) * args.normloss_weight

        # perceptual loss: train the big decoder
        ploss1 = nn.MSELoss()(feats2[0], feats1[0].data) * args.ploss_weight * ploss_lw[0]
        ploss2 = nn.MSELoss()(feats2[1], feats1[1].data) * args.ploss_weight * ploss_lw[1]
        ploss3 = nn.MSELoss()(feats2[2], feats1[2].data) * args.ploss_weight * ploss_lw[2]
        ploss4 = nn.MSELoss()(feats2[3], feats1[3].data) * args.ploss_weight * ploss_lw[3]
        
        # soft loss (KL Divergence) and hard target loss
        logprob1 = F.log_softmax(logits1/args.Temp, dim=1)
        logprob2 = F.log_softmax(logits2/args.Temp, dim=1)
        softloss1 = nn.KLDivLoss()(logprob1, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        softloss2 = nn.KLDivLoss()(logprob2, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        hardloss1 = nn.CrossEntropyLoss()(logits1, label.data) * args.hardloss_weight
        hardloss2 = nn.CrossEntropyLoss()(logits2, label.data) * args.hardloss_weight
        hardloss1_definedDA = nn.CrossEntropyLoss()(logits1_definedDA, label.data) * args.daloss_weight
        pred_BE = logits1.detach().max(1)[1]; trainacc_BE = pred_BE.eq(label.view_as(pred_BE)).sum().cpu().data.numpy() / float(args.batch_size)
        
        # total loss
        loss_BD = tvloss1 + img_norm1 + tvloss2 + img_norm2 + \
                  ploss1 + ploss2 + ploss3 + ploss4 + \
                  softloss1 + hardloss1 + hardloss1_definedDA + softloss2 + hardloss2 \
                  
        loss_BD.backward()
        optimizer_BD.step()
        for name, param in ae.dec.named_parameters():
          if param.requires_grad:
            param.data = ema_BD(name, param.data)
        
        ## update AdvBE
        ae.advbe.zero_grad()
        logits_AdvBE = ae.advbe(img_rec1.detach()); hardloss_AdvBE = nn.CrossEntropyLoss()(logits_AdvBE, label.data) * args.hardloss_weight
        pred_AdvBE = logits_AdvBE.detach().max(1)[1]; trainacc_AdvBE = pred_AdvBE.eq(label.view_as(pred_AdvBE)).sum().cpu().data.numpy() / float(args.batch_size)
        
        # img_rec1_DA = ae.learned_trans(img_rec1.detach())
        # logits_DA_AdvBE = ae.advbe(img_rec1_DA.detach()); hardloss_DA_AdvBE = nn.CrossEntropyLoss()(logits_DA_AdvBE, label.data) * args.hardloss_weight
        
        loss_AdvBE = hardloss_AdvBE # + 0.1 * hardloss_DA_AdvBE
        loss_AdvBE.backward()
        optimizer_AdvBE.step()
        for name, param in ae.advbe.named_parameters():
          if param.requires_grad:
            param.data = ema_AdvBE(name, param.data)
        
        # # AdvBE2
        # ae.advbe2.zero_grad()
        # logits_AdvBE2 = ae.advbe2(img_rec1.detach()); hardloss_AdvBE2 = nn.CrossEntropyLoss()(logits_AdvBE2, label.data) * args.hardloss_weight
        # loss_AdvBE2 = hardloss_AdvBE2
        # loss_AdvBE2.backward()
        # optimizer_AdvBE2.step()
        # for name, param in ae.advbe2.named_parameters():
          # if param.requires_grad:
            # param.data = ema_AdvBE2(name, param.data)

        ## update learned transform
        ae.learned_trans.zero_grad()
        img_rec1_DA = ae.learned_trans(img_rec1.detach()); img_rec1_DA_definedDA = ae.defined_trans(img_rec1_DA)
        logits1_DA        = ae.enc(img_rec1_DA);    loss_trans_BE    = nn.CrossEntropyLoss()(logits1_DA,       label.data) * args.hardloss_weight
        logits1_DA_AdvBE  = ae.advbe(img_rec1_DA);  loss_trans_AdvBE = nn.CrossEntropyLoss()(logits1_DA_AdvBE, label.data) * args.hardloss_weight
        logits1_DA_definedDA = ae.enc(img_rec1_DA_definedDA); loss_trans_BE_definedDA = nn.CrossEntropyLoss()(logits1_DA_definedDA, label.data) * args.daloss_weight
        
        # DA img loss
        tvloss1_DA = args.tvloss_weight * (torch.sum(torch.abs(img_rec1_DA[:, :, :, :-1] - img_rec1_DA[:, :, :, 1:])) + 
                                           torch.sum(torch.abs(img_rec1_DA[:, :, :-1, :] - img_rec1_DA[:, :, 1:, :]))) * 0.25
        img_norm1_DA = torch.pow(torch.norm(img_rec1_DA, p=6), 6) * args.normloss_weight * 0.25
        
        # # LT2
        # img_rec1_DA2 = ae.learned_trans2(img_rec1.detach())
        # logits1_DA2       = ae.enc(img_rec1_DA2);    loss_trans_BE2    = nn.CrossEntropyLoss()(logits1_DA2,       label.data) * args.hardloss_weight
        # logits1_DA_AdvBE2 = ae.advbe2(img_rec1_DA2); loss_trans_AdvBE2 = nn.CrossEntropyLoss()(logits1_DA_AdvBE2, label.data) * args.hardloss_weight
        
        loss_trans  = loss_trans_BE / (loss_trans_AdvBE * args.alpha) + tvloss1_DA + img_norm1_DA # + (loss_trans_BE  - loss_trans_AdvBE)  * (loss_trans_BE  - loss_trans_AdvBE)  * args.beta
        # loss_trans2 = loss_trans_BE2 / (loss_trans_AdvBE2 * args.alpha) + (loss_trans_BE2 - loss_trans_AdvBE2) * (loss_trans_BE2 - loss_trans_AdvBE2) * args.beta
        
        pred_DA_BE    = logits1_DA.detach().max(1)[1];       trainacc_DA_BE    = pred_DA_BE.eq(label.view_as(pred_DA_BE)).sum().cpu().data.numpy() / float(args.batch_size)
        pred_DA_AdvBE = logits1_DA_AdvBE.detach().max(1)[1]; trainacc_DA_AdvBE = pred_DA_AdvBE.eq(label.view_as(pred_DA_AdvBE)).sum().cpu().data.numpy() / float(args.batch_size)
        
        loss_trans.backward();  optimizer_trans.step()
        # loss_trans2.backward(); optimizer_trans2.step()
        for name, param in ae.learned_trans.named_parameters():
          if param.requires_grad:
            param.data = ema_trans(name, param.data)
        # for name, param in ae.learned_trans2.named_parameters():
          # if param.requires_grad:
            # param.data = ema_trans2(name, param.data)
        
        ## update SE
        ae.small_enc.zero_grad()
        Sfeats1 = ae.small_enc.forward_branch(img_rec1.detach()); Slogits = Sfeats1[-1]
        Slogprob = F.log_softmax(Slogits/args.Temp, dim=1); prob_BE = F.softmax(logits1/args.Temp, dim=1)
        Ssoftloss = nn.KLDivLoss()(Slogprob, prob_BE.data) * (args.Temp*args.Temp) * args.softloss_weight
        Shardloss = nn.CrossEntropyLoss()(Slogits, label.data) * args.hardloss_weight
        
        # feature reconstruction loss: train the small encoder
        floss3 = nn.MSELoss()(Sfeats1[2], feats1[2].data) * args.floss_weight * floss_lw[2]
        floss4 = nn.MSELoss()(Sfeats1[3], feats1[3].data) * args.floss_weight * floss_lw[3]
        
        Slogits_DA  = ae.small_enc(img_rec1_DA.detach());  Shardloss_DA  = nn.CrossEntropyLoss()(Slogits_DA,  label.data) * (hardloss1.data / loss_trans_BE.data)
        # Slogits_DA2 = ae.small_enc(img_rec1_DA2.detach()); Shardloss_DA2 = nn.CrossEntropyLoss()(Slogits_DA2, label.data)
        
        Spred    = Slogits.detach().max(1)[1];    trainacc_SE    = Spred.eq(label.view_as(Spred)).sum().cpu().data.numpy() / float(args.batch_size)
        Spred_DA = Slogits_DA.detach().max(1)[1]; trainacc_DA_SE = Spred_DA.eq(label.view_as(Spred_DA)).sum().cpu().data.numpy() / float(args.batch_size)
        
        loss_SE = Ssoftloss + Shardloss + Shardloss_DA + (floss3 + floss4) #+ Shardloss_DA2
        loss_SE.backward()
        optimizer_SE.step()
        for name, param in ae.small_enc.named_parameters():
          if param.requires_grad:
            param.data = ema_SE(name, param.data)
        
      if args.adv_train == 3:
        # update decoder
        imgrec = []; imgrec_DT = []; hardloss_dec = []; trainacc_dec = []; ave_imgrec = 0
        for i in range(NUM_DEC):
          dec = eval("ae.d%s" % (i+1)); optimizer = eval("optimizer_d%s" % (i+1)); ema = eval("ema_d%s" % (i+1))
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
                  softloss1 + hardloss1 + hardloss1_DT + softloss2 + hardloss2 \
                  + 0.5 / hardloss_dse
          loss.backward()
          optimizer.step()
          for name, param in dec.named_parameters():
            if param.requires_grad:
              param.data = ema(name, param.data)
        ave_imgrec /= NUM_DEC
        
        ## update SE
        ae.se.zero_grad()
        loss_se = 0
        hardloss_se = []; trainacc_se = []
        for di in range(NUM_DEC):
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
        # logprint("E{}S{} grad x lr:\n{}".format(epoch, step, ave_grad), log)
      
      # Print training loss
      if step % SHOW_INTERVAL == 0:
        if args.adv_train:
          if args.adv_train == 1:
            format_str = "E{}S{} | BE: {:.5f}({:.3f}) {:.5f}({:.3f}) | AdvBE: {:.5f}({:.3f}) | SE: {:.5f}({:.3f}) {:.5f}({:.3f}) | soft: {:.5f} {:.5f} tv: {:.5f} {:.5f} norm: {:.5f} {:.5f} p: {:.5f} {:.5f} {:.5f} {:.5f} ({:.3f}s/step)"
            logprint(format_str.format(epoch, step,
                
                hardloss1.data.cpu().numpy(), trainacc_BE, hardloss1_DA.data.cpu().numpy(), trainacc_BE_DA,
                hardloss_AdvBE.data.cpu().numpy(), trainacc_AdvBE,
                Shardloss.data.cpu().numpy(), trainacc_SE, Shardloss_DA.data.cpu().numpy(), trainacc_DA_SE,
                
                softloss1.data.cpu().numpy(), softloss2.data.cpu().numpy(),
                tvloss1.data.cpu().numpy(), tvloss2.data.cpu().numpy(),
                img_norm1.data.cpu().numpy(), img_norm2.data.cpu().numpy(),
                ploss1.data.cpu().numpy(), ploss2.data.cpu().numpy(), ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(),
                (time.time()-t1)/SHOW_INTERVAL), log)
          
          elif args.adv_train == 2:
            format_str = "E{}S{} | BE: {:.5f}({:.3f}) AdvBE: {:.5f}({:.3f}) | DA_BE: {:.5f}({:.3f}) DA_AdvBE: {:.5f}({:.3f}) | SE: {:.5f}({:.3f}) {:.5f}({:.3f}) | soft: {:.5f} tv: {:.5f} norm: {:.5f} p: {:.5f} {:.5f} {:.5f} {:.5f} ({:.3f}s/step)"
            logprint(format_str.format(epoch, step,
                hardloss1.data.cpu().numpy(),        trainacc_BE,
                hardloss_AdvBE.data.cpu().numpy(),   trainacc_AdvBE, # cls loss before the learned transform
                loss_trans_BE.data.cpu().numpy(),    trainacc_DA_BE, # cls loss after the learned transform
                loss_trans_AdvBE.data.cpu().numpy(), trainacc_DA_AdvBE, 
                Shardloss.data.cpu().numpy(), trainacc_SE, Shardloss_DA.data.cpu().numpy(), trainacc_DA_SE, # cls loss for SE
                softloss1.data.cpu().numpy(), tvloss1.data.cpu().numpy(), img_norm1.data.cpu().numpy(), ploss1.data.cpu().numpy(), ploss2.data.cpu().numpy(), ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(),
                (time.time()-t1)/SHOW_INTERVAL), log)
          
          elif args.adv_train == 3:
            format_str1 = "E{}S{}"
            format_str2 = " | dec: {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f})"
            format_str3 =  " | se: {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f}) {:.4f}({:.3f})"
            format_str4 = " | tv: {:.5f} norm: {:.5f} p: {:.5f} {:.5f} {:.5f} {:.5f} ({:.3f}s/step)"
            format_str = "".join([format_str1, format_str2, format_str3, format_str4])
            logprint(format_str.format(epoch, step,
                hardloss_dec[0], trainacc_dec[0], hardloss_dec[1], trainacc_dec[1], hardloss_dec[2], trainacc_dec[2], hardloss_dec[3], trainacc_dec[3], hardloss_dec[4], trainacc_dec[4], hardloss_dec[5], trainacc_dec[5], hardloss_dec[6], trainacc_dec[6], hardloss_dec[7], trainacc_dec[7], hardloss_dec[8], trainacc_dec[8],
                hardloss_dec[9], trainacc_dec[9], hardloss_dec[10], trainacc_dec[10], hardloss_dec[11], trainacc_dec[11], hardloss_dec[12], trainacc_dec[12], hardloss_dec[13], trainacc_dec[13], hardloss_dec[14], trainacc_dec[14],
                hardloss_se[0 ], trainacc_se[0 ], hardloss_se[1 ], trainacc_se[1 ], hardloss_se[2 ], trainacc_se[2 ], hardloss_se[3 ], trainacc_se[3 ], hardloss_se[4 ], trainacc_se[4 ], hardloss_se[5 ], trainacc_se[5 ], hardloss_se[6 ], trainacc_se[6 ], hardloss_se[7 ], trainacc_se[7 ], hardloss_se[8 ], trainacc_se[8 ],
                hardloss_se[9], trainacc_se[9], hardloss_se[10], trainacc_se[10], hardloss_se[11], trainacc_se[11], hardloss_se[12], trainacc_se[12], hardloss_se[13], trainacc_se[13], hardloss_se[14], trainacc_se[14],
                softloss1.data.cpu().numpy(), tvloss1.data.cpu().numpy(), imgnorm1.data.cpu().numpy(), ploss1.data.cpu().numpy(), ploss2.data.cpu().numpy(), ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(),
                (time.time()-t1)/SHOW_INTERVAL), log)
            
        else:
          if args.mode in ["BD", "BDSE"]:
            format_str = "E{}S{} loss: {:.3f} | soft: {:.5f} {:.5f} | tv: {:.5f} {:.5f} | norm: {:.5f} {:.5f} | hard: {:.5f}({:.4f}) {:.5f}({:.4f}) {:.5f} | p: {:.5f} {:.5f} {:.5f} {:.5f} ({:.3f}s/step)"
            logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), softloss1.data.cpu().numpy(), softloss2.data.cpu().numpy(),
                tvloss1.data.cpu().numpy(), tvloss2.data.cpu().numpy(),
                img_norm1.data.cpu().numpy(), img_norm2.data.cpu().numpy(),
                hardloss1.data.cpu().numpy(), trainacc1, hardloss1_DA.data.cpu().numpy(), trainacc2, Shardloss1_DA.data.cpu().numpy(),
                ploss1.data.cpu().numpy(), ploss2.data.cpu().numpy(), ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(),
                (time.time()-t1)/SHOW_INTERVAL), log)
          
          elif args.mode == "SE":
            format_str = "E{}S{} loss: {:.3f} | soft: {:.5f} {:.5f} | tv: {:.5f} {:.5f} | hard: {:.5f}({:.4f}) {:.5f}({:.4f}) | f: {:.5f} {:.5f} | p: {:.5f} {:.5f} {:.5f} {:.5f} ({:.3f}s/step)"
            logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), softloss1.data.cpu().numpy(), softloss2.data.cpu().numpy(),
                tvloss1.data.cpu().numpy(), tvloss2.data.cpu().numpy(),
                hardloss1.data.cpu().numpy(), trainacc1, hardloss2.data.cpu().numpy(), trainacc2,
                floss3.data.cpu().numpy(), floss4.data.cpu().numpy(), 
                ploss1.data.cpu().numpy(), ploss2.data.cpu().numpy(), ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(), 
                (time.time()-t1)/SHOW_INTERVAL), log)
        t1 = time.time()
      
      # Test and save models
      if step % SAVE_INTERVAL == 0:
        if args.adv_train == 3:
          ae.dec = ae.d1; ae.small_enc = ae.se; ae.learned_trans = ae.defined_trans       
          ae.enc = ae.be
        ae.eval()
        # save some test images
        for i in range(len(test_codes)):
          x = test_codes[i].cuda()
          if args.adv_train == 3:
            for di in range(7):
              dec = eval("ae.d%s" % (di+1))
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
        
        format_str = "E{}S{} | =======> Test softloss with real logits: BE: {:.5f}({:.3f}) SE: {:.5f}({:.3f}) | test accuracy on SE: {:.4f} | test accuracy on AdvBE: {:.4f}"
        logprint(format_str.format(epoch, step, softloss1_test, test_acc1, Ssoftloss1_test, Stest_acc, test_acc, test_acc_advbe), log)
        if args.adv_train == 0:
          torch.save(ae.dec.state_dict(), pjoin(weights_path, "%s_BD_E%sS%s_testacc1=%.4f.pth" % (TIME_ID, epoch, step, test_acc1)))
          torch.save(ae.small_enc.state_dict(), pjoin(weights_path, "%s_SE_E%sS%s_testacc=%.4f.pth" % (TIME_ID, epoch, step, test_acc)))
        elif args.adv_train == 2:
          torch.save(ae.dec.state_dict(), pjoin(weights_path, "%s_BD_E%sS%s_testacc1=%.4f.pth" % (TIME_ID, epoch, step, test_acc1)))
          torch.save(ae.small_enc.state_dict(), pjoin(weights_path, "%s_SE_E%sS%s_testacc=%.4f.pth" % (TIME_ID, epoch, step, test_acc)))
          torch.save(ae.learned_trans.state_dict(), pjoin(weights_path, "%s_LT_E%sS%s.pth" % (TIME_ID, epoch, step)))
          torch.save(ae.advbe.state_dict(), pjoin(weights_path, "%s_AdvBE_E%sS%s.pth" % (TIME_ID, epoch, step)))
        elif args.adv_train == 3:
          torch.save(ae.d1.state_dict(), pjoin(weights_path, "%s_d1_E%sS%s_testacc1=%.4f.pth" % (TIME_ID, epoch, step, test_acc1)))
          torch.save(ae.d2.state_dict(), pjoin(weights_path, "%s_d2_E%sS%s.pth" % (TIME_ID, epoch, step)))
          torch.save(ae.d3.state_dict(), pjoin(weights_path, "%s_d3_E%sS%s.pth" % (TIME_ID, epoch, step)))
          torch.save(ae.d4.state_dict(), pjoin(weights_path, "%s_d4_E%sS%s.pth" % (TIME_ID, epoch, step)))
          torch.save(ae.d5.state_dict(), pjoin(weights_path, "%s_d5_E%sS%s.pth" % (TIME_ID, epoch, step)))
          torch.save(ae.d6.state_dict(), pjoin(weights_path, "%s_d6_E%sS%s.pth" % (TIME_ID, epoch, step)))
          torch.save(ae.d7.state_dict(), pjoin(weights_path, "%s_d7_E%sS%s.pth" % (TIME_ID, epoch, step)))
          torch.save(ae.d8.state_dict(), pjoin(weights_path, "%s_d8_E%sS%s.pth" % (TIME_ID, epoch, step)))
          torch.save(ae.d9.state_dict(), pjoin(weights_path, "%s_d9_E%sS%s.pth" % (TIME_ID, epoch, step)))
          torch.save(ae.se.state_dict(), pjoin(weights_path, "%s_se_E%sS%s_testacc=%.4f.pth" % (TIME_ID, epoch, step, test_acc)))
  
  log.close()
