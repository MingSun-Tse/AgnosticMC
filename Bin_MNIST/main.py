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
  parser.add_argument('--d',   type=str,   default=None)#"../Ex*/*81/w*/*BD*E76S0*.pth")
  parser.add_argument('--gpu', type=int,   default=0)
  parser.add_argument('--lr',  type=float, default=5e-4)
  # ----------------------------------------------------------------
  # various losses
  parser.add_argument('--floss_weight',    type=float, default=1)
  parser.add_argument('--ploss_weight',    type=float, default=2)
  parser.add_argument('--softloss_weight', type=float, default=10) # According to the paper KD, the soft target loss weight should be considarably larger than that of hard target loss.
  parser.add_argument('--hardloss_weight', type=float, default=1)
  parser.add_argument('--tvloss_weight',   type=float, default=1e-6)
  parser.add_argument('--normloss_weight', type=float, default=1e-4)
  parser.add_argument('--daloss_weight',   type=float, default=10)
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
  parser.add_argument('--begin', type=float, default=40)
  parser.add_argument('--end',   type=float, default=25)
  parser.add_argument('--Temp',  type=float, default=1, help="the Tempature in KD")
  args = parser.parse_args()
  
  # Get path
  args.e1 = glob.glob(args.e1)[0] if args.e1 != None else None
  args.e2 = glob.glob(args.e2)[0] if args.e2 != None else None
  args.d  = glob.glob(args.d )[0] if args.d  != None else None
  
  # Check mode
  assert(args.mode in AutoEncoders.keys())
  
  # Set up directories and logs, etc.
  project_path = pjoin("../Experiments", args.project_name)
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
  TIME_ID = "SERVER" + os.environ["SERVER"] + time.strftime("-%Y%m%d-%H%M")
  log_path = pjoin(weights_path, "log_" + TIME_ID + ".txt")
  log = sys.stdout if args.debug else open(log_path, "w+")
  
  # Set up model
  AE = AutoEncoders[args.mode]
  ae = AE(args.e1, args.d, args.e2)
  ae.cuda()
  
  # Set up exponential moving average
  ema = EMA(0.9)
  for name, param in ae.named_parameters():
    if param.requires_grad:
      ema.register(name, param.data)

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
  train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,      shuffle=True, **kwargs)
  test_loader  = torch.utils.data.DataLoader(data_test,  batch_size=args.test_batch_size, shuffle=True, **kwargs)
  
  
  # Prepare transform and one hot generator
  one_hot = OneHotCategorical(torch.Tensor([1./args.num_class] * args.num_class))
  
  # Prepare test code
  _, (test_imgs, test_labels) = list(enumerate(test_loader))[0]
  test_codes = ae.enc(test_imgs.cuda())
  np.save(pjoin(rec_img_path, "test_codes.npy"), test_codes.data.cpu().numpy())
  
  # Print setting for later check
  logprint(str(args._get_kwargs()), log)
  
  # Parse to get stage loss weight
  floss_lw = [float(x) for x in args.floss_lw.split("-")]
  ploss_lw = [float(x) for x in args.ploss_lw.split("-")]
  
  # Optimization  
  optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr)
  loss_func = nn.MSELoss()
  t1 = time.time()
  
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
  
  for epoch in range(previous_epoch, args.num_epoch):
    for step, (img, label) in enumerate(train_loader):
      ae.train()
      # Generate codes randomly
      if args.use_pseudo_code:
        onehot_label = one_hot.sample_n(args.batch_size)
        x = torch.randn([args.batch_size, args.num_class]) * 5.0 + onehot_label * (args.begin - (args.begin-args.end)/args.num_epoch * epoch) # logits
        x = x.cuda() / args.Temp
        label = onehot_label.data.numpy().argmax(axis=1)
        label = torch.from_numpy(label).long()
      else:
        x = ae.enc(img.cuda()) / args.Temp
      prob_gt = F.softmax(x, dim=1) # prob, ground truth
      label = label.cuda()
              
      # forward
      if args.mode == "BD":
        img_rec1, feats1, img_rec2, feats2 = ae(x)
        # total variation loss
        tvloss1 = args.tvloss_weight * (torch.sum(torch.abs(img_rec1[:, :, :, :-1] - img_rec1[:, :, :, 1:])) + 
                                        torch.sum(torch.abs(img_rec1[:, :, :-1, :] - img_rec1[:, :, 1:, :])))
        tvloss2 = args.tvloss_weight * (torch.sum(torch.abs(img_rec2[:, :, :, :-1] - img_rec2[:, :, :, 1:])) + 
                                        torch.sum(torch.abs(img_rec2[:, :, :-1, :] - img_rec2[:, :, 1:, :])))
        # code loss: KL Divergence
        logits1 = feats1[-1]; logprob1 = F.log_softmax(logits1/args.Temp, dim=1) 
        logits2 = feats2[-1]; logprob2 = F.log_softmax(logits2/args.Temp, dim=1)
        softloss1 = nn.KLDivLoss()(logprob1, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        softloss2 = nn.KLDivLoss()(logprob2, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        # perceptual loss
        ploss1 = loss_func(feats2[0], feats1[0].data) * args.ploss_weight * ploss_lw[0]
        ploss2 = loss_func(feats2[1], feats1[1].data) * args.ploss_weight * ploss_lw[1]
        ploss3 = loss_func(feats2[2], feats1[2].data) * args.ploss_weight * ploss_lw[2]
        ploss4 = loss_func(feats2[3], feats1[3].data) * args.ploss_weight * ploss_lw[3]
        # hard target loss
        hardloss1 = nn.CrossEntropyLoss()(logits1, label.data) * args.hardloss_weight
        hardloss2 = nn.CrossEntropyLoss()(logits2, label.data) * args.hardloss_weight
        # total loss
        loss = softloss1 + softloss2 + ploss1 + ploss2 + ploss3 + ploss4 + tvloss1 + tvloss2 + hardloss1 + hardloss2
        # train cls accuracy
        pred1 = logits1.detach().max(1)[1]; train_acc1 = pred1.eq(label.view_as(pred1)).sum().cpu().data.numpy() / float(args.batch_size)
        pred2 = logits2.detach().max(1)[1]; train_acc2 = pred2.eq(label.view_as(pred2)).sum().cpu().data.numpy() / float(args.batch_size)
        
      elif args.mode == "SE":
        img_rec1, feats1, Sfeats1, img_rec2, feats2 = ae(x)
        # total variation loss
        tvloss1 = args.tvloss_weight * (torch.sum(torch.abs(img_rec1[:, :, :, :-1] - img_rec1[:, :, :, 1:])) + 
                                        torch.sum(torch.abs(img_rec1[:, :, :-1, :] - img_rec1[:, :, 1:, :])))
        tvloss2 = args.tvloss_weight * (torch.sum(torch.abs(img_rec2[:, :, :, :-1] - img_rec2[:, :, :, 1:])) + 
                                        torch.sum(torch.abs(img_rec2[:, :, :-1, :] - img_rec2[:, :, 1:, :])))
        # code loss: KL Divergence
        logits1 = Sfeats1[-1]; logprob1 = F.log_softmax(logits1/args.Temp, dim=1)
        logits2 =  feats2[-1]; logprob2 = F.log_softmax(logits2/args.Temp, dim=1)
        softloss1 = nn.KLDivLoss()(logprob1, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        softloss2 = nn.KLDivLoss()(logprob2, prob_gt.data) * (args.Temp*args.Temp) * args.softloss_weight
        # feature reconstruction loss
        floss3 = loss_func(Sfeats1[2], feats1[2].data) * args.floss_weight * floss_lw[2]
        floss4 = loss_func(Sfeats1[3], feats1[3].data) * args.floss_weight * floss_lw[3]
        # perceptual loss
        ploss1 = loss_func(feats2[0], feats1[0].data) * args.ploss_weight * ploss_lw[0]
        ploss2 = loss_func(feats2[1], feats1[1].data) * args.ploss_weight * ploss_lw[1]
        ploss3 = loss_func(feats2[2], feats1[2].data) * args.ploss_weight * ploss_lw[2]
        ploss4 = loss_func(feats2[3], feats1[3].data) * args.ploss_weight * ploss_lw[3]
        # hard classification loss
        hardloss1 = nn.CrossEntropyLoss()(logits1, label.data) * args.hardloss_weight
        hardloss2 = nn.CrossEntropyLoss()(logits2, label.data) * args.hardloss_weight
        # total loss
        loss = softloss1 + softloss2 + ploss1 + ploss2 + ploss3 + ploss4 + floss3 + floss4 + tvloss1 + tvloss2 + hardloss1 + hardloss2
        # train cls accuracy
        pred1 = logits1.detach().max(1)[1]; train_acc1 = pred1.eq(label.view_as(pred1)).sum().cpu().data.numpy() / float(args.batch_size)
        pred2 = logits2.detach().max(1)[1]; train_acc2 = pred2.eq(label.view_as(pred2)).sum().cpu().data.numpy() / float(args.batch_size)
      
      elif args.mode == "BDSE":
        # forward
        img_rec1, feats1, logits1_trans, Sfeats1, Slogits1_trans, img_rec2, feats2 = ae(x)
        
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
        floss3 = loss_func(Sfeats1[2], feats1[2].data) * args.floss_weight * floss_lw[2]
        floss4 = loss_func(Sfeats1[3], feats1[3].data) * args.floss_weight * floss_lw[3]
        
        # perceptual loss: train the big decoder
        ploss1 = loss_func(feats2[0], feats1[0].data) * args.ploss_weight * ploss_lw[0]
        ploss2 = loss_func(feats2[1], feats1[1].data) * args.ploss_weight * ploss_lw[1]
        ploss3 = loss_func(feats2[2], feats1[2].data) * args.ploss_weight * ploss_lw[2]
        ploss4 = loss_func(feats2[3], feats1[3].data) * args.ploss_weight * ploss_lw[3]
        
        # hard target loss: train both 
        hardloss1   = nn.CrossEntropyLoss()( logits1, label.data) * args.hardloss_weight
        hardloss2   = nn.CrossEntropyLoss()( logits2, label.data) * args.hardloss_weight
        Shardloss1  = nn.CrossEntropyLoss()(Slogits1, label.data) * args.hardloss_weight
        
        # semantic consistency loss
        hardloss1_DA  = nn.CrossEntropyLoss()( logits1_trans, label.data) * args.daloss_weight
        Shardloss1_DA = nn.CrossEntropyLoss()(Slogits1_trans, label.data) * args.daloss_weight
        
        # Total loss settings ----------------------------------------------
        # (1.1) basic setting: BD fixed, train SE 
        # loss = Ssoftloss1 + Shardloss1 + floss3 + floss4 
        # (1.2) train SE, add DA loss
        # loss = Ssoftloss1 + Shardloss1 + floss3 + floss4 + Shardloss1_DA
               
        # (2) joint-training: both BD and SE are trainable
        loss = softloss1 + hardloss1 + softloss2 + hardloss2 + ploss1 + ploss2 + ploss3 + ploss4 + tvloss1 + tvloss2 + img_norm1 + img_norm2 + hardloss1_DA + \
               Ssoftloss1 + Shardloss1 + floss3 + floss4 + \
               softloss1 / Ssoftloss1.data * 20
        # ------------------------------------------------------------------
        
        # train cls accuracy
        pred1 =       logits1.detach().max(1)[1]; train_acc1 = pred1.eq(label.view_as(pred1)).sum().cpu().data.numpy() / float(args.batch_size)
        pred2 = logits1_trans.detach().max(1)[1]; train_acc2 = pred2.eq(label.view_as(pred2)).sum().cpu().data.numpy() / float(args.batch_size)
      
      optimizer.zero_grad()
      loss.backward()
      
      # check the gradient
      if step % 2000 == 0:
        ave_grad = []
        model = ae.dec if args.mode =="BD" else ae.small_enc
        for p in model.named_parameters(): # get the params in each layer
          layer_name = p[0].split(".")[0]
          layer_name = "  "+layer_name if "fc" in layer_name else layer_name
          if p[1].grad is not None:
            ave_grad.append([layer_name, np.average(p[1].grad.abs()) * args.lr, np.average(p[1].data.abs())])
        ave_grad = ["{}: {:.6f} / {:.6f} ({:.10f})\n".format(x[0], x[1], x[2], x[1]/x[2]) for x in ave_grad]
        ave_grad = "".join(ave_grad)
        logprint("\n=> E{}S{} grad x lr:\n{}".format(epoch, step, ave_grad), log)
      
      optimizer.step()
      # apply EMA, after updating params
      for name, param in ae.named_parameters():
        if param.requires_grad:
          param.data = ema(name, param.data)
      
      # Print training loss
      if step % SHOW_INTERVAL == 0:
        if args.mode in ["BD", "BDSE"]:
          format_str = "E{}S{} loss: {:.3f} | soft: {:.5f} {:.5f} | tv: {:.5f} {:.5f} | norm: {:.5f} {:.5f} | hard: {:.5f}({:.4f}) {:.5f}({:.4f}) {:.5f} | p: {:.5f} {:.5f} {:.5f} {:.5f} ({:.3f}s/step)"
          logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), softloss1.data.cpu().numpy(), softloss2.data.cpu().numpy(),
              tvloss1.data.cpu().numpy(), tvloss2.data.cpu().numpy(),
              img_norm1.data.cpu().numpy(), img_norm2.data.cpu().numpy(),
              hardloss1.data.cpu().numpy(), train_acc1, hardloss1_DA.data.cpu().numpy(), train_acc2, Shardloss1_DA.data.cpu().numpy(),
              ploss1.data.cpu().numpy(), ploss2.data.cpu().numpy(), ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(),
              (time.time()-t1)/SHOW_INTERVAL), log)
        
        elif args.mode == "SE":
          format_str = "E{}S{} loss: {:.3f} | soft: {:.5f} {:.5f} | tv: {:.5f} {:.5f} | hard: {:.5f}({:.4f}) {:.5f}({:.4f}) | f: {:.5f} {:.5f} | p: {:.5f} {:.5f} {:.5f} {:.5f} ({:.3f}s/step)"
          logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), softloss1.data.cpu().numpy(), softloss2.data.cpu().numpy(),
              tvloss1.data.cpu().numpy(), tvloss2.data.cpu().numpy(),
              hardloss1.data.cpu().numpy(), train_acc1, hardloss2.data.cpu().numpy(), train_acc2,
              floss3.data.cpu().numpy(), floss4.data.cpu().numpy(), 
              ploss1.data.cpu().numpy(), ploss2.data.cpu().numpy(), ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(), 
              (time.time()-t1)/SHOW_INTERVAL), log)
        
        t1 = time.time()
      
      # Test and save models
      if step % SAVE_INTERVAL == 0:
        ae.eval()
        # save some test images
        for i in range(len(test_codes)):
          x = test_codes[i].cuda()
          img1 = ae.dec(x)
          img2 = ae.dec(ae.enc(img1))
          out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_img%s-rec1_label=%s.jpg" % (TIME_ID, epoch, step, i, test_labels[i].data.numpy()))
          out_img2_path = pjoin(rec_img_path, "%s_E%sS%s_img%s-rec2_label=%s.jpg" % (TIME_ID, epoch, step, i, test_labels[i].data.numpy()))
          vutils.save_image(img1.data.cpu().float(), out_img1_path) # save some samples to check
          vutils.save_image(img2.data.cpu().float(), out_img2_path) # save some samples to check
        
        # test with the real codes generated from test set
        test_loader = torch.utils.data.DataLoader(data_test,  batch_size=64, shuffle=True, **kwargs)
        if args.mode == "BD":
          softloss1_test = softloss2_test = test_acc1 = test_acc2 = cnt = 0
          for i, (img, label) in enumerate(test_loader):
            x = ae.enc(img.cuda())
            label = label.cuda()
            prob_gt = F.softmax(x, dim=1)
            img_rec1, feats1, img_rec2, feats2 = ae(x)
            logits1 = feats1[-1]; logprob1 = F.log_softmax(logits1/args.Temp, dim=1)
            logits2 = feats2[-1]; logprob2 = F.log_softmax(logits2/args.Temp, dim=1)
            softloss1_ = nn.KLDivLoss()(logprob1, prob_gt.data) * args.softloss_weight
            softloss2_ = nn.KLDivLoss()(logprob2, prob_gt.data) * args.softloss_weight
            softloss1_test += softloss1_.data.cpu().numpy()
            softloss2_test += softloss2_.data.cpu().numpy()
            # test cls accuracy
            pred1 = logits1.detach().max(1)[1]; test_acc1 += pred1.eq(label.view_as(pred1)).sum().cpu().data.numpy()
            pred2 = logits2.detach().max(1)[1]; test_acc2 += pred2.eq(label.view_as(pred2)).sum().cpu().data.numpy()
            cnt += 1
          softloss1_test /= cnt; test_acc1 /= float(len(data_test))
          softloss2_test /= cnt; test_acc2 /= float(len(data_test))
          
          format_str = "E{}S{} test softloss: {:.5f}({:.4f}) {:.5f}({:.4f})"
          logprint(format_str.format(epoch, step, softloss1_test, test_acc1, softloss2_test, test_acc2), log)
          torch.save(ae.dec.state_dict(), pjoin(weights_path, "%s_BD_E%sS%s_testacc1=%.4f.pth" % (TIME_ID, epoch, step, test_acc1)))
          
        elif args.mode == "SE":
          test_acc = 0
          for i, (img, label) in enumerate(test_loader):
            label = label.cuda()
            pred = ae.small_enc(img.cuda()).detach().max(1)[1]
            test_acc += pred.eq(label.view_as(pred)).sum().cpu().data.numpy()
          test_acc /= float(len(data_test))
          
          logprint("E{}S{} test accuracy: {:.4f}".format(epoch, step, test_acc), log)
          torch.save(ae.small_enc.state_dict(), pjoin(weights_path, "%s_SE_E%sS%s_testacc=%.4f.pth" % (TIME_ID, epoch, step, test_acc)))
          
        elif args.mode == "BDSE":
          softloss1_test = softloss2_test = Ssoftloss1_test = test_acc1 = test_acc2 = Stest_acc1 = test_acc = cnt = 0
          for i, (img, label) in enumerate(test_loader):
            x = ae.enc(img.cuda())
            label = label.cuda()
            prob_gt = F.softmax(x, dim=1)
            
            # forward
            img_rec1, feats1, logits1_trans, Sfeats1, Slogits1_trans, img_rec2, feats2 = ae(x)
            
            logits1  =  feats1[-1];  logprob1 = F.log_softmax( logits1, dim=1)
            logits2  =  feats2[-1];  logprob2 = F.log_softmax( logits2, dim=1)
            Slogits1 = Sfeats1[-1]; Slogprob1 = F.log_softmax(Slogits1, dim=1)
            
            # code reconstruction loss
            softloss1_  = nn.KLDivLoss()(logprob1,  prob_gt.data) * args.softloss_weight
            softloss2_  = nn.KLDivLoss()(logprob2,  prob_gt.data) * args.softloss_weight
            Ssoftloss1_ = nn.KLDivLoss()(Slogprob1, prob_gt.data) * args.softloss_weight
            
            softloss1_test  +=  softloss1_.data.cpu().numpy()
            softloss2_test  +=  softloss2_.data.cpu().numpy()
            Ssoftloss1_test += Ssoftloss1_.data.cpu().numpy()
            
            # test cls accuracy
            pred1  =  logits1.detach().max(1)[1];  test_acc1 +=  pred1.eq(label.view_as( pred1)).sum().cpu().data.numpy()
            pred2  =  logits2.detach().max(1)[1];  test_acc2 +=  pred2.eq(label.view_as( pred2)).sum().cpu().data.numpy()
            Spred1 = Slogits1.detach().max(1)[1]; Stest_acc1 += Spred1.eq(label.view_as(Spred1)).sum().cpu().data.numpy()
            cnt += 1
             
            # test acc for small enc
            pred = ae.small_enc(img.cuda()).detach().max(1)[1]
            test_acc += pred.eq(label.view_as(pred)).sum().cpu().data.numpy()
          
          softloss1_test  /= cnt;  test_acc1 /= float(len(data_test))
          softloss2_test  /= cnt;  test_acc2 /= float(len(data_test))
          Ssoftloss1_test /= cnt; Stest_acc1 /= float(len(data_test))
          test_acc /= float(len(data_test))
          
          format_str = "E{}S{} test softloss: {:.5f}({:.4f}) {:.5f}({:.4f}) {:.5f}({:.4f}) | test accuracy: {:.4f}"
          logprint(format_str.format(epoch, step, softloss1_test, test_acc1, softloss2_test, test_acc2, Ssoftloss1_test, Stest_acc1, test_acc), log)
          torch.save(ae.dec.state_dict(), pjoin(weights_path, "%s_BD_E%sS%s_testacc1=%.4f.pth" % (TIME_ID, epoch, step, test_acc1)))
          torch.save(ae.small_enc.state_dict(), pjoin(weights_path, "%s_SE_E%sS%s_testacc=%.4f.pth" % (TIME_ID, epoch, step, test_acc)))
  
  log.close()
