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
from torch.distributions.one_hot_categorical import OneHotCategorical
import torchvision.datasets as datasets
# my libs
from model import AutoEncoders


def logprint(some_str, f=sys.stdout):
  print(time.strftime("[%s" % os.getpid() + "-%Y/%m/%d-%H:%M] ") + str(some_str), file=f, flush=True)
  
  
if __name__ == "__main__":
  # Some settings
  SHOW_INTERVAL = 100
  SAVE_INTERVAL = 2000

  # Passed-in params
  parser = argparse.ArgumentParser(description="Knowledge Transfer")
  parser.add_argument('--e1', type=str, default="train*/*2/w*/*20190222-1834_E17S0_acc=0.9919.pth")
  parser.add_argument('--e2', type=str, default=None)
  parser.add_argument('--d',  type=str, default=None)
  parser.add_argument('--gpu', type=int, help="which gpu to run on. default is 0", default=0)
  parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
  parser.add_argument('--test_batch_size', type=int, help='batch size', default=5)
  parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
  parser.add_argument('--floss_weight', type=float, help='loss weight to balance multi-losses', default=1.0)
  parser.add_argument('--ploss_weight', type=float, help='loss weight to balance multi-losses', default=1.0)
  parser.add_argument('--closs_weight', type=float, help='loss weight to balance multi-losses', default=1.0)
  parser.add_argument('--floss_lw', type=str, default="1-1-1-1-1-1-1")
  parser.add_argument('--ploss_lw', type=str, default="1-1-1-1-1-1-1")
  parser.add_argument('--closs_lw', type=str, default="10-10")
  parser.add_argument('-p', '--project_name', type=str, help='the name of project, to save logs etc., will be set in directory, "Experiments"')
  parser.add_argument('-r', '--resume', action='store_true', help='if resume, default=False')
  parser.add_argument('-m', '--mode', type=str, help='the training mode name.')
  parser.add_argument('--epoch', type=int, default=61)
  parser.add_argument('--num_step_per_epoch', type=int, default=10000)
  parser.add_argument('--debug', action="store_true")
  parser.add_argument('--num_class', type=int, default=10)
  parser.add_argument('--num_test_sample', type=int, default=5)
  args = parser.parse_args()
  
  # Get path
  args.e1 = glob.glob(args.e1)[0] if args.e1 != None else None
  args.e2 = glob.glob(args.e2)[0] if args.e2 != None else None
  args.d  = glob.glob(args.d )[0] if args.d  != None else None
  
  # Check mode
  assert(args.mode in AutoEncoders.keys())
  
  # Set up directories and logs etc
  if args.debug:
    args.project_name = "test"
  project_path = pjoin("../Experiments", args.project_name)
  rec_img_path = pjoin(project_path, "reconstructed_images")
  weights_path = pjoin(project_path, "weights") # to save torch model
  if not args.resume:
    if os.path.exists(project_path):
      respond = "Y" # input("The appointed project name has existed. Do you want to overwrite it (everything inside will be removed)? (y/n) ")
      if str.upper(respond) in ["Y", "YES"]:
        shutil.rmtree(project_path)
      else:
        exit(1)
    if not os.path.exists(rec_img_path):
      os.makedirs(rec_img_path)
    if not os.path.exists(weights_path):
      os.makedirs(weights_path)
  TIME_ID = os.environ["SERVER"] + time.strftime("-%Y%m%d-%H%M")
  log_path = pjoin(weights_path, "log_" + TIME_ID + ".txt")
  log = sys.stdout if args.debug else open(log_path, "w+")
  
  # Set up model
  AE = AutoEncoders[args.mode]
  ae = AE(args.e1, args.d, args.e2)
  ae.cuda()

  
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
  train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,      shuffle=True,  **kwargs)
  test_loader  = torch.utils.data.DataLoader(data_test,  batch_size=args.test_batch_size, shuffle=False, **kwargs)
  
  
  # Prepare test code
  # one_hot = OneHotCategorical(torch.Tensor([1./args.num_class] * args.num_class))
  # test_codes = one_hot.sample_n(args.num_test_sample) * 100 # logits
  _, (test_img, _) = list(enumerate(test_loader))[0]
  test_codes = ae.enc(test_img.cuda())
  np.save(pjoin(rec_img_path, "test_codes.npy"), test_codes.data.cpu().numpy())
  
  # print setting for later check
  logprint(str(args._get_kwargs()), log)
  
  # Parse to get stage loss weight
  floss_lw = [float(x) for x in args.floss_lw.split("-")]
  ploss_lw = [float(x) for x in args.ploss_lw.split("-")]
  closs_lw = [float(x) for x in args.closs_lw.split("-")]
  
  # Optimize
  ## set lr
  # fc8_params = list(map(id, ae.dec.fc8.parameters()))
  # fc7_params = list(map(id, ae.dec.fc7.parameters()))
  # fc6_params = list(map(id, ae.dec.fc6.parameters()))
  # conv5_params = list(map(id, ae.dec.conv5.parameters()))
  # conv4_params = list(map(id, ae.dec.conv4.parameters()))
  # conv3_params = list(map(id, ae.dec.conv3.parameters()))
  # base_params = filter(lambda p: id(p) not in fc8_params+fc7_params+fc6_params+conv5_params+conv4_params+conv3_params, ae.dec.parameters())
  # optimizer = torch.optim.Adam([
            # {'params': base_params},
            # {'params': ae.dec.fc8.parameters(), 'lr': args.lr * 20},
            # {'params': ae.dec.fc7.parameters(), 'lr': args.lr * 40},
            # {'params': ae.dec.fc6.parameters(), 'lr': args.lr * 40},
            # {'params': ae.dec.conv5.parameters(), 'lr': args.lr * 5},
            # {'params': ae.dec.conv4.parameters(), 'lr': args.lr * 5},
            # {'params': ae.dec.conv3.parameters(), 'lr': args.lr * 5},
            # ], lr=args.lr)  
  
  

  
  optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr)
  loss_func = nn.MSELoss()
  t1 = time.time()
  for epoch in range(args.epoch):
    for step, (img, label) in enumerate(train_loader):
      # Generate codes randomly
      # x = torch.randn([args.batch_size, args.num_class]) + one_hot.sample_n(args.batch_size) * 3 # logits
      x = ae.enc(img.cuda())
      x = x.cuda()
      prob_gt = nn.functional.softmax(x, dim=1) # prob, ground truth
      
      # forward
      if args.mode == "BD":
        feats1, feats2 = ae(x)
      elif args.mode == "SE":
        feats1, small_feats1, feats2 = ae(x) # feats1: feats from encoder. small_feats1: feats from small encoder. feats2: feats from encoder.
      
      # code loss: cross entropy
      if args.mode == "BD":
        logits1 = feats1[-1]; logprob_1 = nn.functional.log_softmax(logits1, dim=1) 
        logits2 = feats2[-1]; logprob_2 = nn.functional.log_softmax(logits2, dim=1)
        closs1 = nn.KLDivLoss()(logprob_1, prob_gt.data) * args.closs_weight * closs_lw[0]
        closs2 = nn.KLDivLoss()(logprob_2, prob_gt.data) * args.closs_weight * closs_lw[1]
        # perceptual loss
        ploss1 = loss_func(feats2[0], feats1[0].data) * args.ploss_weight * ploss_lw[0]
        ploss2 = loss_func(feats2[1], feats1[1].data) * args.ploss_weight * ploss_lw[1]
        ploss3 = loss_func(feats2[2], feats1[2].data) * args.ploss_weight * ploss_lw[2]
        ploss4 = loss_func(feats2[3], feats1[3].data) * args.ploss_weight * ploss_lw[3]
        # total loss
        loss = closs1 + closs2 + ploss1 + ploss2 + ploss3 + ploss4
      
      elif args.mode == "SE":
        logits1 = small_feats1[-1]; logprob_1 = nn.functional.log_softmax(logits1, dim=1)
        logits2 =       feats2[-1]; logprob_2 = nn.functional.log_softmax(logits2, dim=1)
        closs1 = nn.KLDivLoss()(logprob_1, prob_gt.data) * args.closs_weight * closs_lw[0]
        closs2 = nn.KLDivLoss()(logprob_2, prob_gt.data) * args.closs_weight * closs_lw[1]
        # feature reconstruction loss
        floss1 = loss_func(small_feats1[0], feats1[0].data) * args.floss_weight * floss_lw[0]
        floss2 = loss_func(small_feats1[1], feats1[1].data) * args.floss_weight * floss_lw[1]
        floss3 = loss_func(small_feats1[2], feats1[2].data) * args.floss_weight * floss_lw[2]
        floss4 = loss_func(small_feats1[3], feats1[3].data) * args.floss_weight * floss_lw[3]
        # perceptual loss
        ploss1 = loss_func(feats2[0], feats1[0].data) * args.ploss_weight * ploss_lw[0]
        ploss2 = loss_func(feats2[1], feats1[1].data) * args.ploss_weight * ploss_lw[1]
        ploss3 = loss_func(feats2[2], feats1[2].data) * args.ploss_weight * ploss_lw[2]
        ploss4 = loss_func(feats2[3], feats1[3].data) * args.ploss_weight * ploss_lw[3]
        # total loss
        loss = closs1 + closs2 + \
               ploss1 + ploss2 + ploss3 + ploss4 + \
               floss1 + floss2 + floss3 + floss4
      
      optimizer.zero_grad()
      loss.backward()
      
      # check the gradient
      if step % 2000 == 0:
        ave_grad = []
        for p in ae.dec.named_parameters(): # get the params in each layer
          layer_name = p[0].split(".")[0]
          layer_name = "  "+layer_name if "fc" in layer_name else layer_name
          if p[1].grad is not None:
            ave_grad.append([layer_name, np.average(p[1].grad.abs()) * args.lr, np.average(p[1].data.abs())])
        ave_grad = ["{}: {:.6f} / {:.6f} ({:.10f})\n".format(x[0], x[1], x[2], x[1]/x[2]) for x in ave_grad]
        ave_grad = "".join(ave_grad)
        logprint("\n=> E{}S{} grad x lr:\n{}".format(epoch, step, ave_grad), log)
      
      optimizer.step()
      
      # Print training loss
      if step % SHOW_INTERVAL == 0:
        if args.mode == "BD":
          format_str = "E{}S{} loss={:.3f} | closs: {:.5f} {:.5f} | ploss: {:.5f} {:.5f} {:.5f} {:.5f} ({:.3f}s/step)"
          logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), closs1.data.cpu().numpy(), closs2.data.cpu().numpy(),
              ploss1.data.cpu().numpy(), ploss2.data.cpu().numpy(), ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(),
              (time.time()-t1)/SHOW_INTERVAL), log)
        
        elif args.mode == "SE":
          format_str = "E{}S{} loss={:.3f} | closs: {:.5f} {:.5f} | floss: {:.5f} {:.5f} {:.5f} {:.5f} | ploss: {:.5f} {:.5f} {:.5f} {:.5f} ({:.3f}s/step)"
          logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), closs1.data.cpu().numpy(), closs2.data.cpu().numpy(),
              floss1.data.cpu().numpy(), floss2.data.cpu().numpy(), floss3.data.cpu().numpy(), floss4.data.cpu().numpy(), 
              ploss1.data.cpu().numpy(), ploss2.data.cpu().numpy(), ploss3.data.cpu().numpy(), ploss4.data.cpu().numpy(), 
              (time.time()-t1)/SHOW_INTERVAL), log)
        
        t1 = time.time()
      
      # Test and save models
      if step % SAVE_INTERVAL == 0:
        for i in range(len(test_codes)):
          x = test_codes[i].cuda()
          img1 = ae.dec(x)
          img2 = ae.dec(ae.enc(img1))
          out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_img%s-rec1.jpg" % (TIME_ID, epoch, step, i))
          out_img2_path = pjoin(rec_img_path, "%s_E%sS%s_img%s-rec2.jpg" % (TIME_ID, epoch, step, i))
          vutils.save_image(img1.data.cpu().float(), out_img1_path) # save some samples to check
          vutils.save_image(img2.data.cpu().float(), out_img2_path) # save some samples to check
        
        # save model
        if args.mode == "BD":
          torch.save(ae.dec.state_dict(), pjoin(weights_path, "%s_%s_E%sS%s.pth" % (TIME_ID, args.mode, epoch, step)))
        elif args.mode == "SE":
          torch.save(ae.small_enc.state_dict(), pjoin(weights_path, "%s_%s_E%sS%s.pth" % (TIME_ID, args.mode, epoch, step)))
  log.close()
