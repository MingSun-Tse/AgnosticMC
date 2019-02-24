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
import torch.utils.data as Data
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# my libs
from model import LeNet5


def logprint(some_str, f=sys.stdout):
  print(time.strftime("[%s" % os.getpid() + "-%Y/%m/%d-%H:%M] ") + str(some_str), file=f, flush=True)
  
  
if __name__ == "__main__":
  # Some settings
  SHOW_INTERVAL = 10
  SAVE_INTERVAL = 1000

  # Passed-in params
  parser = argparse.ArgumentParser(description="LeNet5")
  parser.add_argument('--model', type=str, default=None)
  parser.add_argument('--gpu', type=int)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--test_batch_size', type=int, default=100)
  parser.add_argument('--lr', type=float, help='learning rate', default=2e-3)
  parser.add_argument('-p', '--project_name', type=str, default="trained_weights")
  parser.add_argument('-r', '--resume', action='store_true', help='if resume, default=False')
  parser.add_argument('--epoch', type=int, default=31)
  parser.add_argument('--debug', action="store_true")
  args = parser.parse_args()
  
  # Get path
  args.model = glob.glob(args.model)[0] if args.model != None else None
  
  # Set up directories and logs etc
  if args.debug:
    args.project_name = "test"
  project_path = pjoin("./", args.project_name)
  weights_path = pjoin(project_path, "weights") # to save torch model
  if not args.resume:
    if os.path.exists(project_path):
      respond = "Y" # input("The appointed project name has existed. Do you want to overwrite it (everything inside will be removed)? (y/n) ")
      if str.upper(respond) in ["Y", "YES"]:
        shutil.rmtree(project_path)
      else:
        exit(1)
    if not os.path.exists(weights_path):
      os.makedirs(weights_path)
  TIME_ID = "SERVER" + os.environ["SERVER"] + time.strftime("-%Y%m%d-%H%M")
  log_path = pjoin(weights_path, "log_" + TIME_ID + ".txt")
  log = sys.stdout if args.debug else open(log_path, "w+")
  
  # Set up model
  net = LeNet5(args.model)
  net.cuda()

  # Prepare data
  data_train = datasets.MNIST('../data',
                              train=True,
                              download=True,
                              transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
                             )
  data_test = datasets.MNIST('../data',
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

  
  # print setting for later check
  logprint(str(args._get_kwargs()), log)
  
  
  # Optimize  
  optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
  loss_func = nn.CrossEntropyLoss()
  t1 = time.time()
  for epoch in range(args.epoch):
    net.train()
    for step, (x, y) in enumerate(train_loader):
      x, y = x.cuda(), y.cuda()
      y_ = net(x) # logits
      loss = loss_func(y_, y.data)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      # Print training loss
      if step % SHOW_INTERVAL == 0:
        format_str = "E{}S{} train_loss: {:.5f} ({:.3f}s/step)"
        logprint(format_str.format(epoch, step, loss.data.cpu().numpy(), (time.time()-t1)/SHOW_INTERVAL), log)
        t1 = time.time()
      
      # Test and save model
      if step % SAVE_INTERVAL == 0:
        net.eval()
        num_right = 0; avg_loss = 0
        for _, (x, y) in enumerate(test_loader):
          x, y = x.cuda(), y.cuda()
          y_ = net(x)
          avg_loss += loss_func(y_, y.data).sum()
          pred = y_.detach().max(1)[1]
          num_right += pred.eq(y.view_as(pred)).sum()
        avg_loss /= len(data_test)
        test_acc = float(num_right) / len(data_test)
        logprint("E{}S{} test_loss: {:.5f} | test accuracy: {:.4f}".format(epoch, step, avg_loss.detach().cpu().item(), test_acc), log)
        torch.save(net.state_dict(), pjoin(weights_path, "{}_E{}S{}_acc={:.4f}.pth".format(TIME_ID, epoch, step, test_acc)))
  log.close()
