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
parser.add_argument('--num_dec', type=int, default=9)
parser.add_argument('--num_se', type=int, default=1)
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
parser.add_argument('--num_epoch', type=int, default=96)
parser.add_argument('--debug', action="store_true")
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
parser.add_argument('--use_ave_img', action="store_true")
parser.add_argument('--acc_thre_reset_dec', type=float, default=0)
parser.add_argument('--history_acc_weight', type=float, default=0.25)
parser.add_argument('--num_z', type=int, default=100, help="the dimension of hidden z")
parser.add_argument('--msgan_option', type=str, default="pixel")
args = parser.parse_args()

# Update and check args
assert(args.num_dec >= 1)
assert(args.mode in AutoEncoders.keys())
args.e1 = path_check(args.e1)
args.e2 = path_check(args.e2)
args.pretrained_dir = path_check(args.pretrained_dir)
args.adv_train = int(args.mode[-1])
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
  ae = AE(args).cuda()
  
  # Set up exponential moving average
  history_acc_se = []
  history_acc_dec = []
  if args.adv_train == 4:
    ema_dec = []; ema_se = []
    for di in range(1, args.num_dec+1):
      ema_dec.append(EMA(args.ema_factor))
      dec = eval("ae.d%s"  % di)
      for name, param in dec.named_parameters():
        if param.requires_grad:
          ema_dec[-1].register(name, param.data)
      history_acc_se.append(0) # to cache history accuracy
      history_acc_dec.append(0)
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
  if args.adv_train == 4:
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
  
  
  # Get the pre-image
  # pre_image = []
  # num_step = 150
  # for cls in range(args.num_class):
    # created_image = np.uint8(np.random.uniform(0, 255, (32, 32, 3)))
    # for i in range(num_step): # 150 steps
      # processed_image = preprocess_image(created_image, False) # normalize
      # optimizer_preimage = torch.optim.SGD([processed_image], lr=6)
      # logits = ae.be(processed_image)
      # activmax_loss = -logits[0, cls]
      # ae.be.zero_grad()
      # activmax_loss.backward()
      # optimizer_preimage.step() # optimize the image
      # created_image = recreate_image(processed_image) # 0-1 image to 0-255 image
      # if i == num_step-1:
        # outpath = pjoin(rec_img_path, "%s_preimage_label=%s.jpg" % (TIME_ID, cls))
        # vutils.save_image(processed_image.cpu().data.float(), outpath)
    # pre_image.append(processed_image)
    
    
  # Optimization
  t1 = time.time(); total_step = 0
  for epoch in range(previous_epoch, args.num_epoch):
    for step, (img, label) in enumerate(train_loader):
      total_step += 1
      ae.train()
      # Generate codes randomly
      if args.use_pseudo_code:
        random_z1 = torch.cuda.FloatTensor(args.batch_size, args.num_z); random_z1.copy_(torch.randn(args.batch_size, args.num_z))
        random_z2 = torch.cuda.FloatTensor(args.batch_size, args.num_z); random_z2.copy_(torch.randn(args.batch_size, args.num_z))
        z_concat = torch.cat([random_z1, random_z2], dim=0)
        onehot_label = one_hot.sample_n(args.batch_size).view([args.batch_size, args.num_class]).cuda()
        label_concat = torch.cat([onehot_label, onehot_label], dim=0)
        x = torch.cat([z_concat, label_concat], dim=1).detach() # input to the Generator network
        label = label_concat.data.cpu().numpy().argmax(axis=1)
        label = torch.from_numpy(label).long().detach()
      else:
        x = ae.be(img.cuda()) / args.temp # TODO
      # prob_gt = F.softmax(x, dim=1) # prob, ground truth
      label = label.cuda()
        
      if args.adv_train == 4:
        # update decoder
        imgrec = []; imgrec_DT = []; hardloss_dec = []; trainacc_dec = []; ave_imgrec = Variable(torch.zeros([label.size(0), 3, 32, 32], requires_grad=True)).cuda()
        for di in range(1, args.num_dec+1):
          dec = eval("ae.d" + str(di)); optimizer = optimizer_dec[di-1]; ema = ema_dec[di-1]
          imgdecs = torch.split(dec(x), 3, dim=1) # 3 channels
          total_loss = 0
          for imgrec1 in imgdecs:
            feats1 = ae.be.forward_branch(tensor_normalize(imgrec1)); logits1 = feats1[-1]
            imgrec1_DT = ae.defined_trans(imgrec1); logits1_DT = ae.be(tensor_normalize(imgrec1_DT)) # DT: defined transform
            imgrec.append(imgrec1); imgrec_DT.append(imgrec1_DT) # for SE
            if args.use_ave_img:
              ave_imgrec += imgrec1
            
            tvloss1 = args.lw_tv * (torch.sum(torch.abs(imgrec1[:, :, :, :-1] - imgrec1[:, :, :, 1:])) + 
                                    torch.sum(torch.abs(imgrec1[:, :, :-1, :] - imgrec1[:, :, 1:, :])))
            imgnorm1 = torch.pow(torch.norm(imgrec1, p=6), 6) * args.lw_norm
            
            ploss = 0
            ploss_print = []
            for i in range(len(feats1)-1):
              if "feats2" in dir():
                ploss_print.append(nn.MSELoss()(feats2[i], feats1[i].data) * args.lw_perc)
              else:
                ploss_print.append(torch.cuda.FloatTensor(0))
              ploss += ploss_print[-1]
              
            ## Classification loss, the bottomline loss
            # logprob1 = F.log_softmax(logits1/args.temp, dim=1)
            # logprob2 = F.log_softmax(logits2/args.temp, dim=1)
            # softloss1 = nn.KLDivLoss()(logprob1, prob_gt.data) * (args.temp*args.temp) * args.lw_soft
            # softloss2 = nn.KLDivLoss()(logprob2, prob_gt.data) * (args.temp*args.temp) * args.lw_soft
            hardloss1 = nn.CrossEntropyLoss()(logits1, label) * args.lw_hard
            hardloss1_DT = nn.CrossEntropyLoss()(logits1_DT, label) * args.lw_DA
            
            pred = logits1.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().cpu().data.numpy() / label.size(0)
            hardloss_dec.append(hardloss1.data.cpu().numpy()); trainacc_dec.append(trainacc)
            history_acc_dec[di-1] = history_acc_dec[di-1] * args.history_acc_weight + trainacc * (1 - args.history_acc_weight)
            
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
              feats1_1, feats1_2 = torch.split(feats1[2], args.batch_size, dim=0)
              lz = torch.mean(torch.abs(feats1_1 - feats1_2)) / torch.mean(torch.abs(random_z1 - random_z2))
            eps = 1e-5
            loss_diversity = args.lw_msgan / (lz + eps)
            
            ## Diversity encouraging loss 2
            # ref: 2017 CVPR Diversified Texture Synthesis with Feed-forward Networks
            
            
            ## Activation maximization loss
            activmax_loss = 0
            for i in range(logits1.size(0)):
              activmax_loss += -logits1[i, label[i]] * args.lw_actimax
            activmax_loss /= logits1.size(0)
            
            
            ## total loss
            loss = hardloss1 + hardloss1_DT + \
                   advloss + \
                   tvloss1 + imgnorm1 + \
                   loss_diversity
            if np.random.rand() < 1./args.classloss_update_interval:
              loss += activmax_loss # do not let the class loss update too often
            total_loss += loss
          
          dec.zero_grad()
          total_loss.backward(retain_graph=args.use_ave_img)
        
        # average image loss
        if args.use_ave_img:
          ave_imgrec /= args.num_dec
          logits_ave = ae.be(ave_imgrec)
          hardloss_ave = nn.CrossEntropyLoss()(logits_ave, label) * args.lw_hard
          hardloss_ave.backward()
        
        # update params
        for di in range(1, args.num_dec+1):
          dec = eval("ae.d" + str(di)); optimizer = optimizer_dec[di-1]; ema = ema_dec[di-1]
          if args.acc_thre_reset_dec and history_acc_se[di-1] > args.acc_thre_reset_dec \
            and history_acc_dec[di-1] > args.acc_thre_reset_dec:
            logprint("E{}S{} | ==> Reset decoder {}, history_acc_se = {:.4f}, history_acc_dec = {:.4f}".format(epoch, step, di, 
                history_acc_se[di-1], history_acc_dec[di-1]))
            # reset the history_acc
            history_acc_dec[di-1] = 0
            history_acc_se[di-1] = 0
            # reset weights
            for m in dec.modules():
              if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()              
            # reset ema, after reseting the weights
            for name, param in dec.named_parameters():
              if param.requires_grad:
                ema.register(name, param.data)
          else:
            optimizer.step()
            for name, param in dec.named_parameters():
              if param.requires_grad:
                param.data = ema(name, param.data)

        # update SE
        hardloss_se = []; trainacc_se = []
        for sei in range(1, args.num_se+1):
          se = eval("ae.se" + str(sei)); optimizer = optimizer_se[sei-1]; ema = ema_se[sei-1]
          se.zero_grad()
          loss_se = 0
          for di in range(1, args.num_dec+1):
            logits = se(tensor_normalize(imgrec[di-1].detach()))
            logits_DT = se(tensor_normalize(imgrec_DT[di-1].detach()))
            hardloss = nn.CrossEntropyLoss()(logits, label) * args.lw_hard
            hardloss_DT = nn.CrossEntropyLoss()(logits_DT, label) * args.lw_DA
            loss_se += hardloss + hardloss_DT
            pred = logits.detach().max(1)[1]; trainacc = pred.eq(label.view_as(pred)).sum().cpu().data.numpy() / label.size(0)
            hardloss_se.append(hardloss.data.cpu().numpy()); trainacc_se.append(trainacc)
            if sei == 1:
              history_acc_se[di-1] = history_acc_se[di-1] * args.history_acc_weight + trainacc * (1 - args.history_acc_weight)
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
              out_img1_path = pjoin(rec_img_path, "%s_E%sS%s_imgrec%s_label=%s_d%s_%s.jpg" % (TIME_ID, epoch, step, i, test_labels[i], di, j))
              vutils.save_image(img1.data.cpu().float(), out_img1_path)
            
      
      # Test and save models
      if step % args.test_interval == 0:
        ae.dec = ae.d1
        ae.small_enc = ae.se1
        ae.enc = ae.be
        ae.eval()
        # test with the true logits generated from test set
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, **kwargs)
        test_acc = 0
        for i, (img, label) in enumerate(test_loader):
          label = label.cuda()
          # test acc for small enc
          pred = ae.small_enc(img.cuda()).detach().max(1)[1]
          test_acc += pred.eq(label.view_as(pred)).sum().cpu().data.numpy()
        test_acc /= float(len(data_test))
        format_str = "E{}S{} | =======> Test softloss with real logits: test accuracy on SE: {:.4f}"
        logprint(format_str.format(epoch, step, test_acc))
        
        # torch.save(ae.se1.state_dict(), pjoin(weights_path, "%s_se_E%sS%s_testacc=%.4f.pth" % (TIME_ID, epoch, step, test_acc)))
        # torch.save(ae.d1.state_dict(), pjoin(weights_path, "%s_d1_E%sS%s.pth" % (TIME_ID, epoch, step)))
        # for di in range(2, args.num_dec+1):
          # dec = eval("ae.d" + str(di))
          # torch.save(dec.state_dict(), pjoin(weights_path, "%s_d%s_E%sS%s.pth" % (TIME_ID, di, epoch, step)))

      # Print training loss
      if step % args.show_interval == 0:
        if args.adv_train:
          if args.adv_train in [3, 4]:
            format_str1 = "E{}S{}"
            format_str2 = " | dec:" + " {:.3f}({:.3f}-{:.3f})" * args.num_dec
            format_str3 = " | se:" + " {:.3f}({:.3f}-{:.3f})" * args.num_dec
            format_str4 = " | tv: {:.3f} norm: {:.3f} diversity: {:.3f}"
            format_str5 = "" # " p:" + " {:.4f}" * len(ploss_print)
            format_str6 = " ({:.3f}s/step)"
            format_str = "".join([format_str1, format_str2, format_str3, format_str4, format_str5, format_str6])
            strvalue2 = []; strvalue3 = []
            for i in range(args.num_dec):
              strvalue2.append(hardloss_dec[i]); strvalue2.append(trainacc_dec[i]); strvalue2.append(history_acc_dec[i])
              strvalue3.append(hardloss_se[i]);  strvalue3.append(trainacc_se[i]); strvalue3.append(history_acc_se[i])
            # strvalue5 = [x.data.cpu().numpy() for x in ploss_print]
            logprint(format_str.format(
                epoch, step,
                *strvalue2,
                *strvalue3,
                tvloss1.data.cpu().numpy(), imgnorm1.data.cpu().numpy(), loss_diversity.data.cpu().numpy(),
                (time.time()-t1)/args.show_interval))

        t1 = time.time()
        
        