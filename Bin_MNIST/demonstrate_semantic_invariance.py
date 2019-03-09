from PIL import Image
import glob
import numpy as np
import time
import cv2
import argparse

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch

from model import LeNet5, SmallLeNet5
from PhotoWCT_Model import PhotoWCT

parser = argparse.ArgumentParser()
parser.add_argument('--floss_weight', type=float, default=1)



# Set up autoencoders
AE = PhotoWCT().cuda()
AE.load_state_dict(torch.load("photo_wct.pth"))
def deep_transform(img_path, level=4):
  enc = eval("AE.e" + str(level))
  dec = eval("AE.d" + str(level))
  img = Image.open(img_path).convert("RGB")
  img = transforms.ToTensor()(img).unsqueeze(0).cuda()
  img_rec = dec(*enc(img))[0]
  img_out_path = img_path.replace(".jpg", "_rec.jpg")
  vutils.save_image(img_rec.data.cpu().float(), img_out_path)
  return img_out_path

# Set up erode and dilate transform
def dilate(img_path, kernel_size=3, num_iter=1):
  img = cv2.imread(img_path)
  kernel = np.ones((kernel_size,kernel_size), np.uint8)
  img_dilated = cv2.dilate(img, kernel, iterations=num_iter)
  img_out_path = img_path.replace(".jpg", "_dil.jpg")
  cv2.imwrite(img_out_path, img_dilated)
  return img_out_path

def erode(img_path, kernel_size=3, num_iter=1):
  img = cv2.imread(img_path)
  kernel = np.ones((kernel_size,kernel_size), np.uint8)
  img_eroded = cv2.erode(img, kernel, iterations=num_iter)
  img_out_path = img_path.replace(".jpg", "_ero.jpg")
  cv2.imwrite(img_out_path, img_eroded)
  return img_out_path
  
# Set up random affine
randomaffine = transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.2))
# randomaffine = transforms.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(1, 1))

# Prepare images
## fake images
fake_img_path = "*1606*label=4.jpg"  #"../Experiments/test_xx/reconstructed_images/SERVER12-20190303-1607_E509S0_img3-rec1_label=4.jpg"
fake_img_path = glob.glob(fake_img_path)[0]
# fake_img_rec_path = fake_img_path.replace(".jpg", "_rec.jpg"); deep_transform(fake_img_path, fake_img_rec_path)
fake_img_dil_ero_path = dilate(erode((fake_img_path)))
fake_img = Image.open(fake_img_dil_ero_path).convert("L")
fake_img_label = int(fake_img_path.split("=")[1][0])

## real images
data_test = datasets.MNIST('./MNIST_data',
                            train=False,
                            download=True,
                            transform=transforms.Compose([
                              transforms.Resize((32, 32)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
                          )
index = np.random.randint(len(data_test)) # get random test image
real_img_label = data_test[index][1].numpy()
TIME_ID = time.time(); save_real_img_path = "real_img_label=%s_%s.jpg" % (real_img_label, TIME_ID)
test_img_tensor = data_test[index][0][0]; vutils.save_image(test_img_tensor.data.cpu().float(), save_real_img_path)
real_img_path = dilate(erode((save_real_img_path)))
real_img = Image.open(real_img_path).convert("L")


# Set up models
BE_path = "train_baseline_lenet5/trained_weights2/weights/SERVER12-20190222-1834_E17S0_acc=0.9919.pth"
SE_path = "../Experiments/test_xx/weights/SERVER12-20190303-1607_SE_E508S0_testacc=0.5533.pth"
BE = LeNet5(BE_path).cuda()
SE = SmallLeNet5(SE_path).cuda()
# BE = SE

num_test = 100
fake_pred = []
for i in range(num_test):
  fake_pred.append(BE(transforms.ToTensor()(randomaffine(fake_img)).unsqueeze(0).cuda()).argmax().data.cpu().item())
fake_pred = np.array(fake_pred)
print("\nfake label = %s\n" % fake_img_label, fake_pred, np.sum(fake_pred==fake_img_label))

real_pred = []
for i in range(num_test):
  real_pred.append(BE(transforms.ToTensor()(randomaffine(real_img)).unsqueeze(0).cuda()).argmax().data.cpu().item())
real_pred = np.array(real_pred)
print("\nreal label = %s\n" % real_img_label, real_pred, np.sum(real_pred==real_img_label))


