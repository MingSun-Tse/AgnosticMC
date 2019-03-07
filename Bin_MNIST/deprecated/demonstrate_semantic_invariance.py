from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import LeNet5, SmallLeNet5
import glob
import numpy as np

# Set up random affine
randomaffine = transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.8, 1.2))
# randomaffine = transforms.RandomAffine(degrees=0, translate=(0.0, 0.0), scale=(1, 1))

# Prepare images
fake_img_path = "../Experiments/test2/reconstructed_images/*E12S0*img2*rec1*.jpg"
fake_img_path = glob.glob(fake_img_path)[0]
fake_img = Image.open(fake_img_path).convert("L")
fake_img_label = int(fake_img_path.split("=")[1].split(".")[0])
data_test = datasets.MNIST('./MNIST_data',
                            train=False,
                            download=True,
                            transform=transforms.Compose([
                              transforms.Resize((32, 32)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
                          )
index = np.random.randint(5000)
real_img = Image.fromarray(data_test[index][0].numpy()[0])
real_img_label = data_test[index][1].numpy()

# Set up models
BE_path = "train_baseline_lenet5/trained_weights2/weights/SERVER12-20190222-1834_E17S0_acc=0.9919.pth"
SE_path = "../Experiments/test2/weights/12-20190225-2209_SE_E11S0_testacc=0.2570.pth"
BE = LeNet5(BE_path).cuda()
SE = SmallLeNet5(SE_path).cuda()


num_test = 100
fake_pred = []
for i in range(num_test):
  fake_pred.append(BE(transforms.ToTensor()(randomaffine(fake_img)).unsqueeze(0).cuda(0)).argmax().data.cpu().item())
fake_pred = np.array(fake_pred)
print("\nfake label = %s" % fake_img_label, fake_pred, np.sum(fake_pred==fake_img_label))

real_pred = []
for i in range(num_test):
  real_pred.append(BE(transforms.ToTensor()(randomaffine(real_img)).unsqueeze(0).cuda(0)).argmax().data.cpu().item())
real_pred = np.array(real_pred)
print("\nfake label = %s" % real_img_label, real_pred, np.sum(real_pred==real_img_label))