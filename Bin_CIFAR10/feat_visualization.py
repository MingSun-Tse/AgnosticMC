from model import LeNet5_2neurons
from data import set_up_data
from util import check_path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors  = ["red", "blue", "black", "yellow", "green", "yellowgreen", "gold", "royalblue", "peru", "purple"]
pretrained = "train_baseline_lenet5/trained_weights_*2neurons/w*/*E21S0*.pth"
pretrained = check_path(pretrained)
gpu_id = 4 # change your GPU here
net = LeNet5_2neurons(pretrained).cuda(gpu_id)

train_loader, num_train, test_loader, num_test = set_up_data("MNIST", 500)
for step, (img, label) in enumerate(train_loader):
  print("train step %s / %s" % (step, len(train_loader)))
  img = img.cuda(gpu_id)
  feat = net.forward_2neurons(img)
  feat = feat.data.cpu().numpy(); label = label.data.cpu().numpy()
  for x, y in zip(feat, label):
    plt.scatter(x[0], x[1], color=colors[y])
plt.savefig("./mnist_trainset_feat_visualization.jpg")
plt.close()

for step, (img, label) in enumerate(test_loader):
  print("test step %s / %s" % (step, len(test_loader)))
  img = img.cuda(gpu_id)
  feat = net.forward_2neurons(img)
  feat = feat.data.cpu().numpy(); label = label.data.cpu().numpy()
  for x, y in zip(feat, label):
    plt.scatter(x[0], x[1], color=colors[y])
plt.savefig("./mnist_testset_feat_visualization.jpg")
plt.close()