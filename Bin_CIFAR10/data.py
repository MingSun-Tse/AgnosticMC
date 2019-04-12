import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def set_up_data(dataset, train_batch_size):
  # ref: https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py
  if dataset == "CIFAR10":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    data_train = datasets.CIFAR10('./CIFAR10_data', train=True, download=True,
                                transform=transforms.Compose([
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomCrop(32, 4),
                                  transforms.ToTensor(),
                                  normalize,
                                ]))
    data_test = datasets.CIFAR10('./CIFAR10_data', train=False, download=True,
                              transform=transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                              ]))
  elif dataset == "MNIST":                   
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
  train_loader = torch.utils.data.DataLoader(data_train, batch_size=train_batch_size, shuffle=True, **kwargs)
  test_loader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, **kwargs)
  return train_loader, len(data_train), test_loader, len(data_test)