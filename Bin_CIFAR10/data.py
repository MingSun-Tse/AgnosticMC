import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def set_up_data(dataset, train_batch_size):
  # ref: https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py
  if dataset == "CIFAR10":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    data_train = datasets.CIFAR10('./data_CIFAR10', train=True, download=False,
                                transform=transforms.Compose([
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.ToTensor(),
                                  normalize,
                                ]))
    data_test = datasets.CIFAR10('./data_CIFAR10', train=False, download=False,
                              transform=transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                              ]))
  elif dataset == "MNIST":                   
    data_train = datasets.MNIST('./data_MNIST', train=True, download=False,
                                transform=transforms.Compose([
                                  transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]))
    data_test = datasets.MNIST('./data_MNIST', train=False, download=False,
                                transform=transforms.Compose([
                                  transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]))
  elif dataset == "CIFAR100":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    data_train = datasets.CIFAR100('./data_CIFAR100', train=True, download=False,
                                transform=transforms.Compose([
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.ToTensor(),
                                  normalize,
                                ]))
    data_test = datasets.CIFAR100('./data_CIFAR100', train=False, download=False,
                              transform=transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                              ]))
                            
  kwargs = {'num_workers': 4, 'pin_memory': True}
  train_loader = torch.utils.data.DataLoader(data_train, batch_size=train_batch_size, shuffle=True, **kwargs)
  test_loader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, **kwargs)
  return train_loader, len(data_train), test_loader, len(data_test)