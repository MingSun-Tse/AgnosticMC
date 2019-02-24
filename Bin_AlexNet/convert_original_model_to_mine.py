import torchvision.models as models
import torch
from model import AlexNet_Encoder

alex = models.alexnet(pretrained=True)
alex_odict = alex.state_dict()
my_alex = AlexNet_Encoder()
my_alex_odict = my_alex.state_dict()

tensor_map = dict(zip(my_alex_odict.keys(), alex_odict.keys()))
print(tensor_map)
for tensor_name in my_alex_odict:
  print("Processing tensor '%s'" % tensor_name)
  tensor_value = alex_odict[tensor_map[tensor_name]]
  my_alex_odict[tensor_name].data.copy_(tensor_value)
torch.save(my_alex_odict, "models/my_alexnet.pth")