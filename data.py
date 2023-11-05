import os
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import LFWPeople, MNIST

import torchvision.transforms as transforms



transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels = 3),
    transforms.Resize((64, 64)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: (x - 0.5) * 2)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_data = LFWPeople(root = "./Data", split = 'train', image_set = 'original', 
                      transform = transform, download = True)

test_data = LFWPeople(root = "./Data", split = 'test', image_set = 'original', 
                      transform = transform, download = True)

# train_data = MNIST(root = "./Data", train = True, 
#                       transform = transform, download = True)
# test_data = MNIST(root = "./Data", train = False, 
#                       transform = transform, download = True)


lfw_train_dataloader = DataLoader(train_data, batch_size = 32, shuffle = True, drop_last = True)
lfw_test_dataloader = DataLoader(test_data, batch_size = 603, shuffle = False, drop_last = True)