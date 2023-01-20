import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from glob import glob
import cv2
import numpy as np


class CIFAR10(Dataset):
    def __init__(self, train=True, transform=None):
        super(CIFAR10, self).__init__()

        self.train = train
        if train:
            self.img_dir = './datasets/train/'
            self.img_paths = glob(os.path.join(self.img_dir, '*', '*.jpg'))
        else:
            self.img_dir = './datasets/test/'
            self.img_paths = glob(os.path.join(self.img_dir, '*.jpg'))

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.label_dict = {
            "airplane" : 0,
            "automobile" : 1,
            "bird" : 2,
            "cat" : 3,
            "deer" : 4,
            "dog" : 5,
            "frog" : 6,
            "horse" : 7,
            "ship" : 8,
            "truck" : 9
        }

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image_numpy = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_numpy)

        if self.train:
            name = os.path.basename(os.path.split(img_path)[0])
            label = np.zeros([10])
            label[self.label_dict[name]] = 1
            label = torch.FloatTensor(label)
            return image_tensor, label
        else:
            name = os.path.basename(img_path)
            return image_tensor, name

    def __len__(self):
        return len(self.img_paths)

