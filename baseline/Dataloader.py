import torch
import torch.utils.data as data
import random
import numpy as np
import pandas as pd
import time
import os
import cv2
import h5py # why exists?
from glob import glob
import warnings
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore", ".*output shape of zoom.*") # study later


class CIFAR10_Dataset(data.Dataset):
    def __init__(self, img_root, transforms=None):
        self.img_paths = glob(img_root + '/*/*')
        self.transforms = transforms
        if len(self.img_paths) == 0:
            raise ValueError("Check data path : %s"%(img_root))
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

    def __getitem__(self, idx):
        path_file = self.img_paths[idx]
        #image = cv2.imread(path_file)
        #image = np.array(image, dtype=np.float32)
        image = Image.open(path_file)
        if self.transforms is not None:
            image = self.transforms(image)
        label = np.zeros([10])
        label[self.label_dict[path_file.split("/")[-2]]] = 1
        
        return image, label

    def __len__(self):
        return len(self.img_paths)

class Inference_Dataset(data.Dataset):
    def __init__(self, img_root, csv_path, transforms=None):
        self.img_root = img_root
        self.img_paths = glob(img_root + '/*')
        self.pd_csv = pd.read_csv(csv_path)
        self.transforms = transforms
        if len(self.img_paths) == 0:
            raise ValueError("Check data path : %s"%(img_root))

    def __getitem__(self, idx):
        id = self.pd_csv["id"][idx]
        path_file = os.path.join(str(self.img_root), id)
        image = Image.open(path_file)
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image.float()

    def __len__(self):
        return len(self.img_paths)

class WrapperDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, answer = self.dataset[index]

        if self.transform is not None:
            image = self.transform(image)

        answer = torch.from_numpy(answer)
        return image.float(), answer.float()

    def __len__(self):
        return len(self.dataset)

    