# utils.py
from pathlib import Path
import os
import numpy as np
import scipy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch
import math
import h5py
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def cal_acc_recal_pre_f1(outputs, targets):
    target = targets.reshape(-1)
    output = outputs.reshape(-1)
    acc = accuracy_score(target, output)
    recall = recall_score(target, output, average='macro', zero_division = 0)
    precision = precision_score(target, output, average='macro', zero_division = 0)
    F1_score = f1_score(target, output, average='macro', zero_division = 0)
    return acc, recall, precision, F1_score
    
# from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img