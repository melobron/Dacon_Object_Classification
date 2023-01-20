from dataloaders import CIFAR10
from torch.utils.data import Dataset, DataLoader

import numpy as np

a = np.mean([1, 2, 3])
b = np.mean([2, 3, 4])
print(np.mean([a, b]))
