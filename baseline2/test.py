import torch
import torchvision.transforms as transforms

from dataloaders import CIFAR10
from models import ConvNextModel, ViTModel

import argparse
import numpy as np
import csv

# Parameters
parser = argparse.ArgumentParser(description='Dacon Classification')
parser.add_argument("--model_name", type=str, default='ConvNext', help="model_name")
parser.add_argument("--version", type=int, default=1, help="ConvNext Version")

parser.add_argument("--imageSize", type=int, default=128, help="imageSize")
parser.add_argument("--batchSize", type=int, default=128, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="Epochs")

parser.add_argument("--optimizer", type=str, default='Adam', help="Optimizer")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
parser.add_argument("--scheduler", type=str, default='Plateau', help="lr Scheduler")

# Training parameters
parser.add_argument("--gpu_num", type=int, default=0, help="select gpu")

args = parser.parse_args()

# Device
gpu_num = 0
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda:{}".format(gpu_num) if use_cuda else "cpu")
print(device)

# Train Dataset
train_dataset = CIFAR10.CIFAR10(train=True)

# Test Dataset
test_dataset = CIFAR10.CIFAR10(train=False)

# Model
model = ConvNextModel(version=3)
model.load_state_dict(torch.load('./trained_models/ConvNext_128batch_SGDoptimizer_Plateauscheduler_500epochs', map_location='cuda'))
# model = ViTModel(image_size=512)
# model.load_state_dict(torch.load('./trained_models/ViT_128batch_Adamoptimizer_Lambdascheduler_500epochs', map_location='cuda'))

model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4913, 0.4821, 0.4465], std=[0.2470, 0.2434, 0.2615]),
    # transforms.Resize(args.imageSize, interpolation=transforms.InterpolationMode.BICUBIC),
])

label_dict = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}

f = open('result.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['id', 'target'])

for index, (image, name) in enumerate(test_dataset):
    image = torch.unsqueeze(image, dim=0).to(device)
    prediction = np.argmax(model(image).detach().cpu().numpy())
    class_name = list(label_dict.keys())[list(label_dict.values()).index(prediction)]
    wr.writerow([name, class_name])
    print('{}th image'.format(index+1))

f.close()

# accuracy = 0
# for index, (image, label) in enumerate(train_dataset):
#     image = torch.unsqueeze(image, dim=0).to(device)
#     prediction = np.argmax(model(image).detach().cpu().numpy())
#     answer = np.argmax(label.numpy())
#     print('prediction: {}, answer: {}'.format(prediction, answer))
#     if prediction == answer:
#         accuracy += 1/len(train_dataset)
#
# print(accuracy)
