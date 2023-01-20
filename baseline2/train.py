import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

import argparse
import numpy as np
import time
import os

# from models import EfficientNetModel, ViTModel, ConvNextModel
from models import ConvNextModel
from dataloaders import CIFAR10

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
gpu_num = args.gpu_num
no_cuda = False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda:{}".format(gpu_num) if use_cuda else "cpu")
print(device)

# Model
if args.model_name == 'EfficientNet':
    model = EfficientNetModel(out_channels=10)
elif args.model_name == 'ViT':
    model = ViTModel(image_size=args.imageSize, in_channels=3, num_classes=10)
elif args.model_name == 'ConvNext':
    if args.version == 1:
        model = ConvNextModel(version=1)
    elif args.version == 2:
        model = ConvNextModel(version=2)
    elif args.version == 3:
        model = ConvNextModel(version=3)

model.to(device)

# Dataloader
dataset = CIFAR10.CIFAR10(train=True)
test_dataset = CIFAR10.CIFAR10(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=args.batchSize, shuffle=True)

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4913, 0.4821, 0.4465], std=[0.2470, 0.2434, 0.2615]),
    transforms.Resize(args.imageSize, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(90, expand=True),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomPerspective()
])

# Criterion
criterion = nn.CrossEntropyLoss().to(device)

# Optimizer
if args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

# Scheduler
if args.scheduler == 'Lambda':
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
elif args.scheduler == 'Plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
elif args.scheduler == 'Cyclic':
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00001, max_lr=0.01, step_size_up=50, step_size_down=50,
                                            mode='exp_range', gamma=0.995)
elif args.scheduler == 'Cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=2, eta_min=0)

# Loss list
train_loss_list = {}
validation_loss_list = {}
accuracy_list = {}

# Training
splits = StratifiedKFold(n_splits=3, shuffle=True)
start = time.time()

count_label = []
for index in range(len(dataset)):
    count_label.append(np.argmax(dataset[index][1]))

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)), count_label)):
    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_dataloader = DataLoader(dataset, batch_size=args.batchSize, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=args.batchSize, sampler=val_sampler)

    train_loss_list[fold] = []
    validation_loss_list[fold] = []
    accuracy_list[fold] = []

    for epoch in range(1, args.nEpochs + 1):
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        # Train Set
        model.train()

        for batch, (image, label) in enumerate(train_dataloader):
            image, label = image.to(device), label.to(device)
            output = model(image)

            train_loss = criterion(output, torch.argmax(label, axis=1))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            accuracy = accuracy_score(torch.argmax(output, axis=1).cpu(), torch.argmax(label, axis=1).cpu())

            if batch % 10 == 0:
                print('Train Fold: {} Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                    fold+1, epoch, batch * args.batchSize, len(train_dataloader.dataset),
                    100. * batch / len(train_dataloader), np.mean(train_loss.item())))
                train_loss_list[fold].append(train_loss.item())
                accuracy_list[fold].append(accuracy)

        # Validation
        val_loss = 0
        model.eval()

        for batch, (image, label) in enumerate(val_dataloader):
            image, label = image.to(device), label.to(device)

            with torch.no_grad():
                output = model(image)

            val_loss += criterion(output, torch.argmax(label, axis=1)).item()
            accuracy = accuracy_score(torch.argmax(output, axis=1).cpu(), torch.argmax(label, axis=1).cpu())

            if batch % 10 == 0:
                validation_loss_list[fold].append(val_loss)
                accuracy_list[fold].append(accuracy)

        print('Validation Fold: {} Epoch: {} Loss: {:.6f} Accuracy: {:.6f}'.format(fold+1, epoch, val_loss / len(val_dataloader),
                                                                                   np.mean(accuracy_list[fold])))

        if args.scheduler == 'Plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

# total_accuracy = np.mean([np.mean(accuracy_list[fold] for fold in range(5))])

# Save Model
save_name = '{}_{}batch_{}optimizer_{}scheduler_{}epochs'.format(args.model_name, args.batchSize, args.optimizer,
                                                                     args.scheduler, args.nEpochs)
torch.save(model.state_dict(), os.path.join('./trained_models/', save_name))

# Save Time
print("time :", time.time() - start)
history = open('runtime_history.txt', 'a')
history.write(save_name + '\truntime:{}'.format(time.time() - start) + '\n')
