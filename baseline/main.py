import os
import random
import numpy as np
import torch
import argparse

from utils.arg_parser import parse_args
from Dataloader import CIFAR10_Dataset, Inference_Dataset
from trainers.Trainer import Trainer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True 

def main():

    # Input argument from shell script
    argparser = argparse.ArgumentParser()
    # INI script file name
    argparser.add_argument('inifile')
    # TODO: Add TensorboardX to store losses, metrics, and output images
    argparser.add_argument('--log_dir', nargs='?', help='Dir to save logs')
    cmd_args = argparser.parse_args()

    args = parse_args(cmd_args.inifile)

    # overwrite ini args with cmd args
    for k, v in cmd_args.__dict__.items():
        if v:
            # create non-existing directory
            if k.endswith('dir'):
                if not os.path.exists(v):
                    os.makedirs(v)

            args.__setattr__(k, v)

    # GPU selection
    #if args.gpus is not None:
    #    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    seed_everything(args.seed)

    data_set = CIFAR10_Dataset(os.path.join(str(args.data_path), "new_train"))
    test_dataset = CIFAR10_Dataset(os.path.join(str(args.data_path), "new_test"))
    trainer = Trainer(args, data_set, test_dataset)

    if args.phase == 'train':
        trainer.train()
        trainer.test()
    elif args.phase == 'test':
        trainer.test()
    elif args.phase == 'inference':
        trainer.inference()
if __name__ == "__main__":
    main()
