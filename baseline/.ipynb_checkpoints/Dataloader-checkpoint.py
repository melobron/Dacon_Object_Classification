import torch
import torch.utils.data as data
import random
import numpy as np
import time
import os
import h5py
from glob import glob
from utils import preprocess
import warnings
from collections import namedtuple
warnings.filterwarnings("ignore", ".*output shape of zoom.*")


class CT_Dataset(data.Dataset):
    def __init__(self, img_root, infer=False, augment=None, torch_type="float", augment_rate=0.3, zoom_factor=1):
        if type(img_root) == list:
            img_paths = [p for path in img_root for p in glob(path + "/*.npy")]
        else:
            img_paths = glob(img_root + '/*.npy')

        if len(img_paths) == 0:
            raise ValueError("Check data path : %s"%(img_root))

        self.origin_image_len = len(img_paths)
        self.img_paths = img_paths
        if augment is not None:
            self.img_paths += random.sample(img_paths, int(self.origin_image_len * augment_rate))

        self.transform = [] if augment is None else preprocess.get_preprocess(augment)
        self.torch_type = torch.float if torch_type == float else torch.half
        self.zoom_factor = zoom_factor
        self.data_type = data_type

    def __getitem__(self, idx):
        return self._loader(idx)

    def __len__(self):
        return len(self.img_paths)

    def _np2tensor(self, np):
        tmp = torch.from_numpy(np).view(1, *np.shape)
        return tmp.to(dtype=self.torch_type)

    def _loader(self, idx):
        img_path = self.img_paths[idx]
        img = np.load(img_path)

        input_np = img[:, :512] / 4095.0      # MAR Knee 2D
        target_np = img[:, 512:] / 4095.0     # MAR Knee 2D

        input_ = self._np2tensor(input_np)
        target_ = self._np2tensor(target_np)

        return input_, target_, os.path.basename(img_path)


def load_dataset(args):
    data_loader = {}

    if args.fold == None:    # Fold 가 None 일 때
        train_dataset = CT_Dataset(args.data_path + '/train', infer=False, augment=args.augment, torch_type=args.torch_type,
                                augment_rate=args.augment_rate, zoom_factor=args.zoom_factor, data_type=args.data)
        val_dataset = CT_Dataset(args.data_path + '/val', infer=True, augment=None, torch_type=args.torch_type,
                                augment_rate=0, zoom_factor=args.zoom_factor, data_type=args.data)

    else:  # Fold number 가 있을 때
        train_dataset = CT_Dataset(args.data_path + '/fold%s/train' % args.fold, infer=False, augment=args.augment,
                                torch_type=args.torch_type, augment_rate=args.augment_rate,
                                zoom_factor=args.zoom_factor, data_type=args.data)
        val_dataset = CT_Dataset(args.data_path + '/fold%s/val' % args.fold, infer=True, augment=None,
                                torch_type=args.torch_type, augment_rate=0,
                                zoom_factor=args.zoom_factor, data_type=args.data)
    
    test_dataset = CT_Dataset(args.data_path + '/test', infer=True, augment=None, torch_type=args.torch_type,
                             augment_rate=0, zoom_factor=args.zoom_factor, data_type=args.data)

    data_loader['train'] = data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.cpus,
                                        drop_last=False, pin_memory=True)
    data_loader['val'] = data.DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.cpus,
                                        drop_last=False, pin_memory=True)
    data_loader['test'] = data.DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.cpus,
                                        drop_last=False, pin_memory=True)

    return data_loader


def __test_npy(npy, txt):
    print(txt)
    print("shape : ", npy.shape)
    print("type : ", npy.dtype)
    print("min : ", npy.min())
    print("max : ", npy.max())


if __name__ == "__main__":
    data_path = 'dataset_HN'
    aug = 'flip'

    Arg = namedtuple('Arg', ['data_path', 'augment', 'torch_type', 'augment_rate', 'zoom_factor',
                             'batch_size', 'cpus', 'fold', 'data_type', 'phase'])
    args = Arg(data_path, aug, 'float', 0.3, 1, 4, 4, None, 'dicom', 'train')

    test_loader = load_dataset(args)
    print(test_loader)

    t = time.time()
    for i, (input_np, target_np, fname) in enumerate(test_loader['train']):
        print(time.time() - t, fname, input_np.shape, target_np.shape)
        t = time.time()

