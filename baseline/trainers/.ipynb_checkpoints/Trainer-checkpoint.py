from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from time import time
import os
import shutil
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from .BaseTrainer import BaseTrainer
from models.KMARNet import weights_init_kaiming, Fusionnet
from utils.loss import TvLoss
from skimage.measure import compare_ssim, compare_psnr

torch.backends.cudnn.benchmark = True

def _metric(t, o, pcc, ssim, psnr, mse):
    _pcc = utils.pearson_correlation_coeff(t, o)
    _ssim = compare_ssim(np.moveaxis(t, 0, -1), np.moveaxis(o, 0, -1), multichannel=True)
    _psnr = compare_psnr(t, o, 1)
    _mse = (np.square(t - o)).mean(axis=None)

    pcc.append(_pcc)
    ssim.append(_ssim)
    psnr.append(_psnr)
    mse.append(_mse)


class RegressionTrainer(BaseTrainer):
    def __init__(self, args, data_loaders):
        super(RegressionTrainer, self).__init__(args, data_loaders)

        self.model = torch.nn.DataParallel(eval(args.model)(args), output_device=0).to(self.torch_device)
        self.data_type = args.data_type        
        
        self.recon_loss = torch.nn.MSELoss()
        self.ssim_loss = ssim_loss.SSIM(torch_device=self.torch_device)
        self.ssim_rate = args.ssim_rate
        self.tv_loss   = TvLoss()
        self.tv_rate = args.tv_rate
        
        self.optim = torch.optim.Adam(self.model.parameters())
        self.best_metric = 1000000   # MSE

        self.load()
        self.prev_epoch_loss = 0
        
        self.test_save_path = utils.get_save_dir(args) + '/test'
        if os.path.exists(self.test_save_path) is False:
            os.mkdir(self.test_save_path)

    def save(self, epoch, metric, filename="models"):
        save_path = Path(self.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save({"model_type": self.model_type,
                    "epoch": epoch + 1,
                    "param": self.model.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "score": metric,
                    }, str(save_path / "{}.pth.tar".format(filename)))

        if metric < self.best_metric:             # MSE
            print('{} -> {}'.format(self.best_metric, metric), end=', ')
            self.best_metric = metric
            shutil.copyfile(str(save_path / "{}.pth.tar".format(filename)), str(save_path / "best.pth.tar"))
            print("Model saved %d epoch" % (epoch))

    def load(self, file_name="models.pth.tar"):
        save_path = Path(self.save_path)
        if (save_path/file_name).exists():
            print("Load %s File" % (save_path))
            ckpoint = torch.load(str(save_path / file_name))
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s" % (ckpoint["model_type"]))

            self.model.load_state_dict(ckpoint['param'])
            self.optim.load_state_dict(ckpoint['optimizer'])
            self.epoch = ckpoint['epoch']
            self.best_metric = ckpoint["score"]
            print("Load Model Type : %s, epoch : %d" % (ckpoint["model_type"], self.epoch))
        else:
            print("Load Failed, not exists file")

    def _init_model(self):
        for m in self.model.modules():
            m.apply(weights_init_kaiming)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def train(self):
        print("\nStart Train")
        start_time = time.time()

        for epoch in range(self.epoch, self.args.num_epoch):
            loader = tqdm(self.data_loaders['train'],
                          total=len(self.data_loaders['train']),
                          desc="[Train epoch {}]".format(epoch))

            losses = []
            for i, (input_, target_, path) in enumerate(train_loader):
                self.model.train()
                
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_ = self.model(input_)

                # Loss definition
                if self.loss_ssim is None:
                    loss = self.recon_loss(output_, target_)  # Only MSE loss
                elif self.loss_mse is None:
                    loss = self.ssim_loss(output_, target_)   # Only ssim loss
                else:
                    if self.loss_tv is None:
                        loss = self.recon_loss(output_, target_) + self.ssim_loss(output_, target_) * self.ssim_rate     # MSE + ssim loss
                    else:
                        loss = self.recon_loss(output_, target_) + self.ssim_loss(output_, target_) * self.ssim_rate \
                               + self.tv_loss(output_) * self.tv_rate       # MSE + ssim loss + TV loss

                self.optim.zero_grad()
                recon_loss.backward()
                self.optim.step()
                
                loss_np = loss.cpu().detach().numpy()
                losses += [loss_np]

                loader.set_postfix(loss='{:f} / {:f}'.format(loss_np, np.average(losses)))

                del input_, target_, output_, loss
            
            self.logger.will_write("[Train] epoch:%d loss:%f" % (epoch, np.average(losses)), is_print=False)

            self.valid(epoch)

        print("\nEnd Train")
        total_time = time.time() - start_time
        print("\nTraining Time:", total_time, 'sec')
        self.logger.will_write("Total training time: %f sec" %total_time)
        

    def _test_foward(self, input_, target_):
        input_ = input_.to(self.torch_device)
        output_ = self.model(input_).type(torch.FloatTensor).numpy()
        target_ = target_.type(torch.FloatTensor).numpy()
        input_ = input_.type(torch.FloatTensor).numpy()
        return input_, output_, target_

    def valid(self, epoch):
        self.model.eval()
        num_workers = self.args.batch_size if self.args.batch_size < 11 else 10
        
        with torch.no_grad(), ThreadPoolExecutor(max_workers=num_workers) as executor:
            pcc = []
            ssim = []
            psnr = []
            mse = []
            futures = []
            
            for i, (input_np, target_np, path) in enumerate(self.data_loaders['val']):
                _, output_, target_ = self._test_foward(input_np, target_np)
                
                futures = []
                for batch_idx in range(target_.shape[0]):         # batch 안 output / target 한장씩 꺼내오기
                    o, t = output_[batch_idx], target_[batch_idx]
                    
                    # Metric calculation
                    future = executor.submit(_metric, t, o, pcc, ssim, psnr, mse)
                    futures.append(future)
                    
            for f in futures:
                f.result()

            pcc_ = np.mean(pcc)
            ssim_ = np.mean(ssim)
            psnr_ = np.mean(psnr)
            mse_ = np.mean(mse)

            metric = mse_   # metric 조합 가능: 10*pcc_ + ssim_ + psnr_ + mse_

            self.save(epoch, metric)
            self.logger.write("[Val] epoch:%d pcc:%f ssim:%f psnr:%f mse:%f" % (epoch, pcc_, ssim_, psnr_, mse_))

            del input_, target_, pcc_, ssim_, psnr_, metric

    def test(self):
        self.load(file_name='best.pth.tar')
        self.model.eval()
        num_workers = self.args.batch_size if self.args.batch_size < 11 else 10
        
        with torch.no_grad(), ThreadPoolExecutor(max_workers=num_workers) as executor:
            pcc = []
            ssim = []
            psnr = []
            mse = []
            futures = []
            
            loader = tqdm(self.data_loaders['test'],
                          total=len(self.data_loaders['test']))
            
            for i, (input_np, target_np, path) in enumerate(loader):
                input_, output_, target_ = self._test_foward(input_np, target_np)
                
                for batch_idx in range(target_.shape[0]): 
                    o, t = output_[batch_idx], target_[batch_idx]
                    future = executor.submit(_metric, t, o, pcc, ssim, psnr, mse)
                    futures.append(future)                    
                    
                    if self.data_type == 'dicom':
                        input_b = input_[batch_idx, 0, :, :] * 4095
                        output_b = output_[batch_idx, 0, :, :] * 4095
                        target_b = target_[batch_idx, 0, :, :] * 4095  # 512 x 512
                    
                    else:
                        input_b = input_[batch_idx, 0, :, :]
                        output_b = output_[batch_idx, 0, :, :]
                        target_b = target_[batch_idx, 0, :, :]                       
                    
                    ### Test results save ###
                    fname = path[batch_idx]
                    save_path = "%s/%s" % (self.test_save_path, fname[:-4])
                    
                    total = np.concatenate([input_b, target_b, output_b], axis=1)                    
                    np.save(save_path + '.npy', total)

            for f in futures:
                f.result()
            
            pcc_ = np.mean(pcc)
            ssim_ = np.mean(ssim)
            psnr_ = np.mean(psnr)
            mse_ = np.mean(mse)
            
            self.logger.write("[Summary] pcc:%f ssim:%f psnr:%f mse:%f" % (pcc_, ssim_, psnr_, mse_))

            print("End Test\n")