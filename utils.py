import logging
import os.path
import numpy as np
import scipy.fftpack
import skimage.metrics
import torch
import torch.fft
import torchvision.transforms as transforms
from math import ceil

# Early stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, model_name='unet', mask_name='gaussian2d', mask_perc=30,
                 verbose=False, delta=0, checkpoint_path='./checkpoint_dir', log_path='./log_dir',
                 log_all=False, log_eval=False, trace_func=print):

        self.patience = patience
        self.model_name = model_name
        self.mask_name = mask_name
        self.mask_perc = mask_perc
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.nmse = None
        self.early_stop = False
        self.val_nmse_min = np.Inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.log_all = log_all
        self.log_eval = log_eval
        self.trace_func = trace_func

    def __call__(self, nmse, model_g, model_d, epoch):

        # nmse(+ --> 0) score(- --> 0)
        score = -nmse
        # init score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(nmse, model_g, model_d, epoch)
        # new model is worse
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        # new model is better
        else:
            self.best_score = score
            self.save_checkpoint(nmse, model_g, model_d, epoch)
            self.counter = 0

        log = f'EarlyStopping counter of epoch {epoch + 1} : {self.counter} out of {self.patience}'
        # print(log)
        self.log_all.debug(log)

    def save_checkpoint(self, val_nmse, model_g, model_d, epoch):
        # save model when validation loss decrease.
        if self.verbose:
            log = f'Validation loss decreased ({self.val_nmse_min:.6f} --> {val_nmse:.6f}).  Saving model ...'
            # print(log)
            self.log_all.debug(log)
            self.log_eval.info(log)
        # update the min nmse
        self.val_nmse_min = val_nmse
        torch.save(model_g.state_dict(),
                   os.path.join(self.checkpoint_path,
                                "best_checkpoint_generator_{}_{}_{}_epoch_{}_nmse_{}.pt"
                                .format(self.model_name, self.mask_name, self.mask_perc, epoch + 1, self.val_nmse_min)))
        torch.save(model_d.state_dict(),
                   os.path.join(self.checkpoint_path,
                                "best_checkpoint_discriminator_{}_{}_{}_epoch_{}_nmse_{}.pt"
                                .format(self.model_name, self.mask_name, self.mask_perc, epoch + 1, self.val_nmse_min)))


# Data Augment
class DataAugment:
    def __init__(self):
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0)
        ])

    def __call__(self, x):
        x = torch.div(torch.add(x, torch.ones_like(x)), 2)
        x = self.transform_train(x)
        x = torch.sub(torch.mul(x, 2), torch.ones_like(x))

        return x

def to_bad_img(x, mask):
    y = x.copy()
    for i in range(x.shape[0]):
        xx = x[i, :, :, 0]
        xx = (xx + 1.) / 2.
        fft = scipy.fftpack.fft2(xx)
        fft = scipy.fftpack.fftshift(fft)
        fft = fft * mask
        fft = scipy.fftpack.ifftshift(fft)
        xx = scipy.fftpack.ifft2(fft)
        xx = np.abs(xx)
        xx = xx * 2 - 1
        y[i, :, :, :] = xx[:, :, np.newaxis]
    return y


# Fourier Transform
def fft_abs_for_map_fn(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.div(torch.add(x, torch.ones_like(x)), 2)
    fft_x = torch.fft.fftn(x)
    fft_abs = torch.abs(fft_x)

    return fft_abs


# Structural Similarity
def ssim(x_good, x_bad):
    x_good = np.squeeze(x_good.numpy())
    x_bad = np.squeeze(x_bad.numpy())
    ssim_res = []
    for idx in range(x_good.shape[0]):
        ssim_res.append(skimage.metrics.structural_similarity(x_good[idx], x_bad[idx]))

    return ssim_res


# Peak Signal to Noise Ratio
def psnr(x_good, x_bad):
    x_good = np.squeeze(x_good.numpy())
    x_bad = np.squeeze(x_bad.numpy())
    psnr_res = []
    for idx in range(x_good.shape[0]):
        psnr_res.append(skimage.metrics.peak_signal_noise_ratio(x_good[idx], x_bad[idx]))

    return psnr_res


# Preparation for VGG
class VGG_PRE:
    def __init__(self):
        self.transform_vgg = transforms.Compose([transforms.Resize((244, 244))])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __call__(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = torch.mul(torch.add(x, torch.ones_like(x)), 127.5)
        mean = torch.from_numpy(np.array([123.68, 116.779, 103.939], dtype=np.float32)
                                .reshape((1, 3, 1, 1)))
        x = torch.sub(x, mean.to(self.device))
        x = self.transform_vgg(x)

        return x


# Preparation for UNet
class PREPROCESS:
    def __init__(self):
        pass

    def __call__(self, x):
        x = x.permute(0, 3, 1, 2)

        h_padding = 256 - x.shape[2]
        w_padding = 256 - x.shape[3]
        if h_padding > 0:
            h_padding_t = ceil(h_padding / 2)  # 128 + ceil(x.shape[2]/2)
            h_padding_b = h_padding - h_padding_t  # 128 - ceil(x.shape[2]/2) - x.shape[2]
            h_cutting_t = 0
            h_cutting_b = 256
        else:
            h_padding_t = 0
            h_padding_b = 0
            h_cutting_t = ceil(x.shape[2] / 2) - 128
            h_cutting_b = ceil(x.shape[2] / 2) + 128
        if w_padding > 0:
            w_padding_l = ceil(w_padding / 2)  # 128 + ceil(x.shape[3]/2)
            w_padding_r = w_padding - w_padding_l  # 128 - ceil(x.shape[3]/2) - x.shape[3]
            w_cutting_t = 0
            w_cutting_b = 256
        else:
            w_padding_l = 0
            w_padding_r = 0
            w_cutting_t = ceil(x.shape[3] / 2) - 128
            w_cutting_b = ceil(x.shape[3] / 2) + 128

        constant_padding = torch.nn.ConstantPad2d((w_padding_l, w_padding_r, h_padding_t, h_padding_b), -1)
        x = constant_padding(x)
        x = x[:, :, h_cutting_t:h_cutting_b, w_cutting_t:w_cutting_b]

        x = x.permute(0, 2, 3, 1)

        return x


# Logger Setup for Train and Val
def logging_setup(log_dir):
    # generate train log filename
    log_all_filename = os.path.join(log_dir, 'log_all.log')
    # generate eval log filename
    log_eval_filename = os.path.join(log_dir, 'log_eval.log')

    # set train log
    log_all = logging.getLogger('log_all')
    log_all.setLevel(logging.DEBUG)
    log_all.addHandler(logging.FileHandler(log_all_filename))

    # set eval log
    log_eval = logging.getLogger('log_eval')
    log_eval.setLevel(logging.INFO)
    log_eval.addHandler(logging.FileHandler(log_eval_filename))

    return log_all, log_eval, log_all_filename, log_eval_filename


# Logger Setup for Test
def logging_test_setup(log_dir):
    # generate test log filename
    log_test_filename = os.path.join(log_dir, 'log_test.log')

    # set test log
    log_test = logging.getLogger('log_test')
    log_test.setLevel(logging.INFO)
    log_test.addHandler(logging.FileHandler(log_test_filename))

    return log_test, log_test_filename




if __name__ == "__main__":
    pass
