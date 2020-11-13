import numpy as np
import skimage.metrics
from time import localtime, strftime
import logging
import os
import torch.fft
import torchvision.transforms as transforms


# Data Augment
class DataAugment:
    def __init__(self):
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=45),
            transforms.RandomAffine(0)])

    def __call__(self, x):
        x = torch.div(torch.add(x, torch.ones_like(x)), 2)
        x = self.transform_train(x)
        x = torch.sub(torch.mul(x, 2), torch.ones_like(x))

        return x


# Filtering
def to_bad_img(x, mask):
    x = torch.div(torch.add(x, torch.ones_like(x)), 2)
    fft_x = torch.fft.fftn(x)
    fft_x = torch.mul(fft_x, mask)
    x = torch.fft.ifftn(fft_x)
    x = torch.abs(x)
    x = torch.sub(torch.mul(x, 2), torch.ones_like(x))

    return x


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
        psnr_res = skimage.metrics.peak_signal_noise_ratio(x_good[idx], x_bad[idx])

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


# Logger Setup for Train and Val
def logging_setup(log_dir, current_time_str):

    # generate train log filename
    log_all_filename = os.path.join(log_dir, current_time_str, 'log_all_{}.log'.format(current_time_str))
    # generate eval log filename
    log_eval_filename = os.path.join(log_dir, current_time_str, 'log_eval_{}.log'.format(current_time_str))

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
def logging_test_setup(log_dir, current_time_str):

    # generate test log filename
    log_test_filename = os.path.join(log_dir, current_time_str, 'log_test_{}.log'.format(current_time_str))

    # set test log
    log_test = logging.getLogger('log_test')
    log_test.setLevel(logging.INFO)
    log_test.addHandler(logging.FileHandler(log_test_filename))

    return log_test, log_test_filename

if __name__ == "__main__":
    pass
