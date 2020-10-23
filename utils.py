from tensorlayer.prepro import *
import numpy as np
import skimage.measure
import scipy
from time import localtime, strftime
import logging

# 图像失真
def distort_img(x):
    x = (x + 1.) / 2.
    x = flip_axis(x, axis=1, is_random=True)
    x = elastic_transform(x, alpha=255 * 3, sigma=255 * 0.10, is_random=True)
    x = rotation(x, rg=10, is_random=True, fill_mode='constant')
    x = shift(x, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    x = zoom(x, zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
    x = brightness(x, gamma=0.05, is_random=True)
    x = x * 2 - 1
    return x

# 滤波
def to_bad_img(x, mask):
    x = (x + 1.) / 2.
    fft = scipy.fftpack.fft2(x[:, :, 0])
    fft = scipy.fftpack.fftshift(fft)
    fft = fft * mask
    fft = scipy.fftpack.ifftshift(fft)
    x = scipy.fftpack.ifft2(fft)
    x = np.abs(x)
    x = x * 2 - 1
    return x[:, :, np.newaxis]


# 傅里叶变换&绝对值
def fft_abs_for_map_fn(x):
    x = (x + 1.) / 2.
    x_complex = tf.complex(x, tf.zeros_like(x))[:, :, 0]
    fft = tf.spectral.fft2d(x_complex)
    fft_abs = tf.abs(fft)
    return fft_abs


# 结构相似性
def ssim(data):
    x_good, x_bad = data
    x_good = np.squeeze(x_good)
    x_bad = np.squeeze(x_bad)
    ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    return ssim_res


# 峰值信噪比
def psnr(data):
    x_good, x_bad = data
    psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
    return psnr_res


# vgg 准备
def vgg_prepro(x):
    x = imresize(x, [244, 244], interp='bilinear', mode=None)
    x = np.tile(x, 3)
    x = x / 127.5 - 1
    return x


# 日志
def logging_setup(log_dir):
    # get current time
    current_time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    # generate train log filename
    log_all_filename = os.path.join(log_dir, 'log_all_{}.log'.format(current_time_str))
    # generate eval log filename
    log_eval_filename = os.path.join(log_dir, 'log_eval_{}.log'.format(current_time_str))
    # set train log
    log_all = logging.getLogger('log_all')
    log_all.setLevel(logging.DEBUG)
    log_all.addHandler(logging.FileHandler(log_all_filename))
    # set eval log
    log_eval = logging.getLogger('log_eval')
    log_eval.setLevel(logging.INFO)
    log_eval.addHandler(logging.FileHandler(log_eval_filename))
    # generate test log filename
    log_50_filename = os.path.join(log_dir, 'log_50_images_testing_{}.log'.format(current_time_str))
    # set test log
    log_50 = logging.getLogger('log_50')
    log_50.setLevel(logging.DEBUG)
    log_50.addHandler(logging.FileHandler(log_50_filename))
    # logging.Logger object, str
    return log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename


if __name__ == "__main__":
    pass
