import pickle
from model import *
from utils import *
from config import config, log_config
from scipy.io import loadmat, savemat


from pickle import load
from model import *
from utils import *
from config import config, log_config
from scipy.io import loadmat
import torch
from torch.utils import data
import torch.optim as optim
from pytorchtools import EarlyStopping
import time



def main_test(device, model_name, mask_name, mask_perc):


    # =================================== BASIC CONFIGS =================================== #

    print('[*] run basic configs ... ')

    # configs
    sample_size = config.TRAIN.sample_size
    image_size = 256

    # setup log
    log_dir = "log_inference_{}_{}_{}".format(model_name, mask_name, mask_perc)
    isExists = os.path.exists(log_dir)
    if not isExists:
        os.makedirs(log_dir)
    _, _, log_inference, _, _, log_inference_filename = logging_setup(log_dir)

    # setup checkpoint
    checkpoint_dir = "checkpoint_{}_{}_{}".format(model_name, mask_name, mask_perc)
    isExists = os.path.exists(checkpoint_dir)
    if not isExists:
        os.makedirs(checkpoint_dir)

    # setup save dir
    save_dir = "sample_{}_{}_{}".format(model_name, mask_name, mask_perc)
    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)

    # ==================================== PREPARE DATA ==================================== #

    print('[*] load data ... ')
    testing_data_path = config.TRAIN.testing_data_path

    data_augment = DataAugment()
    with open(testing_data_path, 'rb') as f:
        X_test = torch.from_numpy(load(f))
        X_test = data_augment(X_test)

    log = 'X_test shape:{}/ min:{}/ max:{}\n'.format(X_test.shape, X_test.min(), X_test.max())
    print(log)
    log_.debug(log)
    log_eval.info(log)

    print('[*] loading mask ... ')
    if mask_name == "gaussian2d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian2D_path, "GaussianDistribution2DMask_{}.mat".format(mask_perc)))[
                'maskRS2']
    elif mask_name == "gaussian1d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian1D_path, "GaussianDistribution1DMask_{}.mat".format(mask_perc)))[
                'maskRS1']
    elif mask_name == "poisson2d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian1D_path, "PoissonDistributionMask_{}.mat".format(mask_perc)))[
                'population_matrix']
    else:
        raise ValueError("no such mask exists: {}".format(mask_name))
    mask = torch.from_numpy(mask).to(device)

    # ==================================== DEFINE MODEL ==================================== #

    # pre-processing for vgg
    vgg_pre = VGG_PRE(image_size)

    # load vgg
    vgg16_cnn = VGG_CNN()
    vgg16_cnn = vgg16_cnn.to(device)
    # load unet
    generator = UNet()
    generator = generator.to(device)

    # loss function
    bce = nn.BCELoss(reduction='mean').to(device)
    mse = nn.MSELoss(reduction='mean').to(device)

    # real and fake label
    real = 1.
    fake = 0.

    X_test = X_test.to(device)

    # (N, H, W, C)-->(N, C, H, W)
    X_test = X_test.permute(0, 3, 1, 2)
    X_bad_test = to_bad_img(X_test, mask)

    # generator
    if model_name == 'unet':
        X_generated_test = generator(X_bad_test, is_train=False, is_refine=False)
    elif model_name == 'unet_refine':
        X_generated_test = generator(X_bad_test, is_train=False, is_refine=True)
    else:
        raise Exception("unknown model")

    # generator loss (pixel-wise)
    g_nmse_a = mse(X_generated_test, X_test)
    g_nmse_b = mse(X_generated_test, torch.zeros_like(X_test).to(device))
    g_nmse = torch.div(g_nmse_a, g_nmse_b)




    print('[*] start testing ... ')

    x_gen = sess.run(net_test.outputs, {t_image_bad: X_samples_bad})
    x_gen_0_1 = (x_gen + 1) / 2

    # evaluation for generated data

    nmse_res = sess.run(nmse_0_1, {t_gen: x_gen_0_1, t_image_good: x_good_sample_rescaled})
    ssim_res = threading_data([_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=ssim)
    psnr_res = threading_data([_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=psnr)

    log = "NMSE testing: {}\nSSIM testing: {}\nPSNR testing: {}\n\n".format(
        nmse_res,
        ssim_res,
        psnr_res)

    log_inference.debug(log)

    log = "NMSE testing average: {}\nSSIM testing average: {}\nPSNR testing average: {}\n\n".format(
        np.mean(nmse_res),
        np.mean(ssim_res),
        np.mean(psnr_res))

    log_inference.debug(log)

    log = "NMSE testing std: {}\nSSIM testing std: {}\nPSNR testing std: {}\n\n".format(np.std(nmse_res),
                                                                                        np.std(ssim_res),
                                                                                        np.std(psnr_res))

    log_inference.debug(log)

    # evaluation for zero-filled (ZF) data
    nmse_res_zf = sess.run(nmse_0_1,
                           {t_gen: x_bad_sample_rescaled, t_image_good: x_good_sample_rescaled})
    ssim_res_zf = threading_data([_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=ssim)
    psnr_res_zf = threading_data([_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=psnr)

    log = "NMSE ZF testing: {}\nSSIM ZF testing: {}\nPSNR ZF testing: {}\n\n".format(
        nmse_res_zf,
        ssim_res_zf,
        psnr_res_zf)

    log_inference.debug(log)

    log = "NMSE ZF average testing: {}\nSSIM ZF average testing: {}\nPSNR ZF average testing: {}\n\n".format(
        np.mean(nmse_res_zf),
        np.mean(ssim_res_zf),
        np.mean(psnr_res_zf))

    log_inference.debug(log)

    log = "NMSE ZF std testing: {}\nSSIM ZF std testing: {}\nPSNR ZF std testing: {}\n\n".format(
        np.std(nmse_res_zf),
        np.std(ssim_res_zf),
        np.std(psnr_res_zf))

    log_inference.debug(log)

    # sample testing images
    tl.visualize.save_images(x_gen,
                             [5, 10],
                             os.path.join(save_dir, "final_generated_image.png"))

    tl.visualize.save_images(np.clip(10 * np.abs(X_samples_good - x_gen) / 2, 0, 1),
                             [5, 10],
                             os.path.join(save_dir, "final_generated_image_diff_abs_10_clip.png"))

    tl.visualize.save_images(np.clip(10 * np.abs(X_samples_good - X_samples_bad) / 2, 0, 1),
                             [5, 10],
                             os.path.join(save_dir, "final_bad_image_diff_abs_10_clip.png"))

    print("[*] Job finished!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')
    parser.add_argument('--mask', type=str, default='gaussian2d', help='gaussian1d, gaussian2d, poisson2d')
    parser.add_argument('--maskperc', type=int, default='30', help='10,20,30,40,50')

    args = parser.parse_args()

    tl.global_flag['model'] = args.model
    tl.global_flag['mask'] = args.mask
    tl.global_flag['maskperc'] = args.maskperc

    main_test()
