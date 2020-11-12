from pickle import load
from model import *
from utils import *
from config import config, log_config
from scipy.io import loadmat
import torch
from torch.utils import data
import torchvision


def main_test(device, model_name, mask_name, mask_perc):
    # configs
    batch_size = config.TRAIN.batch_size
    weight_unet = config.TRAIN.weight_unet

    print('[*] Run Basic Configs ... ')
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
    save_dir = "sample_{}_{}_{}/{}".format(model_name, mask_name, mask_perc, weight_unet)
    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)

    print('[*] Loading data ... ')
    # data path
    testing_data_path = config.TRAIN.testing_data_path

    # load data (augment)
    with open(testing_data_path, 'rb') as f:
        X_test = torch.from_numpy(load(f))

    log = 'X_test shape:{}/ min:{}/ max:{}'.format(X_test.shape, X_test.min(), X_test.max())
    # print(log)
    log_inference.info(log)

    print('[*] Loading Mask ... ')
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

    print('[*] Loading Network ... ')
    # data loader
    dataloader_test = torch.utils.data.DataLoader(X_test[0:100], batch_size=batch_size, pin_memory=True, timeout=0,
                                                  shuffle=True)
    # load unet
    generator = UNet()
    generator = generator.to(device)
    generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, weight_unet)))

    # loss function
    mse = nn.MSELoss(reduction='mean').to(device)

    # real and fake label
    real = 1.
    fake = 0.

    print('[*] Testing  ... ')
    # initialize testing
    total_nmse_test = 0
    total_ssim_test = 0
    total_psnr_test = 0
    num_test_temp = 0

    X_good_test_sample = []
    X_bad_test_sample = []
    X_generated_test_sample = []

    with torch.no_grad():
        # testing
        for step, X_good in enumerate(dataloader_test):
            X_good = X_good.to(device)
            print(step)
            # (N, H, W, C)-->(N, C, H, W)
            X_good = X_good.permute(0, 3, 1, 2)
            X_bad = to_bad_img(X_good, mask)

            # generator
            if model_name == 'unet':
                X_generated = generator(X_bad, is_train=False, is_refine=False)
            elif model_name == 'unet_refine':
                X_generated = generator(X_bad, is_train=False, is_refine=True)
            else:
                raise Exception("unknown model")

            # gpu --> cpu
            X_good = X_good.cpu()
            X_generated = X_generated.cpu()
            X_bad = X_bad.cpu()

            # (-1,1)-->(0,1)
            X_good_0_1 = torch.div(torch.add(X_good, torch.ones_like(X_good)), 2)
            X_generated_0_1 = torch.div(torch.add(X_generated, torch.ones_like(X_generated)), 2)
            X_bad_0_1 = torch.div(torch.add(X_bad, torch.ones_like(X_bad)), 2)

            # eval for validation
            nmse_a = mse(X_generated_0_1, X_good_0_1)
            nmse_b = mse(X_generated_0_1, torch.zeros_like(X_generated_0_1))
            nmsn_res = torch.div(nmse_a, nmse_b).cpu().numpy()
            ssim_res = ssim(X_generated_0_1.cpu(), X_bad_0_1.cpu())
            psnr_res = psnr(X_generated_0_1.cpu(), X_bad_0_1.cpu())

            total_nmse_test = total_nmse_test + np.sum(nmsn_res)
            total_ssim_test = total_ssim_test + np.sum(ssim_res)
            total_psnr_test = total_psnr_test + np.sum(psnr_res)

            num_test_temp = num_test_temp + batch_size

            # output the sample
            X_good_test_sample.append(X_good_0_1[0, :, :, :])
            X_generated_test_sample.append(X_generated_0_1[0, :, :, :])
            X_bad_test_sample.append(X_bad_0_1[0, :, :, :])

        total_nmse_test = total_nmse_test / num_test_temp
        total_ssim_test = total_ssim_test / num_test_temp
        total_psnr_test = total_psnr_test / num_test_temp

        # record testing eval
        log = "NMSE testing average: {:8}\nSSIM testing average: {:8}\nPSNR testing average: {:8}\n\n".format(
            total_nmse_test, total_ssim_test, total_psnr_test)
        print(log)
        log_inference.debug(log)

        # save image

        for i in range(len(X_good_test_sample)):
            torchvision.utils.save_image(X_good_test_sample[i],
                                         os.path.join(save_dir, 'GroundTruth_{}.png'.format(i)))
            torchvision.utils.save_image(X_generated_test_sample[i],
                                         os.path.join(save_dir, 'Generated_{}.png'.format(i)))
            torchvision.utils.save_image(X_bad_test_sample[i],
                                         os.path.join(save_dir, 'Bad_{}.png'.format(i)))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')
    parser.add_argument('--mask', type=str, default='gaussian2d', help='gaussian1d, gaussian2d, poisson2d')
    parser.add_argument('--maskperc', type=int, default='30', help='10,20,30,40,50')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main_test(device, args.model, args.mask, args.maskperc)
