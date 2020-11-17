from pickle import load

import torchvision
from scipy.io import loadmat
from torch.utils import data

from config import config
from model import *
from utils import *


def main_test(device, model_name, mask_name, mask_perc):
    print('[*] Run Basic Configs ... ')
    # configs
    batch_size = config.TRAIN.batch_size
    train_date = config.TRAIN.train_date
    weight_unet = config.TRAIN.weight_unet
    is_mini_dataset = config.TRAIN.is_mini_dataset
    size_mini_testset = config.TRAIN.size_mini_testset

    # setup log
    log_dir = os.path.join("log_test_{}_{}_{}"
                           .format(model_name, mask_name, mask_perc),
                           train_date,
                           weight_unet)
    isExists = os.path.exists(log_dir)
    if not isExists:
        os.makedirs(log_dir)
    log_test, log_test_filename = logging_test_setup(log_dir)

    # setup checkpoint
    checkpoint_dir = os.path.join("checkpoint_{}_{}_{}"
                                  .format(model_name, mask_name, mask_perc),
                                  train_date)
    isExists = os.path.exists(checkpoint_dir)
    if not isExists:
        os.makedirs(checkpoint_dir)

    # setup save dir
    save_dir = os.path.join("sample_test_{}_{}_{}".
                            format(model_name, mask_name, mask_perc),
                            train_date,
                            weight_unet)
    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)

    print('[*] Loading data ... ')
    # data path
    testing_data_path = config.TRAIN.testing_data_path

    # load data (augment)
    with open(testing_data_path, 'rb') as f:
        X_test = torch.from_numpy(load(f))
        if is_mini_dataset:
            X_test = X_test[0:size_mini_testset]

    log = 'X_test shape:{}/ min:{}/ max:{}'.format(X_test.shape, X_test.min(), X_test.max())
    # print(log)
    log_test.info(log)

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

    print('[*] Loading Network ... ')
    # data loader
    dataloader_test = torch.utils.data.DataLoader(X_test, batch_size=batch_size, pin_memory=True, timeout=0,
                                                  shuffle=True)
    # load unet
    generator = UNet()
    generator = generator.to(device)
    generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, weight_unet)))

    # loss function
    mse = nn.MSELoss(reduction='mean').to(device)

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

            print("step={:3}".format(step))

            # good-->bad
            X_bad = torch.from_numpy(to_bad_img(X_good.numpy(), mask))

            # cpu-->gpu
            X_good = X_good.to(device)
            X_bad = X_bad.to(device)

            # (N, H, W, C)-->(N, C, H, W)
            X_good = X_good.permute(0, 3, 1, 2)
            X_bad = X_bad.permute(0, 3, 1, 2)

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
            nmsn_res = torch.div(nmse_a, nmse_b).numpy()
            ssim_res = ssim(X_generated_0_1, X_good_0_1)
            psnr_res = psnr(X_generated_0_1, X_good_0_1)

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
        log_test.debug(log)

        # save image
        for i in range(len(X_good_test_sample)):
            torchvision.utils.save_image(X_good_test_sample[i],
                                         os.path.join(save_dir,
                                                      'GroundTruth_{}.png'.format(i)))
            torchvision.utils.save_image(X_generated_test_sample[i],
                                         os.path.join(save_dir,
                                                      'Generated_{}.png'.format(i)))
            torchvision.utils.save_image(X_bad_test_sample[i],
                                         os.path.join(save_dir,
                                                      'Bad_{}.png'.format(i)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')
    parser.add_argument('--mask', type=str, default='gaussian2d', help='gaussian1d, gaussian2d, poisson2d')
    parser.add_argument('--maskperc', type=int, default='30', help='10,20,30,40,50')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main_test(device, args.model, args.mask, args.maskperc)
