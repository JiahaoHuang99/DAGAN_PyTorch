from pickle import load

import torchvision
from scipy.io import loadmat

from config import config
from model import *
from utils import *


def main_test(device, model_name, mask_name, mask_perc):
    print('[*] Run Basic Configs ... ')
    # configs
    is_mini_dataset = config.TRAIN.is_mini_dataset
    size_mini_testset = config.TRAIN.size_mini_testset

    print('[*] Loading data ... ')
    # data path
    testing_data_path = config.TRAIN.testing_data_path

    isExists = os.path.exists('test_img')
    if not isExists:
        os.makedirs('test_img')

    # load data (augment)
    with open(testing_data_path, 'rb') as f:
        X_good = load(f)
        if is_mini_dataset:
            X_good = X_good[50:size_mini_testset + 50, :, :, :]

    mse = nn.MSELoss(reduction='mean').to(device)

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

    X_bad = to_bad_img(X_good, mask)

    X_good = torch.from_numpy(X_good)
    X_good = X_good.to(device)
    X_good = X_good.permute(0, 3, 1, 2)

    X_bad = torch.from_numpy(X_bad)
    X_bad = X_bad.to(device)
    X_bad = X_bad.permute(0, 3, 1, 2)

    X_good_0_1 = torch.div(torch.add(X_good, torch.ones_like(X_good)), 2)
    X_bad_0_1 = torch.div(torch.add(X_bad, torch.ones_like(X_bad)), 2)

    X_diff_0_1 = torch.sub(X_bad_0_1, X_good_0_1)
    # save image
    for i in range(len(X_good_0_1)):
        torchvision.utils.save_image(X_good_0_1[i], os.path.join('test_img', 'GroundTruth_{}.png'.format(i)))
        torchvision.utils.save_image(X_bad_0_1[i], os.path.join('test_img', 'Bad_{}.png'.format(i)))
        torchvision.utils.save_image(X_diff_0_1[i], os.path.join('test_img', 'Diff_{}.png'.format(i)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')
    parser.add_argument('--mask', type=str, default='gaussian2d', help='gaussian1d, gaussian2d, poisson2d')
    parser.add_argument('--maskperc', type=int, default='30', help='10,20,30,40,50')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main_test(device, args.model, args.mask, args.maskperc)
