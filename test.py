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
    batch_size = config.TEST.batch_size
    diff_rate = config.TEST.diff_rate
    train_date = config.TEST.train_date
    weight_unet = config.TEST.weight_unet
    is_mini_dataset = config.TEST.is_mini_dataset
    size_mini_testset = config.TEST.size_mini_dataset

    # setup log
    log_dir = os.path.join("log_test_{}_{}_{}"
                           .format(model_name, mask_name, mask_perc),
                           train_date,
                           weight_unet)
    isExists = os.path.exists(log_dir)
    if not isExists:
        os.makedirs(log_dir)
    log_test, log_test_filename = logging_test_setup(log_dir)

    # setup checkpoint dir
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

    isExists = os.path.exists(os.path.join(save_dir, 'GroundTruth'))
    if not isExists:
        os.makedirs(os.path.join(save_dir, 'GroundTruth'))
    isExists = os.path.exists(os.path.join(save_dir, 'Generated'))
    if not isExists:
        os.makedirs(os.path.join(save_dir, 'Generated'))
    isExists = os.path.exists(os.path.join(save_dir, 'Bad'))
    if not isExists:
        os.makedirs(os.path.join(save_dir, 'Bad'))
    isExists = os.path.exists(os.path.join(save_dir, 'Diff'))
    if not isExists:
        os.makedirs(os.path.join(save_dir, 'Diff'))

    print('[*] Loading data ... ')
    # data path
    testing_data_path = config.TEST.testing_data_path

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
                os.path.join(config.TEST.mask_Gaussian2D_path, "GaussianDistribution2DMask_{}.mat".format(mask_perc)))[
                'maskRS2']
    elif mask_name == "gaussian1d":
        mask = \
            loadmat(
                os.path.join(config.TEST.mask_Gaussian1D_path, "GaussianDistribution1DMask_{}.mat".format(mask_perc)))[
                'maskRS1']
    elif mask_name == "poisson2d":
        mask = \
            loadmat(
                os.path.join(config.TEST.mask_Gaussian1D_path, "PoissonDistributionMask_{}.mat".format(mask_perc)))[
                'population_matrix']
    else:
        raise ValueError("no such mask exists: {}".format(mask_name))

    print('[*] Loading Network ... ')
    # data loader
    dataloader_test = torch.utils.data.DataLoader(X_test, batch_size=batch_size, pin_memory=True, timeout=0,
                                                  shuffle=True)

    # pre-processing for vgg
    preprocessing = PREPROCESS()

    # load unet
    generator = UNet().eval()
    generator = generator.to(device)
    generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, weight_unet)))

    # load loss function
    mse = nn.MSELoss(reduction='mean').to(device)

    print('[*] Testing  ... ')
    # initialize testing
    total_nmse_test = []
    total_ssim_test = []
    total_psnr_test = []
    num_test_temp = 0

    X_good_test_sample = []
    X_bad_test_sample = []
    X_generated_test_sample = []
    X_diff_gen_0_1_x_test_sample = []
    X_diff_bad_0_1_x_test_sample = []

    with torch.no_grad():
        # testing
        for step, X_good in enumerate(dataloader_test):

            # print("step={:3}".format(step))

            # pre-processing for unet
            X_good = preprocessing(X_good)

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
                X_generated = generator(X_bad, is_refine=False)
            elif model_name == 'unet_refine':
                X_generated = generator(X_bad, is_refine=True)
            else:
                raise Exception("unknown model")

            # gpu --> cpu
            X_good = X_good.cpu()
            X_bad = X_bad.cpu()
            X_generated = X_generated.cpu()

            # (-1,1)-->(0,1)
            X_good_0_1 = torch.div(torch.add(X_good, torch.ones_like(X_good)), 2)
            X_bad_0_1 = torch.div(torch.add(X_bad, torch.ones_like(X_bad)), 2)
            X_generated_0_1 = torch.div(torch.add(X_generated, torch.ones_like(X_generated)), 2)

            # X_diff_x10
            X_diff_gen_0_1_x = torch.mul(torch.abs(torch.sub(X_good_0_1, X_generated_0_1)), diff_rate)
            X_diff_bad_0_1_x = torch.mul(torch.abs(torch.sub(X_good_0_1, X_bad_0_1)), diff_rate)

            # eval for validation
            nmse_a = mse(X_generated_0_1, X_good_0_1)
            nmse_b = mse(X_good_0_1, torch.zeros_like(X_good_0_1))
            nmse_res = torch.div(nmse_a, nmse_b).numpy()
            ssim_res = ssim(X_generated_0_1, X_good_0_1)
            psnr_res = psnr(X_generated_0_1, X_good_0_1)

            total_nmse_test.append(nmse_res)
            total_ssim_test.append(np.mean(ssim_res))
            total_psnr_test.append(np.mean(psnr_res))

            num_test_temp = num_test_temp + batch_size

            # output the sample
            X_good_test_sample.append(X_good_0_1[0, :, :, :])
            X_bad_test_sample.append(X_bad_0_1[0, :, :, :])
            X_generated_test_sample.append(X_generated_0_1[0, :, :, :])
            X_diff_gen_0_1_x_test_sample.append(X_diff_gen_0_1_x[0, :, :, :])
            X_diff_bad_0_1_x_test_sample.append(X_diff_bad_0_1_x[0, :, :, :])

        ave_nmse_test = np.mean(total_nmse_test)
        ave_ssim_test = np.mean(total_ssim_test)
        ave_psnr_test = np.mean(total_psnr_test)

        std_nmse_test = np.std(total_nmse_test, ddof=1)
        std_ssim_test = np.std(total_ssim_test, ddof=1)
        std_psnr_test = np.std(total_psnr_test, ddof=1)

        # record testing eval
        log = "NMSE testing average: {:8}\nSSIM testing average: {:8}\nPSNR testing average: {:8}\n".format(
            ave_nmse_test, ave_ssim_test, ave_psnr_test)
        print(log)
        log_test.info(log)

        log = "NMSE testing std: {:8}\nSSIM testing std: {:8}\nPSNR testing std: {:8}\n".format(
            std_nmse_test, std_ssim_test, std_psnr_test)
        print(log)
        log_test.info(log)

        # save image
        for i in range(len(X_good_test_sample)):
            torchvision.utils.save_image(X_good_test_sample[i],
                                         os.path.join(save_dir, 'GroundTruth',
                                                      'GroundTruth_{}.png'.format(i)))
            torchvision.utils.save_image(X_bad_test_sample[i],
                                         os.path.join(save_dir, 'Bad',
                                                      'Bad_{}.png'.format(i)))
            torchvision.utils.save_image(X_generated_test_sample[i],
                                         os.path.join(save_dir, 'Generated',
                                                      'Generated_{}.png'.format(i)))
            torchvision.utils.save_image(X_diff_gen_0_1_x_test_sample[i],
                                         os.path.join(save_dir, 'Diff',
                                                      'Diff_gen_{}.png'.format(i)))
            torchvision.utils.save_image(X_diff_bad_0_1_x_test_sample[i],
                                         os.path.join(save_dir, 'Diff',
                                                      'Diff_bad_{}.png'.format(i)))

        # FID
        log = os.popen("python -m pytorch_fid {} {} ".format(
            os.path.join('.', save_dir, 'Generated'),
            os.path.join('.', save_dir, 'GroundTruth'))).read()
        print(log)
        log_test.info(log)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')
    parser.add_argument('--mask', type=str, default='gaussian2d', help='gaussian1d, gaussian2d, poisson2d')
    parser.add_argument('--maskperc', type=int, default='30', help='10,20,30,40,50')
    parser.add_argument('--gpu', type=str, default='0', help='0, 1, 2, 3')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main_test(device, args.model, args.mask, args.maskperc)
