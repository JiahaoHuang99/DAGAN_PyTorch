import torchvision
from scipy.io import loadmat
from skimage.io import imread

from config import config
from model import *
from utils import *

from cv2 import resize

def main_test(device, model_name, mask_name, mask_perc, input_image_path, train_date, weight_unet):

    print('[*] Run Basic Configs ... ')
    # input size
    h_size = 200
    w_size = 300

    # setup checkpoint dir
    checkpoint_dir = os.path.join("checkpoint_{}_{}_{}"
                                  .format(model_name, mask_name, mask_perc),
                                  train_date)
    isExists = os.path.exists(checkpoint_dir)
    if not isExists:
        os.makedirs(checkpoint_dir)

    # setup save dir
    save_dir = os.path.join("sample", "sample_test_{}_{}_{}".
                            format(model_name, mask_name, mask_perc),
                            train_date,
                            weight_unet)
    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)

    print('[*] Loading data ... ')

    input_image = imread(input_image_path, as_gray=True)

    input_image = resize(input_image, (h_size, w_size))

    input_image = input_image * 2 - 1

    input_image = input_image[np.newaxis, :, :, np.newaxis]
    X_good = (torch.from_numpy(input_image)).type(torch.float32)

    log = 'X_good shape:{}/ min:{}/ max:{}'.format(X_good.shape, X_good.min(), X_good.max())
    print(log)

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

    # pre-processing for vgg
    preprocessing = PREPROCESS()

    # load unet
    generator = UNet().eval()
    generator = generator.to(device)
    generator.load_state_dict(torch.load(os.path.join(checkpoint_dir, weight_unet)))

    print('[*] Testing  ... ')

    with torch.no_grad():
        # testing
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
        X_diff_0_1_x10 = torch.mul(torch.abs(torch.sub(X_good_0_1, X_generated_0_1)), 10)

        # save image
        torchvision.utils.save_image(X_good_0_1,
                                     os.path.join(save_dir,
                                                  'GroundTruth_sample.png'))
        torchvision.utils.save_image(X_bad_0_1,
                                     os.path.join(save_dir,
                                                  'Bad_sample.png'))
        torchvision.utils.save_image(X_generated_0_1,
                                     os.path.join(save_dir,
                                                  'Generated_sample.png'))
        torchvision.utils.save_image(X_diff_0_1_x10,
                                     os.path.join(save_dir,
                                                  'Diff_sample.png'))
    print('finish!')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')
    parser.add_argument('--mask', type=str, default='gaussian2d', help='gaussian1d, gaussian2d, poisson2d')
    parser.add_argument('--maskperc', type=int, default='30', help='10,20,30,40,50')

    parser.add_argument('--input_image', type=str, default=os.path.join('sample','sample1.png'))
    parser.add_argument('--train_date', type=str, default='2020_11_17_22_19_14')
    parser.add_argument('--weight_unet', type=str,
                        default='best_checkpoint_generator_unet_gaussian2d_30_epoch_30_nmse_0.000142498295917324.pt')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main_test(device, args.model, args.mask, args.maskperc, args.input_image, args.train_date, args.weight_unet)
