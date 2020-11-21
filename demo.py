from pickle import load

from torch.utils import data

from config import config
from model import *
from utils import *


def main_train(device, model_name, mask_name, mask_perc):
    is_mini_dataset = config.TRAIN.is_mini_dataset
    size_mini_valset = config.TRAIN.size_mini_valset

    print('[*] Loading Data ... ')
    val_data_path = config.TRAIN.val_data_path

    # load data (augment)
    data_augment = DataAugment()
    with open(val_data_path, 'rb') as f:
        X_val = torch.from_numpy(load(f))
        if is_mini_dataset:
            X_val = X_val[0:size_mini_valset]

    dataloader_val = torch.utils.data.DataLoader(X_val, batch_size=2, pin_memory=True, timeout=0, shuffle=True)

    # pre-processing for vgg
    vgg_pre = VGG_PRE()

    # load vgg
    vgg16_cnn = VGG_CNN()
    vgg16_cnn = vgg16_cnn
    # load unet
    generator = UNet()
    generator = generator
    # load discriminator
    discriminator = Discriminator()
    discriminator = discriminator

    print('[*] Training  ... ')
    # initialize global step
    global GLOBAL_STEP
    GLOBAL_STEP = 0

    # training
    for step_val, X_good_256 in enumerate(dataloader_val):
        t288 = transforms.Resize((288, 288))
        t384 = transforms.Resize((384, 384))

        X_good_256 = X_good_256.permute(0, 3, 1, 2)
        X_good_288 = t288(X_good_256)
        X_good_384 = t384(X_good_256)
        print(step_val)

        X_good_288 = discriminator(X_good_288)

        # # cpu-->gpu
        # X_good = X_good.to(device)
        #
        #
        # # (N, H, W, C)-->(N, C, H, W)
        # X_good = X_good.permute(0, 3, 1, 2)
        #
        # # # generator
        # # if model_name == 'unet':
        # #     X_generated = generator(X_good_256, is_refine=False)
        # # elif model_name == 'unet_refine':
        # #     X_generated = generator(X_good_256, is_refine=True)
        # # else:
        # #     raise Exception("unknown model")
        #
        # # discriminator
        # # logits_fake = discriminator(X_generated)
        # logits_real = discriminator(X_good_256)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')
    parser.add_argument('--mask', type=str, default='gaussian2d', help='gaussian1d, gaussian2d, poisson2d')
    parser.add_argument('--maskperc', type=int, default='30', help='10,20,30,40,50')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main_train(device, args.model, args.mask, args.maskperc)
