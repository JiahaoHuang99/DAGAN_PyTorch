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


def main_train(device, model_name, mask_name, mask_perc):
    # =================================== BASIC CONFIGS =================================== #

    print('[*] Run Basic Configs ... ')

    # setup log
    log_dir = "log_{}_{}_{}".format(model_name, mask_name, mask_perc)
    isExists = os.path.exists(log_dir)
    if not isExists:
        os.makedirs(log_dir)
    log_all, log_eval, log_all_filename, log_eval_filename, = logging_setup(log_dir)

    # setup checkpoint
    checkpoint_dir = "checkpoint_{}_{}_{}".format(model_name, mask_name, mask_perc)
    isExists = os.path.exists(checkpoint_dir)
    if not isExists:
        os.makedirs(checkpoint_dir)

    # read parameters
    image_size = 256
    batch_size = config.TRAIN.batch_size
    early_stopping_num = config.TRAIN.early_stopping_num
    save_epoch_every = config.TRAIN.save_every_epoch
    g_alpha = config.TRAIN.g_alpha
    g_beta = config.TRAIN.g_beta
    g_gamma = config.TRAIN.g_gamma
    g_adv = config.TRAIN.g_adv
    lr = config.TRAIN.lr
    lr_decay = config.TRAIN.lr_decay
    lr_decay_every = config.TRAIN.decay_every
    beta1 = config.TRAIN.beta1
    n_epoch = config.TRAIN.n_epoch
    log_config(log_all_filename, config)
    log_config(log_eval_filename, config)

    # data path
    print('[*] Loading Data ... ')
    training_data_path = config.TRAIN.training_data_path
    val_data_path = config.TRAIN.val_data_path

    # data augment
    data_augment = DataAugment()
    with open(training_data_path, 'rb') as f:
        X_train = torch.from_numpy(load(f))
        X_train = data_augment(X_train)
    with open(val_data_path, 'rb') as f:
        X_val = torch.from_numpy(load(f))
        X_val = data_augment(X_val)

    log = 'X_train shape:{}/ min:{}/ max:{}\n'.format(X_train.shape, X_train.min(), X_train.max()) \
          + 'X_val shape:{}/ min:{}/ max:{}'.format(X_val.shape, X_val.min(), X_val.max())
    print(log)
    log_all.debug(log)
    log_eval.info(log)

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
    dataloader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, pin_memory=True, timeout=0,
                                             shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(X_val, batch_size=batch_size, pin_memory=True, timeout=0,
                                                 shuffle=True)
    # early stopping
    early_stopping = EarlyStopping(early_stopping_num,
                                   model_name=model_name, mask_name=mask_name, mask_perc=mask_perc,

                                   verbose=True, checkpoint_path=checkpoint_dir, log_path=log_dir,
                                   log_all=log_all, log_eval=log_eval)
    # pre-processing for vgg
    vgg_pre = VGG_PRE(image_size)
    # load vgg
    vgg16_cnn = VGG_CNN()
    # load unet
    generator = UNet()
    # load discriminator
    discriminator = Discriminator()

    # loss function
    bce = nn.BCELoss(reduction='mean')
    mse = nn.MSELoss(reduction='mean')

    # real and fake label
    real = 1.
    fake = 0.

    # optimizer
    g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optim.lr_scheduler.StepLR(g_optim, lr_decay_every, gamma=lr_decay, last_epoch=-1)

    d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optim.lr_scheduler.StepLR(d_optim, lr_decay_every, gamma=lr_decay, last_epoch=-1)

    # loss
    train_losses = {'g_loss': [], 'g_adversarial': [], 'g_perceptual': [], 'g_nmse': [], 'g_fft': [],
                    'd_loss': [], 'd_loss_real': [], 'd_loss_fake': []}

    valid_losses = {'g_loss': [], 'g_adversarial': [], 'g_perceptual': [], 'g_nmse': [], 'g_fft': [],
                    'd_loss': [], 'd_loss_real': [], 'd_loss_fake': []}

    print('[*] Training  ... ')
    for epoch in range(0, n_epoch):
        # initialize training
        total_nmse_training = 0
        total_ssim_training = 0
        total_psnr_training = 0
        num_training_temp = 0

        # training
        for step, X_good in enumerate(dataloader):
            # starting time for step
            step_time = time.time()

            # (N, H, W, C)-->(N, C, H, W)
            X_good = X_good.permute(0, 3, 1, 2)
            X_bad = to_bad_img(X_good, mask)

            # generator
            if model_name == 'unet':
                X_generated = generator(X_bad, is_train=True, is_refine=False)
            elif model_name == 'unet_refine':
                X_generated = generator(X_bad, is_train=True, is_refine=True)
            else:
                raise Exception("unknown model")

            # discriminator
            _, logits_fake = discriminator(X_generated, is_train=True)
            _, logits_real = discriminator(X_good, is_train=True)

            # vgg
            X_good_244 = vgg_pre(X_good)
            net_vgg_conv4_good = vgg16_cnn(X_good_244)
            X_generated_244 = vgg_pre(X_generated)
            net_vgg_conv4_gen = vgg16_cnn(X_generated_244)

            # discriminator loss
            d_loss_real = bce(logits_real, torch.full((logits_real.size()), real))
            d_loss_fake = bce(logits_fake, torch.full((logits_fake.size()), fake))

            d_loss = d_loss_real + d_loss_fake

            # generator loss (adversarial)
            g_adversarial = bce(logits_fake, torch.full((logits_fake.size()), real))

            # generator loss (perceptual)
            g_perceptual = mse(net_vgg_conv4_good, net_vgg_conv4_gen)

            # generator loss (pixel-wise)
            g_nmse_a = mse(X_generated, X_good)
            g_nmse_b = mse(X_generated, torch.zeros_like(X_generated))
            g_nmse = torch.div(g_nmse_a, g_nmse_b)

            # generator loss (frequency)
            g_fft = mse(fft_abs_for_map_fn(X_generated), fft_abs_for_map_fn(X_good))

            # generator loss (total)
            g_loss = g_adv * g_adversarial + g_alpha * g_nmse + g_gamma * g_perceptual + g_beta * g_fft

            # clear gradient (discriminator)
            d_optim.zero_grad()
            # back propagation (discriminator)
            d_loss.backward(retain_graph=True)

            # clear gradient (generator)
            g_optim.zero_grad()
            # back propagation (generator)
            g_loss.backward()

            # update weight
            d_optim.step()
            g_optim.step()

            # record train loss
            train_losses['g_loss'].append(g_loss.item())
            train_losses['g_adversarial'].append(g_adversarial.item())
            train_losses['g_perceptual'].append(g_perceptual.item())
            train_losses['g_nmse'].append(g_nmse.item())
            train_losses['g_fft'].append(g_fft.item())
            train_losses['d_loss'].append(d_loss.item())
            train_losses['d_loss_real'].append(d_loss_real.item())
            train_losses['d_loss_fake'].append(d_loss_fake.item())

            log = "Epoch[{:3}/{:3}] step={:3} d_loss={:5} g_loss={:5} g_adversarial={:5} g_perceptual_loss={:5} g_mse={:5} g_freq={:5} took {:3}s".format(
                epoch + 1, n_epoch, step,
                round(float(d_loss), 3),
                round(float(g_loss), 3),
                round(float(g_adversarial), 3),
                round(float(g_perceptual), 3),
                round(float(g_nmse), 3),
                round(float(g_fft), 3),
                round(time.time() - step_time, 2))
            print(log)
            log_all.debug(log)

            # eval for training
            nmsn_res = g_loss.detach().numpy()
            ssim_res = ssim(X_good, X_bad)
            psnr_res = psnr(X_good, X_bad)

            total_nmse_training = total_nmse_training + np.sum(nmsn_res)
            total_ssim_training = total_ssim_training + np.sum(ssim_res)
            total_psnr_training = total_psnr_training + np.sum(psnr_res)

            num_training_temp = num_training_temp + batch_size

        total_nmse_training = total_nmse_training / num_training_temp
        total_ssim_training = total_ssim_training / num_training_temp
        total_psnr_training = total_psnr_training / num_training_temp

        log = "Epoch: {}  NMSE training: {:8}, SSIM training: {:8}, PSNR training: {:8}".format(
            epoch + 1, total_nmse_training, total_ssim_training, total_psnr_training)
        print(log)
        log_all.debug(log)
        log_eval.info(log)

        # initialize training
        total_nmse_val = 0
        total_ssim_val = 0
        total_psnr_val = 0
        num_val_temp = 0

        # validation
        for step_val, X_good_val in enumerate(dataloader_val):
            # (N, H, W, C)-->(N, C, H, W)
            X_good_val = X_good_val.permute(0, 3, 1, 2)
            X_bad_val = to_bad_img(X_good_val, mask)

            # generator
            if model_name == 'unet':
                X_generated_val = generator(X_bad_val, is_train=False, is_refine=False)
            elif model_name == 'unet_refine':
                X_generated_val = generator(X_bad_val, is_train=False, is_refine=True)
            else:
                raise Exception("unknown model")

            # discriminator
            _, logits_fake = discriminator(X_generated_val, is_train=False)
            _, logits_real = discriminator(X_good_val, is_train=False)

            # vgg
            X_good_244_val = vgg_pre(X_good_val)
            net_vgg_conv4_good_val = vgg16_cnn(X_good_244_val)
            X_generated_244_val = vgg_pre(X_generated_val)
            net_vgg_conv4_gen_val = vgg16_cnn(X_generated_244_val)

            # discriminator loss
            d_loss_real = bce(logits_real, torch.full((logits_real.size()), real))
            d_loss_fake = bce(logits_fake, torch.full((logits_fake.size()), fake))

            d_loss = d_loss_real + d_loss_fake

            # generator loss (adversarial)
            g_adversarial = bce(logits_fake, torch.full((logits_fake.size()), real))

            # generator loss (perceptual)
            g_perceptual = mse(net_vgg_conv4_good_val, net_vgg_conv4_gen_val)

            # generator loss (pixel-wise)
            g_nmse_a = mse(X_generated_val, X_good_val)
            g_nmse_b = mse(X_generated_val, torch.zeros_like(X_generated_val))
            g_nmse = torch.div(g_nmse_a, g_nmse_b)

            # generator loss (frequency)
            g_fft = mse(fft_abs_for_map_fn(X_generated_val), fft_abs_for_map_fn(X_good_val))

            # generator loss (total)
            g_loss = g_adv * g_adversarial + g_alpha * g_nmse + g_gamma * g_perceptual + g_beta * g_fft

            # record validation loss
            valid_losses['g_loss'].append(g_loss.item())
            valid_losses['g_adversarial'].append(g_adversarial.item())
            valid_losses['g_perceptual'].append(g_perceptual.item())
            valid_losses['g_nmse'].append(g_nmse.item())
            valid_losses['g_fft'].append(g_fft.item())
            valid_losses['d_loss'].append(d_loss.item())
            valid_losses['d_loss_real'].append(d_loss_real.item())
            valid_losses['d_loss_fake'].append(d_loss_fake.item())

            # eval for validation
            nmsn_res = g_loss.detach().numpy()
            ssim_res = ssim(X_good_val, X_bad_val)
            psnr_res = psnr(X_good_val, X_bad_val)

            total_nmse_val = total_nmse_val + np.sum(nmsn_res)
            total_ssim_val = total_ssim_val + np.sum(ssim_res)
            total_psnr_val = total_psnr_val + np.sum(psnr_res)

            num_val_temp = num_val_temp + batch_size

        total_nmse_val = total_nmse_val / num_val_temp
        total_ssim_val = total_ssim_val / num_val_temp
        total_psnr_val = total_psnr_val / num_val_temp

        log = "Epoch: {}  NMSE val: {:8}, SSIM val: {:8}, PSNR val: {:8}".format(
            epoch + 1, total_nmse_val, total_ssim_val, total_psnr_val)
        print(log)
        log_all.debug(log)
        log_eval.info(log)

        # saving checkpoint
        if (epoch + 1) % save_epoch_every == 0:
            torch.save(generator.state_dict(),
                       "./" + checkpoint_dir + "checkpoint_generator_{}_{}_{}_epoch_{}_nmse_{}.pkl"
                       .format(model_name, mask_name, mask_perc, (epoch + 1), total_nmse_val))
            torch.save(discriminator.state_dict(),
                       "./" + checkpoint_dir + "checkpoint_discriminator_{}_{}_{}_epoch_{}_nmse_{}.pkl"
                       .format(model_name, mask_name, mask_perc, (epoch + 1), total_nmse_val))

        # early stopping
        early_stopping(total_nmse_val, generator, discriminator, epoch)
        if early_stopping.early_stop:
            print("Early stopping!")
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet, unet_refine')
    parser.add_argument('--mask', type=str, default='gaussian2d', help='gaussian1d, gaussian2d, poisson2d')
    parser.add_argument('--maskperc', type=int, default='30', help='10,20,30,40,50')

    args = parser.parse_args()

    torch.cuda.is_available()

    # %%

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main_train(device, args.model, args.mask, args.maskperc)
