from pickle import load
from time import localtime, strftime, time

import torch.optim as optim
import torchvision
from scipy.io import loadmat
from tensorboardX import SummaryWriter
from torch.utils import data

from config import config, log_config
from model import *
from pytorchtools import EarlyStopping
from utils import *


def main_train(device, model_name, mask_name, mask_perc):
    # current time
    current_time = strftime("%Y_%m_%d_%H_%M_%S", localtime())

    print('[*] Run Basic Configs ... ')
    # setup log
    log_dir = os.path.join("log_{}_{}_{}".format(model_name, mask_name, mask_perc), current_time)
    isExists = os.path.exists(log_dir)
    if not isExists:
        os.makedirs(log_dir)

    log_all, log_eval, log_all_filename, log_eval_filename = logging_setup(log_dir)

    # tensorbordX logger
    logger_tensorboard = SummaryWriter(os.path.join('tensorboard', log_dir))

    # setup checkpoint dir
    checkpoint_dir = os.path.join("checkpoint_{}_{}_{}"
                                  .format(model_name, mask_name, mask_perc),
                                  current_time)
    isExists = os.path.exists(checkpoint_dir)
    if not isExists:
        os.makedirs(os.path.join(checkpoint_dir))

    # setup save dir
    save_dir = os.path.join("sample_{}_{}_{}".
                            format(model_name, mask_name, mask_perc),
                            current_time)
    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)

    # configs
    batch_size = config.TRAIN.batch_size
    early_stopping_num = config.TRAIN.early_stopping_num
    save_epoch_every = config.TRAIN.save_every_epoch
    save_img_every_val_step = config.TRAIN.save_img_every_val_step
    g_alpha = config.TRAIN.g_alpha
    g_beta = config.TRAIN.g_beta
    g_gamma = config.TRAIN.g_gamma
    g_adv = config.TRAIN.g_adv
    lr = config.TRAIN.lr
    lr_decay = config.TRAIN.lr_decay
    lr_decay_every = config.TRAIN.lr_decay_every
    beta1 = config.TRAIN.beta1
    n_epoch = config.TRAIN.n_epoch
    is_mini_dataset = config.TRAIN.is_mini_dataset
    size_mini_trainset = config.TRAIN.size_mini_trainset
    size_mini_valset = config.TRAIN.size_mini_valset
    log_config(log_all_filename, config)
    log_config(log_eval_filename, config)

    print('[*] Loading Data ... ')
    # data path
    training_data_path = config.TRAIN.training_data_path
    val_data_path = config.TRAIN.val_data_path

    # load data (augment)
    data_augment = DataAugment()
    with open(training_data_path, 'rb') as f:
        X_train = torch.from_numpy(load(f))
        if is_mini_dataset:
            X_train = X_train[0:size_mini_trainset]
        X_train = data_augment(X_train)
    with open(val_data_path, 'rb') as f:
        X_val = torch.from_numpy(load(f))
        if is_mini_dataset:
            X_val = X_val[0:size_mini_valset]

    log = 'X_train shape:{}/ min:{}/ max:{}\n'.format(X_train.shape, X_train.min(), X_train.max()) \
          + 'X_val shape:{}/ min:{}/ max:{}'.format(X_val.shape, X_val.min(), X_val.max())
    # print(log)
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
    vgg_pre = VGG_PRE()

    # load vgg
    vgg16_cnn = VGG_CNN()
    vgg16_cnn = vgg16_cnn.to(device)
    # load unet
    generator = UNet()
    generator = generator.to(device)
    # load discriminator
    discriminator = Discriminator()
    discriminator = discriminator.to(device)

    # loss function
    bce = nn.BCELoss(reduction='mean').to(device)
    mse = nn.MSELoss(reduction='mean').to(device)

    # real and fake label
    real = 1.
    fake = 0.

    # optimizer
    g_optim = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optim.lr_scheduler.StepLR(g_optim, lr_decay_every, gamma=lr_decay, last_epoch=-1)

    d_optim = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optim.lr_scheduler.StepLR(d_optim, lr_decay_every, gamma=lr_decay, last_epoch=-1)

    print('[*] Training  ... ')
    # initialize global step
    global GLOBAL_STEP
    GLOBAL_STEP = 0

    for epoch in range(0, n_epoch):
        # initialize training
        total_nmse_training = 0
        total_ssim_training = 0
        total_psnr_training = 0
        num_training_temp = 0

        # training
        for step, X_good in enumerate(dataloader):

            # starting time for step
            step_time = time()

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

            # discriminator
            logits_fake = discriminator(X_generated)
            logits_real = discriminator(X_good)

            # vgg
            X_good_244 = vgg_pre(X_good)
            net_vgg_conv4_good = vgg16_cnn(X_good_244)
            X_generated_244 = vgg_pre(X_generated)
            net_vgg_conv4_gen = vgg16_cnn(X_generated_244)

            # discriminator loss
            d_loss_real = bce(logits_real, torch.full((logits_real.size()), real).to(device))
            d_loss_fake = bce(logits_fake, torch.full((logits_fake.size()), fake).to(device))

            d_loss = d_loss_real + d_loss_fake

            # generator loss (adversarial)
            g_adversarial = bce(logits_fake, torch.full((logits_fake.size()), real).to(device))

            # generator loss (perceptual)
            g_perceptual = mse(net_vgg_conv4_good, net_vgg_conv4_gen)

            # generator loss (pixel-wise)
            g_nmse_a = mse(X_generated, X_good)
            g_nmse_b = mse(X_generated, torch.zeros_like(X_generated).to(device))
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

            with torch.no_grad():
                # record train loss
                logger_tensorboard.add_scalar('TRAIN Generator LOSS/G_LOSS', g_loss.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('TRAIN Generator LOSS/g_adversarial', g_adversarial.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('TRAIN Generator LOSS/g_perceptual', g_perceptual.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('TRAIN Generator LOSS/g_nmse', g_nmse.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('TRAIN Generator LOSS/g_fft', g_fft.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('TRAIN Discriminator LOSS/D_LOSS', d_loss.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('TRAIN Discriminator LOSS/d_loss_real', d_loss_real.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('TRAIN Discriminator LOSS/d_loss_fake', d_loss_fake.item(),
                                              global_step=GLOBAL_STEP)

                log = "Epoch[{:3}/{:3}] step={:3} d_loss={:5} g_loss={:5} g_adversarial={:5} g_perceptual_loss={:5} g_mse={:5} g_freq={:5} took {:3}s".format(
                    epoch + 1, n_epoch, step,
                    round(float(d_loss), 3),
                    round(float(g_loss), 3),
                    round(float(g_adversarial), 3),
                    round(float(g_perceptual), 3),
                    round(float(g_nmse), 3),
                    round(float(g_fft), 3),
                    round(time() - step_time, 2))
                # print(log)
                log_all.debug(log)

                # gpu --> cpu
                X_good = X_good.cpu()
                X_generated = X_generated.cpu()
                X_bad = X_bad.cpu()

                # (-1,1)-->(0,1)
                X_good_0_1 = torch.div(torch.add(X_good, torch.ones_like(X_good)), 2)
                X_generated_0_1 = torch.div(torch.add(X_generated, torch.ones_like(X_generated)), 2)
                X_bad_0_1 = torch.div(torch.add(X_bad, torch.ones_like(X_bad)), 2)

                # eval for training
                nmse_a = mse(X_generated_0_1, X_good_0_1)
                nmse_b = mse(X_generated_0_1, torch.zeros_like(X_generated_0_1))
                nmse_res = torch.div(nmse_a, nmse_b).numpy()
                ssim_res = ssim(X_generated_0_1, X_good_0_1)
                psnr_res = psnr(X_generated_0_1, X_good_0_1)

                total_nmse_training = total_nmse_training + np.sum(nmse_res)
                total_ssim_training = total_ssim_training + np.sum(ssim_res)
                total_psnr_training = total_psnr_training + np.sum(psnr_res)

                num_training_temp = num_training_temp + batch_size
                GLOBAL_STEP = GLOBAL_STEP + 1

        total_nmse_training = total_nmse_training / num_training_temp
        total_ssim_training = total_ssim_training / num_training_temp
        total_psnr_training = total_psnr_training / num_training_temp

        # record training eval
        logger_tensorboard.add_scalar('Training/NMSE', total_nmse_training, global_step=epoch)
        logger_tensorboard.add_scalar('Training/SSIM', total_ssim_training, global_step=epoch)
        logger_tensorboard.add_scalar('Training/PSNR', total_psnr_training, global_step=epoch)

        log = "Epoch: {}  NMSE training: {:8}, SSIM training: {:8}, PSNR training: {:8}".format(
            epoch + 1, total_nmse_training, total_ssim_training, total_psnr_training)
        # print(log)
        log_all.debug(log)
        log_eval.info(log)

        # initialize validation
        total_nmse_val = 0
        total_ssim_val = 0
        total_psnr_val = 0
        num_val_temp = 0

        X_good_val_sample = []
        X_bad_val_sample = []
        X_generated_val_sample = []

        with torch.no_grad():
            # validation
            for step_val, X_good in enumerate(dataloader_val):

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

                # discriminator
                logits_fake = discriminator(X_generated)
                logits_real = discriminator(X_good)

                # vgg
                X_good_244 = vgg_pre(X_good)
                net_vgg_conv4_good = vgg16_cnn(X_good_244)
                X_generated_244 = vgg_pre(X_generated)
                net_vgg_conv4_gen = vgg16_cnn(X_generated_244)

                # discriminator loss
                d_loss_real = bce(logits_real, torch.full((logits_real.size()), real).to(device))
                d_loss_fake = bce(logits_fake, torch.full((logits_fake.size()), fake).to(device))

                d_loss = d_loss_real + d_loss_fake

                # generator loss (adversarial)
                g_adversarial = bce(logits_fake, torch.full((logits_fake.size()), real).to(device))

                # generator loss (perceptual)
                g_perceptual = mse(net_vgg_conv4_good, net_vgg_conv4_gen)

                # generator loss (pixel-wise)
                g_nmse_a = mse(X_generated, X_good)
                g_nmse_b = mse(X_generated, torch.zeros_like(X_generated).to(device))
                g_nmse = torch.div(g_nmse_a, g_nmse_b)

                # generator loss (frequency)
                g_fft = mse(fft_abs_for_map_fn(X_generated), fft_abs_for_map_fn(X_good))

                # generator loss (total)
                g_loss = g_adv * g_adversarial + g_alpha * g_nmse + g_gamma * g_perceptual + g_beta * g_fft

                # record validation loss
                logger_tensorboard.add_scalar('VALIDATION Generator LOSS/G_LOSS', g_loss.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Generator LOSS/g_adversarial', g_adversarial.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Generator LOSS/g_perceptual', g_perceptual.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Generator LOSS/g_nmse', g_nmse.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Generator LOSS/g_fft', g_fft.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Discriminator LOSS/D_LOSS', d_loss.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Discriminator LOSS/d_loss_real', d_loss_real.item(),
                                              global_step=GLOBAL_STEP)
                logger_tensorboard.add_scalar('VALIDATION Discriminator LOSS/d_loss_fake', d_loss_fake.item(),
                                              global_step=GLOBAL_STEP)

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
                nmse_res = torch.div(nmse_a, nmse_b).numpy()
                ssim_res = ssim(X_generated_0_1, X_good_0_1)
                psnr_res = psnr(X_generated_0_1, X_good_0_1)

                total_nmse_val = total_nmse_val + np.sum(nmse_res)
                total_ssim_val = total_ssim_val + np.sum(ssim_res)
                total_psnr_val = total_psnr_val + np.sum(psnr_res)

                num_val_temp = num_val_temp + batch_size

                # output the sample
                if step_val % save_img_every_val_step == 0:
                    X_good_val_sample.append(X_good_0_1[0, :, :, :])
                    X_generated_val_sample.append(X_generated_0_1[0, :, :, :])
                    X_bad_val_sample.append(X_bad_0_1[0, :, :, :])

            total_nmse_val = total_nmse_val / num_val_temp
            total_ssim_val = total_ssim_val / num_val_temp
            total_psnr_val = total_psnr_val / num_val_temp

            # record validation eval
            logger_tensorboard.add_scalar('Validation/NMSE', total_nmse_val, global_step=epoch)
            logger_tensorboard.add_scalar('Validation/SSIM', total_ssim_val, global_step=epoch)
            logger_tensorboard.add_scalar('Validation/PSNR', total_psnr_val, global_step=epoch)

            log = "Epoch: {}  NMSE val: {:8}, SSIM val: {:8}, PSNR val: {:8}".format(
                epoch + 1, total_nmse_val, total_ssim_val, total_psnr_val)
            # print(log)
            log_all.debug(log)
            log_eval.info(log)

            # saving checkpoint
            if (epoch + 1) % save_epoch_every == 0:
                torch.save(generator.state_dict(),
                           os.path.join(checkpoint_dir,
                                        "checkpoint_generator_{}_{}_{}_epoch_{}_nmse_{}.pt"
                                        .format(model_name, mask_name, mask_perc, (epoch + 1), total_nmse_val)))
                torch.save(discriminator.state_dict(),
                           os.path.join(checkpoint_dir,
                                        "checkpoint_discriminator_{}_{}_{}_epoch_{}_nmse_{}.pt"
                                        .format(model_name, mask_name, mask_perc, (epoch + 1), total_nmse_val)))

            # save image
            for i in range(len(X_good_val_sample)):
                torchvision.utils.save_image(X_good_val_sample[i],
                                             os.path.join(save_dir,
                                                          'Epoch_{}_GroundTruth_{}.png'.format(epoch, i)))
                torchvision.utils.save_image(X_generated_val_sample[i],
                                             os.path.join(save_dir,
                                                          'Epoch_{}_Generated_{}.png'.format(epoch, i)))
                torchvision.utils.save_image(X_bad_val_sample[i],
                                             os.path.join(save_dir,
                                                          'Epoch_{}_Bad_{}.png'.format(epoch, i)))

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main_train(device, args.model, args.mask, args.maskperc)
