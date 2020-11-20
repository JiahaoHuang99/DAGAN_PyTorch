import json
import os

from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.TEST = edict()
config.VAL = edict()

# train & val
config.TRAIN.image_size = 256
config.TRAIN.batch_size = 16
config.TRAIN.early_stopping_num = 30
config.TRAIN.save_every_epoch = 2
config.TRAIN.save_img_every_val_step = 30
config.TRAIN.lr = 0.0001  # init learning rate
config.TRAIN.lr_decay = 0.5  # learning rate decay rate
config.TRAIN.lr_decay_every = 5  # decay every epoch
config.TRAIN.beta1 = 0.5  # beta1 in Adam optimiser
config.TRAIN.n_epoch = 9999  # total epoch

config.TRAIN.is_mini_dataset = True  # for debug
config.TRAIN.size_mini_trainset = 40
config.TRAIN.size_mini_valset = 40

config.TRAIN.g_alpha = 15  # weight for pixel loss
config.TRAIN.g_gamma = 0.0025  # weight for perceptual loss
config.TRAIN.g_beta = 0.1  # weight for frequency loss
config.TRAIN.g_adv = 1  # weight for adv loss

config.TRAIN.VGG16_path = os.path.join('trained_model', 'VGG16', 'vgg16_weights.npz')
config.TRAIN.training_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'training.pickle')
config.TRAIN.val_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'validation.pickle')
config.TRAIN.mask_Gaussian1D_path = os.path.join('mask', 'Gaussian1D')
config.TRAIN.mask_Gaussian2D_path = os.path.join('mask', 'Gaussian2D')
config.TRAIN.mask_Poisson2D_path = os.path.join('mask', 'Poisson2D')

# test
config.TEST.image_size = 256
config.TEST.batch_size = 7

config.TEST.is_mini_dataset = True  # for debug
config.TEST.size_mini_dataset = 300

config.TEST.train_date = '2020_11_18_16_07_46'
config.TEST.weight_unet = 'best_checkpoint_generator_unet_gaussian2d_30_epoch_1_nmse_0.1069742100579398.pt'

config.TEST.VGG16_path = os.path.join('trained_model', 'VGG16', 'vgg16_weights.npz')
config.TEST.testing_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'testing.pickle')
config.TEST.mask_Gaussian1D_path = os.path.join('mask', 'Gaussian1D')
config.TEST.mask_Gaussian2D_path = os.path.join('mask', 'Gaussian2D')
config.TEST.mask_Poisson2D_path = os.path.join('mask', 'Poisson2D')


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
