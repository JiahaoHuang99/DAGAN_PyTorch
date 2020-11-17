import json
import os

from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()

config.TRAIN.image_size = 256
config.TRAIN.batch_size = 16
config.TRAIN.early_stopping_num = 30
config.TRAIN.save_every_epoch = 2
config.TRAIN.lr = 0.0001
config.TRAIN.lr_decay = 0.5
config.TRAIN.decay_every = 5
config.TRAIN.beta1 = 0.5  # beta1 in Adam optimiser
config.TRAIN.n_epoch = 9999
config.TRAIN.is_mini_dataset = False
config.TRAIN.size_mini_trainset = 40
config.TRAIN.size_mini_valset = 10
config.TRAIN.size_mini_testset = 300
config.TRAIN.sample_size = 50
config.TRAIN.g_alpha = 15  # weight for pixel loss
config.TRAIN.g_gamma = 0.0025  # weight for perceptual loss
config.TRAIN.g_beta = 0.1  # weight for frequency loss
config.TRAIN.g_adv = 1  # weight for adv loss

config.TRAIN.seed = 100
config.TRAIN.epsilon = 0.000001

config.TRAIN.VGG16_path = os.path.join('trained_model', 'VGG16', 'vgg16_weights.npz')
config.TRAIN.training_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'training.pickle')
config.TRAIN.val_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'validation.pickle')
config.TRAIN.testing_data_path = os.path.join('data', 'MICCAI13_SegChallenge', 'testing.pickle')
config.TRAIN.mask_Gaussian1D_path = os.path.join('mask', 'Gaussian1D')
config.TRAIN.mask_Gaussian2D_path = os.path.join('mask', 'Gaussian2D')
config.TRAIN.mask_Poisson2D_path = os.path.join('mask', 'Poisson2D')
config.TRAIN.train_date = '2020_11_17_01_00_44'
config.TRAIN.weight_unet = 'checkpoint_generator_unet_gaussian2d_30_epoch_26_nmse_0.002987398299578411.pt'


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
