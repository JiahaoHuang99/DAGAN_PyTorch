# DAGAN_PyTorch

This is a re-implementation code in PyTorch by Jiahao Huang for [DAGAN: Deep De-Aliasing Generative Adversarial Networks for Fast Compressed Sensing MRI Reconstruction](https://ieeexplore.ieee.org/document/8233175/) published in IEEE Transactions on Medical Imaging (2018).  
[Guang Yang](https://www.imperial.ac.uk/people/g.yang)\, [Simiao Yu](https://nebulav.github.io/)\, et al.  
(* equal contributions) 

Official code : [DAGAN](https://github.com/tensorlayer/DAGAN).

If you use this code for your research, please cite our paper.

```
@article{yang2018_dagan,
	author = {Yang, Guang and Yu, Simiao and Dong, Hao and Slabaugh, Gregory G. and Dragotti, Pier Luigi and Ye, Xujiong and Liu, Fangde and Arridge, Simon R. and Keegan, Jennifer and Guo, Yike and Firmin, David N.},
	journal = {IEEE Trans. Med. Imaging},
	number = 6,
	pages = {1310--1321},
	title = {{DAGAN: deep de-aliasing generative adversarial networks for fast compressed sensing MRI reconstruction}},
	volume = 37,
	year = 2018
}
```

If you have any questions about this code, please feel free to contact Jiahao Huang (huangjiahao0711@gmail.com).


# Prerequisites

The original code is in python 3.6 under the following dependencies:
1. torch (v1.7.0+cu101)
2. torchvision (v0.8.0)
2. tensorlayer (v1.7.2)
3. easydict (v1.9)
4. nibabel (v2.1.0)
5. scikit-image (v0.17.2)
6. tensorboard (v2.2.2)
7. tensorboardX (v2.1)
8. crc32c (v2.2)
9. soundfile (v0.10.3.post1)

Thr
Code tested in Ubuntu 16.04 with Nvidia GPU + CUDA(10.1) CuDNN (v7.6.0.64)

# How to use

1. Prepare data

    1) PUT 
    testing.pickle
    training.pickle
    validation.pickle
    into data/MICCAI13_SegChallenge/
2. Download pretrained VGG16 model

    1) run python setup_vgg.py
    
3. Train model
    1) run 'CUDA_VISIBLE_DEVICES=0 python train.py --model MODEL --mask MASK --maskperc MASKPERC' where you should specify MODEL, MASK, MASKPERC respectively:
    - MODEL: choose from 'unet' or 'unet_refine'
    - MASK: choose from 'gaussian1d', 'gaussian2d', 'poisson2d'
    - MASKPERC: choose from '10', '20', '30', '40', '50' (percentage of mask)
 
4. Test trained model
    
    1) run 'CUDA_VISIBLE_DEVICES=0 python test.py --model MODEL --mask MASK --maskperc MASKPERC' where you should specify MODEL, MASK, MASKPERC respectively (as above).

# Results

Please refer to the paper for the detailed results.
