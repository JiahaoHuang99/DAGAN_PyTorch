import numpy as np
import os
from pickle import load, dump
from config import config
import cv2

# data path
training_data_path = config.TRAIN.training_data_path
val_data_path = config.TRAIN.val_data_path
testing_data_path = config.TEST.testing_data_path

# load data
print('Loading Data...')
with open(training_data_path, 'rb') as f:
    X_train = load(f)
with open(val_data_path, 'rb') as f:
    X_val = load(f)
with open(testing_data_path, 'rb') as f:
    X_test = load(f)

print('X_train shape:{}/ min:{}/ max:{}\n'.format(X_train.shape, X_train.min(), X_train.max())
      + 'X_val shape:{}/ min:{}/ max:{}\n'.format(X_val.shape, X_val.min(), X_val.max())
      + 'X_test shape:{}/ min:{}/ max:{}'.format(X_test.shape, X_test.min(), X_test.max()))

# config
preserving_ratio = 0.3  # filter out 2d images containing < x% non-zeros
clip = 0.1
# processing
print('Processing Data...')

# training set
X_train_pro = []
min_nonzero_pixel = np.Inf
min_nonzero_idx = 0
for idx, img in enumerate(X_train):
    print("Processing [{}/{}] image for training set ...".format(idx + 1, len(X_train)))
    # -1~1-->0~1
    img_01 = np.clip(((img + 1) / 2 - clip), 0.0, 1.0)

    if float(np.count_nonzero(img_01)) / img_01.size >= preserving_ratio:
        if float(np.count_nonzero(img_01)) < min_nonzero_pixel:
            min_nonzero_pixel = np.count_nonzero(img_01)
            min_nonzero_idx = idx
        X_train_pro.append(img)

min_nonzero_img = (X_train[min_nonzero_idx, :, :, :] + 1) * 127.5

isExists = os.path.exists('./tmp')
if not isExists:
    os.makedirs('./tmp')
cv2.imwrite("./tmp/min_nonzero{}_train.png".format(int(preserving_ratio*100)), min_nonzero_img)

# validation set
X_val_pro = []
min_nonzero_pixel = np.Inf
min_nonzero_idx = 0
for idx, img in enumerate(X_val):
    print("Processing [{}/{}] image for validation set ...".format(idx + 1, len(X_val)))
    # -1~1-->0~1
    img_01 = np.clip(((img + 1) / 2 - clip), 0.0, 1.0)

    if float(np.count_nonzero(img_01)) / img_01.size >= preserving_ratio:
        if float(np.count_nonzero(img_01)) < min_nonzero_pixel:
            min_nonzero_pixel = np.count_nonzero(img_01)
            min_nonzero_idx = idx
        X_val_pro.append(img)

min_nonzero_img = (X_val[min_nonzero_idx, :, :, :] + 1) * 127.5
cv2.imwrite("./tmp/min_nonzero{}_val.png".format(int(preserving_ratio*100)), min_nonzero_img)

# testing set
X_test_pro = []
min_nonzero_pixel = np.Inf
min_nonzero_idx = 0
for idx, img in enumerate(X_test):
    print("Processing [{}/{}] image for testing set ...".format(idx + 1, len(X_test)))
    # -1~1-->0~1-->-0.1~0.9
    img_01 = np.clip(((img + 1) / 2 - clip), 0.0, 1.0)

    if float(np.count_nonzero(img_01)) / img_01.size >= preserving_ratio:
        if float(np.count_nonzero(img_01)) < min_nonzero_pixel:
            min_nonzero_pixel = np.count_nonzero(img_01)
            min_nonzero_idx = idx
        X_test_pro.append(img)

min_nonzero_img = (X_test[min_nonzero_idx, :, :, :] + 1) * 127.5
cv2.imwrite("./tmp/min_nonzero{}_test.png".format(int(preserving_ratio*100)), min_nonzero_img)


X_train_pro = np.asarray(X_train_pro)
X_val_pro = np.asarray(X_val_pro)
X_test_pro = np.asarray(X_test_pro)

# save data into pickle format
data_saving_path = 'data/MICCAI13_SegChallenge/'

print("Saving training set into pickle format...")
with open(os.path.join(data_saving_path, 'training_pro{}.pickle'.format(int(preserving_ratio*100))), 'wb') as f:
    dump(X_train_pro, f, protocol=4)

print("Saving validation set into pickle format...")
with open(os.path.join(data_saving_path, 'validation_pro{}.pickle'.format(int(preserving_ratio*100))), 'wb') as f:
    dump(X_val_pro, f, protocol=4)

print("Saving test set into pickle format...")
with open(os.path.join(data_saving_path, 'testing_pro{}.pickle'.format(int(preserving_ratio*100))), 'wb') as f:
    dump(X_test_pro, f, protocol=4)

print('X_train_pro shape:{}/ min:{}/ max:{}\n'.format(X_train_pro.shape, X_train_pro.min(), X_train_pro.max())
      + 'X_val_pro shape:{}/ min:{}/ max:{}\n'.format(X_val_pro.shape, X_val_pro.min(), X_val_pro.max())
      + 'X_test_pro shape:{}/ min:{}/ max:{}'.format(X_test_pro.shape, X_test_pro.min(), X_test_pro.max()))
print("Processing data finished!")
