#Reference:https://keras.io/examples/vision/deeplabv3_plus/

#mounting from google drive:
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
print(tf.__version__)

import keras as k
print(k.__version__)

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Activation, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
import cv2
from cv2 import fastNlMeansDenoising, medianBlur, findContours,drawContours

eps = np.finfo(float).eps
print(eps)

#set parameters
IMG_WIDTH = 176
IMG_HEIGHT = 176
IMG_CHANNELS = 1
start_frame = [62,65,65]
end_frame = [86,88,85]
TRAIN_PATH = '/content/drive/MyDrive/Brain_MRI_Segmentation/Hammer/train/'
TEST_PATH = '/content/drive/MyDrive/13_t1_mprage_sag_p2_iso_mpr_tra.nii'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

X_train = np.zeros((560, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
Y_train = np.zeros((560, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
X_test = np.zeros((10, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
Y_test = np.zeros((10, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
n = 0
i = 0
Num_files = 29
for i in range(1,Num_files):
    if i<10:
      name = 'a0'
    else:
      name = 'a'
    train_file_names = TRAIN_PATH + 'MRI_data/' + name + str(i)+ '.nii'
    label_file_names = TRAIN_PATH + 'Labels/' + name + str(i)+ '-seg.nii'
    print(train_file_names)
    img_seq = nib.load(train_file_names)
    label_seq = nib.load(label_file_names)
    data = img_seq.get_fdata()
    label = label_seq.get_fdata()
    for f in range(65, 85):
        img = data[:,:,f]
        img = cv2.resize(img,(IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_NEAREST)
        #non-local means filtering
        filterStrength = 15
        templateWindowSize = 11
        searchWindowSize = 23

        X_train[n] = img

        multi_mask = label[:,:,f]
        multi_mask = np.squeeze(multi_mask)

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
        mask_ix = np.where(multi_mask == 40)
        mask[mask_ix] = 1
        mask_ix = np.where(multi_mask == 41)
        mask[mask_ix] = 2
        mask = cv2.resize(mask,(IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_NEAREST)

        Y_train[n] = mask
        n = n + 1
    i = i + 1
X_train = np.expand_dims(X_train, axis=3)


print('Getting and resizing test images and masks ... ')
sys.stdout.flush()

n = 0
test_file_names = TEST_PATH   
img_seq = nib.load(test_file_names)
data = img_seq.get_fdata()
print (data.shape)
for f in range(35, 45):
    img = data[:,:,f]
    print(img.shape)
    img = cv2.resize(img,(IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_NEAREST)
    imshow(img, cmap='gray')
    plt.show()
    X_test[n] = img
    n = n+1

print(X_test.shape)
imshow(X_test[5], cmap='gray')
plt.show()

# Predict on train, val and test
model = load_model('/content/drive/MyDrive/model_mri_seg.h5')
print(X_test.shape)
preds_test = model.predict(X_test, verbose=1)
print(preds_test[5].max())

for ix in range(1, 5):
    print('test image num = ', ix)
    img = X_test[ix]
    if ix < 21:
        IMG_HEIGHT = 182
        IMG_WIDTH = 198
    else:
        IMG_HEIGHT = 178
        IMG_WIDTH = 198

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
    print(img.shape)
    img3 = np.zeros((IMG_HEIGHT,IMG_WIDTH,3),dtype=np.uint8)
    print(img3.shape)
    img3[:,:,0] = img
    img3[:,:,1] = img
    img3[:,:,2] = img

    pred = np.argmax(preds_test[ix], axis=-1)
    print('pred = ', pred.shape)
    mask1 = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)


    mask_ix1 = np.where(pred == 1)
    mask1[mask_ix1] = 1
    print('mask1 = ', mask1.shape)
    contours1, hierarchy = findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img3 = drawContours(img3, contours1, -1, (255, 0, 0), 1)

    mask2 = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    mask_ix2 = np.where(pred == 2)
    mask2[mask_ix2] = 1
    contours2, hierarchy = findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img3 = drawContours(img3, contours2, -1, (255, 0, 0), 1)

    print('img3 = ', img3.shape)
    plt.imshow(img3)
    plt.show()


