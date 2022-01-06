# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2019/4/2

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics
from keras import backend as K
import tensorflow as tf
from models.several_losses import create_loss_cell,create_loss_nr

def pad_1x1(x):
    return K.spatial_2d_padding(x, ((0, 1), (1, 0)))



def Residual_block(x_input, channels_num):
    x = Activation('elu')(x_input)
    conv1 = Conv2D(channels_num, 3, padding='same', kernel_initializer='he_normal')(x)
    conv1 = Dropout(0.5)(conv1)
    conv1 = Activation('elu')(conv1)
    conv2 = Conv2D(channels_num, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv2 = Lambda(lambda a: a * 0.3)(conv2)
    # se = squeeze_excite_block(conv2)
    result = add([x_input, conv2])
    return result


def unet(pretrained_weights=None, loss='mapping_loss', input_size=(None, None, 3),nb_classes=3,
                  times=5, beta=0.2, lambd=1):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(inputs)  # 256x256x32
    # residual block
    conv2 = Residual_block(conv1, channels_num=32)  # 256x256x32    conv2 concate1
    # convolution 3x3
    conv3 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv2)  # 256x256x32
    # conv 1x1
    conv3 = Conv2D(64, 1, activation='elu', padding='same', kernel_initializer='he_normal')(conv3)  # 256x256x64
    # mean pooling 2x2  1st DownSample
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv3)  # 128x128x64

    # residual block
    conv4 = Residual_block(pool1, channels_num=64)  # 128x128x64    conv4 concate2
    # convolution 3x3
    conv5 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv4)  # 128x128x64
    # conv 1x1
    conv5 = Conv2D(128, 1, activation='elu', padding='same', kernel_initializer='he_normal')(conv5)  # 128x128x128
    # mean pooling 2x2 down sample  2nd Down Sample
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv5)  # 64x64x128

    # residual block
    conv6 = Residual_block(pool2, channels_num=128)  # 64x64x128    conv6 concate3
    # convolution 3x3
    conv7 = Conv2D(128, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv6)  # 64x64x128
    # convolution 1x1
    conv7 = Conv2D(256, 1, activation='elu', padding='same', kernel_initializer='he_normal')(conv7)  # 64x64x256
    # mean pooling 2x2 down sample  3rd Down Sample
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv7)  # 32x32x256

    # residual block
    conv8 = Residual_block(pool3, channels_num=256)  # 32x32x256    conv8  concate4
    # mean pooling 2x2 down sample  4th Down Sample
    pool4 = AveragePooling2D(pool_size=(2, 2))(conv8)  # 16x16x256
    # residual block
    conv9 = Residual_block(pool4, channels_num=256)  # 16x16x256
    #1st  up sampleing
    up9 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv9)  # 32x32x256
    # # keep size equal
    # up9_1 = Lambda(pad_1x1)(up9)
    # up9 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv8)[2] % 2, 0), up9_1, up9))(up9)

    # concatenate
    merge9 = concatenate([conv8, up9], axis=-1)  # 32x32x512
    conv10 = Residual_block(merge9, channels_num=512)  # 32x32x512
    conv10 = Conv2D(256, 1, activation='elu', padding='same', kernel_initializer='he_normal')(conv10)  # 32x32x256
    # 2nd up sample
    up10 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv10)  # 64x64x256
    # keep size equal
    # conditional padding
    # up10_1 = Lambda(pad_1x1)(up10)
    # up10 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv6)[2] % 2, 0), up10_1, up10))(up10)

    # concatenate
    merge10 = concatenate(([conv6, up10]), axis=-1)  # 64x64x384
    res10 = Residual_block(merge10, channels_num=384)  # 64x64x384
    res10 = Conv2D(128, 1, activation='elu', padding='same', kernel_initializer='he_normal')(res10)  # 64x64x128
    # 3rd up sample
    up11 = UpSampling2D(size=(2, 2), interpolation='bilinear')(res10)  # 128x128x128
    # conditional padding
    # keep size equal
    # up11_1 = Lambda(pad_1x1)(up11)
    # up11 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv4)[2] % 2, 0), up11_1, up11))(up11)

    # concatenate
    merge11 = concatenate(([conv4, up11]), axis=-1)  # 128x128x192
    res11 = Residual_block(merge11, channels_num=192)  # 128x128x192
    res11 = Conv2D(64, 1, activation='elu', padding='same', kernel_initializer='he_normal')(res11)  # 128x128x64
    # 4th up sample
    up12 = UpSampling2D(size=(2, 2), interpolation='bilinear')(res11)  # 256x256x64
    # conditional padding
    # keep size equal
    # up12_1 = Lambda(pad_1x1)(up12)
    # up12 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv2)[2] % 2, 0), up12_1, up12))(up12)

    merge12 = concatenate(([conv2, up12]), axis=-1)  # 256x256x96
    res12 = Residual_block(merge12, channels_num=96)  # 256x256x96
    res12 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(res12)  # 256x256x32
    conv12 = Conv2D(nb_classes, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(res12)  # 256x256x3

    model = Model(input=inputs, output=conv12)
    if loss == 'mse':
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    elif loss == 'mapping_loss':
        model.compile(optimizer=Adam(lr=1e-4), loss=create_loss_cell(times=times,beta=beta,lambd=lambd))
    elif loss == 'reverseMapping_loss':
        model.compile(optimizer=Adam(lr=1e-4), loss=create_loss_nr(nbclasses=nb_classes,
                                                                       times=times,beta=beta,lambd=lambd))

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

if __name__ == '__main__':
    model = unet(input_size=(256, 256, 3))
    model.summary()