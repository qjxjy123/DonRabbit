# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2018/12/23
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import metrics
from keras import backend as K
from models.several_losses import create_loss_cell, create_loss_nr
def pad_1x1(x):
    return K.spatial_2d_padding(x, ((0, 1), (1, 0)))


def squeeze_excite_block(input, ratio=16):
    weight_decay = 0.0005
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # compute channel axis
    filters = K.shape(init)[channel_axis]  # infer input number of filters
    se_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (filters, 1, 1)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Conv2D(3,1 ,activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(se)
    se = Conv2D(filters,1, activation='sigmoid', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
    x = multiply([init, se])
    return x

def unet(pretrained_weights=None, loss = 'mse', input_size=(256, 256, 3),times = 5,nb_classes=3,beta=0.2,lambd=1):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # conv4 = squeeze_excite_block(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    up6_1 = Lambda(pad_1x1)(up6)
    # up6 = Lambda(lambda x: K.switch(K.not_equal(K.shape(drop4)[2] % 2, 0), up6_1, up6))(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    up7_1 = Lambda(pad_1x1)(up7)
    # up7 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv3)[2] % 2, 0), up7_1, up7))(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    up8_1 = Lambda(pad_1x1)(up8)
    # up8 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv2)[2] % 2, 0), up8_1, up8))(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    up9_1 = Lambda(pad_1x1)(up9)
    # up9 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv1)[2] % 2, 0), up9_1, up9))(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(8, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(nb_classes, 1, padding='same', kernel_initializer='he_normal')(conv9)

    model = Model(input=inputs, output=conv10)
    if loss == 'mse':
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    elif loss == 'reverseMapping_loss':
        model.compile(optimizer=Adam(lr=1e-4), loss=create_loss_nr(nbclasses=nb_classes))
    elif loss == 'mapping_loss':
        model.compile(optimizer=Adam(lr=1e-4), loss=create_loss_cell(times=times,beta=beta,lambd=lambd))
    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


if __name__ == '__main__':
    model = unet(input_size=(256, 256, 3))
    model.summary()