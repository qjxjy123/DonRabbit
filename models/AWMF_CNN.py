# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2019/5/26

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from models.unet_vanilla import unet
from keras.applications import xception
from keras import metrics
import tensorflow as tf
from keras.utils.vis_utils import plot_model


class AWMF:
    def __init__(self, img_dim, batch_size, times=5):    # img_dim 默认为288 batch_size 2
        self.img_dim = img_dim
        self.input_size = (img_dim, img_dim, 3)
        self.batch_size = batch_size
        self.fE1 = self.inference_E(input_size=(img_dim, img_dim, 3), e_id=0)
        self.fE2 = self.inference_E(input_size=(img_dim, img_dim, 3), e_id=1)
        self.fE3 = self.inference_E(input_size=(img_dim, img_dim, 3), e_id=2)
        self.fW = self.xception_module(input_size=(img_dim, img_dim, 3))
        self.fA = self.inference_A(input_size=(img_dim, img_dim, 9))
        self.nb_classes = 3
        self.times = times


    def pad_1x1(x):
        return K.spatial_2d_padding(x, ((0, 1), (1, 0)))

    def weighting_attention(self, fW, output_E1, output_E2, output_E3):
        output_E1_withWeight = multiply([output_E1, fW[0]])
        output_E2_withWeight = multiply([output_E2, fW[1]])
        output_E3_withWeight = multiply([output_E3, fW[2]])
        return output_E1_withWeight, output_E2_withWeight, output_E3_withWeight

    def get_slice_fromE2(self, x):  # return 256x256x3
        height = width = 128
        center_x = center_y = 319 // 2
        return x[:, center_x - height:center_x + height, center_y - width:center_y + width, :]

    def get_slice_fromE3(self, x):  # return 154x154x3
        height = width = 107
        center_x = center_y = 319 // 2
        return x[:, center_x - height:center_x + height, center_y - width:center_y + width, :]

    def loss_cell_v3(self, y_true, y_pred, beta=0.2, lambd=1):

        y_true_reverse = tf.manip.roll(y_true, shift=1, axis=-1) + tf.roll(y_true, shift=2, axis=-1)
        loss_1 = 0.5 * K.mean(
            (lambd * K.mean(y_true, axis=(-2, -3), keepdims=True) + beta * y_true) * K.square(self.times * y_true - y_pred))
        loss_2 = 0.5 * K.mean(
            (lambd * K.mean(y_true_reverse, axis=(-2, -3), keepdims=True) + beta * y_true_reverse) * K.square(
                self.times * y_true - y_pred))
        return 0.75 * loss_1 + 0.25 * loss_2

    def squeeze_excite_block(self, input, ratio=16):
        weight_decay = 0.0005
        init = input
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # compute channel axis
        filters = init._keras_shape[channel_axis]  # infer input number of filters
        se_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (
            filters, 1, 1)  # determine Dense matrix shape

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Flatten()(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        se = Reshape(se_shape)(se)
        x = multiply([init, se])
        return x

    def inference_E(self, pretrained_weights=None, input_size=(320, 320, 3), e_id=0):
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

        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))

        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))

        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))

        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(3, 1, padding='same', kernel_initializer='he_normal')(conv9)
        # conv10 = Lambda(lambda x: x, name='output_E'+str(e_id))(conv10)
        model = Model(input=inputs, output=conv10, name='output_E' + str(e_id))

        # model.compile(optimizer=Adam(lr=1e-4), loss=loss_cell_v3, metrics=[metrics.categorical_accuracy])

        # model.summary()

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

    def xception_module(self, input_size=(None, None, 3)):
        # Pre-Trained CNN Model using imagenet dataset for pre-trained weights
        base_model = xception.Xception(input_shape=input_size,
                                       weights=None,
                                       include_top=False)

        # Top Model Block
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(3, activation='sigmoid')(x)
        model = Model(base_model.input, predictions)
        # recording to the result, decide whether to train these layers
        for layer in base_model.layers:
            layer.trainable = False
        # optimizer loss
        return model

    def inference_A(self, input_size=(256, 256, 9)):
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv5 = Conv2D(3, 1, padding='same', kernel_initializer='he_normal')(conv4)
        # conv5 = Lambda(lambda x: x)(conv5)
        model = Model(input=inputs, output=conv5, name='output_A')
        return model

    def build_AWMF(self):
        input_x_E1 = Input(shape=self.input_size, name='input_x_E1')
        output_x_E1 = self.fE1(input_x_E1)
        input_x_E2 = Input(shape=self.input_size, name='input_x_E2')
        output_x_E2 = self.fE2(input_x_E2)
        input_x_E3 = Input(shape=self.input_size, name='input_x_E3')
        output_x_E3 = self.fE3(input_x_E3)
        # labels = Input(input_size)
        #
        # loss_E0 = self.loss_cell_v3()
        # get w1,w2,w3 from output_x_Ek
        # 设定满足dice coefficient的条仿

        # get y1,y2,y3 from Xception
        output_fW = self.fW(input_x_E2)

        upsample_bilinear = Lambda(lambda x: tf.image.resize_bilinear(x, size=[256, 256]))
        output_x_E1_crop = upsample_bilinear(output_x_E1)
        output_x_E2_crop = Lambda(self.get_slice_fromE2)(output_x_E2)
        output_x_E3_crop = upsample_bilinear(Lambda(self.get_slice_fromE3)(output_x_E3))
        '''
        output_x_E2_crop = UpSampling2D(size=(2, 2))(Lambda(self.get_slice_fromE1)(output_x_E2))

        output_x_E3_crop = UpSampling2D(size=(4, 4))(Lambda(self.get_slice_fromE2)(output_x_E3))
        '''
        # multiply Xception results with Ek_output
        # output_E1_withWeight, output_E2_withWeight, output_E3_withWeight = Lambda(self.weighting_attention, arguments={'output_E1':output_x_E1,'output_E2':output_x_E2_crop,'output_E3':output_x_E3_crop})(output_fW)
        # crop && upsample output_x_E2,output_x_E3

        merge_input = concatenate([output_x_E1_crop, output_x_E2_crop, output_x_E3_crop], axis=-1)
        # SE_block
        ratio = 16
        weight_decay = 0.0005
        init = merge_input
        filters = 9
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Conv2D(3, 1, activation='relu', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        se = Conv2D(filters, 1, activation='sigmoid', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
        merge_input = multiply([init, se])

        final_output = self.fA(merge_input)
        model = Model(inputs=[input_x_E1, input_x_E2, input_x_E3],
                      outputs=[output_x_E1, output_x_E2, output_x_E3, final_output])
        # model.compile(optimizer=Nadam(lr=1e-4), loss={'output_E0': self.loss_cell_v3, 'output_E1': self.loss_cell_v3,
        #                                               'output_E2': self.loss_cell_v3, 'output_A': self.loss_cell_v3},
        #               metrics=[metrics.binary_accuracy])
        model.compile(optimizer=Nadam(lr=1e-4), loss={'output_E0': self.loss_cell_v3, 'output_E1': self.loss_cell_v3,
                                                      'output_E2': self.loss_cell_v3, 'output_A': self.loss_cell_v3}
                      , loss_weights={'output_E0': 0.2, 'output_E1': 0.2, 'output_E2': 0.2, 'output_A': 0.5}
                      )
        return model


if __name__ == '__main__':
    model = AWMF(img_dim=320, batch_size=2).build_AWMF()
    print(model.summary())
    plot_model(model, to_file='model_AWMF.png', show_shapes=True)
