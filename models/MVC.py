from keras.models import *
from keras.layers import *
from keras.optimizers import *

from models.FRCN import Residual_block
import tensorflow as tf
from models.modules.encoding import EncodingLayer as Encoding
from keras.layers import Lambda
from models.modules.utils import softmax
from keras.losses import logcosh
from experiments.constant import const

class MVC:
    def __init__(self, img_dim, batch_size,nb_classes = 3,times=5):
        self.img_dim = img_dim
        self.input_size = (img_dim, img_dim, 3)
        self.batch_size = batch_size
        self.fE1 = self.inference_E(input_size=(img_dim, img_dim, 3), e_id=0)
        self.fE2 = self.inference_E(input_size=(img_dim, img_dim, 3), e_id=1)
        self.fA = self.inference_A(input_size=(img_dim, img_dim, 6))
        self.nb_classes = nb_classes
        self.times = times

    def pad_1x1(x):
        return K.spatial_2d_padding(x, ((0, 1), (1, 0)))

    def weighting_attention(self, fW, output_E1, output_E2):
        output_E1_withWeight = multiply([output_E1, fW[0]])
        output_E2_withWeight = multiply([output_E2, fW[1]])
        return output_E1_withWeight, output_E2_withWeight

    def get_slice_fromE2(self, x):  # return 256x256x3
        height = width = 128
        center_x = center_y = 255 // 2
        return x[:, center_x - height:center_x + height, center_y - width:center_y + width, :]

    def get_slice_fromE3(self, x):  # return 154x154x3
        height = width = 107
        center_x = center_y = 319 // 2
        return x[:, center_x - height:center_x + height, center_y - width:center_y + width, :]

    def loss_cell_v3(self,y_true, y_pred, times=5, beta=0.2, lambd=1):
        y_true_reverse = tf.roll(y_true, shift=1, axis=-1) + tf.roll(y_true, shift=2, axis=-1)
        loss_1 = 0.5 * K.mean(
            (lambd * K.mean(y_true, axis=(-2, -3), keepdims=True) + beta * y_true) * K.square(times * y_true - y_pred))
        loss_2 = 0.5 * K.mean(
            (lambd * K.mean(y_true_reverse, axis=(-2, -3), keepdims=True) + beta * y_true_reverse) * K.square(
                times * y_true - y_pred))
        return 0.75 * loss_1 + 0.25 * loss_2

    def log_cosh(self,y_true,y_pred):
        return logcosh(y_true/const.Z, y_pred/const.Z)

    # def squeeze_excite_block(self, input, ratio=16):
    #     weight_decay = 0.0005
    #     init = input
    #     channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # compute channel axis
    #     filters = init._keras_shape[channel_axis]  # infer input number of filters
    #     se_shape = (1, 1, filters) if K.image_data_format() == 'channels_last' else (
    #         filters, 1, 1)  # determine Dense matrix shape
    #
    #     se = GlobalAveragePooling2D()(init)
    #     se = Reshape(se_shape)(se)
    #     se = Flatten()(se)
    #     se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal',
    #                kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
    #     se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal',
    #                kernel_regularizer=regularizers.l2(weight_decay), use_bias=False)(se)
    #     se = Reshape(se_shape)(se)
    #     x = multiply([init, se])
    #     return x

    def inference_E(self, pretrained_weights=None, input_size=(320, 320, 3), e_id=0):
        inputs = Input(input_size)
        conv1 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(inputs)  # 256x256x32
        # residual block
        conv2 = Residual_block(conv1, channels_num=32)  # 256x256x32
        # convolution 3x3
        conv3 = Conv2D(32, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv2)  # 256x256x32
        # conv 1x1
        conv3 = Conv2D(64, 1, activation='elu', padding='same', kernel_initializer='he_normal')(conv3)  # 256x256x64
        # mean pooling 2x2  1st DownSample
        pool1 = AveragePooling2D(pool_size=(2, 2))(conv3)  # 128x128x64

        # residual block
        conv4 = Residual_block(pool1, channels_num=64)  # 128x128x64
        # convolution 3x3
        conv5 = Conv2D(64, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv4)  # 128x128x64
        # conv 1x1
        conv5 = Conv2D(128, 1, activation='elu', padding='same', kernel_initializer='he_normal')(conv5)  # 128x128x128
        # mean pooling 2x2 down sample  2nd Down Sample
        pool2 = AveragePooling2D(pool_size=(2, 2))(conv5)  # 64x64x128

        # residual block
        conv6 = Residual_block(pool2, channels_num=128)  # 64x64x128
        # convolution 3x3
        conv7 = Conv2D(128, 3, activation='elu', padding='same', kernel_initializer='he_normal')(conv6)  # 64x64x128
        # convolution 1x1
        conv7 = Conv2D(256, 1, activation='elu', padding='same', kernel_initializer='he_normal')(conv7)  # 64x64x256
        # mean pooling 2x2 down sample  3rd Down Sample
        pool3 = AveragePooling2D(pool_size=(2, 2))(conv7)  # 32x32x256

        # residual block
        conv8 = Residual_block(pool3, channels_num=256)  # 32x32x256
        # mean pooling 2x2 down sample  4th Down Sample
        pool4 = AveragePooling2D(pool_size=(2, 2))(conv8)  # 16x16x256
        # residual block
        conv9 = Residual_block(pool4, channels_num=256)  # 16x16x256
        # 1st  up sampleing
        up9 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv9)  # 32x32x256
        # keep size equal
        up9_1 = Lambda(self.pad_1x1)(up9)
        up9 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv8)[2] % 2, 0), up9_1, up9))(up9)

        # concatenate
        merge9 = concatenate([conv8, up9], axis=-1)  # 32x32x512
        conv10 = Residual_block(merge9, channels_num=512)  # 32x32x512
        conv10 = Conv2D(256, 1, activation='elu', padding='same', kernel_initializer='he_normal')(conv10)  # 32x32x256
        # 2nd up sample
        up10 = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv10)  # 64x64x256
        # keep size equal
        # conditional padding
        up10_1 = Lambda(self.pad_1x1)(up10)
        up10 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv6)[2] % 2, 0), up10_1, up10))(up10)

        # concatenate
        merge10 = concatenate(([conv6, up10]), axis=-1)  # 64x64x384
        res10 = Residual_block(merge10, channels_num=384)  # 64x64x384
        res10 = Conv2D(128, 1, activation='elu', padding='same', kernel_initializer='he_normal')(res10)  # 64x64x128
        # 3rd up sample
        up11 = UpSampling2D(size=(2, 2), interpolation='bilinear')(res10)  # 128x128x128
        # conditional padding
        # keep size equal
        up11_1 = Lambda(self.pad_1x1)(up11)
        up11 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv4)[2] % 2, 0), up11_1, up11))(up11)

        # concatenate
        merge11 = concatenate(([conv4, up11]), axis=-1)  # 128x128x192
        res11 = Residual_block(merge11, channels_num=192)  # 128x128x192
        res11 = Conv2D(64, 1, activation='elu', padding='same', kernel_initializer='he_normal')(res11)  # 128x128x64
        # 4th up sample
        up12 = UpSampling2D(size=(2, 2), interpolation='bilinear')(res11)  # 256x256x64
        # conditional padding
        # keep size equal
        up12_1 = Lambda(self.pad_1x1)(up12)
        up12 = Lambda(lambda x: K.switch(K.not_equal(K.shape(conv2)[2] % 2, 0), up12_1, up12))(up12)

        merge12 = concatenate(([conv2, up12]), axis=-1)  # 256x256x96
        res12 = Residual_block(merge12, channels_num=96)  # 256x256x96
        res12 = Conv2D(32, 1, activation='elu', padding='same', kernel_initializer='he_normal')(res12)  # 256x256x32
        conv12 = Conv2D(3, 1)(res12)  # 256x256x3

        model = Model(input=inputs, output=conv12, name='output_E' + str(e_id))

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

    def inference_A(self, input_size=(256, 256, 6)):
        inputs = Input(input_size)
        conv1 = Conv2D(16, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('elu')(conv1)
        conv2 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('elu')(conv2)
        conv3 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('elu')(conv3)

        encode = self.encoding_module(conv3)
        encode = softmax(encode, axis=-1)
        conv4 = multiply(encode, conv3)
        conv5 = Conv2D(3,3, padding='same', kernel_initializer='he_normal')(conv4)
        model = Model(input=inputs, output=conv5, name='output_A')
        return model, conv5

    def encoding_module(self,input_size=(256, 256, 64)):
        input =Input(input_size)
        conv1 = Conv2D(64,1, padding='same', kernel_initializer='he_normal',use_bias=False)(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        encode = Encoding(K=32, D=64)(conv1)
        encode = K.mean(encode, axis=1)
        fc1 = Dense(64)(encode)
        model = Model(input=input,output=fc1,name='output_Enc')
        return model


    def build_MVC(self):
        input_x_E1 = Input(shape=self.input_size, name='input_x_E1')
        output_x_E1 = self.fE1(input_x_E1)
        input_x_E2 = Input(shape=self.input_size, name='input_x_E2')
        output_x_E2 = self.fE2(input_x_E2)

        upsample_bilinear = Lambda(lambda x: tf.image.resize_bilinear(x, size=[256, 256]))
        output_x_E1_crop = upsample_bilinear(output_x_E1)
        output_x_E2_crop = upsample_bilinear(Lambda(self.get_slice_fromE2)(output_x_E2))

        merge_input = concatenate([output_x_E1_crop, output_x_E2_crop], axis=-1)

        final_output, output_encode = self.fA(merge_input)
        model = Model(inputs=[input_x_E1, input_x_E2],
                      outputs=[output_x_E1, output_x_E2, output_encode, final_output])

        model.compile(optimizer=Nadam(lr=1e-4), loss={'output_E0': self.loss_cell_v3, 'output_E1': self.loss_cell_v3,
                                                      'output_Enc': logcosh, 'output_A': self.loss_cell_v3}
                      , loss_weights={'output_E0': 1, 'output_E1': 1, 'output_A': 1, 'output_Enc': 1}
                      )
        return model

