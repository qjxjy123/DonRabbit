import keras
from functools import partial

from keras.optimizers import Adam

from models.several_losses import create_loss_cell, create_loss_nr


class FCN8s():
    def __init__(self):
        self.DefaultMaxPool2D = partial(keras.layers.MaxPooling2D,
                                   padding='valid', pool_size=(2, 2), strides=(2, 2))
        self.DefaultConv2DTranspose = partial(keras.layers.Conv2DTranspose,
                                         padding='valid',
                                         kernel_size=(2, 2), strides=(2, 2))
        self.DefaultUpSampling2D = partial(keras.layers.UpSampling2D,
                                      size=(2, 2), data_format='channels_last',
                                      interpolation='bilinear')
        self.Default1x1Conv2d = partial(keras.layers.Conv2D, kernel_size=(1, 1))

    def conv2d_block(self,inputs, filters, kernel_size=(3, 3), n_layers=2, alpha=0.2,
                     use_batch_norm=True, padding='same'):
        for i in range(n_layers):
            inputs = keras.layers.Conv2D(
                filters, kernel_size, padding=padding)(inputs)
            inputs = keras.layers.LeakyReLU(alpha=alpha)(inputs)
            if use_batch_norm:
                inputs = keras.layers.BatchNormalization()(inputs)
        return inputs


    def downsample_block(self,inputs, filters, conv_params=None):
        if conv_params is None:
            conv_params = {}
        conv_layer = self.conv2d_block(inputs, filters, **conv_params)
        maxpool_layer = self.DefaultMaxPool2D()(conv_layer)
        return conv_layer, maxpool_layer


    def upsample_block(self,filters, prev_layer, kernel_size=(2, 2), padding='same',
                       upsampling_size=(2, 2)):
        upsampling = self.DefaultUpSampling2D(size=upsampling_size)(prev_layer)
        conv = keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, padding=padding)(upsampling)
        return conv


    def fcn8s(self,pretrained_weights=None, loss='mapping_loss', input_size=(None, None, 3),nb_classes=3,
                      times=5, beta=0.2, lambd=1):
        inputs = keras.layers.Input(shape=input_size)

        # downsampling stack
        conv1, down1 = self.downsample_block(inputs, 64, conv_params={'n_layers': 2})
        conv2, down2 = self.downsample_block(down1, 128, conv_params={'n_layers': 2})
        conv3, down3 = self.downsample_block(down2, 256, conv_params={'n_layers': 3})
        conv4, down4 = self.downsample_block(down3, 256, conv_params={'n_layers': 3})
        conv5, down5 = self.downsample_block(down4, 256, conv_params={'n_layers': 3})

        # bottleneck
        conv6 = self.conv2d_block(down5, 4096, kernel_size=(1, 1), n_layers=1)
        conv7 = self.conv2d_block(conv6, 4096, kernel_size=(1, 1), n_layers=1)
        conv7_preds = self.Default1x1Conv2d(filters=nb_classes)(conv7)

        # upsampling stack
        up1 = self.upsample_block(nb_classes, conv7_preds)  # if upsampled x32 - FCN32s
        down4_preds = keras.layers.Conv2D(
            filters=nb_classes, kernel_size=(1, 1))(down4)
        add1 = keras.layers.Add()([up1, down4_preds])
        add1_preds = self.Default1x1Conv2d(filters=nb_classes)(add1)

        up2 = self.upsample_block(nb_classes, add1_preds)  # if upsampled x16 - FCN16s
        down3_preds = keras.layers.Conv2D(
            filters=nb_classes, kernel_size=(1, 1))(down3)
        add2 = keras.layers.Add()([up2, down3_preds])
        add2_preds = self.Default1x1Conv2d(filters=nb_classes)(add2)

        last = self.upsample_block(nb_classes, add2_preds, upsampling_size=(8, 8))
        outputs = keras.layers.Conv2D(nb_classes,1)(last)

        model = keras.models.Model(inputs=inputs, outputs=outputs)
        if loss == 'mse':
            model.compile(optimizer=Adam(lr=1e-4), loss='mse')
        elif loss == 'mapping_loss':
            model.compile(optimizer=Adam(lr=1e-4), loss=create_loss_cell(times=times,beta=beta,lambd=lambd))
        elif loss == 'reverseMapping_loss':
            model.compile(optimizer=Adam(lr=1e-4), loss=create_loss_nr(nbclasses=nb_classes,
                                                                           times=times,beta=beta,lambd=lambd))
        return model

if __name__ == '__main__':
    model = FCN8s().fcn8s(input_size=(256,256,3))
    model.summary()