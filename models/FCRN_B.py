from keras.models import *
from keras.layers import *
from keras.optimizers import *
from models.several_losses import create_loss_nr, create_loss_cell
from models.unet_vanilla import unet
from keras.applications import xception
from keras import metrics
import tensorflow as tf
from keras.utils.vis_utils import plot_model

class FCRN_B:
    def __init__(self, img_dim, batch_size):    # img_dim 默认为288 batch_size 2
        self.img_dim = img_dim
        self.input_size = (img_dim, img_dim, 3)
        self.batch_size = batch_size
        self.fE = self.inference(input_size=(img_dim, img_dim, 3))
        self.nb_classes = 3

    def inference(self, pretrained_weights=None, input_size=(256, 256, 3),LOSS_INFO='mapping_loss',nb_classes=3,
                  times=5, beta=0.2, lambd=1):
        inputs = Input(input_size)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)

        conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)

        conv6 = UpSampling2D(size=(2,2),interpolation='bilinear')(conv5)
        conv6 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        conv7 = UpSampling2D(size=(2,2),interpolation='bilinear')(conv6)
        conv7 = Conv2D(nb_classes, 5, padding='same', kernel_initializer='he_normal')(conv7)

        model = Model(input=inputs, output=conv7)

        if LOSS_INFO == 'reverseMapping_loss':
            model.compile(optimizer=Adam(lr=1e-4),
                          loss=create_loss_nr(nbclasses=nb_classes, times=times, beta=beta, lambd=lambd))
        if LOSS_INFO == 'mapping_loss':
            model.compile(optimizer=Adam(lr=1e-4), loss=create_loss_cell(times=times, beta=beta, lambd=lambd))
        # model.summary()
        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

if __name__ == '__main__':
    model = FCRN_B(256,2).inference()
    model.summary()
