from keras.models import *
from keras.layers import *
from keras.optimizers import *
from models.several_losses import create_loss_nr,create_loss_cell

class FCRN_A:
    def __init__(self, img_dim, batch_size):    # img_dim 默认为288 batch_size 2
        self.img_dim = img_dim
        self.input_size = (img_dim, img_dim, 3)
        self.batch_size = batch_size


    def inference(self, pretrained_weights=None, input_size=(256, 256, 3),LOSS_INFO='mapping_loss',nb_classes=3,
                  times=5, beta=0.2, lambd=1):
        inputs = Input(input_size)
        conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)

        conv5 = UpSampling2D(size=(2,2),interpolation='bilinear')(conv4)
        conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

        conv6 = UpSampling2D(size=(2,2),interpolation='bilinear')(conv5)
        conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        conv7 = UpSampling2D(size=(2,2),interpolation='bilinear')(conv6)
        conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        conv8 = Conv2D(nb_classes, 3, padding='same', kernel_initializer='he_normal')(conv7)

        model = Model(input=inputs, output=conv8)

        if LOSS_INFO == 'reverseMapping_loss':
            model.compile(optimizer=Adam(lr=1e-4), loss=create_loss_nr(nbclasses=nb_classes,times=times,beta=beta,lambd=lambd))
        if LOSS_INFO == 'mapping_loss':
            model.compile(optimizer=Adam(lr=1e-4), loss=create_loss_cell(times=times,beta=beta,lambd=lambd))

        # model.summary()
        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

if __name__ == '__main__':
    model = FCRN_A(256,4).inference(nb_classes=3)
    model.summary()
