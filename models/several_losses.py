
from keras.optimizers import *
from keras import backend as K
from keras.losses import logcosh
from experiments.constant import const

def create_loss_cell(times=5, beta=0.2, lambd=1):
    def loss_cell(y_true, y_pred):
        return 0.5 * K.mean(
            (lambd * K.mean(y_true, axis=(-2, -3), keepdims=True) + beta * y_true) * K.square(times * y_true - y_pred))
    return loss_cell

def create_loss_nr(times=5, beta=0.2, lambd=1, nbclasses=3):
    def loss_nr(y_true,y_pred):
        y_true_reverse = tf.roll(y_true, shift=1, axis=-1)
        for i in range(nbclasses-1):
            y_true_reverse += tf.roll(y_true, shift=2+i, axis=-1)
        loss_1 = 0.5 * K.mean((lambd * K.mean(y_true, axis=(-2, -3), keepdims=True) + beta * y_true) * K.square(times * y_true - y_pred))
        loss_2 = 0.5 * K.mean((lambd * K.mean(y_true_reverse, axis=(-2, -3), keepdims=True) + beta * y_true_reverse) * K.square(times * y_true - y_pred))
        return 0.75 * loss_1 + 0.25 * loss_2
    return loss_nr

def log_cosh(self,y_true,y_pred):
    return logcosh(y_true/const.Z, y_pred/const.Z)
