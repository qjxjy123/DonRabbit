import tensorflow as tf
from keras import layers
from keras.initializers import RandomUniform
import keras.backend as K
K.set_floatx('float64')


def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


class EncodingLayer(layers.Layer):
    def __init__(self, K, D):
        super(EncodingLayer, self).__init__()
        self.K = K
        self.D = D

    def build(self, input_shape):
        # self.D = input_shape[-1]
        std = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords = self.add_weight(shape=(self.K, self.D), initializer=RandomUniform(minval=-std, maxval=std),
                                         trainable=True, name='codewords')
        self.scale = self.add_weight(shape=(self.K,), initializer=RandomUniform(minval=-std, maxval=std),
                                     trainable=True, name='scale')
        super(EncodingLayer, self).build(input_shape)

    def call(self, x):
        # forward x.shape Bx(HxW)xD
        temp = self.scale_l2(x)
        A = softmax(self.scale_l2(x), axis=2)
        E = self.aggregate(A, x)
        return E

    @tf.custom_gradient
    def scale_l2(self, x, c, s):
        x_explode = K.tile(K.expand_dims(x, axis=-2),
                           (K.int_shape(x)[0], K.int_shape(x)[1], self.K, self.D))  # x.shape Bx(HxW)xKxD
        c_explode = K.expand_dims(K.expand_dims(c, axis=0), axis=0)  # c.shape 1x1xKxD
        s_explode = K.reshape(s, (1, 1, self.K))  # s.shape 1x1xK
        res_vector = K.sum(K.pow(x_explode - c_explode, 2), axis=3)  # r.shape Bx(HxW)xK
        smooth_vector = s_explode * res_vector  # r.shape Bx(HxW)xK

        def grad(dA):
            # dy.shape Bx(HxW)xK
            tmp = K.expand_dims(2 * dA * s_explode, -1) * (
                        x_explode - c_explode)  # Bx(HxW)xKx1 * Bx(HxW)xKxD -> Bx(HxW)xKxD
            grad_x = K.sum(tmp, 2)
            grad_c = K.sum(K.sum(tmp, 0), 0)
            grad_s = K.sum(K.sum(dA * (smooth_vector / s_explode), 0), 0)
            return grad_x, grad_c, grad_s

        return smooth_vector, grad

    @tf.custom_gradient
    def aggregate(self, A, x, c):
        A_explode = K.expand_dims(A, -1)  # A.shape Bx(HxW)xKx1
        x_explode = K.tile(K.expand_dims(x, axis=-2),
                           (K.int_shape(x)[0], K.int_shape(x)[1], self.K, self.D))  # x.shape Bx(HxW)xKxD
        c_explode = K.expand_dims(K.expand_dims(c, axis=0), axis=0)  # c.shape 1x1xKxD
        E = K.sum(A_explode * (x_explode - c_explode), 1)  # Bx(HxW)xKx1* Bx(HxW)xKxD -> Bx(HxW)xKxD -> BxKxD

        def grad(dE):
            grad_A = K.sum((K.expand_dims(dE, 1) * (x_explode - c_explode)),
                           3)  # [BxKxD -> Bx1xKxD] * Bx(HxW)xKxD ->Bx(HxW)xKxD ->(sum) Bx(HxW)xK
            grad_x = K.dot(A, dE)  # Bx(HxW)xK * BxKxD -> Bx(HxW)xD
            grad_c = K.sum((-dE * K.expand_dims(K.sum(A, 1), 2)),
                           0)  # BxKxD * [Bx(HxW)xK -> BxK -> BxKx1] -> BxKxD -> KxD
            return grad_A, grad_x, grad_c

        return E, grad

    def compute_output_shape(self, input_shape):
        return (list(self.K), input_shape[-1])