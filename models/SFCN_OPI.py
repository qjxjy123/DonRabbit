import keras
from keras.layers import Input,Conv2D,Add,BatchNormalization,Activation, Lambda, Multiply, Conv2DTranspose, Concatenate, Reshape
from keras.models import Model
from keras.utils import np_utils,print_summary
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint, Callback
import os, time
import numpy as np
import tensorflow as tf
from models.several_losses import create_loss_nr,create_loss_cell
epsilon = 1e-7
cls_threshold = 0.8



def get_slice_c0(self, x):  # return detection result of channel 0
    return K.expand_dims(x[:, :, :,0],axis=-1)



class Conv3l2(keras.layers.Conv2D):
    """
    Custom convolution layer with default 3*3 kernel size and L2 regularization.
    Default padding change to 'same' in this case.
    """
    def __init__(self, filters, kernel_regularizer_weight,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(self.__class__, self).__init__(filters,
                                             kernel_size=(3, 3),
                                             strides=strides,
                                             padding=padding,
                                             data_format=data_format,
                                             dilation_rate=dilation_rate,
                                             activation=activation,
                                             use_bias=use_bias,
                                             kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer,
                                             kernel_regularizer=keras.regularizers.l2(kernel_regularizer_weight),
                                             bias_regularizer=bias_regularizer,
                                             activity_regularizer=activity_regularizer,
                                             kernel_constraint=kernel_constraint,
                                             bias_constraint=bias_constraint,
                                             **kwargs)


class SFCNnetwork:
    """
    Backbone of SFCN-OPI.
    """
    def __init__(self, input_shape=(64, 64, 3)):
        # self.inputs = inputs
        self.input_shape = input_shape
        self.l2r = 1

    def first_layer(self, inputs, trainable=True):
        """
        First convolution layer.
        """
        x = Conv3l2(filters=32, name='Conv_1',
                    kernel_regularizer_weight=self.l2r,
                    trainable=trainable)(inputs)
        x = BatchNormalization(name='BN_1',trainable=trainable)(x)
        x = Activation('relu', name='act_1',trainable=trainable)(x)
        return x

    def get_slice_c0(self, x):  # return detection result of channel 0
        return K.expand_dims(x[:, :, :,0],axis=-1)
    ###########################################
    # ResNet Graph
    ###########################################
    def identity_block(self, f, stage, block, inputs, trainable=True):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        :param trainable: freeze layer if false
        """
        x_shortcut = inputs

        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                    name=str(block)+'_'+str(stage) + '_idblock_conv_1',
                    trainable=trainable)(inputs)
        x = BatchNormalization(name=str(block)+'_'+str(stage) +'_idblock_BN_1',
                               trainable=trainable)(x)
        x = Activation('relu', name=str(block)+'_'+str(stage) + '_idblock_act_1',
                       trainable=trainable)(x)

        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                    name=str(block)+'_'+str(stage) + '_idblock_conv_2')(x)
        x = BatchNormalization(name=str(block)+'_'+str(stage) + '_idblock_BN_2',
                               trainable=trainable)(x)
        x = Add(name=str(block)+'_'+str(stage) + '_idblock_add',trainable=trainable)([x, x_shortcut])
        x = Activation('relu', name=str(block)+'_'+str(stage)+ '_idblock_act_2',
                       trainable=trainable)(x)
        return x

    def convolution_block(self, f, stage, block, inputs, trainable=True):
        """
        :param f: number of filters
        :param stage: stage of residual blocks
        :param block: ith module
        """
        x = Conv3l2(filters=f, strides=(2,2), kernel_regularizer_weight=self.l2r,
                    name=str(block)+'_'+str(stage) + '_convblock_conv_1',
                    trainable=trainable)(inputs)
        x = BatchNormalization(name=str(block)+'_'+str(stage) + '_convblock_BN_1',
                               trainable=trainable)(x)
        x = Activation('relu', name=str(block)+'_'+str(stage) + '_convblock_act_1',
                       trainable=trainable)(x)
        x = Conv3l2(filters=f, kernel_regularizer_weight=self.l2r,
                   name=str(block) + '_' + str(stage) + '_convblock_conv_2',
                    trainable=trainable)(x)
        x = BatchNormalization(name=str(block)+'_'+str(stage) + '_convblock_BN_2',
                               trainable=trainable)(x)

        x_shortcut = Conv2D(f, kernel_size=(1,1), strides=(2,2), padding='same',
                            kernel_regularizer=keras.regularizers.l2(self.l2r),
                            name=str(block)+'_'+str(stage) + '_convblock_shortcut_conv',
                            trainable=trainable)(inputs)
        x_shortcut = BatchNormalization(name = str(block)+'_'+str(stage) + '_convblock_shortcut_BN_1',
                                        trainable=trainable)(x_shortcut)
        x = Add(name= str(block) + '_'+str(stage) + '_convblock_add',
                trainable=trainable)([x, x_shortcut])
        x = Activation('relu', name = str(block) + '_' + str(stage) + '_convblock_merge_act',
                       trainable=trainable)(x)
        return x

    def res_block(self, inputs, filter, stages, block, trainable=True, if_conv=False):
        x = inputs
        if not if_conv: #if_conv is False
            for stage in range(stages):
                x = self.identity_block(f=filter, stage=stage, block=block,
                                        inputs=x, trainable=trainable)
        else:
            for stage in range(stages):
                if stage == 0:
                    x = self.convolution_block(f=filter, stage=stage, block=block, inputs=inputs,
                                               trainable=trainable)
                else:
                    x = self.identity_block(f=filter, stage=stage, block=block, inputs=x,
                                            trainable=trainable)
        return x

    ######################
    # FCN BACKBONE
    # ####################
    def first_and_second_res_blocks(self, inputs, first_filter, second_filter, trainable=True):
        """
        Shared residual blocks for detection and classification layers.
        """
        x = self.res_block(inputs, filter=first_filter, stages=9, block=1,
                           trainable=trainable)
        x = self.res_block(x, filter=second_filter, stages=9, block=2, if_conv=True,
                           trainable=trainable)
        return x

    def share_layer(self, input, trainable=True):
        with tf.variable_scope("shared_layer"):
            x = self.first_layer(input, trainable=trainable)
            x_future_det_one = self.first_and_second_res_blocks(x, 32, 64, trainable=trainable)
            x_future_cls_det_two = self.res_block(x_future_det_one, filter=128, stages=9, block=3, if_conv=True,
                                                  trainable=trainable)
            #print(tf.trainable_variables())
        return x_future_det_one, x_future_cls_det_two
    ###################
    # Detection Branch
    ###################
    def detection_branch_wrapper(self, input_one, input_two, trainable=True, softmax_trainable=False):
        x_divergent_one = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                                 name='conv2D_diverge_one',
                                 trainable=trainable)(input_one)
        x_divergent_one = BatchNormalization(name='bn_diverge_one',
                                             trainable=trainable)(x_divergent_one)
        x_divergent_one = Activation('relu', trainable=trainable)(x_divergent_one)

        x_divergent_two = Conv2D(filters=2, kernel_size=(1, 1), padding='same',
                                 kernel_regularizer=keras.regularizers.l2(self.l2r),
                                 name='conv_diverge_two',
                                 trainable=trainable)(input_two)
        x_divergent_two = BatchNormalization(name='bn_diverge_two',
                                             trainable=trainable)(x_divergent_two)
        x_divergent_two = Activation('relu',
                                     trainable=trainable)(x_divergent_two)

        x_divergent_two = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                          kernel_regularizer=keras.regularizers.l2(self.l2r),
                                          name='deconv_before_summation',
                                          trainable=trainable)(x_divergent_two)
        x_divergent_two = BatchNormalization(name='bn_deconv_diverge_two',
                                             trainable=trainable)(x_divergent_two)
        x_divergent_two = Activation('relu', name='last_detection_act',
                                     trainable=trainable)(x_divergent_two)

        x_merge = Add(name='merge_two_divergence',
                      trainable=trainable)([x_divergent_one, x_divergent_two])
        x_detection = Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                      kernel_regularizer=keras.regularizers.l2(self.l2r),
                                      name='Deconv_detection_final_layer',
                                      trainable=trainable)(x_merge)
        x_detection = BatchNormalization(name='last_detection_bn',
                                         trainable=trainable)(x_detection)
        # The detection output
        if softmax_trainable == True:
            x_detection = Activation('softmax', name='Detection_output',
                                     trainable=trainable)(x_detection)
        return x_detection

    def detection_branch(self, trainable=True, softmax_trainable=False):
        input_img = Input(shape=self.input_shape)
        x_future_det_one, x_future_cls_det_two = self.share_layer(input_img, trainable=trainable)
        # The detection output
        x_detection = self.detection_branch_wrapper(x_future_det_one, x_future_cls_det_two, softmax_trainable=softmax_trainable)
        # The classification output
        det_model = Model(inputs=input_img,
                      outputs=x_detection)
        return det_model

    ############################
    # Classification Branch
    ############################
    def classification_branch_wrapper(self, input, softmax_trainable=False):
        x = self.res_block(input, filter=128, stages=9, block=4)
        # all layers before OPI
        x = Conv2D(filters=4, kernel_size=(1, 1), padding='same', name='conv2d_after_fourth_resblock',
                   kernel_regularizer=keras.regularizers.l2(self.l2r))(x)
        x = BatchNormalization(name='bn_after_fourth_resblock')(x)
        x = Activation('relu',name='relu_after_fourth_resblock')(x)
        x = Conv2DTranspose(filters=4, kernel_size=(3, 3),
                            strides=(2, 2), padding='same',
                            kernel_regularizer=keras.regularizers.l2(self.l2r),
                            name='secondlast_deconv_before_cls')(x)
        x = BatchNormalization(name='secondlast_bn_before_cls')(x)
        x = Activation('relu', name='last_relu_before_cls')(x)
        x = Conv2DTranspose(filters=4, kernel_size=(3, 3),
                            strides=(2, 2), padding='same',
                            kernel_regularizer=keras.regularizers.l2(self.l2r),
                            name='last_deconv_before_cls')(x)
        x_output = BatchNormalization(name='last_bn_before_cls')(x)
        if softmax_trainable == True:
            x_output = Activation('softmax', name='Classification_output')(x_output)
        return x_output

    def classification_branch(self, trainable=False, softmax_trainable=False):
        """classification branch, seprate from detection branch.
        """
        input_img = Input(shape=self.input_shape)
        # this shared layer is frozen before joint training
        x_useless, x_cls = self.share_layer(input_img, trainable=trainable)
        cls_output = self.classification_branch_wrapper(x_cls, softmax_trainable=softmax_trainable)
        cls_model = Model(inputs=input_img,
                          outputs=cls_output)
        return cls_model

    #########################
    # Joint training
    #########################
    def joint_branch(self, trainable=True, softmax_trainable=False):
        """
        joint branch of detection and classification
        :param trainable: unfreeze detection branch layer if set to true
        """
        input_img = Input(shape=self.input_shape)
        x_future_det_one, x_future_cls_det_two = self.share_layer(input_img, trainable=trainable)
        x_detection = self.detection_branch_wrapper(x_future_det_one, x_future_cls_det_two, trainable=trainable,
                                                    softmax_trainable=softmax_trainable)
        x_classification = self.classification_branch_wrapper(x_future_cls_det_two,
                                                              softmax_trainable=softmax_trainable)
        # x_detection_c0 = Lambda(lambda x: x[:,:,:,0][:,:,:,np.newaxis],name='extract_layer')(x_detection)
        x_detection_c0 = Lambda(self.get_slice_c0,name='crop_layer')(x_detection)
        joint_x = Multiply(name='joint_multiply_layer')([x_detection_c0, x_classification])
        # input_img = Input(shape=self.input_shape)#,name='input_layer')

        joint_model = Model(inputs=input_img,outputs=[x_detection,joint_x])
        return joint_model


class TimerCallback(Callback):
    """Tracking time spend on each epoch as well as whole training process.
    """
    def __init__(self):
        super(TimerCallback, self).__init__()
        self.epoch_time = 0
        self.training_time = 0

    def on_train_begin(self, logs=None):
        self.training_time = time.time()

    def on_train_end(self, logs=None):
        end_time = np.round(time.time() - self.training_time, 2)
        time_to_min = end_time / 60
        print('training takes {} minutes'.format(time_to_min))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print('epoch takes {} seconds to train'.format(np.round(time.time() - self.epoch_time), 2))


def det_model_compile(nn, times, softmax_trainable, optimizer, summary=False):
    """

    :param times:
    :param kernel_weight:
    :param summary:
    :return:
    """
    print('detection model is set')
    det_model=nn.detection_branch(softmax_trainable=softmax_trainable)
    det_model.compile(optimizer=optimizer,
                      loss=create_loss_cell(times=times),
                      metrics=['accuracy'])
    if summary==True:
        det_model.summary()
    return det_model


def cls_model_compile(nn, times, optimizer, summary=False):
    """

    :param det_loss_weight:
    :param kernel_weight:
    :param summary:
    :return:
    """
    print('classification model is set')
    cls_model=nn.classification_branch(trainable=False)
    # cls_model.load_weights(load_weights, by_name=True)
    cls_model.compile(optimizer=optimizer,
                      loss=create_loss_cell(times))
    if summary==True:
        cls_model.summary()
    return cls_model


def joint_model_compile(nn, times,
                        load_weights, optimizer, summary=False):
    """
    :param nn: network
    :param det_loss_weight: detection weight for joint loss.
    :param cls_loss_in_joint: cls weight for joint loss.
    :param joint_loss_weight: regularizer for classification loss part.
    :param load_weights: load weights for this model.
    :param optimizer: optimizer for this model.
    :param summary: if print model summary
    :return:
    """
    print('classification model is set')
    joint_model=nn.joint_branch()
    print(joint_model.output_names)
    # joint_model.load_weights(load_weights, by_name=True)
    joint_model.compile(optimizer=optimizer,
                        loss=[create_loss_cell(times), create_loss_cell(times)])
    # ,loss_weights={'output_dectection': 0.5, 'output_classification': 0.5}
    if summary==True:
        joint_model.summary()
    return joint_model

if __name__ == '__main__':
    CROP_SIZE = 256
    TRAIN_STEP_PER_EPOCH = 20
    EPOCHS = 500
    NUM_TO_CROP, NUM_TO_AUG = 20, 10

    network = SFCNnetwork(input_shape=(CROP_SIZE, CROP_SIZE, 3))
    optimizer = Adam(lr=1e-4)
    cls_model = cls_model_compile(nn=network,
                                  optimizer=optimizer,times=5)
    cls_model.summary()