# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2018/12/23

import os

from keras.optimizers import SGD

os.chdir('../../')    # to the top directory
from experiments.unet_vanilla.data_process import *
from utils.utils_process.common_utils import seconds_regulate
from models.SFCN_OPI import tune_loss_weight, SFCNnetwork, joint_model_compile
import keras.callbacks
import time
import shutil
from experiments.constant import const

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# define the function
PREDICT_NUM = 200
HEIGHT = WIDTH = const.HEIGHT
MODEL_NAME = 'SFCN-OPI'
TASK = 'cell_recognition'
DATASET_NAME = const.DATASET_NAME

LOSS_INFO = 'joint_loss'
METRICS_NAME = 'metrics.txt'

MODEL_INFO = MODEL_NAME + '_' + TASK + '_' + LOSS_INFO
VISUALIZE_PATH = os.path.join('visualization', DATASET_NAME, MODEL_INFO)
SAVE_WEIGHTS = os.path.join('weights', MODEL_INFO + '.hdf5')
LOG_DIR = os.path.join(VISUALIZE_PATH, 'log_dir')
DRAWCIRCLE_PATH_SAVE = os.path.join(VISUALIZE_PATH, 'result_visualization')

print('model info:' + MODEL_INFO) # normalized print function

DATASET_PATH = const.DATASET_PATH


train_path = os.path.join(DATASET_PATH, 'train')
valid_path = os.path.join(DATASET_PATH, 'valid')
test_path = os.path.join(DATASET_PATH, 'test')
predict_save_path = os.path.join(test_path, 'predict')

if os.path.isdir(LOG_DIR):
    shutil.rmtree(LOG_DIR, True)

start_time = time.time()
train_data_gen_args = const.GENERATOR_PARAMS
valid_data_gen_args = dict()
# data augmentation
myGeneTrain = trainGenerator(4, train_path, 'image', 'label',
                             train_data_gen_args,target_size=(HEIGHT, WIDTH), crop_pattern='4_size')  # ,save_to_dir = r'data/cell/train/aug/'
myGeneValid = trainGenerator(4, valid_path, 'image', 'label', valid_data_gen_args,target_size=(HEIGHT, WIDTH), crop_pattern='None')

# model:
weights = tune_loss_weight()
network = SFCNnetwork(l2_regularizer=weights[-1],input_shape=(256,256,3))
optimizer = SGD(lr=1e-4, momentum=0.9, decay=1e-6, nesterov=True)
joint_model = joint_model_compile(nn=network, det_loss_weight=weights[0], cls_loss_in_joint=weights[2],
                                  joint_loss_weight=weights[3],  optimizer=optimizer,
                                  load_weights=None,summary=True)

model_checkpoint = keras.callbacks.ModelCheckpoint(SAVE_WEIGHTS, monitor='val_loss', verbose=1, save_best_only=True,
                                                   save_weights_only=True)
model_earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
tbcallbacks = keras.callbacks.TensorBoard(log_dir=LOG_DIR,
                                          histogram_freq=0,
                                          write_graph=True,
                                          write_images=True)

history = joint_model.fit_generator(myGeneTrain, steps_per_epoch=121, epochs=500, validation_data=myGeneValid,
                              validation_steps=41, callbacks=[model_checkpoint, tbcallbacks,model_earlystopping])

elapse = time.time() - start_time

print('train time used : %s seconds' % (seconds_regulate(elapse)))  # seconds