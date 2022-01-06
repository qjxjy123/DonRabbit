import os
os.chdir('../../')    # to the top directory
from experiments.unet_vanilla.data_process import *
from utils.utils_process.common_utils import seconds_regulate
from models.FCRN_A import FCRN_A
import keras.callbacks
import time
import shutil
from experiments.constant import const

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# define the function
HEIGHT = WIDTH = const.HEIGHT
MODEL_NAME = 'FCRN_A'
TASK = 'cell_recognition'
DATASET_NAME = const.DATASET_NAME

LOSS_INFO = 'mapping_loss'
METRICS_NAME = 'metrics.txt'

MODEL_INFO = MODEL_NAME + '_' + TASK + '_' + LOSS_INFO + '_' + DATASET_NAME
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
myGeneValid = trainGenerator(4, valid_path, 'image', 'label', valid_data_gen_args,target_size=(HEIGHT, WIDTH), crop_pattern='4_size')

model = FCRN_A(img_dim=256,batch_size=4).inference(nb_classes=3,LOSS_INFO=LOSS_INFO)
model_checkpoint = keras.callbacks.ModelCheckpoint(SAVE_WEIGHTS, monitor='loss', verbose=1, save_best_only=True,
                                                   save_weights_only=True)
model_earlystopping = keras.callbacks.EarlyStopping(monitor='loss', patience=25, restore_best_weights=True)
tbcallbacks = keras.callbacks.TensorBoard(log_dir=LOG_DIR,
                                          histogram_freq=0,
                                          write_graph=True,
                                          write_images=True)

history = model.fit_generator(myGeneTrain, steps_per_epoch=121, epochs=500, validation_data=myGeneValid,
                              validation_steps=41, callbacks=[model_checkpoint, tbcallbacks,model_earlystopping])

# history = model.fit_generator(myGeneTrain, steps_per_epoch=121, epochs=500,
#                               callbacks=[model_checkpoint, tbcallbacks,model_earlystopping])
elapse = time.time() - start_time

print('train time used : %s seconds' % (seconds_regulate(elapse)))  # seconds

