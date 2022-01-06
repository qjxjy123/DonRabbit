import os

from keras.optimizers import SGD

from models.SFCN_OPI import joint_model_compile, SFCNnetwork, tune_loss_weight

os.chdir('../../')
import time
from utils.utils_process.common_utils import seconds_regulate
from models.unet_vanilla import unet
from experiments.unet_vanilla.data_process import testGenerator, saveResult
import shutil
from experiments.constant import const

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# define the function
PREDICT_NUM = const.PREDICT_NUM
MODEL_NAME = 'SFCN-OPI'
TASK = 'cell_recognition'
DATASET_NAME = const.DATASET_NAME

LOSS_INFO = 'joint_loss'
METRICS_NAME = 'metrics.txt'
IMAGE_SIZE = 256


MODEL_INFO = MODEL_NAME + '_' + TASK + '_' + LOSS_INFO
VISUALIZE_PATH = os.path.join('visualization', DATASET_NAME, MODEL_INFO)
SAVE_WEIGHTS = os.path.join('weights', MODEL_INFO + '.hdf5')
LOG_DIR = os.path.join(VISUALIZE_PATH, 'log_dir')
DRAWCIRCLE_PATH_SAVE = os.path.join(VISUALIZE_PATH, 'result_visualization')

print('model info:' + MODEL_INFO) # normalized print function
# TASK_DICT = TASK.split('_')[0]+'_' + str(IMAGE_SIZE)
DATASET_PATH = const.DATASET_PATH


train_path = os.path.join(DATASET_PATH, 'train')
valid_path = os.path.join(DATASET_PATH, 'valid')
test_path = os.path.join(DATASET_PATH, 'test')
predict_save_path = os.path.join(test_path, 'predict')

if os.path.isdir(predict_save_path):
    shutil.rmtree(predict_save_path, True)

start_time = time.time()
print('MODEL_INFO = ', MODEL_INFO)

testGene = testGenerator(test_path,crop_mode=const.MODE)
# model
# model:
weights = tune_loss_weight()
network = SFCNnetwork(l2_regularizer=weights[-1], input_shape=(256,256,3))
optimizer = SGD(lr=1e-4, momentum=0.9, decay=1e-6, nesterov=True)
joint_model = joint_model_compile(nn=network, det_loss_weight=weights[0], cls_loss_in_joint=weights[2],
                                  joint_loss_weight=weights[3],  optimizer=optimizer,
                                  load_weights=None,summary=True)

joint_model.load_weights(SAVE_WEIGHTS)
results = joint_model.predict_generator(testGene, PREDICT_NUM, verbose=1)
saveResult(save_path=predict_save_path, mode=const.MODE, npyfile=results, drawCircle_path=DRAWCIRCLE_PATH_SAVE, metrics_path=VISUALIZE_PATH)

elapse = time.time() - start_time

print('predict time used : %s ' % (seconds_regulate(elapse)))  # seconds
