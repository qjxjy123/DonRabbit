import os
os.chdir('../../')
import time
from utils.utils_process.common_utils import seconds_regulate
from models.FCRN_B import FCRN_B
from experiments.unet_vanilla.data_process import testGenerator, saveResult
import shutil
from experiments.constant import const

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(os.getcwd())
PREDICT_NUM = const.PREDICT_NUM
MODEL_NAME = 'FCRN_B'
TASK = 'cell_recognition'
DATASET_NAME = const.DATASET_NAME

LOSS_INFO = 'reverseMapping_loss'
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
model = FCRN_B()
model.load_weights(SAVE_WEIGHTS)
results = model.predict_generator(testGene, PREDICT_NUM, verbose=1)
saveResult(save_path=predict_save_path, mode=const.MODE, npyfile=results, drawCircle_path=DRAWCIRCLE_PATH_SAVE, metrics_path=VISUALIZE_PATH)

elapse = time.time() - start_time

print('predict time used : %s ' % (seconds_regulate(elapse)))  # seconds
