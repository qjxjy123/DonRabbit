# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2019/3/1

from skimage.feature import peak_local_max
import numpy as np
from matplotlib import image
import copy
import os
import scipy.misc
from utils.utils_process.common_utils import save_to_file, getSomeFile
from skimage import filters
import shutil

def sub_local_maxima_process(files, save_path='DownNoise/',useGaussian=False):
    '''
    :param files: 需要进行局部最大值处理的文件
    :param save_path: 处理后的结果存放的文件夹
    :param useGaussian: 是否使用高斯模糊
    :return:
    '''
    if os.path.isdir(save_path):
        shutil.rmtree(save_path, True)

    for file in files:
        img = image.imread(file)
        # img[img<0.5] = 0
        if useGaussian:
            img = filters.gaussian(img, sigma=2)
        a = []
        temp = np.zeros_like(img)
        for i in range(img.shape[2]):
            temp1 = np.zeros_like(img)
            b = peak_local_max(copy.deepcopy(img[:, :, i]) , threshold_rel=0.5, min_distance=5)
            for n in range(b.shape[0]):
                temp[b[n][0], b[n][1], i] = temp1[b[n][0], b[n][1], i] = img[b[n][0], b[n][1], i]

            a.append(b)
        _, orig_name = os.path.split(file)
        name = orig_name.replace('_predict.png', '_localMask.png')

        save_to_file(image=temp, name=name, path=save_path)

# path='data/cell_256/test/predict'
def local_maxima_process(path='data/cell_256/test/predict/',save_path='DownNoise/',useGaussian=False):
    '''
    :param path: 需要取局部极值的文件夹
    :param save_path: 处理后的结果存放的文件夹
    :param useGaussian:是否使用高斯模糊
    :return: None
    '''
    files = getSomeFile(path, '_predict.png')
    sub_local_maxima_process(files, save_path=save_path, useGaussian=useGaussian)
    print('local maxima process finished.')


if __name__ == '__main__':
    local_maxima_process('data/cell/test/predict/')
