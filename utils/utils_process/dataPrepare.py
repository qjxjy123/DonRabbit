# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2018/12/24

# find all file with extension mat
# path example: '/data/cell/train/label_orig/'
import os
import scipy.io as sio
from matplotlib import image
import numpy as np
import scipy.misc
# from utils.utils_process.utils_imageProcess import Mask_val
from .utils_imageProcess import Mask_val
# get MATFiles' names in path 
def getMATFile(path):
    file_orig = os.listdir(path)
    files = []
    for i in file_orig:
        if str.endswith(i, '.mat'):
            i = os.path.join(path, i)
            files.append(i)
    return files

#get Mask_val aug and save 
def getMask_png(path, files):
    # path = os.path.join(path,'label')
    for (num, file) in enumerate(files,start=1):
        # get centers and labels from mat file
        print('file ' + str(num) + ' : ' + file + ' ' + 'is under process')
        pic_info = sio.loadmat(file)

        centers_orig = pic_info['Centers']
        labels = pic_info['Labels']
        centers = np.round(centers_orig).astype(int)
        # create mask picture
        pic_array = np.zeros((1000, 1000, 3))
        for i in range(centers.shape[1]):
            if centers[0, i] == 1000:
                centers[0, i] = 999
            elif centers[0,i] > 1000:
              continue
            elif centers[0,i] < 0:
              continue
            if centers[1, i] == 1000:
                centers[1, i] = 999
            elif centers[1,i] > 1000:
              continue
            elif centers[1,i] < 0:
              continue
            pic_array[centers[0, i], centers[1, i], labels[0, i] - 1] = 1
            
        # get the transpose of the matrix to be consistent with orig_img
        for i in range(3):
            pic_array[:,:,i]=pic_array[:,:,i].T
        
        # create dictionary
        #get Mask_plus matrix
        mask_plus = Mask_val(pic_array,d = 10)
        if not os.path.exists(path):
            os.mkdir(path)
        _,mat_name = os.path.split(file)
        name = os.path.splitext(mat_name)[0] + '.png'
        path_label_orig = path.replace('label','label_orig')
        full_name = os.path.join(path_label_orig, name)
        # print(full_name)
        # print(pic_array.shape)
        # input()
        #image.imsave(full_name, pic_array) # forced to 4 channels
        scipy.misc.toimage(pic_array).save(full_name)
        # save mask_plus
        plus_name = os.path.splitext(mat_name)[0] + '_plus' + '.png'
        full_name_plus = os.path.join(path,plus_name)
        scipy.misc.toimage(mask_plus).save(full_name_plus)

if __name__ == '__main__':

    print(os.getcwd())
    path = os.path.join(os.getcwd(), 'data/cell_153/mat')
    path2 = os.path.join(os.getcwd(), 'data/cell_153')
    files = getMATFile(path)
    print(path2)
    print(files)
    #input()
    # getMask_png(path2, files)
    