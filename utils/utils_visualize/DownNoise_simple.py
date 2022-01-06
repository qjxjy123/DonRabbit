# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2019/3/1

import copy
import os
import time

import munkres
import numpy as np
from matplotlib import image

from utils.utils_process.common_utils import save_to_file, getSomeFile, seconds_regulate

def deleteSquare(img,point,d):
    (x,y,c) = point
    #temp = img[x,y,c]

    img[x-d:x+d+1,y-d:y+d+1,:] = 0
    #img[x,y,c] = temp

def deleteSphere(img,point,d):
    (center_x,center_y,center_c) = point
    radius = d
    min_x = 0 if center_x - radius < 0 else center_x - radius
    min_y = 0 if center_y - radius < 0 else center_y - radius
    max_x = img.shape[0] if center_x + radius > img.shape[0] else center_x + radius
    max_y = img.shape[1] if center_y + radius > img.shape[1] else center_y + radius
    for i in range(min_x,max_x):
        for j in range(min_y,max_y):
            if d<np.sqrt((center_x-i)**2+(center_y-j)**2):
                pass
            else:
                img[i,j,:]=0

def getImages(files,type='_localMask.png'):
    images = dict()
    for file in files:
        _, orig_name = os.path.split(file)
        img = image.imread(file)
        images[orig_name] = img
    return images

def downNoise(path='DownNoise/',distance=16):
    # for file in files:
    files = getSomeFile(path=path,type='_localMask.png')
    images = getImages(files)
    for img_name in images:
        img = images[img_name]
        img_DownNoise = np.zeros_like(img)
        while(np.max(img)!=0):
            pos = np.unravel_index(np.argmax(img),img.shape)
            # print(pos)
            img_DownNoise[pos]=img[pos]
            deleteSphere(img=img,d=distance,point=pos)
        save_to_file(image=img_DownNoise,name=img_name,path=path)
        print('DownNoise complete, %s' %(img_name))

if __name__ == '__main__':
    start_time = time.time()
    downNoise(path='DownNoise/')
    end_time = time.time()
    print('duration is %s' % (str(seconds_regulate(end_time - start_time))))
