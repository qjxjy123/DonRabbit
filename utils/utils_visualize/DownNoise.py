# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2019/3/1

import copy
import os
import time

import munkres
import numpy as np
from matplotlib import image

from utils.utils import getMaskPoints_orig
from utils.common_utils import save_to_file, getSomeFile, seconds_regulate

def getPoints(path='DownNoise/',Type='_localMask.png'):

    files = getSomeFile(path=path,type=Type)
    points = getMaskPoints_orig(files,Type)
    #print(points)
    return points

def getImages(files,type='_localMask.png'):
    images = dict()
    for file in files:
        _, orig_name = os.path.split(file)
        i = int(orig_name.replace(type, ''))

        img = image.imread(file)
        images['img_'+str(i)] = img
    return images

def downNoise(path='DownNoise/'):
    # for file in files:
    files = getSomeFile(path=path,type='_localMask.png')
    images = getImages(files)
    img = None
    mm = munkres.Munkres()
    flags = dict()
    HAs = dict()
    points = getPoints(path)
    points_num = dict()
    for point_name in points:
        n = int(point_name.replace('point_',''))
        point_num = dict()
        for i in range(3):
            point_num['num_' + str(i)] = points['point_' + str(n)]['num_' + str(i)]
        points_num['point_' + str(n)] = point_num


    for nn in range(5):
        if nn != 0:
            points = getPoints(path)
        for point_name in points:  # represent the num of Masks
            n = int(point_name.replace('point_',''))
            img = images['img_'+str(n)]
            if nn != 0:
                if points_num['point_' + str(n)]['num_' + str(0)] == points['point_' + str(n)]['num_' + str(0)] and  \
                    points_num['point_' + str(n)]['num_' + str(1)] == points['point_' + str(n)]['num_' + str(1)] and  \
                    points_num['point_' + str(n)]['num_' + str(2)] == points['point_' + str(n)]['num_' + str(2)]:
                    print('epoch_%d ,skip img_%d' %(nn,n))
                    continue
            for i in range(3):
                for j in range(3):
                    flag = False
                    num_1 = points_num['point_' + str(n)]['num_' + str(i)] = points['point_' + str(n)]['num_' + str(i)]
                    num_2 = points_num['point_' + str(n)]['num_' + str(i)] = points['point_' + str(n)]['num_' + str(j)]

                    if num_1 == 0 or num_2 == 0:
                        continue
                    if num_1 > num_2:  # promise munkres will be working
                        flag = True
                        temp = num_2
                        num_2 = num_1
                        num_1 = temp
                    flags['flag_img_%d_T_%d_P_%d' % (n, i, j)] = flag  # reverse flag

                    aa = np.zeros([num_1, num_2])

                    if flag == False:
                        for u in range(num_1):
                            for v in range(num_2):
                                x1, y1 = points['point_' + str(n)]['dict_' + str(i)][str(u)]
                                # print('v = ' , v)
                                x2, y2 = points['point_' + str(n)]['dict_' + str(j)][str(v)]
                                if (x1, y1, i) == (x2, y2, j):
                                    # promise the same channel same point won't effect the result
                                    aa[u, v] = 1000
                                else:
                                    d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
                                    aa[u, v] = d
                        # print(aa)
                        # np.savetxt('DownNoise/aa_%d_%d_%d.txt' % (nn+1,i,j),aa,fmt='%8.3f')
                        # save_to_file(qq,qq,qq)
                    else:
                        for u in range(num_1):
                            for v in range(num_2):
                                x1, y1 = points['point_' + str(n)]['dict_' + str(j)][str(u)]
                                x2, y2 = points['point_' + str(n)]['dict_' + str(i)][str(v)]
                                if (x1, y1, i) == (x2, y2, j):
                                    aa[u, v] = 1000
                                else:
                                    d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
                                    aa[u, v] = d
                    bb = mm.pad_matrix(matrix=aa, pad_value=999)
                    cc = mm.compute(copy.deepcopy(bb))
                    HAs['HA_img_%d_T_%d_P_%d' % (n, i, j)] = cc

                    if flag == False:
                        for xx, yy in cc:
                            if xx >= num_1:  # greater than small num
                                break
                            x1, y1 = points['point_' + str(n)]['dict_' + str(i)][str(xx)]
                            x2, y2 = points['point_' + str(n)]['dict_' + str(j)][str(yy)]
                            d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
                            if d < 10:
                                if img[x1, y1, i] < img[x2, y2, j]:
                                    img[x1, y1, i] = 0
                                else:
                                    img[x2, y2, j] = 0
                    else:
                        pass

            save_to_file(image=img, name=str(n)+'_localMask.png', path=path)
            print('DownNoise complete epoch_%d , img_%d' %(nn,n))

if __name__ == '__main__':
    start_time = time.time()
    downNoise(path='DownNoise/')
    end_time = time.time()
    print('duration is %s' % (str(seconds_regulate(end_time - start_time))))
