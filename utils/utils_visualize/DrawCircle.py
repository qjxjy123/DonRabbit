import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import scipy.misc
import munkres
import os
import copy
from skimage.draw import circle, circle_perimeter, circle_perimeter_aa
from utils.utils_process.common_utils import save_to_file, getSomeFile
from utils.utils_process.utils_cut_piece import mask_pieced_up, sensor_piece_effective, mask_pieced_up_9, \
    mask_pieced_up_BCData


def drawCircle_save_contrast(path_readPredict='data/cell/test/predict/', path_readDownNoise='DownNoise/',
                    path_save='observation/', d=7):
    '''
    1.get the files of orignal image,predict_downn_noise,mask
    2.As to Mask,draw circle
    3.As to predict_down_noise,draw point
    4.save to %d_ob.png
    '''
    files = getSomeFile(path_readPredict, '.png')
    for file in files:
        _, orig_name = os.path.split(file)
        n = int(orig_name.replace('.png', ''))

        path_predict = path_readPredict
        path_downNoise = path_readDownNoise

        path_observation = os.path.join(path_save, 'observation_%d/' % d)
        img_name = os.path.join(path_predict, str(n) + '.png')
        # read the num
        img = image.imread(img_name)
        mask_name = os.path.join(path_predict, 'label_orig', str(n) + '.png')
        mask = image.imread(mask_name)
        predict_name = os.path.join(path_downNoise, str(n) + '.png')
        predict = image.imread(predict_name)

        img_new = np.zeros_like(img)
        img_new = img
        for i in range(img_new.shape[0]):
            for j in range(img_new.shape[1]):
                for c in range(img_new.shape[2]):
                    if mask[i, j, c] > 0:
                        rr, cc = circle_perimeter(i, j, d, shape=[img_new.shape[0], img_new.shape[0]])
                        img_new[rr, cc, c] = 1
                        img_new[rr, cc, (c + 1) % 3] = 0
                        img_new[rr, cc, (c + 2) % 3] = 0
                    if predict[i, j, c] > 0:
                        rr, cc = circle(i, j, 2, shape=[img_new.shape[0], img_new.shape[0]])
                        img_new[rr, cc, c] = 1
                        img_new[rr, cc, (c + 1) % 3] = 0
                        img_new[rr, cc, (c + 2) % 3] = 0

        #         for i in range(img_new.shape[0]):
        #             for j in range(img_new.shape[1]):
        #                 for c in range(img_new.shape[2]):
        #                     if img_new[i,j,c] == 0:
        #                         img_new[i,j,c] = img[i,j,c]

        save_to_file(img_new, str(n) + '_ob.png', path_observation)
def print_pic(point_T,point_P,pic_num, cell_num):
    if point_T != None:
        x1, y1, c1 = point_T
    if point_P != None:
        x2, y2, c2 = point_P
    piece_num = pic_num % 25
    path_observation = 'visualization/pan-NETs_1000x1000/cell_calculate'
    d = 7
    if cell_num == 1:
        # 获取predict文件中的image和mask文件
        path_predict = 'datasets/pan-NETs/1000x1000/test'
        path_downNoise = 'DownNoise'
        img_name = os.path.join(path_predict, 'predict', str(pic_num) + '_image.png')
        img = image.imread(img_name)
        img_new = img
        if point_T != None:
            rr, cc = circle_perimeter(x1, y1, d, shape=[img_new.shape[0], img_new.shape[0]])
            img_new[rr, cc, c1] = 1
            img_new[rr, cc, (c1 + 1) % 3] = 0
            img_new[rr, cc, (c1 + 2) % 3] = 0
        if point_P != None:
            rr, cc = circle(x2, y2, 2, shape=[img_new.shape[0], img_new.shape[0]])
            img_new[rr, cc, c2] = 1
            img_new[rr, cc, (c2 + 1) % 3] = 0
            img_new[rr, cc, (c2 + 2) % 3] = 0
    else:
        img = image.imread(os.path.join(path_observation,str(pic_num) + '_ob_pieced_effective_%d_cell.png' % (cell_num-1)))
        img_new = img
        if point_T != None:
            rr, cc = circle_perimeter(x1, y1, d, shape=[img_new.shape[0], img_new.shape[0]])
            img_new[rr, cc, c1] = 1
            img_new[rr, cc, (c1 + 1) % 3] = 0
            img_new[rr, cc, (c1 + 2) % 3] = 0
        if point_P != None:
            rr, cc = circle(x2, y2, 2, shape=[img_new.shape[0], img_new.shape[0]])
            img_new[rr, cc, c2] = 1
            img_new[rr, cc, (c2 + 1) % 3] = 0
            img_new[rr, cc, (c2 + 2) % 3] = 0
    save_to_file(img_new, str(pic_num) + '_ob_pieced_effective_%d_cell.png' % cell_num, path_observation)
    save_to_file(img_new, str(pic_num) + '.png', path_observation)

def drawCircle_save_contrast_pieced(path_readPredict='data/cell/test/predict/', path_readDownNoise='DownNoise/',
                    path_save='observation/', d=7):
    '''
    1.get the files of orignal image,predict_downn_noise,mask
    2.As to Mask,draw circle
    3.As to predict_down_noise,draw point
    4.save to %d_ob.png
    '''
    files = getSomeFile(path_readDownNoise, '.png')
    for file in files:
        _, orig_name = os.path.split(file)
        n = int(orig_name.replace('_localMask.png', ''))

        path_predict = path_readPredict
        path_downNoise = path_readDownNoise

        path_observation = os.path.join(path_save, 'observation_%d/' % d)
        img_name = os.path.join(path_predict,'predict', str(n) + '_image.png')
        # read the num
        img = image.imread(img_name)
        mask_name = os.path.join(path_predict, 'predict', str(n) + '_mask.png')
        mask = image.imread(mask_name)
        predict_name = os.path.join(path_downNoise, str(n) + '_localMask.png')
        predict = image.imread(predict_name)

        img_new = np.zeros_like(img)
        img_new = img
        for i in range(img_new.shape[0]):
            for j in range(img_new.shape[1]):
                for c in range(img_new.shape[2]):
                    if mask[i, j, c] > 0:
                        rr, cc = circle_perimeter(i, j, d, shape=[img_new.shape[0], img_new.shape[0]])
                        img_new[rr, cc, c] = 1
                        img_new[rr, cc, (c+1)%3] = 0
                        img_new[rr, cc, (c+2)%3] = 0

                    if predict[i, j, c] > 0:
                        rr, cc = circle(i, j, 2, shape=[img_new.shape[0], img_new.shape[0]])
                        img_new[rr, cc, c] = 1
                        img_new[rr, cc, (c + 1) % 3] = 0
                        img_new[rr, cc, (c + 2) % 3] = 0
        #         for i in range(img_new.shape[0]):
        #             for j in range(img_new.shape[1]):
        #                 for c in range(img_new.shape[2]):
        #                     if img_new[i,j,c] == 0:
        #                         img_new[i,j,c] = img[i,j,c]
        piece_num = n % 1000 - 1
        i, j = piece_num // 5, piece_num % 5
        scratches = [0, 215, 430, 645, 744]
        (horiz_start, horiz_end) = sensor_piece_effective(light_num=i, scratches=scratches, block_left=20,
                                                          block_right=236)
        (verti_start, verti_end) = sensor_piece_effective(light_num=j, scratches=scratches, block_left=20,
                                                          block_right=236)
        img_new = img_new[horiz_start:horiz_end, verti_start:verti_end, :]
        save_to_file(img_new, str(n) + '_ob_pieced_effective.png', path_observation)

def drawCircle_save_contrast_256(path_readPredict='data/cell/test/predict/', path_readDownNoise='DownNoise/',
                    path_save='observation/', d=7):
    '''
    1.get the files of orignal image,predict_downn_noise,mask
    2.As to Mask,draw circle
    3.As to predict_down_noise,draw point
    4.save to %d_ob.png
    '''
    files = getSomeFile(path_readDownNoise, '.png')
    for file in files:
        _, orig_name = os.path.split(file)
        n = int(orig_name.replace('_localMask.png', ''))

        path_predict = path_readPredict
        path_downNoise = path_readDownNoise

        path_observation = os.path.join(path_save, 'observation_%d/' % d)
        img_name = os.path.join(path_predict,'predict', str(n) + '_image.png')
        # read the num
        img = image.imread(img_name)
        mask_name = os.path.join(path_predict, 'predict', str(n) + '_mask.png')
        mask = image.imread(mask_name)
        predict_name = os.path.join(path_downNoise, str(n) + '_localMask.png')
        predict = image.imread(predict_name)

        img_new = np.zeros_like(img)
        img_new = img
        for i in range(img_new.shape[0]):
            for j in range(img_new.shape[1]):
                for c in range(img_new.shape[2]):
                    if mask[i, j, c] > 0:
                        rr, cc = circle_perimeter(i, j, d, shape=[img_new.shape[0], img_new.shape[0]])
                        if c != 2:
                            img_new[rr, cc, c] = 1
                        else:
                            img_new[rr, cc, 0] = 0
                            img_new[rr, cc, 1] = 0
                            img_new[rr, cc, 2] = 0
                    if predict[i, j, c] > 0:
                        rr, cc = circle(i, j, 2, shape=[img_new.shape[0], img_new.shape[0]])

                        if c != 2:
                            img_new[rr, cc, c] = 1
                        else:
                            img_new[rr, cc, 0] = 0
                            img_new[rr, cc, 1] = 0
                            img_new[rr, cc, 2] = 0
        #         for i in range(img_new.shape[0]):
        #             for j in range(img_new.shape[1]):
        #                 for c in range(img_new.shape[2]):
        #                     if img_new[i,j,c] == 0:
        #                         img_new[i,j,c] = img[i,j,c]

        save_to_file(img_new, str(n) + '_ob.png', path_observation)

def drawCircle_save_predict(path_readPredict='data/cell/test/predict/', path_readDownNoise='DownNoise/',
                    path_save='observation/', d=5):
    '''
    1.get the files of orignal image,predict_downn_noise,mask
    2.As to Mask,draw circle
    3.As to predict_down_noise,draw point
    4.save to %d_ob.png
    '''
    files = getSomeFile(path_readPredict, '_image.png')
    for file in files:
        path_predict = path_readPredict
        path_downNoise = path_readDownNoise
        path_observation = path_save
        img_name = file
        # read the num
        img = image.imread(img_name)

        predict_name = os.path.join(path_downNoise, os.path.split(img_name.replace('_image.png', '_localMask.png'))[1])
        # predict_name = os.path.join(path_downNoise,os.path.split(img_name)[1])
        predict = image.imread(predict_name)

        img_new = np.zeros_like(img)
        img_new = img
        for i in range(img_new.shape[0]):
            for j in range(img_new.shape[1]):
                for c in range(img_new.shape[2]):
                    if predict[i, j, c] > 0:
                        rr, cc = circle(i, j, d, shape=[img_new.shape[0], img_new.shape[0]])

                        if c == 0:
                            img_new[rr, cc, 0] = 1
                            img_new[rr, cc, 1] = 1
                            img_new[rr, cc, 2] = 0
                        elif c == 1:
                            img_new[rr, cc, 0] = 1
                            img_new[rr, cc, 1] = 0
                            img_new[rr, cc, 2] = 0
                        elif c == 2:
                            img_new[rr, cc, 0] = 0
                            img_new[rr, cc, 1] = 1
                            img_new[rr, cc, 2] = 0
        save_to_file(img_new, os.path.split(file.replace('.png', '') + '_ob.png')[1], path_observation)

def drawCircle_save(path_readPredict,path_save, path_readDownNoise='DownNoise/',mode='alex', d=7,lens=1025):
    # mode='256_contrast'

    if mode == 'alex':
        drawCircle_save_predict(path_readPredict=path_readPredict, path_readDownNoise=path_readDownNoise,
                                 path_save=path_save, d=d)
    elif mode == 'mosaic':

        path_pieced_up = os.path.join(path_readPredict,'predict_pieced')
        path_visualized = os.path.join(path_pieced_up,'visualized')
        if lens == 1025:
            mask_pieced_up(path_process=path_readDownNoise,path_save=path_pieced_up)
            drawCircle_save_contrast_pieced(path_readPredict=path_readPredict, path_readDownNoise=path_pieced_up,
                                            path_save=path_visualized, d=d)
        elif lens == 99:
            mask_pieced_up_9(path_process=path_readDownNoise,path_save=path_pieced_up)
        elif lens == 1197:
            mask_pieced_up_BCData(path_process=path_readDownNoise,path_save=path_pieced_up)
        # path_readMask = path_readPredict.replace('predict','label_orig')
        # drawCircle_save_contrast(path_readPredict=path_readMask,path_readDownNoise=path_pieced_up,path_save=path_save, d=d)
    elif mode == '256_contrast':
        path_readPredict = 'datasets/pan-NETs/1000x1000/test'
        path_readDownNoise = 'DownNoise/'
        drawCircle_save_contrast_pieced(path_readPredict=path_readPredict,path_readDownNoise=path_readDownNoise,path_save=path_save, d=d)

if __name__ == '__main__':
    drawCircle_save(path_readPredict='data/process_test/true',path_readDownNoise='data/process_test/predict_1',path_save='data/process_test/save_orig')
    drawCircle_save(path_readPredict='data/process_test/true', path_readDownNoise='data/process_test/predict_2',
                    path_save='data/process_test/save_plus')