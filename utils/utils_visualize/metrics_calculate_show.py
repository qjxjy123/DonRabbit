# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2019/3/1

from matplotlib import image
import numpy as np
import munkres
import os
import copy
np.set_printoptions(suppress=True)
from utils.utils_process.common_utils import getSomeFile, write_message
from utils.utils_process.utils_cut_piece import sensor_point_effective
from utils.utils_visualize.DrawCircle import print_pic

def getMaskPoints(path, files):
    # path is predict_masks' path ,files is truth_masks' files
    points_TMask = dict()
    points_PMask = dict()
    for file in files:
        file_TMask = file
        _, orig_name = os.path.split(file)
        i = int(orig_name.replace('_mask.png', ''))

        mask1 = image.imread(file_TMask)

        mask1_dict = dict()
        point_dict = dict()
        red_num = 0
        blue_num = 0
        green_num = 0
        whole_num = 0

        for u in range(mask1.shape[0]):
            for v in range(mask1.shape[1]):
                for c in range(mask1.shape[2]):
                    if mask1[u, v, c] > 0:
                        if c == 0:
                            red_num = red_num + 1
                        elif c == 1:
                            green_num += 1
                        else:
                            blue_num += 1
                        point_dict[str(whole_num)] = (u, v, c)
                        whole_num += 1


        mask1_dict['dict_point'] = point_dict
        mask1_dict['num_0'] = red_num
        mask1_dict['num_1'] = green_num
        mask1_dict['num_2'] = blue_num
        mask1_dict['num_whole'] = whole_num

        points_TMask['point_%d' % i] = mask1_dict

        # get Predict Mask points
        _, orig_name = os.path.split(file)
        file_PMask = os.path.join(path, orig_name.replace('_mask.png', '_mask.png'))
        mask2 = image.imread(file_PMask)
        mask2_dict = dict()
        point_dict = dict()
        red_num = 0
        blue_num = 0
        green_num = 0
        whole_num = 0

        for u in range(mask2.shape[0]):
            for v in range(mask2.shape[1]):
                for c in range(mask2.shape[2]):
                    if mask2[u, v, c] > 0:
                        if c == 0:
                            red_num = red_num + 1
                        elif c == 1:
                            green_num += 1
                        else:
                            blue_num += 1
                        point_dict[str(whole_num)] = (u, v, c)
                        whole_num += 1

        mask2_dict['dict_point'] = point_dict
        mask2_dict['num_0'] = red_num
        mask2_dict['num_1'] = green_num
        mask2_dict['num_2'] = blue_num
        mask2_dict['num_whole'] = whole_num

        points_PMask['point_%d' % i] = mask2_dict

    return points_TMask, points_PMask


def metrics_calculate(mode='alex', path1='data/cell_256/test/predict/',  # Ground Truth Mask
                      path2='DownNoise/',  # Predict Mask
                      type1='_mask.png'):
    confusion_matrixes = dict()
    files = getSomeFile(path1, type1)

    points_TMask, points_PMask = getMaskPoints(path2, files)
    metrics_all = dict()
    mm = munkres.Munkres()
    flags = dict()
    if mode=='alex':
        for point_name in points_TMask:  # represent the num of Masks
            n = int(point_name.replace('point_', ''))
            confusion_matrix = np.zeros([4, 4])
            # for i in range(3):
            #     for j in range(3):
            flag = False
            num_t = points_TMask['point_' + str(n)]['num_whole']
            num_p = points_PMask['point_' + str(n)]['num_whole']

            # solve exceptional condition, which is the num for one of the mask is zero
            if num_t == 0 or num_p == 0:
                if num_t == 0:
                    for u in range(num_p):
                        x2, y2, c2 = points_PMask['point_' + str(n)]['dict_point'][str(u)]
                        confusion_matrix[3, c2] += 1
                if num_p == 0:
                    for u in range(num_t):
                        x1, y1, c1 = points_TMask['point_' + str(n)]['dict_point'][str(u)]
                        confusion_matrix[c1, 3] += 1
                continue

            if num_t > num_p:  # promise munkres will be working
                flag = True

            if flag == False:
                aa = np.zeros([num_t, num_p])
                for u in range(num_t):
                    for v in range(num_p):
                        x1, y1, c1 = points_TMask['point_' + str(n)]['dict_point'][str(u)]
                        x2, y2, c2 = points_PMask['point_' + str(n)]['dict_point'][str(v)]
                        d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
                        aa[u, v] = d
            else:
                aa = np.zeros([num_p, num_t])
                for u in range(num_p):
                    for v in range(num_t):
                        x1, y1, c1 = points_PMask['point_' + str(n)]['dict_point'][str(u)]
                        x2, y2, c2 = points_TMask['point_' + str(n)]['dict_point'][str(v)]
                        d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
                        aa[u, v] = d
            bb = mm.pad_matrix(matrix=aa, pad_value=999)
            cc = mm.compute(copy.deepcopy(bb))

            # confusion matrix calculation

            if flag == False:
                for xx, yy in cc:

                    x2, y2, c2 = points_PMask['point_' + str(n)]['dict_point'][str(yy)]

                    if xx >= num_t:  # greater than small num
                        confusion_matrix[3, c2] += 1
                        continue

                    x1, y1, c1 = points_TMask['point_' + str(n)]['dict_point'][str(xx)]

                    d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

                    if d < 16:
                        confusion_matrix[c1, c2] += 1
                    else:
                        confusion_matrix[c1, 3] += 1
                        confusion_matrix[3, c2] += 1

            else:
                for yy, xx in cc:
                    x1, y1, c1 = points_TMask['point_' + str(n)]['dict_point'][str(xx)]
                    if yy >= num_p:  # greater than small num
                        confusion_matrix[c1, 3] += 1
                        continue
                    x2, y2, c2 = points_PMask['point_' + str(n)]['dict_point'][str(yy)]
                    d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
                    if d < 16:
                        confusion_matrix[c1, c2] += 1
                    else:
                        confusion_matrix[c1, 3] += 1
                        confusion_matrix[3, c2] += 1

            confusion_matrixes['matrix_' + str(n)] = confusion_matrix
            print('confusion_matrix %d has completed.' % (n))
    if mode == 'mosaic':
        for point_name in points_TMask:  # represent the num of Masks
            n = int(point_name.replace('point_', ''))
            confusion_matrix = np.zeros([4, 4])
            # for i in range(3):
            #     for j in range(3):
            flag = False
            num_t = points_TMask['point_' + str(n)]['num_whole']
            num_p = points_PMask['point_' + str(n)]['num_whole']

            # solve exceptional condition, which is the num for one of the mask is zero
            if num_t == 0 or num_p == 0:
                if num_t == 0:
                    for u in range(num_p):
                        x2, y2, c2 = points_PMask['point_' + str(n)]['dict_point'][str(u)]
                        confusion_matrix[3, c2] += 1
                if num_p == 0:
                    for u in range(num_t):
                        x1, y1, c1 = points_TMask['point_' + str(n)]['dict_point'][str(u)]
                        confusion_matrix[c1, 3] += 1
                continue

            num_max = num_p
            if num_t > num_p:  # promise munkres will be working
                num_max = num_t

            aa = np.full([num_max, num_max], 999)
            for u in range(num_t):
                for v in range(num_p):
                    x1, y1, c1 = points_TMask['point_' + str(n)]['dict_point'][str(u)]
                    x2, y2, c2 = points_PMask['point_' + str(n)]['dict_point'][str(v)]
                    d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
                    aa[u, v] = d

            bb = mm.pad_matrix(matrix=aa, pad_value=999)
            cc = mm.compute(copy.deepcopy(bb))
            # 纵列是num_t,横行是num_p
            # confusion matrix calculation
            for xx, yy in cc:

                if xx >= num_t and yy < num_p:  # greater than small num
                    x2, y2, c2 = points_PMask['point_' + str(n)]['dict_point'][str(yy)]
                    if sensor_point_effective(x=x2, y=y2, light_num=n % 1000 - 1):
                        confusion_matrix[3, c2] += 1
                        # print_pic(None, points_PMask['point_' + str(n)]['dict_point'][str(yy)], pic_num=n, cell_num=100)
                    continue
                elif xx < num_t and yy >= num_p:  # greater than small num
                    x1, y1, c1 = points_TMask['point_' + str(n)]['dict_point'][str(xx)]
                    if sensor_point_effective(x=x1, y=y1, light_num=n % 1000 - 1):
                        confusion_matrix[c1, 3] += 1
                        # print_pic(points_TMask['point_' + str(n)]['dict_point'][str(xx)], None, pic_num=n,cell_num=cell_num)
                    continue
                elif xx >= num_t and yy >= num_p:
                    continue
                x1, y1, c1 = points_TMask['point_' + str(n)]['dict_point'][str(xx)]
                x2, y2, c2 = points_PMask['point_' + str(n)]['dict_point'][str(yy)]
                d = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

                if sensor_point_effective(x=x1, y=y1, light_num=n % 1000 - 1):
                    # print_pic(points_TMask['point_' + str(n)]['dict_point'][str(xx)],points_PMask['point_' + str(n)]['dict_point'][str(yy)], pic_num=n, cell_num=cell_num)
                    if d < 16:
                        confusion_matrix[c1, c2] += 1
                    else:
                        confusion_matrix[c1, 3] += 1
                        confusion_matrix[3, c2] += 1

            confusion_matrixes['matrix_' + str(n)] = confusion_matrix
            print('confusion_matrix %d has completed.' % (n))
    return confusion_matrixes


# calculate metrics(acc/recall/f1) in one confusion_matrix
'''
in confusion matrix where coordinate x represent the ground-truth point's category
coordinate y represent the predict point's category
x==y (x belongs to 0,1,2 ; y belongs to 0,1,2) m[x,x]++
x!=y 1. x belongs to 0,1,2 ; y belongs to 0,1,2 d(x,y)<thresh  m[x,y]++
     2. x belongs to 0,1,2 ; y belongs to 0,1,2 d(x,y)>=thresh m[3,y]++ and m[x,3]++
     3. x belongs to 0,1,2 && no y match or on the contrary    m[x,3]++ or m[3,y]++
'''
def sub_metrics_calculate(confusion_matrix):
    metrics = dict()
    precision = np.zeros([3])
    recall = np.zeros([3])
    f1_score = np.zeros([3])
    whole = [2809, 145087, 37877]
    for i in range(3):
        if confusion_matrix[i, i] != 0:
            precision[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
            recall[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
            f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    correct_num = np.sum([confusion_matrix[i, i] for i in range(3)])

    total_precision = np.sum([whole[i] / np.sum(whole) * precision[i] for i in range(3)])
    total_recall = np.sum([whole[i] / np.sum(whole) * recall[i] for i in range(3)])
    total_f1 = np.sum([whole[i] / np.sum(whole) * f1_score[i] for i in range(3)])

    # total_precision = correct_num / np.sum(confusion_matrix[:, 0:3])
    # total_recall = correct_num / np.sum(confusion_matrix[0:3, :])
    # total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    #
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1_score
    metrics['total_precision'] = total_precision
    metrics['total_recall'] = total_recall
    metrics['total_f1'] = total_f1
    return metrics


def metrics_all_calculate(confusion_matrixes):
    confusion_matrix = np.zeros([4, 4])
    for index in confusion_matrixes:
        confusion_matrix += confusion_matrixes[index]
    print(confusion_matrix)
    return sub_metrics_calculate(confusion_matrix), confusion_matrix

# this function is for 1000x1000x3 images
# def print_every_matrix_file(confusion_matrices):
#     with open('mapping_matrix.txt','r') as f:
#
#         for index in confusion_matrices:
#             num = int(index.replace('matrix_',''))
#




def calculate(message_dir,
              mode, path_pieces_TMask='data/cell_256/test/predict/',
              path_pieces_PMask = 'DownNoise/'
              ):
    # message_dir = 'observation/1000x1000x3/metrics.txt'
    matrices = metrics_calculate(mode=mode, path1=path_pieces_TMask, path2=path_pieces_PMask)

    print('All confusion matrices complete ,mode="%s"' % mode)
    for matrix_name in matrices:
        write_message(matrices[matrix_name], file=os.path.join(message_dir, '%s.txt' % matrix_name))
    metrics, confusion_matrix = metrics_all_calculate(matrices)
    write_message(message=confusion_matrix, file=os.path.join(message_dir, 'metrics.txt'))
    print(metrics)
    write_message(message=metrics, file=os.path.join(message_dir, 'metrics.txt'))


if __name__ == '__main__':
    # calculate(message_dir='observation/unet_vanilla_cell_detection_mse/',path_pieces_TMask='data/cell/test/predict/')
    pass