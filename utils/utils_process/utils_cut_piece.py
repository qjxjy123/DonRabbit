import os
import copy
from utils.utils_process.common_utils import write_message, save_to_file, getSomeFile, visualize_4cto3c
import re
from matplotlib import image
import numpy as np

def crop_to_25(image):
    # effective area 0-199 200-399 400-599 600-799 800-999
    scratches = [0, 172, 372, 572, 744]
    crops = [256, 256, 256, 256, 256]
    sub_images = crop_to_squared_pieces(image, scratches, crops)
    return sub_images



def save_subimages(path = 'cell_1000/test'):

    type = '.png'
    files = getSomeFile(path, type)
    message = ''
    n = 0
    for file in files:
        img = image.imread(file)
        _, orig_name = os.path.split(file)
        sub_images = crop_to_25(img)
        for key in sub_images:
            # print(orig_name,'\t',key,'\t',n)
            message = message + orig_name + '\t' + key + '\t' + str(n) + '\n'
            save_to_file(sub_images[key][0], str(n) + '.png', os.path.join(path, 'crop_img'))
            n = n + 1
    write_message(message, os.path.join(path, 'mapping_2.txt'))

def crop_to_squared_pieces(image,scratches,crops):

    sub_images = dict()
    # this is a example cropped to 16
    # scratches = [0, 240, 460, 700]
    # crops = [300, 300, 300, 300]
    # intersection 270,500,730
    x = 0  # initial point
    y = 0  # y -> vertical axis

    for i, (scratch_verti, crop_vert) in enumerate(zip(scratches, crops)):
        x = 0
        y = scratch_verti
        for j, (scratch_horiz, crop_horiz) in enumerate(zip(scratches, crops)):
            x = scratch_horiz

            image_crop = image[y:y + crop_vert, x:x + crop_horiz, :]
            center_verti, center_horiz = y + crop_vert // 2, x + crop_horiz // 2
            sub_images['v_%d_h_%d' % (i, j)] = [copy.deepcopy(image_crop), center_verti, center_horiz]
    return sub_images

# used when pieced up
def sensor_piece_effective(light_num, scratches, block_left, block_right):
    # light_num counts from 0
    block_num = len(scratches)-1
    block_whole = block_left + block_right
    if light_num == 0:
        piece_left = 0
    elif light_num == block_num:
        # piece_left = block_whole - (scratches[-1] + block_whole - (scratches[-2] + block_whole) + block_left)
        piece_left = 20
    else:
        piece_left = block_left

    if light_num == block_num:  # last was ignited
        piece_right = block_whole
    else:
        piece_right = block_right
    return np.round(piece_left), np.round(piece_right)

def sensor_point_effective(x,y,light_num,scratches =[0, 215, 430, 645, 744],block_left=20,block_right=236):
    piece_num = len(scratches)
    x_light_num = light_num // piece_num
    y_light_num = light_num % piece_num
    x_left,x_right = sensor_piece_effective(light_num=x_light_num, scratches=scratches, block_left=block_left, block_right=block_right)
    y_left,y_right = sensor_piece_effective(light_num=y_light_num, scratches=scratches, block_left=block_left, block_right=block_right)
    x_flag = False
    y_flag = False
    # 对纵轴进行边缘判断
    if x<=x_right:
        if x_light_num == 0:
            if x_left <= x:
                x_flag = True
        else:
            if x_left < x:
                x_flag = True
    else:
        x_flag = False
    # 对横轴进行边缘判断
    if y <= y_right:
        if y_light_num == 0:
            if y_left <= y:
                y_flag = True
        else:
            if y_left < y:
                y_flag = True
    else:
        y_flag = False

    if x_flag and y_flag:
        return True
    else:
        return False


# used when pieced up
def mapping_to_dict(mapping_file):
    words_list = dict()
    with open(mapping_file, 'r') as f:
        list = f.readlines()
        for line in list:
            words = line.split('\t')
            if len(words) < 2:
                break
            sub_img_num = re.findall('\d+', words[-1])[0]
            words_list[sub_img_num] = words
    return words_list

def mask_pieced_up_9(path_process='DownNoise', path_save='datasets/pan-NETs/1000x1000/test/predict_pieced'):

    scratches = [0, 215, 244]

    for num in np.arange(start=1, stop=12):
        img_num = num * 1000
        for piece_num in range(9):
            ii = img_num + piece_num + 1
            if int(piece_num) % 9 == 0:
                whole_mask = np.zeros([500, 500, 4])
            piece_mask = image.imread(os.path.join(path_process, str(ii) + '_localMask.png'))
            i, j = piece_num // 3, piece_num % 3
            (horiz_start, horiz_end) = sensor_piece_effective(light_num=i, scratches=scratches, block_left=20,
                                                              block_right=236)
            (verti_start, verti_end) = sensor_piece_effective(light_num=j, scratches=scratches, block_left=20,
                                                              block_right=236)
            whole_mask[scratches[i] + horiz_start:scratches[i] + horiz_end,
            scratches[j] + verti_start:scratches[j] + verti_end, :] = \
                piece_mask[horiz_start:horiz_end, verti_start:verti_end, :]
            if int(piece_num) % 9 == 8:
                save_to_file(whole_mask, '%d.png' % num, path_save)
                print('%d.png is pieced up' % num)
    files = getSomeFile(path_save)
    visualize_4cto3c(files)
def mask_pieced_up_BCData(path_process='DownNoise', path_save='datasets/pan-NETs/1000x1000/test/predict_pieced'):

    scratches = [0, 215, 384]

    for num in np.arange(start=1, stop=134):
        img_num = num * 1000
        for piece_num in range(9):
            ii = img_num + piece_num + 1
            if int(piece_num) % 9 == 0:
                whole_mask = np.zeros([640, 640, 3])
            piece_mask = image.imread(os.path.join(path_process, str(ii) + '_localMask.png'))
            i, j = piece_num // 3, piece_num % 3
            (horiz_start, horiz_end) = sensor_piece_effective(light_num=i, scratches=scratches, block_left=20,
                                                              block_right=236)
            (verti_start, verti_end) = sensor_piece_effective(light_num=j, scratches=scratches, block_left=20,
                                                              block_right=236)
            whole_mask[scratches[i] + horiz_start:scratches[i] + horiz_end,
            scratches[j] + verti_start:scratches[j] + verti_end, :] = \
                piece_mask[horiz_start:horiz_end, verti_start:verti_end, :]
            if int(piece_num) % 9 == 8:
                save_to_file(whole_mask, '%d.png' % num, path_save)
                print('%d.png is pieced up' % num)
    files = getSomeFile(path_save)
    visualize_4cto3c(files)

def mask_pieced_up(path_process='DownNoise', path_save='datasets/pan-NETs/1000x1000/test/predict_pieced'):
    # find all masks
    # every 25 masks piece 1 up
    # save into data/cell_1000/test/predict
    # rename and save into data/cell_1000/test/predict/mask_mat
    path_save_mask_mat = os.path.join(path_save, 'mask_mat')
    scratches = [0, 215, 430, 645, 744]
    effective_region = 256
    type = '_localMask.png'
    # files = getSomeFile(path_process,type)
    # int(os.path.split(files[0])[1].replace(type,''))//1000
    for num in np.arange(start=1, stop=42):
    # for num in np.arange(start=14,stop=15):
        img_num = num * 1000
        for piece_num in range(25):
            ii = img_num + piece_num + 1
            if int(ii) % 25 == 1:
                whole_mask = np.zeros([1000, 1000, 3])
            piece_mask = image.imread(os.path.join(path_process, str(ii) + '_localMask.png'))

            i, j = piece_num // 5, piece_num % 5
            (horiz_start, horiz_end) = sensor_piece_effective(light_num=i, scratches=scratches, block_left=20,
                                                              block_right=236)
            (verti_start, verti_end) = sensor_piece_effective(light_num=j, scratches=scratches, block_left=20,
                                                              block_right=236)
            # print('i,j=', i, j)
            # print('piece_num = ', piece_num)
            # print('verti_start,verti_end,horiz_start,horiz_start =', verti_start, verti_end, horiz_start, horiz_end)
            # print(scratches)
            # print('whole_mask_location = ', scratches[i] + horiz_start, scratches[i] + horiz_end,
            #       scratches[j] + verti_start, scratches[j] + verti_end)
            whole_mask[scratches[i] + horiz_start:scratches[i] + horiz_end,
            scratches[j] + verti_start:scratches[j] + verti_end, :] = \
                piece_mask[horiz_start:horiz_end, verti_start:verti_end, :]
            if int(piece_num) % 25 == 24:
                save_to_file(whole_mask, '%d.png' % num, path_save)

# used when pieced up
# for point_name in points_TMask:  # represent the num of Masks
#         n = int(point_name.replace('point_', ''))
#         # f = open('mappings_metrics.txt','r')
#         [i, j] = re.findall('\d+', words_list[str(n)][2])
#
#         (horiz_start, horiz_end) = sensor_piece_effective(light_num=i, block_num=3, block_left=30, block_right=270)
#         (verti_start, verti_end) = sensor_piece_effective(light_num=j, block_num=3, block_left=30, block_right=270)
