# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2019/4/21
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
from matplotlib import image
import skimage.transform as trans
import scipy.misc
from utils.utils_visualize.resultProcess_plus import local_maxima_process
from utils.utils_visualize.DownNoise_simple import downNoise
from utils.utils_visualize.metrics_calculate import calculate
from utils.utils_visualize.DrawCircle import drawCircle_save
from utils.utils_process.utils_imageProcess import crop_to_four_with_cropSize, my_random_size_crop, \
     my_4size_crop, my_3size_crop, my_random_16_size_crop, img_resize_fixed_4d, my_random_location_crop, crop_to_25
from utils.utils_process.common_utils import save_to_file, getSomeFile,write_message
import copy
from skimage.transform import resize
import time
import datetime


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="rgb",
                   mask_color_mode="rgb", image_save_prefix="image", mask_save_prefix="mask",
                   save_to_dir=None, target_size=(1000, 1000), seed=1, resize_pattern='fixed_256', crop_pattern='4_size'):
                   # , crop_pattern='4_size'
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    if target_size[0] == 500:
        for (img, mask) in train_generator:
            img, mask = adjustData(img, mask)
            # img_random_crop = mask_random_crop = None
            if crop_pattern == 'None':
                while True:
                    np.random.seed()
                    i = np.random.randint(1000, size=1)
                    i = np.squeeze(i)
                    img_random_crop, mask_random_crop = my_random_location_crop(copy.deepcopy(img),i), my_random_location_crop(copy.deepcopy(mask), i)
                    if img_random_crop.shape == mask_random_crop.shape:
                        break
                    else:
                        print('something wrong happened')
                        message = str(datetime.datetime.now()) + '/n' + 'img_shape' + str(
                            img_random_crop.shape) + '/t' + 'mask_shape' + str(
                            mask_random_crop.shape) + '/n' + 'sync_seed' + str(i) + '/n'
                        write_message(message=message, file='log_dir/log.txt')
            else:
                while True:
                    np.random.seed()
                    i = np.random.randint(1000, size=1)
                    i = np.squeeze(i)
                    if crop_pattern == '16_times':
                        img_random_crop, mask_random_crop = my_random_16_size_crop(copy.deepcopy(img),i), my_random_16_size_crop(copy.deepcopy(mask), i)
                    elif crop_pattern == '4_size':
                        img_random_crop, mask_random_crop = my_4size_crop(copy.deepcopy(img), i), my_4size_crop(
                            copy.deepcopy(mask), i)
                    elif crop_pattern == '3_size':
                        img_random_crop, mask_random_crop = my_3size_crop(copy.deepcopy(img), i), my_3size_crop(
                            copy.deepcopy(mask), i)
                    elif crop_pattern == 'random_size':
                        img_random_crop, mask_random_crop = my_random_size_crop(copy.deepcopy(img), i), my_random_size_crop(copy.deepcopy(mask), i)
                    else:
                        img_random_crop, mask_random_crop = copy.deepcopy(img), copy.deepcopy(mask)
                    if img_random_crop.shape == mask_random_crop.shape:
                        break
                    else:
                        print('something wrong happened')
                        message = str(datetime.datetime.now()) + '/n' + 'img_shape' + str(
                            img_random_crop.shape) + '/t' + 'mask_shape' + str(
                            mask_random_crop.shape) + '/n' + 'sync_seed' + str(i) + '/n'
                        write_message(message=message, file='log_dir/log.txt')
            yield (img_resize_fixed_4d(img_random_crop, resized_size=256), img_resize_fixed_4d(mask_random_crop, resized_size=256))
    if target_size[0] == 1000:
        for (img, mask) in train_generator:
            img, mask = adjustData(img, mask)
            # img_random_crop = mask_random_crop = None
            if crop_pattern == 'None':
                while True:
                    np.random.seed()
                    i = np.random.randint(1000, size=1)
                    i = np.squeeze(i)
                    img_random_crop, mask_random_crop = my_random_location_crop(copy.deepcopy(img),
                                                                                i), my_random_location_crop(
                        copy.deepcopy(mask), i)
                    if img_random_crop.shape == mask_random_crop.shape:
                        break
                    else:
                        print('something wrong happened')
                        message = str(datetime.datetime.now()) + '/n' + 'img_shape' + str(
                            img_random_crop.shape) + '/t' + 'mask_shape' + str(
                            mask_random_crop.shape) + '/n' + 'sync_seed' + str(i) + '/n'
                        write_message(message=message, file='log_dir/log.txt')
            else:
                while True:
                    np.random.seed()
                    i = np.random.randint(1000, size=1)
                    i = np.squeeze(i)
                    if crop_pattern == '16_times':
                        img_random_crop, mask_random_crop = my_random_16_size_crop(copy.deepcopy(img),
                                                                                   i), my_random_16_size_crop(
                            copy.deepcopy(mask), i)
                    elif crop_pattern == '4_size':
                        img_random_crop, mask_random_crop = my_4size_crop(copy.deepcopy(img), i), my_4size_crop(
                            copy.deepcopy(mask), i)
                    elif crop_pattern == '3_size':
                        img_random_crop, mask_random_crop = my_3size_crop(copy.deepcopy(img), i), my_3size_crop(
                            copy.deepcopy(mask), i)
                    elif crop_pattern == 'random_size':
                        img_random_crop, mask_random_crop = my_random_size_crop(copy.deepcopy(img),
                                                                                i), my_random_size_crop(
                            copy.deepcopy(mask), i)
                    else:
                        img_random_crop, mask_random_crop = copy.deepcopy(img), copy.deepcopy(mask)
                    if img_random_crop.shape == mask_random_crop.shape:
                        break
                    else:
                        print('something wrong happened')
                        message = str(datetime.datetime.now()) + '/n' + 'img_shape' + str(
                            img_random_crop.shape) + '/t' + 'mask_shape' + str(
                            mask_random_crop.shape) + '/n' + 'sync_seed' + str(i) + '/n'
                        write_message(message=message, file='log_dir/log.txt')
            yield (img_resize_fixed_4d(img_random_crop, resized_size=256),
                   img_resize_fixed_4d(mask_random_crop, resized_size=256))

def adjustData(img, mask):
    img = img / 255
    mask = mask / 255
    return img, mask

def center_crop(x, center_crop_size=[256, 256], **kwargs):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh]


def testGenerator(test_path, crop_mode, num_image=10):
    files = getSomeFile(test_path, '.png')
    if crop_mode=='alex':
        for file in files:
            img = image.imread(file)  # from 0 to 1
            path, orig_name = os.path.split(file)
            mask_name = os.path.join('label_orig/', orig_name)
            mask_file = os.path.join(path, mask_name)
            # mask_file = file.replace('_test.png','.png')
            origMask = image.imread(mask_file)

            # center_crop process
            img_center = center_crop(copy.deepcopy(img))
            mask_center = center_crop(copy.deepcopy(origMask))

            i = int(orig_name.replace('.png', '')) * 1000
            save_to_file(img_center, '%d_image.png' % i, os.path.join(test_path, 'predict/'))
            save_to_file(mask_center, '%d_mask.png' % i, os.path.join(test_path, 'predict/'))

            img_center = img_center.reshape(1, img_center.shape[0], img_center.shape[1], img_center.shape[2])
            yield img_center

            # four_corner_crop process
            img_four = crop_to_four_with_cropSize(copy.deepcopy(img), crop_sz=256)
            mask_four = crop_to_four_with_cropSize(copy.deepcopy(origMask), crop_sz=256)
            # resize  (don't know which interpolation it use.
            for j, (subimg, submask) in enumerate(zip(img_four, mask_four)):
                ii = i + j + 1
                save_to_file(subimg, '%d_image.png' % ii, os.path.join(test_path, 'predict/'))
                save_to_file(submask, '%d_mask.png' % ii, os.path.join(test_path, 'predict/'))
                subimg = subimg.reshape(1, subimg.shape[0], subimg.shape[1], subimg.shape[2])
                yield subimg

    if crop_mode=='mosaic':
        for file in files:
            img = image.imread(file)  # from 0 to 1
            path, orig_name = os.path.split(file)
            mask_name = os.path.join('label_orig/', orig_name)
            mask_file = os.path.join(path, mask_name)
            # mask_file = file.replace('_test.png','.png')
            origMask = image.imread(mask_file)

            # center_crop process
            i = int(orig_name.replace('.png', '')) * 1000


            # 25 cropped patches process
            imgs_25 = crop_to_25(copy.deepcopy(img))
            masks_25 = crop_to_25(copy.deepcopy(origMask))
            # resize  (don't know which interpolation it use.
            for j, (subimg, submask) in enumerate(zip(imgs_25, masks_25)):
                ii = i + j + 1
                save_to_file(subimg, '%d_image.png' % ii, os.path.join(test_path, 'predict/'))
                save_to_file(submask, '%d_mask.png' % ii, os.path.join(test_path, 'predict/'))
                subimg = subimg.reshape(1, subimg.shape[0], subimg.shape[1], subimg.shape[2])
                yield subimg



def saveResult(save_path, mode, npyfile, drawCircle_path, metrics_path):
    # save predict_images
    files = getSomeFile(save_path, '_image.png')
    for file, item in zip(files, npyfile):
        _, orig_name = os.path.split(file)
        i = int(orig_name.replace('_image.png', ''))
        img = item
        img[img < 0] = 0
        img[img > 1] = 1
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        scipy.misc.toimage(img).save(os.path.join(save_path, "%d_predict.png" % i))

    # process predict images for detection
    local_maxima_process(path=save_path)

    downNoise_startTime = time.time()
    downNoise()
    downNoise_endTime = time.time()
    downNoise_duration = downNoise_endTime - downNoise_startTime
    print('downNoise_duration = ', downNoise_duration)

    # calculate the metrics for predict
    calculate(message_dir=metrics_path, path_pieces_TMask=save_path)
    calculate_duration = time.time() - downNoise_endTime
    print('calculate_duration = ', calculate_duration)
    # drawCircle_save(path_readPredict=save_path, path_save=drawCircle_path, d=12)
