# -*- coding:utf-8 -*-
# Created by Xu Jiayu at 2019/5/31
import shutil

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from matplotlib import image
import scipy.misc
from utils.utils_visualize.resultProcess_plus import local_maxima_process
from utils.utils_visualize.DownNoise_simple import downNoise
from utils.utils_visualize.metrics_calculate import calculate
from utils.utils_visualize.DrawCircle import drawCircle_save
from utils.utils_process.common_utils import save_to_file, getSomeFile, write_message
from utils.utils_process.utils_imageProcess import my_random_size_crop, crop_by_center_r, crop_by_center_r_3d,crop_to_25
import copy
from skimage.transform import resize
import time
import datetime
import cv2


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="rgb",
                   mask_color_mode="rgb", image_save_prefix="image", mask_save_prefix="mask",
                   save_to_dir=None, target_size=(500, 500), seed=1):
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
        seed=seed,
        interpolation='bicubic')
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
    for (img_4x, mask_4x) in train_generator:
        img_4x, mask_4x = adjustData(img_4x, mask_4x)
        img_2x = np.zeros([batch_size, 500, 500, 3])
        mask_2x = np.zeros([batch_size, 500, 500, 3])
        img_1x = np.zeros([batch_size, 600, 600, 3])
        mask_1x = np.zeros([batch_size, 600, 600, 3])
        img_2x, mask_2x = copy.deepcopy(img_4x), copy.deepcopy(mask_4x)
        for i in range(batch_size):
            img_1x[i], mask_1x[i] = cv2.copyMakeBorder(img_4x[i], 50, 50, 50, 50,
                                                       cv2.BORDER_REFLECT), cv2.copyMakeBorder(mask_4x[i], 50, 50, 50,
                                                                                               50,
                                                                                               cv2.BORDER_REFLECT)

        # img_random_crop = mask_random_crop = None
        while True:
            np.random.seed()
            i = np.random.randint(1000, size=1)
            i = np.squeeze(i)


            # generate 2x train data
            img_2x_crop, center_x, center_y = my_random_size_crop(copy.deepcopy(img_2x), i, crop_sz=320)
            mask_2x_crop, center_x, center_y = my_random_size_crop(copy.deepcopy(mask_2x), i, crop_sz=320)
            '''
            img_4x_crop_resize = np.zeros([batch_size, 256, 256, 3])
            mask_4x_crop_resize = np.zeros([batch_size, 256, 256, 3])
            for n in range(batch_size):
                img_4x_crop_resize[n] = resize(image=img_4x_crop[n], output_shape=[256, 256], anti_aliasing=True,
                                               mode='reflect')
                mask_4x_crop_resize[n] = resize(image=mask_4x_crop[n], output_shape=[256, 256], anti_aliasing=True,
                                                mode='reflect')
            '''


            # generate 4x train data
            center_x, center_y = center_x, center_y

            img_4x_crop = crop_by_center_r(copy.deepcopy(img_4x), center_x, center_y, 128)
            mask_4x_crop = crop_by_center_r(copy.deepcopy(mask_4x), center_x, center_y, 128)

            img_4x_crop_resize = np.zeros([batch_size, 320, 320, 3])
            mask_4x_crop_resize = np.zeros([batch_size, 320, 320, 3])
            for n in range(batch_size):
                img_4x_crop_resize[n] = resize(image=img_4x_crop[n], output_shape=[320, 320], anti_aliasing=True,
                                               mode='reflect')
                mask_4x_crop_resize[n] = resize(image=mask_4x_crop[n], output_shape=[320, 320], anti_aliasing=True,
                                                mode='reflect')


            # generate 1x train data
            center_x, center_y = center_x * 6 // 5, center_y * 6 // 5
            img_1x_crop = crop_by_center_r(copy.deepcopy(img_1x), center_x, center_y, 194)
            mask_1x_crop = crop_by_center_r(copy.deepcopy(mask_1x), center_x, center_y, 194)
            # assert(img_1x_crop.shape == (batch_size, 388, 388, 3))
            # assert(mask_1x_crop.shape == (batch_size, 388, 388, 3))
            img_1x_crop_resize = np.zeros([batch_size, 320, 320, 3])
            mask_1x_crop_resize = np.zeros([batch_size, 320, 320, 3])
            for n in range(batch_size):
                img_1x_crop_resize[n] = resize(image=img_1x_crop[n], output_shape=[320, 320], anti_aliasing=True,
                                               mode='reflect')
                mask_1x_crop_resize[n] = resize(image=mask_1x_crop[n], output_shape=[320, 320], anti_aliasing=True,
                                                mode='reflect')

            # generate final_ground_truth
            mask_final_ground_truth = copy.deepcopy(mask_4x_crop)
            # circle exit
            if img_2x_crop.shape == mask_2x_crop.shape:
                break
            else:
                print('something wrong happened')
                message = str(datetime.datetime.now()) + '/n' + 'img_shape' + str(
                    img_2x_crop.shape) + '/t' + 'mask_shape' + str(
                    mask_2x_crop.shape) + '/n' + 'sync_seed' + str(i) + '/n'
                write_message(message=message, file='log_dir/log.txt')

        yield ([img_4x_crop_resize, img_2x_crop, img_1x_crop_resize],
               [mask_4x_crop_resize, mask_2x_crop, mask_1x_crop_resize, mask_final_ground_truth])


def adjustData(img, mask):
    img = img / 255
    mask = mask / 255
    return (img, mask)


def center_crop(x, center_crop_size=[256, 256], **kwargs):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh]


def alex_test_crop(x, crop_size=[320, 320], **kwargs):
    '''
    crop center and 4 corner in same size

    :return: cropped images and crop images' center
    '''

    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = crop_size[0] // 2, crop_size[1] // 2
    return [x[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh:], centerw, centerh], \
           [x[:crop_size[0], :crop_size[1], :], crop_size[0] // 2, crop_size[1] // 2], \
           [x[:crop_size[0], -crop_size[1]:, :], crop_size[0] // 2, x.shape[1] - crop_size[1] // 2], \
           [x[-crop_size[0]:, :crop_size[1], :], x.shape[0] - crop_size[0] // 2, crop_size[1] // 2], \
           [x[-crop_size[0]:, -crop_size[1]:, :], x.shape[0] - crop_size[0] // 2, x.shape[1] - crop_size[1] // 2]


def testGenerator(test_path, crop_mode='alex', num_image=10):
    files = getSomeFile(test_path, '.png')
    batch_size = 2
    if crop_mode=='alex':
        for file in files:
            img_4x = image.imread(file)  # from 0 to 1
            path, orig_name = os.path.split(file)
            mask_name = os.path.join('label_orig/', orig_name)
            mask_file = os.path.join(path, mask_name)
            # mask_file = file.replace('_test.png','.png')
            mask_4x = image.imread(mask_file)

            img_2x = np.zeros([batch_size, 500, 500, 3])
            mask_2x = np.zeros([batch_size, 500, 500, 3])
            img_1x = np.zeros([batch_size, 600, 600, 3])
            mask_1x = np.zeros([batch_size, 600, 600, 3])
            img_2x, mask_2x = copy.deepcopy(img_4x), copy.deepcopy(mask_4x)
            img_1x, mask_1x = cv2.copyMakeBorder(img_4x, 50, 50, 50, 50, cv2.BORDER_REFLECT), cv2.copyMakeBorder(
                mask_4x, 50, 50, 50, 50, cv2.BORDER_REFLECT)

            #  generate 2x crop test data
            img_2x_crops = alex_test_crop(copy.deepcopy(img_2x), crop_size=[320, 320])
            mask_2x_crops = alex_test_crop(copy.deepcopy(mask_2x), crop_size=[320, 320])
            j = 0
            for (img_2x_crop_center, mask_2x_crop_center) in zip(img_2x_crops, mask_2x_crops):
                img_2x_crop, center_x, center_y = img_2x_crop_center[0], img_2x_crop_center[1], img_2x_crop_center[2]
                mask_2x_crop, center_x, center_y = mask_2x_crop_center[0], mask_2x_crop_center[1], mask_2x_crop_center[2]
                img_2x_crop_finalTest = center_crop(copy.deepcopy(img_2x_crop), center_crop_size=[256, 256])
                mask_2x_crop_finalGroundTruth = center_crop(copy.deepcopy(mask_2x_crop), center_crop_size=[256, 256])
                # generate 4x crop test data
                center_x, center_y = center_x, center_y
                img_4x_crop = crop_by_center_r_3d(copy.deepcopy(img_4x), center_x, center_y, radius=128)
                img_4x_crop = crop_by_center_r_3d(copy.deepcopy(img_4x), center_x, center_y, radius=128)
                mask_4x_crop = crop_by_center_r_3d(copy.deepcopy(mask_4x), center_x, center_y, radius=128)
                img_4x_crop_resize = resize(image=img_4x_crop, output_shape=[320, 320], anti_aliasing=True, mode='reflect')
                mask_4x_crop_resize = resize(image=mask_4x_crop, output_shape=[320, 320], anti_aliasing=True,
                                             mode='reflect')

                # generate 1x test data
                center_x, center_y = center_x + 50, center_y + 50
                img_1x_crop = crop_by_center_r_3d(copy.deepcopy(img_1x), center_x, center_y, radius=192)
                mask_1x_crop = crop_by_center_r_3d(copy.deepcopy(mask_1x), center_x, center_y, radius=192)
                img_1x_crop_resize = resize(image=img_1x_crop, output_shape=[320, 320], anti_aliasing=True, mode='reflect')
                mask_1x_crop_resize = resize(image=mask_1x_crop, output_shape=[320, 320], anti_aliasing=True,
                                             mode='reflect')

                i = int(orig_name.replace('.png', '')) * 1000
                ii = i + j
                j = j + 1
                # save test input/mask
                save_to_file(img_1x_crop_resize, '%d_image_1x_crop_resize.png' % ii, os.path.join(test_path, 'predict/subsidiary'))
                save_to_file(mask_1x_crop_resize, '%d_mask_1x_crop_resize.png' % ii, os.path.join(test_path, 'predict/subsidiary'))
                save_to_file(img_2x_crop, '%d_image_2x_crop.png' % ii, os.path.join(test_path, 'predict/subsidiary'))
                save_to_file(mask_2x_crop, '%d_mask_2x_crop.png' % ii, os.path.join(test_path, 'predict/subsidiary'))
                save_to_file(img_4x_crop_resize, '%d_image_4x_crop.png' % ii, os.path.join(test_path, 'predict/subsidiary'))
                save_to_file(mask_4x_crop_resize, '%d_mask_4x_crop.png' % ii, os.path.join(test_path, 'predict/subsidiary'))

                save_to_file(img_2x_crop_finalTest, '%d_image.png' % ii, os.path.join(test_path, 'predict'))
                save_to_file(mask_2x_crop_finalGroundTruth, '%d_mask.png' % ii, os.path.join(test_path, 'predict'))

                img_4x_crop_resize = img_4x_crop_resize.reshape(1, img_4x_crop_resize.shape[0], img_4x_crop_resize.shape[1], img_4x_crop_resize.shape[2])
                img_2x_crop = img_2x_crop.reshape(1, img_2x_crop.shape[0], img_2x_crop.shape[1],
                                                                img_2x_crop.shape[2])
                img_1x_crop_resize = img_1x_crop_resize.reshape(1, img_1x_crop_resize.shape[0], img_1x_crop_resize.shape[1],
                                                                img_1x_crop_resize.shape[2])

                yield [img_4x_crop_resize, img_2x_crop, img_1x_crop_resize]
    elif crop_mode=='mosaic':
        img = image.imread(file)  # from 0 to 1
        path, orig_name = os.path.split(file)
        mask_name = os.path.join('label_orig/', orig_name)
        mask_file = os.path.join(path, mask_name)
        # mask_file = file.replace('_test.png','.png')
        origMask = image.imread(mask_file)

        # center_crop process
        i = int(orig_name.replace('.png', '')) * 1000

        # 25 cropped patches process
        img_four = crop_to_25(copy.deepcopy(img))
        mask_four = crop_to_25(copy.deepcopy(origMask))
        # resize  (don't know which interpolation it use.
        for j, (subimg, submask) in enumerate(zip(img_four, mask_four)):
            ii = i + j + 1
            save_to_file(subimg, '%d_image.png' % ii, os.path.join(test_path, 'predict/'))
            save_to_file(submask, '%d_mask.png' % ii, os.path.join(test_path, 'predict/'))
            subimg = subimg.reshape(1, subimg.shape[0], subimg.shape[1], subimg.shape[2])
            yield subimg
        # # four_corner_crop process
        # img_four = crop_to_four_with_cropSize(copy.deepcopy(img), crop_sz=256)
        # mask_four = crop_to_four_with_cropSize(copy.deepcopy(origMask), crop_sz=256)
        # # resize  (don't know which interpolation it use.
        # for j, (subimg, submask) in enumerate(zip(img_four, mask_four)):
        #     ii = i + j + 1
        #     save_to_file(subimg, '%d_image.png' % ii, os.path.join(test_path, 'predict/'))
        #     save_to_file(submask, '%d_mask.png' % ii, os.path.join(test_path, 'predict/'))
        #     subimg = subimg.reshape(1, subimg.shape[0], subimg.shape[1], subimg.shape[2])
        #     yield subimg


def saveResult(save_path,mode, npyfile, times=1, drawCircle_path='observation/', metrics_path='observation/metrics.txt'):
    path_subsidiary = os.path.join(save_path, 'subsidiary')
    # save predict_images
    files = getSomeFile(path_subsidiary, '_image_4x_crop.png')
    for file, im_0, im_1, im_2, item in zip(files, npyfile[0], npyfile[1], npyfile[2], npyfile[3]):
        _, orig_name = os.path.split(file)
        i = int(orig_name.replace('_image_4x_crop.png', ''))
        img = item / times
        im_0, im_1, im_2 = im_0 / times, im_1 / times, im_2 / times
        img[img < 0] = 0
        im_0[im_0 < 0] = 0
        im_1[im_1 < 0] = 0
        im_2[im_2 < 0] = 0

        img[img > 1] = 1
        im_0[im_0 > 1] = 1
        im_1[im_1 > 1] = 1
        im_2[im_2 > 1] = 1


        # print(img.shape)
        # print(npyfile)
        save_to_file(image=img, name="%d_predict.png" % i, path=save_path)
        save_to_file(image=im_0, name="%d_predict_4x_crop.png" % i, path=path_subsidiary)
        save_to_file(image=im_1, name="%d_predict_2x_crop.png" % i, path=path_subsidiary)
        save_to_file(image=im_2, name="%d_predict_1x_crop_resize.png" % i, path=path_subsidiary)

    # process predict images for detection
    local_maxima_process(path=save_path)

    downNoise_startTime = time.time()
    downNoise()
    downNoise_endTime = time.time()
    downNoise_duration = downNoise_endTime - downNoise_startTime
    print('downNoise_duration = ', downNoise_duration)

    # calculate the metrics for predict
    calculate(metrics_path, mode=mode, path_pieces_TMask=save_path)
    calculate_duration = time.time() - downNoise_endTime
    print('calculate_duration = ', calculate_duration)
    drawCircle_save(path_save=drawCircle_path,path_readPredict=save_path, d=12)
