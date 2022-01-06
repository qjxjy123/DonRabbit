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
from utils.utils_process.common_utils import save_to_file, getSomeFile, write_message
import copy
from skimage.transform import resize
import time
import datetime
np.set_printoptions(threshold=509000)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="rgb",
                   mask_color_mode="rgb", image_save_prefix="image", mask_save_prefix="mask",
                   save_to_dir=None, target_size=(1000, 1000), seed=1, resize_pattern='fixed_256',
                   crop_pattern='4_size'):


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
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        imgs_cropped_resized, masks_cropped_resized = crop_2_4_and_resize(img=img, mask=mask, crop_pattern=crop_pattern)
        for i in range(len(imgs_cropped_resized)):
            yield (imgs_cropped_resized[i], masks_cropped_resized[i])


def crop_2_4_and_resize(img, mask, crop_pattern):
    # 1000x1000 None resize
    imgs_crop, masks_crop = crop_imgs_masks_sync8(img, mask)
    return imgs_crop, masks_crop


def crop_imgs_masks_sync8(image, mask, crop_num=8, crop_sz=256):
    imgs_crop, masks_crop = [], []
    img_sz = image.shape[1]
    for i in range(crop_num):
        if img_sz > crop_sz:
            random_arr = np.random.randint(low=0, high=img_sz - crop_sz, size=2)
            y = int(random_arr[0])
            x = int(random_arr[1])
            h = crop_sz
            w = crop_sz
            imgs_crop.append(image[:, y:y + h, x:x + w, :])
            masks_crop.append(mask[:, y:y + h, x:x + w, :])
        else:
            imgs_crop.append(image)
            masks_crop.append(mask)
    return imgs_crop, masks_crop


def adjustData(img, mask):
    img = img / 255
    mask = mask / 255
    return img, mask




def center_crop(x, center_crop_size=[256, 256], **kwargs):
    centerw, centerh = x.shape[1] // 2, x.shape[2] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[:,centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh,:]

# def testGenerator(test_path, crop_mode, num_image=10):
#     image_datagen = ImageDataGenerator(**dict())
#     image_folder = 'img'
#     test_generator = image_datagen.flow_from_directory(
#         test_path,
#         classes=[image_folder],
#         class_mode=None,
#         color_mode='rgb',
#         target_size=[1000, 1000],
#         batch_size=1,
#         seed=1)
#     for img_num, img in enumerate(test_generator):
#         img = img / 255
#         if crop_mode == 'alex':
#             path, orig_name = os.path.split(file)
#             mask_name = os.path.join('label_orig/', orig_name)
#             mask_file = os.path.join(path, mask_name)
#             # mask_file = file.replace('_test.png','.png')
#             origMask = image.imread(mask_file)
#
#             # center_crop process
#             img_center = center_crop(copy.deepcopy(img))
#             mask_center = center_crop(copy.deepcopy(origMask))
#
#             i = int(orig_name.replace('.png', '')) * 1000
#             save_to_file(img_center, '%d_image.png' % i, os.path.join(test_path, 'predict/'))
#             save_to_file(mask_center, '%d_mask.png' % i, os.path.join(test_path, 'predict/'))
#
#             img_center = img_center.reshape(1, img_center.shape[0], img_center.shape[1], img_center.shape[2])
#             yield img_center
def testGenerator_sequence(test_path, crop_mode,predict_num, target_size=1000,mask_color_mode='rgba'):
    print(test_path)
    image_datagen = ImageDataGenerator(**dict())
    image_folder = 'image'
    mask_folder = 'label_orig'
    image_generator = image_datagen.flow_from_directory(
        test_path,
        classes=[image_folder],
        class_mode=None,
        color_mode='rgb',
        target_size=[target_size, target_size],
        batch_size=1,
        shuffle=False,
        seed=1)
    mask_generator = image_datagen.flow_from_directory(
        test_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=[target_size, target_size],
        batch_size=1,
        shuffle=False,
        seed=1)
    file_names = []
    test_generator = zip(image_generator, mask_generator)
    for img_num, (img, mask) in enumerate(test_generator):
        img, mask = img / 255, mask / 255
        if not img_num < predict_num:
            break
        if crop_mode == 'alex':
            # center_crop process
            img_center = center_crop(copy.deepcopy(img))
            mask_center = center_crop(copy.deepcopy(mask))
            file_num = int(os.path.split(image_generator.filenames[img_num])[1].replace('.png', ''))
            image_name = '%d_image.png' % (file_num * 1000)
            mask_name = '%d_mask.png' % (file_num * 1000)
            predict_name = '%d_predict.png' % (file_num * 1000)
            file_names.append(predict_name)
        if crop_mode == 'mosaic':
            imgs_25 = crop_to_25(copy.deepcopy(img))
            masks_25 = crop_to_25(copy.deepcopy(mask))

            for i, _ in enumerate(imgs_25,start=1):
                file_num = int(os.path.split(image_generator.filenames[img_num])[1].replace('.png', ''))
                image_name = '%d_image.png' % (file_num * 1000 + i)
                mask_name = '%d_mask.png' % (file_num * 1000 + i)
                predict_name = '%d_predict.png' % (file_num * 1000 + i)

                file_names.append(predict_name)

                # print('i = ', i, ' file_num = ', file_num)
                # print('predict_name = ', predict_name)
                # print('file_names_num = ',len(file_names),'predict_num = ',predict_num)
    # print(file_names)
    return(file_names)

def testGenerator(test_path, crop_mode, num_image=10,target_size = 1000, mask_color_mode='rgba'):
    image_datagen = ImageDataGenerator(**dict())
    image_folder = 'image'
    mask_folder = 'label_orig'
    image_generator = image_datagen.flow_from_directory(
        test_path,
        classes=[image_folder],
        class_mode=None,
        color_mode='rgb',
        target_size=[target_size, target_size],
        batch_size=1,
        shuffle=False,
        seed=1)
    mask_generator = image_datagen.flow_from_directory(
        test_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=[target_size, target_size],
        batch_size=1,
        shuffle=False,
        seed=1)
    test_generator = zip(image_generator,mask_generator)
    for img_num, (img,mask) in enumerate(test_generator):
        img,mask = img / 255, mask / 255
        if crop_mode == 'alex':
            # center_crop process
            img_center = center_crop(copy.deepcopy(img))
            mask_center = center_crop(copy.deepcopy(mask))
            file_num = int(os.path.split(image_generator.filenames[img_num])[1].replace('.png',''))
            image_name ='%d_image.png' % (file_num*1000)
            mask_name = '%d_mask.png' % (file_num*1000)
            save_to_file(img_center[0,:,:,:], image_name, os.path.join(test_path, 'predict/'))
            save_to_file(mask_center[0,:,:,:], mask_name, os.path.join(test_path, 'predict/'))
            yield img_center

        if crop_mode=='mosaic':
            # 25 images crop
            imgs_25 = crop_to_25(copy.deepcopy(img))
            masks_25 = crop_to_25(copy.deepcopy(mask))
            for i, (img_name_dict, mask_name_dict) in enumerate(zip(imgs_25,masks_25), start=1):
                file_num = int(os.path.split(image_generator.filenames[img_num])[1].replace('.png', ''))
                image_name = '%d_image.png' % (file_num * 1000 + i)
                mask_name = '%d_mask.png' % (file_num * 1000 + i)
                # print(imgs_25[img_name_dict].shape)
                save_to_file(imgs_25[img_name_dict][0, :, :, :], image_name, os.path.join(test_path, 'predict/'))
                save_to_file(masks_25[mask_name_dict][0, :, :, :], mask_name, os.path.join(test_path, 'predict/'))
                yield imgs_25[img_name_dict]



def saveResult(save_path, mode, npyfile, drawCircle_path, metrics_path,predict_names=None):
    # save predict_images
    # files = getSomeFile(save_path, '_image.png')
    # for file, item in zip(files, npyfile):
    #     _, orig_name = os.path.split(file)
    #     i = int(orig_name.replace('_image.png', ''))
    #     img = item
    #     img[img < 0] = 0
    #     img[img > 1] = 1
    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    #     scipy.misc.toimage(img).save(os.path.join(save_path, "%d_predict.png" % i))

    for i,item in enumerate(npyfile):
        img = item
        img[img<0] = 0
        img[img>1] = 1
        save_to_file(image=img,name=predict_names[i],path=save_path)


    # process predict images for detection
    local_maxima_process(path=save_path)

    downNoise_startTime = time.time()
    downNoise(distance=4)
    downNoise_endTime = time.time()
    downNoise_duration = downNoise_endTime - downNoise_startTime
    print('downNoise_duration = ', downNoise_duration)
    #
    # calculate the metrics for predict
    calculate(message_dir=metrics_path, path_pieces_TMask=save_path, path_pieces_PMask='DownNoise', mode=mode)
    calculate_duration = time.time() - downNoise_endTime
    print('calculate_duration = ', calculate_duration)
    # drawCircle_save(path_readPredict=save_path,path_save=drawCircle_path, d=7, mode=mode)
