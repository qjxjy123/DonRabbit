from imgaug import augmenters as iaa
import numpy as np
import dataManager as dm
import cv2
import os

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
            yield (imgs_cropped_resized[i], masks_cropped_resized[i],masks_cropped_resized[i])
















# def crop_on_fly(img, det_mask, cls_mask, crop_size):
#     """
#     Crop image randomly on each training batch
#     """
#     imgcrop = ImageCropping()
#     cropped_img, cropped_det_mask, cropped_cls_mask = imgcrop.crop_image_batch(img, [det_mask, cls_mask], desired_shape=(crop_size, crop_size))
#     return cropped_img, cropped_det_mask, cropped_cls_mask
#
# def aug_on_fly(img, det_mask, cls_mask):
#     """Do augmentation with different combination on each training batch
#     """
#     def image_basic_augmentation(image, masks, ratio_operations=0.9):
#         # without additional operations
#         # according to the paper, operations such as shearing, fliping horizontal/vertical,
#         # rotating, zooming and channel shifting will be apply
#         sometimes = lambda aug: iaa.Sometimes(ratio_operations, aug)
#         hor_flip_angle = np.random.uniform(0, 1)
#         ver_flip_angle = np.random.uniform(0, 1)
#         seq = iaa.Sequential([
#             sometimes(
#                 iaa.SomeOf((0, 5), [
#                 iaa.Fliplr(hor_flip_angle),
#                 iaa.Flipud(ver_flip_angle),
#                 iaa.Affine(shear=(-16, 16)),
#                 iaa.Affine(scale={'x': (1, 1.6), 'y': (1, 1.6)}),
#                 iaa.PerspectiveTransform(scale=(0.01, 0.1))
#             ]))
#         ])
#         det_mask, cls_mask = masks[0], masks[1]
#         seq_to_deterministic = seq.to_deterministic()
#         aug_img = seq_to_deterministic.augment_images(image)
#         aug_det_mask = seq_to_deterministic.augment_images(det_mask)
#         aug_cls_mask = seq_to_deterministic.augment_images(cls_mask)
#         return aug_img, aug_det_mask, aug_cls_mask
#
#     aug_image, aug_det_mask, aug_cls_mask = image_basic_augmentation(image=img, masks=[det_mask, cls_mask])
#     return aug_image, aug_det_mask, aug_cls_mask
#
#
# def generator_with_aug(features, det_labels, cls_labels, batch_size, crop_size,
#                        type,
#                        crop_num=15, aug_num=10):
#     """
#     generator with basic augmentations which have been in the paper.
#     :param features: image.
#     :param det_labels: detection mask as label
#     :param cls_labels: classification mask as label
#     :param batch_size: batch size
#     :param crop_size: default size is 64
#     :param type: type must be one of detection, classification or joint
#     :param crop_num: how many cropped image for a single image.
#     :param aug_num: num of augmentation per cropped image
#     """
#     assert type in ['detection', 'classification', 'joint']
#     batch_features = np.zeros((batch_size * crop_num * aug_num, crop_size, crop_size, 3))
#     batch_det_labels = np.zeros((batch_size * crop_num * aug_num, crop_size, crop_size, 2))
#     batch_cls_labels = np.zeros((batch_size * crop_num * aug_num, crop_size, crop_size, 5))
#     while True:
#         counter = 0
#         for i in range(batch_size):
#             index = np.random.choice(features.shape[0], 1)
#             for j in range(crop_num):
#                 feature_index = features[index]
#                 det_label_index = det_labels[index]
#                 cls_label_index = cls_labels[index]
#                 feature, det_label, cls_label = crop_on_fly(feature_index,
#                                                             det_label_index,
#                                                             cls_label_index,
#                                                             crop_size = crop_size)
#                 for k in range(aug_num):
#                     aug_feature, aug_det_label, aug_cls_label = aug_on_fly(feature, det_label, cls_label)
#                     batch_features[counter] = aug_feature
#                     batch_det_labels[counter] = aug_det_label
#                     batch_cls_labels[counter] = aug_cls_label
#                     counter = counter + 1
#         if type == 'detection':
#             yield batch_features, batch_det_labels
#         elif type == 'classification' or type == 'joint':
#             yield batch_features, batch_cls_labels
#
#
# def generator_without_aug(features, det_labels, cls_labels, batch_size, crop_size,
#                           crop_num = 25):
#     """
#     generator without any augmentation, only randomly crop image into [64, 64, channel].
#     :param features: image.
#     :param det_labels: detection mask as label
#     :param cls_labels: classification mask as label
#     :param batch_size: batch size
#     :param crop_size: default size is 64
#     :param crop_num: how many cropped image for a single image.
#     """
#     batch_features = np.zeros((batch_size * crop_num, crop_size, crop_size, 3))
#     batch_det_labels = np.zeros((batch_size * crop_num, crop_size, crop_size, 2))
#     batch_cls_labels = np.zeros((batch_size * crop_num, crop_size, crop_size, 5))
#     while True:
#         counter = 0
#         for i in range(batch_size):
#             index = np.random.choice(features.shape[0], 1)
#             for j in range(crop_num):
#                 feature, det_label, cls_label = crop_on_fly(features[index],
#                                                             det_labels[index], cls_labels[index], crop_size=crop_size)
#                 batch_features[counter] = feature
#                 batch_det_labels[counter] = det_label
#                 batch_cls_labels[counter] = cls_label
#                 counter += 1
#         yield batch_features, {'Detection_output': batch_det_labels,
#                                'Classification_output': batch_cls_labels,
#                                'Joint_output': batch_cls_labels}
#
#
#
#
# class ImageCropping:
#     def __init__(self, data_path = None, old_filename = None, new_filename = None):
#         self.data_path = data_path
#         self.old_filename = '{}/{}'.format(data_path, old_filename)
#         self.new_filename = '{}/{}'.format(data_path, new_filename)
#         dm.check_directory(self.new_filename)
#         dm.initialize_train_test_folder(self.new_filename)
#
#
#     @staticmethod
#     def crop_image_batch(image, masks=None, if_mask=True, if_det = True, if_cls = True,
#                          origin_shape=(500, 500), desired_shape=(64, 64)):
#         assert image.ndim == 4
#         ori_width, ori_height = origin_shape[0], origin_shape[1]
#         des_width, des_height = desired_shape[0], desired_shape[1]
#
#         max_x = ori_width - des_width
#         max_y = ori_height - des_height
#         ran_x = np.random.randint(0, max_x)
#         ran_y = np.random.randint(0, max_y)
#         cropped_x = ran_x + des_width
#         cropped_y = ran_y + des_height
#         cropped_img = image[:, ran_x:cropped_x, ran_y:cropped_y]
#         if if_mask and masks is not None:
#             if if_det and if_cls:
#                 det_mask = masks[0]
#                 cls_mask = masks[1]
#                 cropped_det_mask = det_mask[:, ran_x:cropped_x, ran_y:cropped_y]
#                 cropped_cls_mask = cls_mask[:, ran_x:cropped_x, ran_y:cropped_y]
#                 return cropped_img, cropped_det_mask, cropped_cls_mask
#             elif if_det and not if_cls:
#                 det_mask = masks
#                 cropped_det_mask = det_mask[:, ran_x:cropped_x, ran_y:cropped_y]
#                 return cropped_img, cropped_det_mask
#             elif if_cls and not if_det:
#                 cls_mask = masks
#                 cropped_cls_mask = cls_mask[:, ran_x:cropped_x, ran_y:cropped_y, :]
#                 return cropped_img, cropped_cls_mask
#         else:
#             return cropped_img
#
#     @staticmethod
#     def crop_image(image, masks=None, if_mask=True, if_det = True, if_cls = True,
#                    origin_shape=(500, 500), desired_shape=(64, 64)):
#         assert image.ndim == 3
#         ori_width, ori_height = origin_shape[0], origin_shape[1]
#         des_width, des_height = desired_shape[0], desired_shape[1]
#
#         max_x = ori_width - des_width
#         max_y = ori_height - des_height
#         ran_x = randint(0, max_x)
#         ran_y = randint(0, max_y)
#         cropped_x = ran_x + des_width
#         cropped_y = ran_y + des_height
#         cropped_img = image[ran_x:cropped_x, ran_y:cropped_y]
#         if if_mask and masks is not None:
#             if if_det and if_cls:
#                 det_mask = masks[0]
#                 cls_mask = masks[1]
#                 cropped_det_mask = det_mask[ran_x:cropped_x, ran_y:cropped_y]
#                 cropped_cls_mask = cls_mask[ran_x:cropped_x, ran_y:cropped_y]
#                 return cropped_img, cropped_det_mask, cropped_cls_mask
#             elif if_det and not if_cls:
#                 det_mask = masks[0]
#                 cropped_det_mask = det_mask[ran_x:cropped_x, ran_y:cropped_y]
#                 return cropped_img, cropped_det_mask
#             elif if_cls and not if_det:
#                 cls_mask = masks[0]
#                 cropped_cls_mask = cls_mask[ran_x:cropped_x, ran_y:cropped_y]
#                 return cropped_img, cropped_cls_mask
#         else:
#             return cropped_img
#
# def load_data(data_path, type, det=True, cls=False, reshape_size=None):
#     path = os.path.join(data_path, type)  # cls_and_det/train
#     imgs, det_masks, cls_masks = [], [], []
#     for i, divide_files in enumerate(os.listdir(path)):
#         for j in range(len(os.listdir(os.path.join(path,divide_files[0])))):
#             img_path = os.path.join(path,'image','{}.png'.format(j))
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             if reshape_size is not None:
#                 img = cv2.resize(img, reshape_size)
#             img = _image_normalization(img)
#             imgs.append(img)
#
#             cls_mask_path = os.path.join(path,'image','{}.png'.format(j))
#
#
#
#             det_mask_path = os.path.join(path, 'label_orig', '{}.png'.format(j))
#             det_mask = cv2.imread(det_mask_path)
#             det_mask = cv2.cvtColor(det_mask, cv2.COLOR_BGR2RGB)
#
#             det_masks.append(det_mask)
#
#
#             elif 'detection.bmp' in img_file and det == True:
#                 det_mask_path = os.path.join(path, file, img_file)
#                 #det_mask = skimage.io.imread(det_mask_path, True).astype(np.bool)
#                 det_mask = cv2.imread(det_mask_path, 0)
#                 if reshape_size is not None:
#                     det_mask = cv2.resize(det_mask, reshape_size)
#                 det_masks.append(det_mask)
#             elif 'classification.bmp' in img_file and cls == True:
#                 if cls == True:
#                     cls_mask_path = os.path.join(path, file, img_file)
#                     cls_mask = cv2.imread(cls_mask_path, 0)
#                     if reshape_size != None:
#                         cls_mask = cv2.resize(cls_mask, reshape_size)
#                     cls_masks.append(cls_mask)
#     return np.array(imgs), np.array(det_masks), np.array(cls_masks)
#
# # def _image_normalization(image):
# #     img = image / 255.
# #     img -= np.mean(img, keepdims=True)
# #     img /= (np.std(img, keepdims=True) + 1e-7)
# #     return img