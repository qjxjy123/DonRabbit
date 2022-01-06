import numpy as np
import copy
from skimage.transform import resize
def D(matrix, u, v, d):
    # get the slice matrix
    slice_matrix, u, v = getSliceMatrix(matrix, u, v, d)
    # get the min distance
    min_dis, channel = getMinDistance(slice_matrix, u, v, d)
    return min_dis, channel


def getMinDistance(matrix, u, v, d):
    min_dis = 11.
    channel = -1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(3):
                if matrix[i, j, k] != 0:
                    l = np.sqrt(np.square(u - i) + np.square(j - v))
                    if l < min_dis:
                        min_dis = l
                        channel = k

    return min_dis, channel

def getSliceMatrix(matrix, u, v, d):
    min_x = 0 if u - d < 0 else u - d
    min_y = 0 if v - d < 0 else v - d
    max_x = matrix.shape[0] if u + d > matrix.shape[0] else u + d
    max_y = matrix.shape[1] if v + d > matrix.shape[1] else v + d
    return matrix[min_x:max_x, min_y:max_y, :], u - min_x, v - min_y


def Mask_val(matrix, d=15, alpha=3):
    a = np.zeros_like(matrix)
    for u in range(a.shape[0]):
        for v in range(a.shape[1]):
            min_dis, channel = D(matrix, u, v, d)
            if min_dis < 15:
                m = (np.exp(alpha * (1 - min_dis / d)) - 1) / (np.exp(alpha) - 1)
                a[u, v, channel] = m
    return a


def crop_to_four(image):
    img_sz = image.shape[0]
    crop_sz = image.shape[0] // 2
    y = 0
    x = 0
    h = crop_sz
    w = crop_sz
    image_crop_1 = image[y:y + h, x:x + w, :]
    image_crop_2 = image[h:h + h, x:x + w, :]
    image_crop_3 = image[y:y + h, w:w + w, :]
    image_crop_4 = image[h:h + h, w:w + w, :]
    return (image_crop_1, image_crop_2, image_crop_3, image_crop_4)


def crop_to_four_with_cropSize(image, crop_sz=None):
    if crop_sz == None:
        crop_sz = image.shape[0] // 2
    img_sz = image.shape[0]
    y = 0
    x = 0
    h = crop_sz
    w = crop_sz
    image_crop_1 = image[y:y + h, x:x + w, :]
    image_crop_2 = image[-h:, x:x + w, :]
    image_crop_3 = image[y:y + h, -w:, :]
    image_crop_4 = image[-h:, -w:, :]
    return (image_crop_1, image_crop_2, image_crop_3, image_crop_4)

def crop_to_four_images(image):
    # this function use for downNoise
    # intersection 250
    sub_images = dict()
    scratches = [0, 200]
    crops = [300, 300]
    x = 0  # initial point
    y = 0  # y -> vertical axis

    for i, (scratch_verti, crop_vert) in enumerate(zip(scratches, crops)):
        x = 0
        y = scratch_verti
        for j, (scratch_horiz, crop_horiz) in enumerate(zip(scratches, crops)):
            x = scratch_horiz

            image_crop = image[y:y + crop_vert, x:x + crop_horiz, :]
            sub_images['v_%d_h_%d' % (i, j)] = copy.deepcopy(image_crop)
    return sub_images

def crop_to_four(image):
    img_sz = image.shape[0]
    crop_sz = image.shape[0] // 2
    y = 0
    x = 0
    h = crop_sz
    w = crop_sz
    image_crop_1 = image[y:y + h, x:x + w, :]
    image_crop_2 = image[h:h + h, x:x + w, :]
    image_crop_3 = image[y:y + h, w:w + w, :]
    image_crop_4 = image[h:h + h, w:w + w, :]
    return (image_crop_1, image_crop_2, image_crop_3, image_crop_4)


def crop_to_sixteen(image):
    # intersection 270,500,730
    sub_images = dict()
    scratches = [0, 240, 460, 700]
    crops = [300, 300, 300, 300]
    x = 0  # initial point
    y = 0  # y -> vertical axis

    for i, (scratch_verti, crop_vert) in enumerate(zip(scratches, crops)):
        x = 0
        y = scratch_verti
        for j, (scratch_horiz, crop_horiz) in enumerate(zip(scratches, crops)):
            x = scratch_horiz

            image_crop = image[y:y + crop_vert, x:x + crop_horiz, :]
            sub_images['v_%d_h_%d' % (i, j)] = copy.deepcopy(image_crop)
    return sub_images

# def crop_to_25(image):
#
#     sub_images = dict()
#     scratches = [0, 235, 470, 705, 744]
#     crops = [256, 256, 256, 256, 256]
#     x = 0  # initial point
#     y = 0  # y -> vertical axis
#
#     for i, (scratch_verti, crop_vert) in enumerate(zip(scratches, crops)):
#         x = 0
#         y = scratch_verti
#         for j, (scratch_horiz, crop_horiz) in enumerate(zip(scratches, crops)):
#             x = scratch_horiz
#
#             image_crop = image[y:y + crop_vert, x:x + crop_horiz, :]
#             sub_images['v_%d_h_%d' % (i, j)] = copy.deepcopy(image_crop)
#     return sub_images
def crop_to_9(image,scratches = [0, 215, 244]):

    sub_images = dict()

    crops = [256, 256, 256]
    x = 0  # initial point
    y = 0  # y -> vertical axis

    for i, (scratch_verti, crop_vert) in enumerate(zip(scratches, crops)):
        x = 0
        y = scratch_verti
        for j, (scratch_horiz, crop_horiz) in enumerate(zip(scratches, crops)):
            x = scratch_horiz
            # print(image.shape)
            image_crop = image[:, y:y + crop_vert, x:x + crop_horiz, :]
            sub_images['v_%d_h_%d' % (i, j)] = copy.deepcopy(image_crop)
            # print(sub_images['v_%d_h_%d' % (i, j)].shape)
    return sub_images

def crop_to_25(image):
    sub_images = dict()
    scratches = [0, 215, 430, 645, 744]
    crops = [256, 256, 256, 256, 256]
    x = 0  # initial point
    y = 0  # y -> vertical axis

    for i, (scratch_verti, crop_vert) in enumerate(zip(scratches, crops)):
        x = 0
        y = scratch_verti
        for j, (scratch_horiz, crop_horiz) in enumerate(zip(scratches, crops)):
            x = scratch_horiz

            image_crop = image[:, y:y + crop_vert, x:x + crop_horiz, :]
            sub_images['v_%d_h_%d' % (i, j)] = copy.deepcopy(image_crop)
    return sub_images

def my_random_crop(image, sync_seed=None):
    np.random.seed(sync_seed)
    img_sz = image.shape[1]
    crop_sz = 256
    # img_sz - 1 - (crop_sz - 1)
    if img_sz > crop_sz:
        random_arr = np.random.randint(low=0, high=img_sz - crop_sz, size=2)
        y = int(random_arr[0])
        x = int(random_arr[1])
        h = crop_sz
        w = crop_sz
        image_crop = image[:, y:y + h, x:x + w, :]
        # print("crop image seed %d ,location - %d:%d,%d:%d" %(sync_seed,y,y+h,x,x+w))
        return image_crop
    else:
        return image


def crop_by_center_r(image, center_x, center_y, radius):
    img_sz = image.shape[1]
    # generate a random integer 256 ~ 500
    crop_sz = 2 * radius
    # crop_sz = np.random.randint(low=13, high=31, size=1)[0] * 16
    # print(crop_sz)
    min_x = 0 if center_x - radius < 0 else center_x - radius
    min_y = 0 if center_y - radius < 0 else center_y - radius
    max_x = image.shape[1] if center_x + radius > image.shape[1] else center_x + radius
    max_y = image.shape[2] if center_y + radius > image.shape[2] else center_y + radius

    if img_sz > crop_sz:
        image_crop = image[:, min_x:max_x, min_y:max_y, :]
        return image_crop
    else:
        return image

def center_crop(x, center_crop_size=[256, 256], **kwargs):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh]

def crop_by_center_r_3d(image, center_x, center_y, radius):
    img_sz = image.shape[1]

    crop_sz = 2 * radius

    # print(crop_sz)
    min_x = 0 if center_x - radius < 0 else center_x - radius
    min_y = 0 if center_y - radius < 0 else center_y - radius
    max_x = image.shape[0] if center_x + radius > image.shape[0] else center_x + radius
    max_y = image.shape[1] if center_y + radius > image.shape[1] else center_y + radius

    if img_sz > crop_sz:
        image_crop = image[min_x:max_x, min_y:max_y, :]
        return image_crop
    else:
        return image

# random_size_crop from 200 to 500
def my_random_size_crop(image, sync_seed=None, crop_sz = 256):
    np.random.seed(sync_seed)
    img_sz = image.shape[1]
    # generate a random integer 256 ~ 500

    # crop_sz = np.random.randint(low=13, high=31, size=1)[0] * 16
    # print(crop_sz)
    if img_sz > crop_sz:
        random_arr = np.random.randint(low=0, high=img_sz - crop_sz, size=2)
        y = int(random_arr[0])
        x = int(random_arr[1])
        h = crop_sz
        w = crop_sz
        image_crop = image[:, y:y + h, x:x + w, :]
        center_x = y + h // 2
        center_y = x + w // 2
        return image_crop, center_x, center_y
    else:
        return image, -1, -1

def my_random_location_crop(image, sync_seed=None, crop_sz = 256):
    np.random.seed(sync_seed)
    img_sz = image.shape[1]
    # generate a random integer 256 ~ 500

    # crop_sz = np.random.randint(low=13, high=31, size=1)[0] * 16
    # print(crop_sz)
    if img_sz > crop_sz:
        random_arr = np.random.randint(low=0, high=img_sz - crop_sz, size=2)
        y = int(random_arr[0])
        x = int(random_arr[1])
        h = crop_sz
        w = crop_sz
        image_crop = image[:, y:y + h, x:x + w, :]
        center_x = y + h // 2
        center_y = x + w // 2
        return image_crop
    else:
        return image




# random_size_crop from 200 to 500
def my_random_16_size_crop(image, sync_seed=None):
    np.random.seed(sync_seed)
    img_sz = image.shape[1]
    # generate a random integer 256 ~ 512
    # crop_sz = np.random.randint(low=200, high=img_sz, size=1)[0]
    crop_sz = np.random.randint(low=13, high=33, size=1)[0] * 16    # 208->516  16
    # print(crop_sz)
    if img_sz > crop_sz:
        random_arr = np.random.randint(low=0, high=img_sz - crop_sz, size=2)
        y = int(random_arr[0])
        x = int(random_arr[1])
        h = crop_sz
        w = crop_sz
        image_crop = image[:, y:y + h, x:x + w, :]
        return image_crop
    else:
        return image


# random crop batch image into 192,256,320,384
def my_4size_crop(image, sync_seed=None):
    # generate a random integer 0,64,128,192
    # crop_sz = np.random.randint(low=200, high=500, size=1)[0]
    # 192 256 320 384
    np.random.seed(sync_seed)
    img_sz = image.shape[1]

    crop_sz = 192 + np.random.randint(low=0, high=4, size=1)[0] * 64
    # generate a random location
    random_arr = np.random.randint(low=0, high=img_sz - crop_sz, size=2)
    y = int(random_arr[0])
    x = int(random_arr[1])
    h = crop_sz
    w = crop_sz
    image_crop = image[:, y:y + h, x:x + w, :]
    return image_crop

# resize all images  channel last.
def img_resize_fixed_4d(img, resized_size = 256):
    batch_size = img.shape[0]
    img_resize = np.zeros([batch_size, resized_size, resized_size, img.shape[3]])
    for n in range(batch_size):
        img_resize[n] = resize(image=img[n], output_shape=[resized_size, resized_size], anti_aliasing=True,
                               mode='reflect')
    return img_resize
# random crop batch image into 256,336,416,496
def my_3size_crop(image, sync_seed=None):
    np.random.seed(sync_seed)
    img_sz = image.shape[1]
    # generate a random integer 200 ~ 500
    # crop_sz = np.random.randint(low=200, high=500, size=1)[0]
    crop_sz = 208 + np.random.randint(low=0, high=3, size=1)[0] * 144
    if img_sz > crop_sz:
        # print(crop_sz)
        random_arr = np.random.randint(low=0, high=img_sz - crop_sz, size=2)
        y = int(random_arr[0])
        x = int(random_arr[1])
        h = crop_sz
        w = crop_sz
        image_crop = image[:, y:y + h, x:x + w, :]
        return image_crop
    else:
        return image

