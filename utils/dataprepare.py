import os
import numpy as np
from matplotlib import image
import random
# 必要的文件
def save_to_file(image, name, path='./'):
    if not os.path.exists(path):
        os.makedirs(path)
    full_name = os.path.join(path, name)
    scipy.misc.imsave(name=full_name, arr=image)


def write_message(message, file, print_log=True):
    file_path, file_name = os.path.split(file)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        with open(file, 'w+') as f:
            print(message, file=f)
            if print_log:
                print('message already write into %s' % file)
    else:
        with open(file, 'a+') as f:
            print(message, file=f)
            if print_log:
                print('message already write into %s' % file)


def getSomeFile(path, type):
    file_orig = os.listdir(path)
    files = []
    for i in file_orig:
        if str.endswith(i, type):
            i = os.path.join(path, i)
            files.append(i)
    return files


path = r'C:\Users\xujiayu\Desktop\临时\203\mask_orig'
files = getSomeFile(path=path, type='.png')
print(files)
matrices = dict()
for file in files:
    img = image.imread(file)
    temp = np.zeros([4])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for c in range(img.shape[2]):
                if img[i, j, c] > 0:
                    temp[c] = temp[c] + 1
                    temp[-1] = temp[-1] + 1
    _, name = os.path.split(file)
    matrices[name] = temp

    print(name, '\t', temp)

print(matrices)

count = np.zeros([4])
for matrix_name in matrices:
    count = count + matrices[matrix_name]
print(count)


def flag_judge(scale_0, scale_1, scale_2):
    scale_0_expected = 0.60
    scale_1_expected = 0.20
    scale_2_expected = 0.20
    return scale_0_expected - precision < scale_0 < scale_0_expected + precision and scale_1_expected - precision < scale_1 < scale_1_expected + precision and scale_2_expected - precision < scale_2 < scale_2_expected + precision


# 随机值保证细胞数目比为6:2:2
scale_0 = 0.
scale_1 = 0.
scale_2 = 0.
count_0 = np.zeros([4])
count_1 = np.zeros([4])
count_2 = np.zeros([4])
precision = 0.01
scale_0_expected = 0.60
scale_1_expected = 0.20
scale_2_expected = 0.20
count_whole = np.array([2806, 145037, 37872, 185715])
keys = list(matrices.keys())  # List of keys
flag = scale_0_expected - precision < scale_0 < scale_0_expected + precision and scale_1_expected - precision < scale_1 < scale_1_expected + precision and scale_2_expected - precision < scale_2 < scale_2_expected + precision
while (True):
    count_0 = np.zeros([4])
    count_1 = np.zeros([4])
    count_2 = np.zeros([4])
    random.shuffle(keys)
    for i, key in enumerate(keys):
        if i < 121:
            count_0 += matrices[key]
        if 121 <= i < 162:
            count_1 += matrices[key]
        if 163 <= i < 203:
            count_2 += matrices[key]
    scale_0_m = count_0 / count_whole
    scale_1_m = count_1 / count_whole
    scale_2_m = count_2 / count_whole
    print(scale_0_m)
    print(scale_1_m)
    print(scale_2_m)
    for (scale_0, scale_1, scale_2) in zip(scale_0_m, scale_1_m, scale_2_m):
        # print(scale_0,scale_1,scale_2)
        if flag_judge(scale_0, scale_1, scale_2):
            print(flag_judge(scale_0, scale_1, scale_2))
        if not flag_judge(scale_0, scale_1, scale_2):
            # print(flag)
            break
    if flag_judge(scale_0, scale_1, scale_2) == False:
        continue
    if flag_judge(scale_0, scale_1, scale_2) == True:
        break

for key in keys:
    print(key, matrices[key])
    write_message(message=key+'\t'+str(matrices[key])+'\n', file='.\out.txt')
print(scale_0_m)
print(scale_1_m)
print(scale_2_m)