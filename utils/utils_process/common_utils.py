# some functions used for other projects
import os
import numpy as np
import time
from PIL import Image
from matplotlib import image
print(os.getcwd())
# os.chdir('../')    # to the top directory
print(os.getcwd())

def time_calculate(func_name, *args,**kws):
    startTime = time.time()
    func_name(*args, **kws)
    endTime = time.time()
    duration = endTime - startTime
    print(func_name.__name__, 'duration = ', seconds_regulate(duration))

def save_to_file(image, name, path='./'):
    if not os.path.exists(path):
        os.makedirs(path)
    full_name = os.path.join(path, name)
    img = Image.fromarray(np.uint8(image*255))
    img.save(full_name)

def save_to_file_255(image, name, path='./'):
    if not os.path.exists(path):
        os.makedirs(path)
    full_name = os.path.join(path, name)
    img = Image.fromarray(np.uint8(image))
    img.save(full_name)

def seconds_regulate(seconds):
    return str(time.strftime('%H:%M:%S', time.gmtime(seconds)))

def write_message(message, file,print_log=True):
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

def getSomeFile(path, type='.png'):
    file_orig = os.listdir(path)
    files = []
    for i in file_orig:
        if str.endswith(i, type):
            i = os.path.join(path, i)
            files.append(i)
    return files


# any names to 0-?
def rename(path, files):
    message = ''
    for i, file in enumerate(files):
        img = image.imread(file)
        _, orig_name = os.path.split(file)
        full_name = str(i) + '.png'
        message = message + orig_name + '\t' + full_name + '\n'
        # full_name = os.path.splitext(orig_name)[0] + '_' + 'test' + '.png'
        # full_name = os.path.splitext(orig_name)[0].replace('_withcontour', '') + '.png'
        save_to_file(img, full_name, path)
    write_path = os.path.join(path, 'mapping.txt')
    write_message(message, write_path)

def rename_retrieve(path, mapping_file):
    with open(mapping_file, 'r') as f:
        list = f.readlines()
        for line in list:
            words = line.split('\t')
            if len(words) < 2:
                break
            img = image.imread(os.path.join(path, words[1].replace('\n', '')))
            save_to_file(img, words[0], path)

def time_calculate(func_name, *args,**kws):
    startTime = time.time()
    func_name(*args, **kws)
    endTime = time.time()
    duration = endTime - startTime
    print(func_name.__name__, 'duration = ', seconds_regulate(duration))

def visualize_4cto3c(files):
    for file in files:
        img = image.imread(file)
        temp = np.zeros([img.shape[0],img.shape[1],3])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for c in range(img.shape[2]):
                    if c < 3:
                        if img[i,j,c] > 0:
                            temp[i,j,c] = 1
                    else:
                        if img[i,j,c] > 0:
                            temp[i,j,c%3] = 1
                            temp[i,j,(c+1)%3] = 1
        save_to_file(temp,os.path.split(file)[1],os.path.join(os.path.split(file)[0],'visualize'))