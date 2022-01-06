import keras
import numpy as np
from matplotlib import image
import os
class DataGenerator(keras.utils.Sequence):
    def __init__(self, file_list):
        self.file_list = os.listdir(file_list)
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.file_list))

    def __getitem__(self, index):
        indexes = self.indexes[index:index+1]

        file_list_temp =[self.file_list[k] for k in indexes]
        x,y = self.__data_generation(file_list_temp)

        return x,y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_list))

    def  __data_generation(self, file_list_temp):
        data_loc = './'
        data_img_loc = os.path.join(data_loc,'img')
        data_mask_loc = os.path.join(data_loc,'mask')
        for ID in file_list_temp:
            x_file_path = os.path.join(data_img_loc,ID)
            y_file_path = os.path.join(data_mask_loc,ID.replace('.png','.npy'))
            x = image.imread(x_file_path)
            y = np.load(y_file_path)
        return x, y

if __name__ == '__main__':
    training_generator = DataGenerator('img/')
    print(training_generator.file_list)
    print(training_generator.on_epoch_end())
    print(training_generator.indexes)
    img = training_generator[0][0]
    mask = training_generator[0][1]
    temp = np.zeros([mask.shape[0],mask.shape[1],3])
    temp[:,:,:2] = mask
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()
    plt.imshow(temp)
    plt.show()