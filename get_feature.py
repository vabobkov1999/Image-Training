import cv2
import numpy as np
import os
import h5py
import functions as fun


def getting_features():

    
    path = 'data/train/'
    titles = os.listdir(path) # получили список покемонов

    labels = []
    features = []
    
    
    for name in titles:
        print(name)
        for image in os.listdir(path + name): # проходимся по всем картинкам в каждой папке
            original_image = cv2.imread(path + name + '/' + image)
            scaled_image = cv2.resize(original_image, (750,750))
            feature1 = fun.fd_hu_moments(scaled_image)
            feature2 = fun.fd_haralick(scaled_image)
            feature3 = fun.fd_histogram(scaled_image)
            global_feature = np.hstack([feature3, feature2, feature1])
            features.append(global_feature)
            labels.append(titles.index(name))

    h5f_data = h5py.File('features/data.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(features))

    h5f_label = h5py.File('features/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(labels))

    h5f_data.close()
    h5f_label.close()

if __name__ == '__main__':
    getting_features()
