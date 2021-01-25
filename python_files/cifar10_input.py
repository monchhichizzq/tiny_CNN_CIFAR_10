import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = '../dataset/data_download/cifar-10-python/cifar-10-batches-py'

def unpickle(file):
    '''
    Define the data-read method
    :param file:
    :return:
    '''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def unpickle_cifar():
    # Read the cifar data
    cifar_train={}
    # Unite 5 training dataset
    for i in range(5):
        print(os.getcwd())
        cifar1=unpickle(os.path.join(data_path, 'data_batch_'+str(i+1)))
        if i==0:
            cifar_train[b'data']=cifar1[b'data']
            cifar_train[b'labels']=cifar1[b'labels']
        else:
            cifar_train[b'data']=np.vstack([cifar1[b'data'],cifar_train[b'data']])
            cifar_train[b'labels']=np.hstack([cifar1[b'labels'],cifar_train[b'labels']])
    # batches.meta restore the label names
    target_name = unpickle(os.path.join(data_path, 'batches.meta'))
    cifar_train[b'label_names'] = [str(x)[2:-1] for i, x in enumerate(target_name[b'label_names'])]
    print(cifar_train[b'label_names'] )
    cifar_train = get_images(cifar_train)

    # load testset
    cifar_test=unpickle(os.path.join(data_path, 'test_batch'))
    cifar_test[b'labels']=np.array(cifar_test[b'labels'])
    cifar_test = get_images(cifar_test)
    return cifar_train, cifar_test

def get_images(cifar):
    # Define the data format (reshape 3072 to 32*32*3)
    # Define an array of rgb images
    blank_image = np.zeros((len(cifar[b'data']), 32, 32, 3), np.uint8)
    # Define an array of grey images
    blank_image2 = np.zeros((len(cifar[b'data']), 32, 32), np.uint8)
    for i in range(len(cifar[b'data'])):
        blank_image[i] = np.zeros((32, 32, 3), np.uint8)
        # Reshape 1024 pixels to 32*32 and write it in the red/green/blue channel
        for j in range(3):
            blank_image[i][:, :, j] = cifar[b'data'][i][j*1024:(j+1)*1024].reshape(32, 32)
    cifar[b'data_rgb'] = blank_image
    cifar[b'data_grey'] = blank_image2
    return cifar

def ouput_labels(cifar):
    target_list = pd.Series(cifar[b'labels']).drop_duplicates()
    target_list = target_list.sort_values()
    target_list = list(target_list.index)
    target_figure = cifar[b'data_rgb'][target_list]
    for i in range(len(cifar[b'label_names'])):
        plt.subplot(2, 5, 1 + i)
        plt.title(cifar[b'label_names'][i])
        plt.imshow(target_figure[i]), plt.axis('off')
    plt.savefig('../output/label-visualization_remove.png', dpi=600)
    plt.show()

def data_generator(cifar_train, cifar_test):
    # Define training and test data
    x_train = cifar_train[b'data_rgb']
    y_train = cifar_train[b'labels']
    x_test = cifar_test[b'data_rgb']
    y_test = cifar_test[b'labels']

    x_train = x_train.reshape(-1, 32, 32, 3)
    x_test = x_test.reshape(-1, 32, 32, 3)
    class_names = cifar_train[b'label_names']

    # x_train = x_train.astype('float16')
    # x_test = x_test.astype('float16')
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    print('train_data:', x_train.shape, 'test_data:', x_test.shape)
    print('train_label:',y_train.shape, 'test_label:', y_test.shape)
    return x_train, x_test, y_train, y_test, class_names

def n_classes(cifar, label_name, del_number, train = True):
    x = cifar[b'data_rgb']
    y = cifar[b'labels']
    if train:
        z = cifar[b'label_names']
        del z[del_number]
        cifar[b'label_names'] = z
    origin_index = [i for i in range(len(x))]
    remove_index = [i for i, index in enumerate(y) if index == label_name]
    new_index = list(set(origin_index) - set(remove_index))
    new_labels = [y[i] for i in new_index]
    new_data = [x[i] for i in new_index]
    cifar[b'labels'] =  np.array(new_labels)
    cifar[b'data_rgb'] = np.array(new_data)
    #print('new_data', np.array(new_data).shape, ' new_label', np.array(new_labels).shape)
    return cifar


if __name__ == '__main__':
    cifar_train, cifar_test = unpickle_cifar()
    y_list_index = list(cifar_test[b'labels']).index(0)
    # Remove some labels (cifar 5)
    del_number = 5
    for i in range(del_number):
        cifar_train = n_classes(cifar_train,label_name=i+del_number, del_number=del_number, train = True)
        cifar_test = n_classes(cifar_test, label_name=i+del_number, del_number=del_number, train = False)
        ouput_labels(cifar_train)



    #ouput_labels(cifar_train)
    #print(cifar_train[b'label_names'])