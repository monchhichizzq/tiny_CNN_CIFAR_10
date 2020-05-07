import warnings
warnings.filterwarnings('ignore')
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import h5py
import cifar10_input
import cifar10_model
from keras import optimizers, losses, utils
import keras.backend as K
import numpy as np
# dtype='float16'
# K.set_floatx(dtype)
# K.set_epsilon(1e-4)
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 70:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def monitoring():
    # Prepare model model saving directory.
    save_dir = os.path.join('../model_save', 'alexnet_cifar_10')
    model_name_save = 'cifar_10_model-{epoch:05d}-{val_accuracy: .2f}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name_save)
    print(filepath)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)

    board = TensorBoard(log_dir='../model_save/logs')

    callbacks= [checkpoint, lr_reducer, lr_scheduler, board]
    #callbacks = [checkpoint]
    return callbacks

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape([-1,28,28,1])
    x_test = x_test.reshape([-1,28,28,1])
    y_train = utils.to_categorical(y_train, num_classes=10)
    y_test = utils.to_categorical(y_test, num_classes=10)
    print('train_data:', x_train.shape, 'test_data:', x_test.shape)
    print('train_label:', y_train.shape, 'test_label:', y_test.shape)

    subtract_pixel_mean = True
    # If subtract pixel mean is enabled
    # if subtract_pixel_mean:
    #     x_train_mean = np.mean(x_train, axis=0)
    #     x_train -= x_train_mean
    #     x_test -= x_train_mean

    # create data generator
    # datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen = ImageDataGenerator(
        #  set rescaling factor (applied before any other transformation)
        rescale=1. / 255,
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0.2,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.2,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.2,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    # prepare iterator
    batch_size = 64
    input_train = datagen.flow(x_train, y_train, batch_size=batch_size)

    # input model
    model = cifar10_model.model_minst(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), keep_prob=0.5,
                                 classes=10)

    adam = optimizers.adam(lr=lr_schedule(0))
    sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False)
    model.compile(optimizer=adam,
              loss= losses.categorical_crossentropy,
              metrics=['accuracy'])

    # fit model
    steps = int(x_train.shape[0] / batch_size)
    callbacks_list = monitoring()
    history = model.fit_generator(input_train,
                                  steps_per_epoch=steps,
                                  epochs=100,
                                  validation_data=(x_test, y_test),
                                  callbacks=callbacks_list)
    # Plot
    cifar10_model.plot_traning_curves(history)
