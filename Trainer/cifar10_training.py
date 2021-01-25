import warnings
warnings.filterwarnings('ignore')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('../../tiny_CNN_CIFAR_10')

import logging
import h5py
from python_files.cifar10_input import unpickle_cifar, data_generator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler,  EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2, MobileNetV2, ResNet50, DenseNet121, MobileNet
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from Models.ResNet_SE import resnet_v1

def plot_traning_curves(history):
    os.makedirs('plots', exist_ok=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('plots/Accuracy_plot.png', dpi=600)

    plt.figure()
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('plots/loss_plot.png', dpi=600)
    plt.show()

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
    # cifar 10
    if epoch > 400:
        lr *= 1e-2
    elif epoch > 200:
        lr *= 1e-1
    # cifar 10
    if epoch > 400:
        lr *= 1e-2
    elif epoch > 200:
        lr *= 1e-1
    # print('Learning rate: ', lr)
    return lr

def monitoring(name):
    # Prepare model model saving directory.
    save_dir = os.path.join('../model_save', name + '_cifar_10', precision)
    model_name_save = 'cifar_10_model-{epoch:05d}-{val_accuracy: .2f}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name_save)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True)

    # Dynamic_lr_Callbacks                                                            
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, verbose=1)

    # Early Stopping Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1)

    board = TensorBoard(log_dir='../model_save/logs')

    callbacks= [checkpoint, reduce_lr]
    return callbacks

def cifar_model(x_test, model_name='MobileNetV2', num_classes=10):
     # Load model from tensorlfow
    include_top = False
    weights = 'imagenet'
    input_tensor = Input(x_test.shape[1:], dtype='float16')
    input_shape = x_test.shape[1:]
    pooling = None
    classes = num_classes

    if model_name == 'resnetv1SE':
        KerasModel = resnet_v1(input_shape,
                               depth=50,
                               SE_impl=True,
                               include_top=include_top,
                               num_classes=classes)

    if model_name == 'resnetv1':
         KerasModel = resnet_v1(input_shape,
                                depth=50,
                                SE_impl=False,
                                include_top=include_top,
                                num_classes=classes)

    if model_name == 'MobileNet':
         KerasModel = MobileNet(include_top=include_top,
                                  weights=weights,
                                  input_tensor=input_tensor,
                                  input_shape=input_shape,
                                  pooling=pooling,
                                  classes=classes)

    elif model_name == 'MobileNetV2':
        KerasModel = MobileNetV2(include_top=include_top,
                                 weights=weights,
                                 input_tensor=input_tensor,
                                 input_shape=input_shape,
                                 pooling=pooling,
                                 classes=classes)

    elif model_name == 'ResNet50V2':
        KerasModel = ResNet50V2(include_top=include_top,
                                weights=weights,
                                input_tensor=input_tensor,
                                input_shape=input_shape,
                                pooling=pooling,
                                # classifier_activation='softmax',
                                classes=classes)

    elif model_name == 'ResNet50':
        KerasModel = ResNet50(include_top=include_top,
                              weights=weights,
                              input_tensor=input_tensor,
                              input_shape=input_shape,
                              pooling=pooling,
                              # classifier_activation='softmax',
                              classes=classes)

    elif model_name == 'DenseNet121':
        KerasModel = DenseNet121(include_top=include_top,
                                 weights=weights,
                                 input_tensor=input_tensor,
                                 input_shape=input_shape,
                                 pooling=pooling,
                                 classes=classes)
    inputs = KerasModel.input
    output = KerasModel.output
    x = GlobalAveragePooling2D()(output)
    x = Flatten()(x)
    x = Dense(num_classes,
                    kernel_initializer='he_normal')(x)
    outputs = Softmax(dtype='float32')(x)
    
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def check_layer_dtype(model): 
    for layer in model.layers:
        # if type(layer.input) is list:
        #     print(layer.input[0])
        input_type = layer.input[0].dtype  if type(layer.input) is list else layer.input.dtype
        output_type = layer.output[0].dtype if type(layer.output) is list else layer.output.dtype
        # input_type = layer.input.dtype
        # output_type = layer.output.dtype
        logger.debug('{} - Input: {} - Output: {}'.format(layer.name, input_type, output_type))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,  # 配置输出层级 INFO, DEBUG
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger('Cifar 10 Training')
    # global logger
    logger.setLevel(logging.DEBUG)

    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    # logger.info('Compute dtype: {}'.format(policy.compute_dtype))
    # logger.info('Variable dtype: {}'.format(policy.variable_dtype))

    # Create data generator
    cifar_train, cifar_test = unpickle_cifar()
    x_train, x_test, y_train, y_test, classnames = data_generator(cifar_train, cifar_test)
    y_train = to_categorical(y_train, num_classes=len(classnames))
    y_test = to_categorical(y_test, num_classes=len(classnames))

    # Create data generator
    datagen = ImageDataGenerator(
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
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
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
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Prepare iterator
    model_name = 'resnetv1'
    batch_size = 128
    learning_rate = 1e-3
    precision = 'fp32'

    logger.info('Model: {}'.format(model_name))
    logger.info('Batch Size: {}'.format(batch_size))
    logger.info('Learning Rate: {}'.format(learning_rate))
    logger.info('Training precision: {}'.format(precision))

    input_train = datagen.flow(x_train, y_train, batch_size=batch_size)

    model = cifar_model(x_test, model_name=model_name, num_classes=len(classnames))

    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam,
                      loss= categorical_crossentropy,
                      metrics=['accuracy'])
    model.summary()
    check_layer_dtype(model)

    # Fit model
    steps = int(x_train.shape[0] / batch_size)
    callbacks_list = monitoring(model_name)
    history = model.fit_generator(input_train,
                                   # steps_per_epoch=steps,
                                   epochs=600,
                                   verbose=2,
                                   validation_data=(x_test, y_test),
                                   callbacks=callbacks_list)

    # Plot
    plot_traning_curves(history)
