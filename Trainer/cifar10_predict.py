import warnings
warnings.filterwarnings('ignore')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('../../tiny_CNN_CIFAR_10')

import h5py
from python_files.cifar10_input import unpickle_cifar, data_generator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler,  EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2, MobileNetV2, ResNet50, DenseNet121
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import load_model

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

def monitoring():
    # Prepare model model saving directory.
    save_dir = os.path.join('../model_save', 'MobileNetV2_cifar_10')
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

    if model_name == 'MobileNetV2':
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
                                classifier_activation='softmax',
                                classes=classes)

    elif model_name == 'ResNet50':
        KerasModel = ResNet50(include_top=include_top,
                              weights=weights,
                              input_tensor=input_tensor,
                              input_shape=input_shape,
                              pooling=pooling,
                              classifier_activation='softmax',
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
       print('')
       print(layer.name)
       print(layer.input)
       print(layer.output)

if __name__ == '__main__':

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s'%policy.compute_dtype) 
    print('Variable dtype: %s'%policy.variable_dtype)

    # Create data generator
    cifar_train, cifar_test = unpickle_cifar()
    x_train, x_test, y_train, y_test, classnames = data_generator(cifar_train, cifar_test)
    y_train = to_categorical(y_train, num_classes=len(classnames))
    y_test = to_categorical(y_test, num_classes=len(classnames))

    # Prepare iterator
    batch_size = 512

    model = cifar_model(x_test, model_name='MobileNetV2', num_classes=len(classnames))

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam,
                      loss= categorical_crossentropy,
                      metrics=['accuracy'])
    

    # Predict model
    model_path = '../model_save/MobileNetV2_cifar_10/cifar_10_model-00355- 0.88.h5'
    model = load_model(model_path)
    model.summary()
    check_layer_dtype(model)

    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss : {}, test acc: {} %".format(np.round(results[0]), np.round(results[1]*100,2)))

    predictions_float32 = model.predict(x_test, batch_size=128)
    print("Prediction shape : {}, dtype: {}".format(np.shape(predictions_float32), predictions_float32.dtype))


    print("Prediction on test data")

    model_cut = Model(inputs=model.input, outputs=model.layers[-2].output)
    outputs = Softmax(dtype='float16')(model_cut.output)
    model_new = Model(inputs=model_cut.input, outputs=outputs)
    
    check_layer_dtype(model_new)
    predictions_float16 = model_new.predict(x_test, batch_size=128)
    print("Prediction shape : {}, dtype: {}".format(np.shape(predictions_float16), predictions_float16.dtype))

    diff = predictions_float16 - predictions_float32
    print('Prediction difference: {}, dtype: {}'.format(diff, diff.dtype))

    model_new.save('../model_save/mobilenetv2_float16.h5')

    # print("Evaluate on test data")
    # model_new.compile(optimizer=adam,
    #                   loss= categorical_crossentropy,
    #                   metrics=['accuracy'])
    # results = model_new.evaluate(x_test, y_test, batch_size=128)
    # print("test loss : {}, test acc: {} %".format(np.round(results[0]), np.round(results[1]*100,2)))
    # model_new
