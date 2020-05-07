import warnings
warnings.filterwarnings('ignore')
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cifar10_input
import os
from keras import regularizers, Input, Model, optimizers
from keras.layers import Conv2D, MaxPooling2D,Dropout, GlobalAveragePooling2D, SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
import keras.backend as K
# dtype='float16'
# K.set_floatx(dtype)
# # # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
# K.set_epsilon(1e-4)
import matplotlib.pyplot as plt
import numpy as np




def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), r = 1e-2, padding='same', add_bn=False, name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(nb_filter,
               kernel_size,
               strides=strides,
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(r),
               use_bias=True,
               bias_initializer='zero',
               padding=padding,
               name=conv_name)(x)
    if add_bn:
        x = BatchNormalization(axis=3, name=bn_name)(x)
    x = Activation('relu')(x)
    return x

def Alexnet(input_shape=None, keep_prob=0.5, classes=10, r=1e-2, name='Alex'):
    Inpt = Input(shape=input_shape, name='Input_'+name)
    # Block 1
    x = Conv2d_BN(Inpt, 96, (11, 11), strides=(4, 4), padding='valid', name='block1_conv1_' + name)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='block1_pool1'+ name)(x)
    # Block 2
    x = Conv2d_BN(x, 256, (5, 5), (1, 1),  padding='same', name='block2_conv1_'+ name)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='block2_pool1' + name)(x)
    # Block 3
    x = Conv2d_BN(x, 384, (3, 3), (1, 1),  padding='same', name='block3_conv1_'+ name)
    # Block 4
    x = Conv2d_BN(x, 384, (3, 3), (1, 1),  padding='same', name='block4_conv1_'+ name)
    # Block 5
    x = Conv2d_BN(x, 256, (3, 3), (1, 1), padding='same', name='block5_conv1_' + name)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='block5_pool1' + name)(x)
    # passsing it to a Fully connected layer
    x = Flatten(name='flatten_'+ name)(x)
    # 1st fully connected layer
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r), name='fc1_'+ name)(x)
    x = Dropout(1 - keep_prob)(x)
    # 2rd fully connected layer
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r), name='fc2_'+ name)(x)
    x = Dropout(1 - keep_prob)(x)
    # Output Layer
    prediction = Dense(classes, activation='softmax', name='final_'+ name)(x)
    # Create model.
    model = Model(Inpt, prediction, name=name)
    model.summary()
    return model

# FitNet-4 https://arxiv.org/abs/1511.06422
def test_1(input_shape=None, keep_prob=0.5, classes=10, r=1e-2, name='test_1'):
    Inpt = Input(shape=input_shape, name='Input_'+name)
    # Block 1
    x = Conv2d_BN(Inpt, 32, (3, 3), padding='same', name='block1_conv1_' + name)
    x = Conv2d_BN(x, 32, (3, 3), padding='same', name='block1_conv2_'+ name)
    x = Conv2d_BN(x, 32, (3, 3),  padding='same', name='block1_conv3_'+ name)
    #x = Conv2d_BN(x, 48, (3, 3), padding='same', name='block1_conv4_'+ name)
    #x = Conv2d_BN(x, 48, (3, 3), padding='same', name='block1_conv5_' + name)
    x = MaxPooling2D((2, 2), strides=(1, 1), name='block1_pool1' + name)(x)

    # Block 2
    x = Conv2d_BN(x, 48, (3, 3), padding='same', name='block2_conv1_' + name)
    x = Conv2d_BN(x, 48, (3, 3), padding='same', name='block2_conv2_'+ name)
    #x = Conv2d_BN(x, 48, (3, 3),  padding='same', name='block2_conv3_'+ name)
    #x = Conv2d_BN(x, 48, (3, 3), padding='same', name='block2_conv4_'+ name)
    #x = Conv2d_BN(x, 48, (3, 3), padding='same', name='block2_conv5_' + name)
    x = MaxPooling2D((2, 2), strides=(1, 1), name='block2_pool1' + name)(x)

    # Block 3
    x = Conv2d_BN(x, 80, (3, 3), padding='same', name='block3_conv1_' + name)
    x = Conv2d_BN(x, 80, (3, 3), padding='same', name='block3_conv2_'+ name)
    x = Conv2d_BN(x, 128, (3, 3),  padding='same', name='block3_conv3_'+ name)
    x = Dropout(1 - 0.5)(x)
    #x = Conv2d_BN(x, 128, (3, 3), padding='same', name='block3_conv4_'+ name)
    #x = Conv2d_BN(x, 128, (3, 3), padding='same', name='block3_conv5_' + name)
    x = MaxPooling2D((8, 8),name='block3_pool1' + name)(x)

    # passsing it to a Fully connected layer
    x = Flatten(name='flatten_'+ name)(x)
    # 1st fully connected layer
    #x = Dense(500, activation='relu', kernel_regularizer=regularizers.l2(r), name='fc1_'+ name)(x)
    #x = Dropout(1 - keep_prob)(x)
    # Output Layer
    prediction = Dense(classes, activation='softmax', name='final_'+ name)(x)
    # Create model.
    model = Model(Inpt, prediction, name=name)
    model.summary()
    return model

def model_88(input_shape=None, keep_prob=0.5, classes=10, r=1e-2, name='model88'):
    Inpt = Input(shape=input_shape, name='Input_' + name)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(Inpt)
    #x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.2)(x)
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    #x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    # x = SpatialDropout2D(0.3)(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    #x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    #x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    # x = Dropout(0.4)(x)
    x = SpatialDropout2D(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    # Orignal network use FC layers with BN and Dropout
    #x = Flatten()(x)
    #x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    prediction = Dense(classes, activation='softmax')(x)
    model = Model(Inpt, prediction, name=name)
    model.summary()
    return model


def model_cifar_5(input_shape=None, keep_prob=0.5, classes=10, r=1e-2, name='model88'):
    Inpt = Input(shape=input_shape, name='Input_' + name)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(Inpt)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.3)(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(classes, activation='softmax')(x)
    model = Model(Inpt, prediction, name=name)
    model.summary()
    return model


def model_88_v1(input_shape=None, keep_prob=0.5, classes=10, r=1e-2, name='model88'):
    Inpt = Input(shape=input_shape, name='Input_' + name)
    # block 1
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(Inpt)
    #x = BatchNormalization()(x)
    # block 2
    x = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)

    x = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)
    # x = Dropout(0.2)(x)
    # block 3
    x = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    # block 4
    x = Conv2D(64, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(256, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)
    #x = Dropout(0.3)(x)
    # block 5
    x = Conv2D(64, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(256, (1, 1), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.4)(x)
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(classes, activation='softmax')(x)
    model = Model(Inpt, prediction, name=name)
    model.summary()
    return model


def model_88_v2(input_shape=None, keep_prob=0.5, classes=10, r=1e-2, name='model88_v2'):
    Inpt = Input(shape=input_shape, name='Input_' + name)
    x = Conv2D(32, (3, 3), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(Inpt)
    x = Conv2D(32, (3, 1), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(32, (1, 3), use_bias=False,activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(64, (3, 1), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(64, (1, 3), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(64, (3, 1), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(64, (1, 3), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)
    x = Conv2D(64, (3, 1), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(64, (1, 3), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(32, (1, 1), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(128, (3, 1), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(128, (1, 3), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    # Orignal network use FC layers with BN and Dropout
    #x = Flatten()(x)
    #x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    prediction = Dense(classes, activation='softmax')(x)
    model = Model(Inpt, prediction, name=name)
    model.summary()
    return model


def model_minst(input_shape=None, keep_prob=0.5, classes=10, r=1e-2, name='model_minst'):
    Inpt = Input(shape=input_shape, name='Input_' + name)
    x = Conv2D(32, (3, 3), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(Inpt)
    x = Conv2D(32, (3, 1), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(32, (1, 3), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    #x = SpatialDropout2D(0.2)(x)
    x = Conv2D(64, (3, 1), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(64, (1, 3), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 1), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = Conv2D(128, (1, 3), use_bias=False, activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    #x = SpatialDropout2D(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(classes, activation='softmax')(x)
    model = Model(Inpt, prediction, name=name)
    model.summary()
    return model


def plot_traning_curves(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    cifar_train, cifar_test = cifar10_input.unpickle_cifar()
    x_train, x_test, y_train, y_test, classnames = cifar10_input.data_generator(cifar_train, cifar_test)
    model = test_1(input_shape= (x_train.shape[1], x_train.shape[2], x_train.shape[3]), keep_prob=0.5, classes=len(classnames), r=1e-2)
    callbacks = monitoring()
    model.compile(optimizer=optimizers.adam(lr=1e-4),  # keras.optimizers.Adadelta()
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model = multi_gpu_model(model, gpus=gpu)
    history = model.fit(x_train,
                        y_train,
                        batch_size=128,
                        epochs= 1000,
                        validation_data= (x_test, y_test),
                        callbacks=callbacks)
    plot_traning_curves(history)



