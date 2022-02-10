from keras import Model, Input
from keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D, Conv2D, Conv1D, MaxPooling2D, Flatten, Dense, Reshape, Dropout
from keras.layers.normalization import BatchNormalization

def make_3d_1(units, dataset):

    if dataset == 'PU':
        shape = (None, 10, 10, 4, 16)
        K = 15

    if dataset == 'IP':
        shape = (None, 10, 10, 12, 16)
        K = 30

    if dataset == 'SA':
        shape = (None, 10, 10, 4, 16)
        K = 15

    input_shape = (25,25,K,1)

    model = Sequential()

    model.add(Conv3D(8, kernel_size=(3, 3, 5),activation='relu', input_shape=input_shape))
    model.add(Conv3D(16, kernel_size=(3, 3, 3),activation='relu'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Reshape((shape[1], shape[2], shape[3] * shape[4])))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units,activation='softmax'))

    return model


def make_3d_2(units, dataset):

    if dataset == 'PU':
        shape = (None, 17, 17, 1, 16)
        K = 15

    if dataset == 'IP':
        shape = (None, 17, 17, 16, 16)
        K = 30

    if dataset == 'SA':
        shape = (None, 17, 17, 1, 16)
        K = 15

    input_shape = (25,25,K,1)


    model = Sequential()

    model.add(Conv3D(8, kernel_size=(3, 3, 7),activation='relu', input_shape=input_shape))
    model.add(Conv3D(16, kernel_size=(3, 3, 5),activation='relu'))
    model.add(Conv3D(16, kernel_size=(3, 3, 3),activation='relu'))
    model.add(Conv3D(16, kernel_size=(3, 3, 3),activation='relu'))

    #model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())
    model.add(Reshape((shape[1], shape[2], shape[3] * shape[4])))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(units,activation='softmax'))

    return model
