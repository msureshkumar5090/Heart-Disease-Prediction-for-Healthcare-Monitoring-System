import tensorflow as tf
import numpy as np
from keras.src.models import Sequential
from keras.src.layers import Input, Conv1D, BatchNormalization, Activation, Dropout, Flatten, Dense
from keras.src.optimizers import Adam
from Evaluation import evaluation


def tcn_model(input_shape, num_classes, sol, num_filters=64, kernel_size=3, dilations=[1, 2, 4, 8, 16]):
    model = Sequential()
    model.add(Input(shape=input_shape))
    # Temporal Convolutional Blocks (TCN)
    for dilation_rate in dilations:
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, padding='causal', dilation_rate=dilation_rate))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(int(sol[0]), activation='relu'))  # 128
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def Model_TCN(trainX, trainY, testX, testy, BS=None, sol=None):
    if BS is None:
        BS = 32
    if sol is None:
        sol = [128, 0.01]
    input_shape = (784, 3)
    Train_Temp = np.zeros((trainX.shape[0], input_shape[0], input_shape[1]))
    for i in range(trainX.shape[0]):
        Train_Temp[i, :] = np.resize(trainX[i], (input_shape[0], input_shape[1]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], input_shape[0], input_shape[1])

    Test_Temp = np.zeros((testX.shape[0], input_shape[0], input_shape[1]))
    for i in range(testX.shape[0]):
        Test_Temp[i, :] = np.resize(testX[i], (input_shape[0], input_shape[1]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], input_shape[0], input_shape[1])

    input_shape = (Train_X.shape[1], Train_X.shape[2])
    num_classes = testy.shape[-1]

    model = tcn_model(input_shape, num_classes, sol)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Train_X, trainY, epochs=10, batch_size=BS, validation_data=(Test_X, testy))

    pred = model.predict(Test_X)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.astype(int)
    Eval = evaluation(testy, pred)
    return Eval, pred

