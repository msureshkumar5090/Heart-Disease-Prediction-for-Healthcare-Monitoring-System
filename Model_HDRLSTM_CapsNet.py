from keras import Sequential
import numpy as np
from keras.src.layers import LSTM, Dense
from keras.src.optimizers import Adam
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from Model_Capsnet import Model_Capsnet


def Model_LSTM_Feat(Data, Target, BS=None, EP=None):
    print('LSTM')

    if BS is None:
        BS = 4
    if EP is None:
        EP = 10

    IMG_SIZE = [1, 100]
    num_classes = Target.shape[-1]

    trainX, testX, trainY, testy = train_test_split(Data, Target,
                                                    random_state=104,
                                                    test_size=0.25,
                                                    shuffle=True)

    Train_Temp = np.zeros((trainX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(trainX.shape[0]):
        Train_Temp[i, :] = np.resize(trainX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Test_Temp = np.zeros((testX.shape[0], IMG_SIZE[0], IMG_SIZE[1]))
    for i in range(testX.shape[0]):
        Test_Temp[i, :] = np.resize(testX[i], (IMG_SIZE[0], IMG_SIZE[1]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE[0], IMG_SIZE[1])

    Activation = ['linear', 'relu', 'tanh', 'sigmoid', 'softmax', 'leaky relu']
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(Train_X.shape[1], Train_X.shape[-1])))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    # model.compile(optimizer='adam', loss='mse')
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, trainY, epochs=EP, steps_per_epoch=5, batch_size=BS, verbose=1,
              validation_data=(Test_X, testy))
    pred = model.predict(Test_X, verbose=2)

    layerNo = 2
    intermediate_model = Model(inputs=model.input, outputs=model.layers[layerNo].output)
    Feats = intermediate_model.predict(np.concatenate((Train_X, Test_X)))
    Feats = np.asarray(Feats)
    Feature = np.resize(Feats, (Feats.shape[0], Feats.shape[-1]))
    return Feature


def Model_HDRLSTM_CapsNet(Data, Target, BS=None):
    if BS is None:
        BS = 32
    Feature = Model_LSTM_Feat(Data, Target, BS=BS)
    trainX, testX, trainY, testy = train_test_split(Feature, Target,
                                                    random_state=104,
                                                    test_size=0.25,
                                                    shuffle=True)
    Eval, pred = Model_Capsnet(trainX, trainY, testX, testy, BS=BS)
    return Eval, pred
