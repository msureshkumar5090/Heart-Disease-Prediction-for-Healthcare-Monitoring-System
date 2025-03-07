import keras
from keras import layers
import numpy as np
from Evaluation import evaluation


def Model_AutoEncoder(Train_Data, Train_Target, Test_Data, Test_Target, BS=None, sol=2):
    if BS is None:
        BS = 256
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    input_img = keras.Input(shape=(Train_Data.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(Train_Target.shape[1], activation='sigmoid')(encoded)
    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    Train_Data = Train_Data.astype('float32') / 255.
    Test_Data = Test_Data.astype('float32') / 255.
    Train_Data = Train_Data.reshape((len(Train_Data), np.prod(Train_Data.shape[1:])))
    Test_Data = Test_Data.reshape((len(Test_Data), np.prod(Test_Data.shape[1:])))
    autoencoder.fit(Train_Data, Train_Target,
                    epochs=int(sol),
                    batch_size=BS,
                    shuffle=True,
                    validation_data=(Test_Data, Test_Target))
    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded_imgs = encoder.predict(Test_Data)
    pred = decoder.predict(encoded_imgs)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(Test_Target, pred)
    return Eval, pred
