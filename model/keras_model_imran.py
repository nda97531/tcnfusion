"""
Non-official implementation of 1D-CNN and Bi-LSTM from https://link.springer.com/article/10.1007/s12652-019-01239-9
"""

from tensorflow import keras

def imran_fusion(acc_input_shape, ske_input_shape, num_class, mode='fusion'):
    inputs_acc = keras.layers.Input(shape=acc_input_shape)
    x = keras.layers.Conv1D(32, kernel_size=3, strides=1)(inputs_acc)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=None)(x)

    x = keras.layers.Conv1D(64, kernel_size=3, strides=1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=None)(x)

    x = keras.layers.Conv1D(128, kernel_size=3, strides=1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=None)(x)

    x = keras.layers.Conv1D(256, kernel_size=3, strides=1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=None)(x)

    x = keras.layers.Conv1D(512, kernel_size=3, strides=1)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.MaxPool1D(pool_size=2, strides=None)(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.Dense(2048, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs_acc = keras.layers.Dense(num_class, activation='softmax')(x)
    if mode == 'acc':
        return keras.Model(inputs_acc, outputs_acc)

    inputs_ske = keras.layers.Input(shape=ske_input_shape)
    y = keras.layers.Bidirectional(keras.layers.GRU(512, return_sequences=True), input_shape=(60, 32))(inputs_ske)
    y = keras.layers.Bidirectional(keras.layers.GRU(512, return_sequences=False))(y)
    y = keras.layers.Dense(2048, activation='relu')(y)
    y = keras.layers.Dropout(0.8)(y)
    outputs_ske = keras.layers.Dense(num_class, activation='softmax')(y)
    if mode == 'ske':
        return keras.Model(inputs_ske, outputs_ske)

    outputs_fusion = keras.layers.Average()([outputs_acc, outputs_ske])
    return keras.Model([inputs_acc, inputs_ske], outputs_fusion)