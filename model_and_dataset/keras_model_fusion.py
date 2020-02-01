import tensorflow as tf
from tensorflow.python import keras


def classifier(x, input_size=256, l2_reg=0., dropout=0.9, n_classes=20, n_FCs=1):
    nameindex = 0
    for i in range(n_FCs):
        x = keras.layers.Dense(input_size, activation=tf.nn.relu, kernel_regularizer=l2_reg, bias_regularizer=l2_reg)(x)
        x = keras.layers.Dropout(dropout)(x)
        nameindex += 1

    x = keras.layers.Dense(n_classes, activation=tf.nn.softmax, kernel_regularizer=l2_reg, bias_regularizer=l2_reg)(x)
    return x


def SoftFusionModel(n_classes=20, l2=0., drop_rate=0., n_FCs=1, seq1_size=128, seq2_size=128):
    """
    Soft fusion model with inputs are two feature vectors.
    :param n_classes: Number of classes you wish to classify.
    :param l2:
    :param drop_rate:
    :param n_FCs: number of FC layers to use after fusion (the last FC layer with softmax is not included here).
    :param seq1_size: int, size of the first feature vector.
    :param seq2_size: int, size of the second feature vector.
    :return: Soft Fusion model, not compiled yet (call model.compile() before training).
    """

    l2_reg = keras.regularizers.l2(l2)
    seq_1 = keras.layers.Input(shape=(seq1_size,))
    seq_2 = keras.layers.Input(shape=(seq2_size,))

    soft_mask_1 = keras.layers.Dense(seq1_size, activation=tf.nn.sigmoid, kernel_regularizer=l2_reg,
                                     bias_regularizer=l2_reg)(seq_1)
    soft_mask_2 = keras.layers.Dense(seq2_size, activation=tf.nn.sigmoid, kernel_regularizer=l2_reg,
                                     bias_regularizer=l2_reg)(seq_2)

    x = keras.layers.Concatenate()([seq_1, seq_2])
    soft_mask = keras.layers.Concatenate()([soft_mask_1, soft_mask_2])

    x = keras.layers.Multiply()([x, soft_mask])

    output = classifier(x, seq1_size + seq2_size, l2_reg, drop_rate, n_classes=n_classes, n_FCs=n_FCs)

    return keras.Model(inputs=(seq_1, seq_2), outputs=output)
