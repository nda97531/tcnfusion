import tensorflow as tf
from tensorflow.python import keras

layers = keras.layers


def receptive_field(k, B, N=2):
    '''
    Calculate TCN's receptive field, given dilation_base = 2.
    :param k: kernel size
    :param B: number of blocks
    :param N: number of Conv 1D layers per block
    :return: receptive field
    '''
    return 1 + N * (k - 1) * (2 ** B - 1)


name_ = 0


def res_tcn_block(x, n_filters, kernel_size, dilation, dropout_rate=0.2, l2=keras.regularizers.l2(0.),
                  use_batch_norm=False):
    global name_
    prev_x = x
    for _ in reversed(range(2)):
        x = layers.ZeroPadding1D(((kernel_size - 1) * dilation, 0))(x)
        x = layers.Conv1D(n_filters, kernel_size=kernel_size, dilation_rate=dilation, kernel_regularizer=l2,
                          bias_regularizer=l2,
                          kernel_initializer=keras.initializers.Orthogonal())(x)

        if use_batch_norm:
            x = layers.BatchNormalization()(x)

        x = layers.Activation(tf.nn.relu)(x)

        x = layers.SpatialDropout1D(rate=dropout_rate)(inputs=x)

    # 1x1 conv to match the shapes (channel dimension).
    if prev_x.shape[-1] != n_filters:
        prev_x = layers.Conv1D(n_filters, 1, padding='same', activation='relu', name='match_channel' + str(name_))(
            prev_x)
        name_ += 1

    res_x = keras.layers.add([x, prev_x])
    res_x = layers.Activation(tf.nn.relu)(res_x)

    return res_x  # , x


def TCN(input_shape=(60, 32)
        , n_classes=20
        , n_tcn_channels=(64,) * 3 + (128,) * 2
        , kernel_size=2
        , dilation_base=2
        , dropout_rate=0.2
        , l2=1e-4
        , use_batchnorm=False
        , classify=True
        ):
    """
    :param input_shape: Shape of the input window.
    :param n_classes: Number of classes you wish to classify. This is only used if ``classify=True``
    :param n_tcn_channels: A tuple specifying the number of Conv filters in each block.
        The length of tuple equals the number of residual blocks.
    :param kernel_size: Conv 1D kernel size
    :param dilation_base: dilation in block number i is (dilation_base^i)
    :param dropout_rate:
    :param l2:
    :param use_batchnorm: If True, apply BatchNorm in res block. Default is False.
    :param classify: If true, output is classes' probabilities, else, output is a feature vector. Default is True.
    :return: TCN model, not compiled yet (call model.compile() before training).
    """

    if (dilation_base > kernel_size):
        raise ValueError('dilation base must be less than or equal to kernel size')

    l2 = keras.regularizers.l2(l2)

    inputs = layers.Input(shape=input_shape)
    x = inputs

    for i in range(len(n_tcn_channels)):
        dilation = dilation_base ** i
        out_channels = n_tcn_channels[i]
        x = res_tcn_block(x, n_filters=out_channels, kernel_size=kernel_size, dilation=dilation,
                          dropout_rate=dropout_rate, l2=l2, use_batch_norm=use_batchnorm)

    outputs = layers.Lambda(lambda tt: tt[:, -1, :])(x)

    if classify:
        outputs = layers.Dense(n_classes, activation=tf.nn.softmax, kernel_regularizer=l2, bias_regularizer=l2,
                               kernel_initializer=keras.initializers.Orthogonal())(outputs)

    return keras.Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    model = TCN()
    print(model.summary())
