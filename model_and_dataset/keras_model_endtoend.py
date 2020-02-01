from tensorflow.python import keras

import model_and_dataset.keras_model_remove_last_layers
from model_and_dataset.keras_model_fusion import SoftFusionModel
from model_and_dataset.keras_model_tcn import TCN

remove_layers = model_and_dataset.keras_model_remove_last_layers.remove_layers


def EndToEndModel(n_classes=20, l2=1e-4, tcndropout=0.2, fusiondropout=0.9, n_FCs_fusion=1,
                  acc_weight=None, ske_weight=None):
    """
    End-to-end model with two TCN feature extractors and soft fusion.
    :param n_classes: Number of classes you wish to classify.
    :param l2:
    :param tcndropout: dropout rate in feature extractors
    :param fusiondropout: dropout rate in fusion phase
    :param n_FCs_fusion: number of FC layers to use after fusion (the last FC layer with softmax is not included here)
    :param acc_weight: trained weights file of input stream 1 (h5 extension)
    :param ske_weight: trained weights file of input stream 1 (h5 extension)
    :return: End-to-end model, not compiled yet. (call model.compile() before training)
    """

    acc_input_shape = (150, 6)
    ske_input_shape = (60, 32)

    acc_seq = keras.layers.Input(shape=acc_input_shape)
    ske_seq = keras.layers.Input(shape=ske_input_shape)

    if acc_weight is None:
        acc_feature = TCN(input_shape=acc_input_shape, n_classes=n_classes, n_tcn_channels=(64,) * 5 + (128,) * 2,
                          dropout_rate=tcndropout, l2=l2, classify=False)(acc_seq)

    else:
        acc_model = TCN(input_shape=acc_input_shape, n_classes=n_classes, n_tcn_channels=(64,) * 5 + (128,) * 2,
                        dropout_rate=tcndropout, l2=l2)
        acc_feature = remove_layers(acc_model, inputlayerindex=0, outputlayerindex=-2, trained_file=acc_weight)(acc_seq)

    if ske_weight is None:
        ske_feature = TCN(n_classes=n_classes, n_tcn_channels=(64,) * 3 + (128,) * 2, l2=l2, dropout_rate=tcndropout,
                          classify=False, input_shape=ske_input_shape)(ske_seq)
    else:
        ske_model = TCN(n_classes=n_classes, n_tcn_channels=(64,) * 3 + (128,) * 2, l2=l2, dropout_rate=tcndropout,
                        input_shape=ske_input_shape)
        ske_feature = remove_layers(ske_model, inputlayerindex=0, outputlayerindex=-2)(ske_seq)

    output = SoftFusionModel(n_classes=n_classes, l2=l2, drop_rate=fusiondropout, n_FCs=n_FCs_fusion)(
        [acc_feature, ske_feature])

    return keras.Model((acc_seq, ske_seq), output)
