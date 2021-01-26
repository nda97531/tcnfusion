from tensorflow.python import keras

from model.keras_model_tcn import TCN


def remove_layers(model, trained_file=None, outputlayerindex=None):
    """
    remove layers at the end of a model
    :param model: keras model
    :param trained_file: trained weights file .h5
    :param outputlayerindex: the last layer that you want to keep
    :return: new model with some layers at the end removed
    """
    if (outputlayerindex is None):
        outputlayerindex = int(input('output layer index= ? '))

    if trained_file is not None:
        model.load_weights(trained_file)

    if outputlayerindex < 0:
        outputlayerindex = len(model.layers) + outputlayerindex

    new_model = keras.models.Model(model.inputs, model.layers[outputlayerindex].output)

    return new_model


if __name__ == '__main__':
    md = TCN()
    md.summary()
    md = remove_layers(md, outputlayerindex=-3)
    md.summary()
