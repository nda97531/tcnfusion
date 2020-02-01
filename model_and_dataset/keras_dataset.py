import math

from tensorflow.python import keras


class KerasDatasetOneArray(keras.utils.Sequence):
    def __init__(self, input, output, batchsize):
        """
        :param input: input data in one array, shape (window, sample, channel)
        :param output: classification label
        :param batchsize: batch size used for training or evaluating
        """
        self.x = input
        self.y = output
        self.batch_size = batchsize

    def __len__(self):
        return math.ceil(1. * len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y


class KerasDatasetFusionOneArray(keras.utils.Sequence):
    def __init__(self, input, output, batchsize):
        """
        :param input: a list of 2 arrays, each of which is an array of shape (window, sample, channel)
        :param output: classification label
        :param batchsize: batch size used for training or evaluating
        """
        self.x1 = input[0]
        self.x2 = input[1]
        self.y = output
        self.batch_size = batchsize

    def __len__(self):
        return math.ceil(1. * len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        batch_x1 = self.x1[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x2 = self.x2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return (batch_x1, batch_x2), batch_y
