from tensorflow.keras.utils import Sequence
import numpy as np


class TrainDataGenerator(Sequence):
    def __init__(
            self,
            input,
            output,
            batch_size,
            augment_rate=0.,
            augmenter=None,
            shuffle=True,
            use_rate=1
    ):
        """
        :param input: data array shape [windows, time axis, channel axis]
        :param output: label array shape [windows, ...]
        :param batch_size: batch size
        :param augment_rate: float from 0 to 1
        :param augmenter: Augmenter object
        :param shuffle: whether to shuffle data after each epoch
        :param use_rate: float from 0 to 1, amount of data in each epoch
        """
        self.input = input
        self.output = output
        self.augment_rate = augment_rate
        self.indices = np.arange(len(self.output)).astype(np.int)
        self.shuffle = shuffle
        self.augmenter = augmenter

        self.input.flags.writeable = False
        self.output.flags.writeable = False

        self.batch_size = batch_size
        self.use_rate = use_rate

        if shuffle:
            self.on_epoch_end()

    def __len__(self):
        """
        number of batches in an epoch
        """
        return np.floor(
            (len(self.output) / self.batch_size) * self.use_rate).astype(
            np.int)

    def __getitem__(self, index):
        """
        get data and label of 1 batch
        """
        X = self._generate_X(index)
        y = self._generate_y(index)
        return X, y

    def on_epoch_end(self):
        """
        shuffle data on epoch end
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _generate_X(self, idx_batch):
        """
        generate batch data with augmentation
        """

        # pick a batch
        new_idx = self.indices[idx_batch * self.batch_size:
                               (idx_batch + 1) * self.batch_size]
        original_data = self.input[new_idx]

        # replace some original instances with augmented instances
        augmented_data = np.copy(original_data)
        for i in range(original_data.shape[0]):
            if np.random.rand() < self.augment_rate:
                augmented_data[i] = self.augmenter.random_augment(
                    original_data[i])

        return augmented_data

    def _generate_y(self, idx_batch):
        """
        generate batch label
        """
        new_idx = self.indices[idx_batch * self.batch_size:
                               (idx_batch + 1) * self.batch_size]
        original_data = self.output[new_idx]
        return original_data
