import os

import numpy as np
from keras_radam import RAdam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from model_and_dataset.keras_dataset import *
from model_and_dataset.keras_model_endtoend import EndToEndModel

os.environ['TF_KERAS'] = "1"


def class_weight(label, return_type='array'):
    """
    :param label: numpy array of training labels, shape (window, )
    :param return_type: 'array' or 'dict'
    :return: class weight
    """
    values, counts = np.unique(label, return_counts=True)
    cweight = 1. * len(label) / counts
    cweight = cweight / cweight.min()

    if return_type == 'array':
        return cweight
    elif return_type == 'dict':
        return dict(zip(values, cweight))
    else:
        raise ValueError('return_type must be array or dict!')


def train(model,
          trainloader,
          validloader,
          weights_save_name,
          max_epoch=100,
          class_weight=None,
          patience=10):
    early_stopping = EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0.0, patience=patience)
    checkpointer = ModelCheckpoint(
        filepath=weights_save_name + '.{epoch:02d}-{val_sparse_categorical_accuracy:.5f}.h5',
        monitor='val_sparse_categorical_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1)

    model.fit_generator(trainloader,
                        epochs=max_epoch,
                        verbose=2,
                        callbacks=[early_stopping, checkpointer],
                        validation_data=validloader,
                        shuffle=True,
                        class_weight=class_weight)

    model.save(f'{weights_save_name}+_earlystopping.h5', overwrite=True)


if __name__ == '__main__':
    traindata1 = np.load('<file name>')  # shape (window, timestep, channel)
    traindata2 = np.load('<file name>')  # shape (window, timestep, channel)
    traindata = (traindata1, traindata2)

    validdata1 = np.load('<file name>')  # shape (window, timestep, channel)
    validdata2 = np.load('<file name>')  # shape (window, timestep, channel)
    validdata = (validdata1, validdata2)

    trainlabel = np.load('<file name>').astype(int)  # shape (window,)

    validlabel = np.load('<file name>').astype(int)  # shape (window,)

    model = EndToEndModel()
    opt = RAdam(1e-3)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    trainloader = KerasDatasetFusionOneArray(traindata, trainlabel, 32)
    validloader = KerasDatasetFusionOneArray(traindata, trainlabel, 32)

    train(model,
          trainloader,
          validloader,
          weights_save_name='param/test_',
          max_epoch=100,
          class_weight=class_weight(trainlabel, 'array'),
          patience=10,
          )
