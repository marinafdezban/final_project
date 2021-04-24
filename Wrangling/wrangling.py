import tensorflow as tf
import itertools
from collections import defaultdict


def autotune_dataset(dataset):
    autotune_local = tf.data.AUTOTUNE
    return dataset.prefetch(buffer_size=autotune_local)


def dataset_augmentation(dataset):
    autotune = autotune_dataset(dataset)
    data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
                                             tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                                             ])

    data_augmentation_train = []
    for image in autotune:
        for i in range(9):
            augmented_image = data_augmentation(tf.expand_dims(image, 0))
            data_augmentation_train.append(augmented_image)
    return data_augmentation_train
