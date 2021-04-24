import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import cv2

from tensorflow.keras.preprocessing import image_dataset_from_directory

PATH = '/home/marina/Bootcamp/final_project/images/'
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

def get_train_dir():
    return os.path.join(PATH, 'train')

def get_validation_dir():
    return os.path.join(PATH, 'validation')

def load_train_dataset():
    train_dir = get_train_dir()
    return image_dataset_from_directory(train_dir,
                                        shuffle=True,
                                        batch_size=BATCH_SIZE,
                                        image_size=IMG_SIZE)

def load_val_dataset():
    validation_dir = get_validation_dir()
    return image_dataset_from_directory(validation_dir,
                                        shuffle=True,
                                        batch_size=BATCH_SIZE,
                                        image_size=IMG_SIZE)

def get_test_dataset():
    validation_dataset = load_val_dataset()
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    return validation_dataset.take(val_batches // 5)

def get_validation_dataset():
    validation_dataset = load_val_dataset()
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    return validation_dataset.skip(val_batches // 5)




