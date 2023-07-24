import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

import tensorflow as tf

img_height, img_width, img_channels = 180, 180, 3
batch_size=32
data_dir = './data/raw/images/training'
test_data_dir = './data/raw/images/testing'
loading_dataste_seed = 0

def get_train_ds():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=loading_dataste_seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return train_ds

def get_validation_ds():
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=loading_dataste_seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return val_ds

def get_test_ds():
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        seed=loading_dataste_seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return test_ds
