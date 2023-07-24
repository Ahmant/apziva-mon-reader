import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from src.data.make_dataset import *
from src.visualization.visualize import *

models_paths = {
    'custom': './models/custom_model.h5',
    'resnet': './models/resnet_model_weights.h5',
    'mobilenet': './models/mobilenet_model_weights.h5',
}

def train_custom_model(train_ds, val_ds, output_length, save_path = None):
    model = models.Sequential([
        # CNN Layers 01
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(img_width, img_height, img_channels)
        ),
        layers.MaxPooling2D((2, 2)),
        
        # CNN Layers 02
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation='relu'
        ),
        layers.MaxPooling2D((2, 2)),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_length, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training
    model_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=7
    )

    # Get Reports
    plot_model_metrics(
        [
            model_history.history['accuracy'],
            model_history.history['val_accuracy']
        ],
        title='Custom Model Accuracy',
        xlabel='Epochs',
        ylabel='Accuracy',
        legends=['train', 'validation'],
        save_path='./reports/figures/custom-model-accuracy.png'
    )
    plot_model_metrics(
        [
            model_history.history['loss'],
            model_history.history['val_loss']
        ],
        title='Custom Model Loss',
        xlabel='Epochs',
        ylabel='Loss',
        legends=['train', 'validation'],
        save_path='./reports/figures/custom-model-loss.png'
    )

    # Save The Model
    if save_path is not None:
        model.save(save_path)


def train_resnet_model(train_ds, val_ds, output_length, save_path = None):
    model = Sequential()

    pretrained_model= tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(img_width, img_height, img_channels),
        pooling='avg',
        weights='imagenet'
    )

    for layer in pretrained_model.layers:
            layer.trainable = False

    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    model.add(Dense(output_length, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training
    model_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )

    # Get Reports
    plot_model_metrics(
        [
            model_history.history['accuracy'],
            model_history.history['val_accuracy']
        ],
        title='Resnet Model Accuracy',
        xlabel='Epochs',
        ylabel='Accuracy',
        legends=['train', 'validation'],
        save_path='./reports/figures/resnet-model-accuracy.png'
    )
    plot_model_metrics(
        [
            model_history.history['loss'],
            model_history.history['val_loss']
        ],
        title='Resnet Model Loss',
        xlabel='Epochs',
        ylabel='Loss',
        legends=['train', 'validation'],
        save_path='./reports/figures/resnet-model-loss.png'
    )

    # Save The Model
    if save_path is not None:
        model.save_weights(save_path)


def train_mobilenet_model(train_ds, val_ds, output_length, save_path = None):
    model = Sequential()

    pretrained_model= tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False,
        input_shape=(img_width, img_height, img_channels),
        pooling='avg',
        weights='imagenet'
    )

    for layer in pretrained_model.layers:
            layer.trainable=False

    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(output_length, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training
    model_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    # Get Reports
    plot_model_metrics(
        [
            model_history.history['accuracy'],
            model_history.history['val_accuracy']
        ],
        title='MobileNet Model Accuracy',
        xlabel='Epochs',
        ylabel='Accuracy',
        legends=['train', 'validation'],
        save_path='./reports/figures/mobilenet-model-accuracy.png'
    )
    plot_model_metrics(
        [
            model_history.history['loss'],
            model_history.history['val_loss']
        ],
        title='MobileNet Model Loss',
        xlabel='Epochs',
        ylabel='Loss',
        legends=['train', 'validation'],
        save_path='./reports/figures/mobilenet-model-loss.png'
    )

    # Save The Model
    if save_path is not None:
        model.save_weights(save_path)




# Get Datasets
train_ds = get_train_ds()
val_ds = get_validation_ds()

# Get Classes Names
classes_names = train_ds.class_names

# Training
train_custom_model(train_ds, val_ds, output_length=len(classes_names), save_path=models_paths['custom'])
train_resnet_model(train_ds, val_ds, output_length=len(classes_names), save_path=models_paths['resnet'])
train_mobilenet_model(train_ds, val_ds, output_length=len(classes_names), save_path=models_paths['mobilenet'])

