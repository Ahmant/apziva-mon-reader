import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.data.make_dataset import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam



models_paths = {
    'custom': './models/custom_model.h5',
    'resnet': './models/resnet_model_weights.h5',
    'mobilenet': './models/mobilenet_model_weights.h5',
}


def evaluate_model(model, test_data, cm_save_path=None):
    results = {}

    # predicted_labels = []
    predicted_labels = []
    true_labels = []
    for images, labels in test_data:
        true_labels.extend(labels.numpy())
        predicted_labels.extend(tf.argmax(model.predict(images), axis=1).numpy())

    # Accuracy
    results['accuracy'] = accuracy_score(true_labels, predicted_labels)

    # F1 Score
    results['f1_score'] = f1_score(true_labels, predicted_labels)

    cm = confusion_matrix(true_labels, predicted_labels)

    class_names = test_data.class_names
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    if cm_save_path is not None:
        plt.savefig(cm_save_path)
    else:
        plt.show()

    return results

test_ds = get_test_ds()
img_height, img_width, img_channels = 180, 180, 3
output_length = len(test_ds.class_names)
for model_name, model_path in models_paths.items():
    print(model_name)
    # if model_path not exists: TODO:
        # continue

    # Load model
    model = Sequential()
    if model_name == 'custom':
        model = tf.keras.models.load_model(model_path)
    elif model_name == 'resnet':
        pretrained_model= tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=(img_width, img_height, img_channels),
            pooling='avg',
            weights=None  # Do not load weights yet
        )

        for layer in pretrained_model.layers:
                layer.trainable = False

        model.add(pretrained_model)
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(output_length, activation='softmax'))

        # Load the saved weights into the model
        model.load_weights(model_path)
    elif model_name == 'mobilenet':
        pretrained_model= tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False,
            input_shape=(img_width, img_height, img_channels),
            pooling='avg',
            weights=None
        )

        for layer in pretrained_model.layers:
                layer.trainable=False

        model.add(pretrained_model)
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(output_length, activation='softmax'))

        # Load the saved weights into the model
        model.load_weights(model_path)
    else:
        model = None

    if model is not None:
        evaluate_model(
            model,
            test_ds,
            './reports/figures/' + model_name + '-confusion-matrix.png'
        )
