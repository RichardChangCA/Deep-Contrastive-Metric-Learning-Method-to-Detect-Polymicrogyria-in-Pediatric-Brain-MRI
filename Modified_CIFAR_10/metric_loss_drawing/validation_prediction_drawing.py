import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import layers

import sklearn
from sklearn.model_selection import train_test_split

from natsort import natsorted

import cv2
import shutil
import glob

base_dir = '.'

def dataset_collection_func(normal_class, abnormal_classes_list, abnormal_ratio=0.1, normal_amount=5000):

    abnormal_classes_array = np.array(abnormal_classes_list)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    
#     print(train_labels)
    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)

    print("train_images.shape:", train_images.shape)

#     print(train_images[0].shape)

#     print(train_images[[1,2,3]].shape)

    print("train_labels.shape:", train_labels.shape)
    
    one_class_idx = np.where(train_labels == normal_class)
#     print("one_class_idx:", one_class_idx)
    
    one_class_train_images = train_images[one_class_idx]
    one_class_train_labels = train_labels[one_class_idx]
    one_class_train_labels[one_class_train_labels==normal_class] = 0
    
    one_class_train_images = one_class_train_images[:normal_amount]
    one_class_train_labels = one_class_train_labels[:normal_amount]

#     print("1 one_class_train_images.shape:", one_class_train_images.shape)

    other_class_idx = np.where(train_labels != normal_class)
    other_class_train_images = train_images[other_class_idx]
    other_class_train_labels = train_labels[other_class_idx]

#     print("2 other_class_train_images.shape:", other_class_train_images.shape)

    other_class_train_should_idx = np.where((other_class_train_labels == abnormal_classes_array[0])
                                            | (other_class_train_labels == abnormal_classes_array[1])
                                            | (other_class_train_labels == abnormal_classes_array[2])
                                            | (other_class_train_labels == abnormal_classes_array[3])
                                            | (other_class_train_labels == abnormal_classes_array[4]))
    other_class_train_images = other_class_train_images[other_class_train_should_idx]
    other_class_train_labels = other_class_train_labels[other_class_train_should_idx]

#     print("3 other_class_train_images.shape:", other_class_train_images.shape)

    np.random.seed(1234) # set seed
    idx = np.random.permutation(len(other_class_train_labels))
    other_class_train_images = other_class_train_images[[idx]]
    other_class_train_labels = other_class_train_labels[[idx]]

#     print("4 other_class_train_images.shape:", other_class_train_images.shape)

#     print("other_class_train_labels:", other_class_train_labels)

    other_class_train_labels[other_class_train_labels !=normal_class] = 1

    train_one_class_len = len(one_class_train_labels)

    train_other_class_should_len = int(train_one_class_len * abnormal_ratio)

    other_class_train_images = other_class_train_images[:train_other_class_should_len]
    other_class_train_labels = other_class_train_labels[:train_other_class_should_len]

#     print("5 other_class_train_images.shape:", other_class_train_images.shape)

    print("majority train number:", len(one_class_train_labels))
    print("minority train number:", len(other_class_train_labels))
    # print("other_class_train_labels:", other_class_train_labels)

    # train:validation = 8:2
    one_class_train_validation_threshold = int(len(one_class_train_labels)*0.8)
    other_class_train_validation_threshold = int(len(other_class_train_labels)*0.8)

    one_class_train_images_train = one_class_train_images[:one_class_train_validation_threshold]
    one_class_train_images_validation = one_class_train_images[one_class_train_validation_threshold:]

    one_class_train_labels_train = one_class_train_labels[:one_class_train_validation_threshold]
    one_class_train_labels_validation = one_class_train_labels[one_class_train_validation_threshold:]

    other_class_train_images_train = other_class_train_images[:other_class_train_validation_threshold]
    other_class_train_images_validation = other_class_train_images[other_class_train_validation_threshold:]

    other_class_train_labels_train = other_class_train_labels[:other_class_train_validation_threshold]
    other_class_train_labels_validation = other_class_train_labels[other_class_train_validation_threshold:]

    train_images = np.concatenate((one_class_train_images_train,other_class_train_images_train),axis=0)
    train_labels = np.concatenate((one_class_train_labels_train,other_class_train_labels_train),axis=0)
    
    print("one_class_train_labels_train:",len(one_class_train_labels_train))
    print("other_class_train_labels_train:",len(other_class_train_labels_train))

    validation_images = np.concatenate((one_class_train_images_validation,other_class_train_images_validation),axis=0)
    validation_labels = np.concatenate((one_class_train_labels_validation,other_class_train_labels_validation),axis=0)
    
    print("one_class_train_labels_validation:",len(one_class_train_labels_validation))
    print("other_class_train_labels_validation:",len(other_class_train_labels_validation))
    
    # test
    
    test_labels[test_labels!=normal_class] = 11
    test_labels[test_labels==normal_class] = 0
    test_labels[test_labels==11] = 1
    
    print("train_images.shape:", train_images.shape)

    return validation_images, validation_labels

test_images, test_labels = dataset_collection_func(normal_class = 1, abnormal_classes_list = [2,3,4,5,6])

image_shape = (32,32,3)

img_dim = image_shape[0]

latent_dim = 128

half_model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=5, strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(
            filters=64, kernel_size=5, strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(
            filters=128, kernel_size=5, strides=(1, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Flatten(),

        # No activation
        tf.keras.layers.Dense(latent_dim),
    ]
)

margin = 5
theta = 5
c = np.ones(latent_dim)

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

checkpoint_dir = os.path.join(base_dir, 'metric_learning_classification')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=half_model)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

def my_metrics(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    y_pred = np.where(y_pred >= 0.5, 1, 0)

    TP, TN, FP, FN = 0, 0, 0, 0
    for prediction, y in zip(y_pred, y_true):

        if(prediction == y):
            if(prediction == 1): # {'No': 0, 'Yes': 1}
                TP += 1
            else:
                TN += 1
        else:
            if(prediction == 1):
                FP += 1
            else:
                FN += 1

    precision = TP/(TP+FP+1.0e-4)

    recall = TP/(TP+FN+1.0e-4)

    f_measure = (2. * precision * recall)/(precision + recall + 1.0e-4)

    accuracy = (TP + TN) / (TP + TN + FP + FN+1.0e-4)

    # print("TP:", TP)
    # print("TN:", TN)
    # print("FP:", FP)
    # print("FN:", FN)

    # print("precision:", precision)
    # print("recall:", recall)
    # print("f_measure:", f_measure)
    # print("accuracy:", accuracy)

    return np.array([TP, TN, FP, FN, precision, recall, f_measure, accuracy])

def evaluation(test_images, test_labels):
    print("testing")

    checkpoint.restore(ckpt_manager.latest_checkpoint)

    labels = []
    predictions = []
    dists = []

    for i in range(len(test_labels)):
        img_array = test_images[i]

        img_array = img_array / 255.

        img_array = np.expand_dims(img_array, axis=0)

        image_batch = img_array

        z = half_model(image_batch, training=False)
        
        z = np.array(z)

        dist = np.sum((z - c) ** 2, axis=-1)
        dist = np.sqrt(dist)

        # if(dist[0] > margin):
        #     prediction = 1
        # else:
        #     prediction = 0

        dists.append(dist[0])
        # predictions.append(prediction)
        
        labels.append(test_labels[i])

    labels = np.array(labels)

    dists = np.array(dists)
    # dists = np.expand_dims(dists,axis=-1)
    # predictions = np.array(predictions)

    # metric_results = my_metrics(labels, predictions)

    # auc_roc = sklearn.metrics.roc_auc_score(labels, dists)

    
    kwargs = dict(alpha=0.5, bins=100)

    plt.hist(dists[np.where((labels==0))], **kwargs, color='g', label='normal')
    plt.hist(dists[np.where((labels==1))], **kwargs, color='b', label='anomaly')
    plt.gca().set(title='Metric loss validation data predictions', ylabel='Amount', xlabel='Prediction value')
    plt.xlim(-1,50)
    plt.legend()
    # plt.show()
    plt.savefig('validation_metric_loss.png')

    # print("TP:", metric_results[0])
    # print("TN:", metric_results[1])
    # print("FP:", metric_results[2])
    # print("FN:", metric_results[3])

    # print("precision:", metric_results[4])
    # print("recall:", metric_results[5])
    # print("f_measure:", metric_results[6])
    # print("accuracy:", metric_results[7])
    # print("auc_roc:", auc_roc)

evaluation(test_images, test_labels)