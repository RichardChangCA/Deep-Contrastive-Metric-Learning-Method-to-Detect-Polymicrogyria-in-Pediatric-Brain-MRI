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

def dataset_collection_func(normal_class, abnormal_classes_list, abnormal_ratio, normal_amount):

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

    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels

train_images, train_labels, validation_images, validation_labels, test_images, test_labels = dataset_collection_func(normal_class = 0, abnormal_classes_list = [1,2,3,4,5], abnormal_ratio = 0.1, normal_amount=5000)

normal_idx = np.where(train_labels==0)[0]
train_images_normal = train_images[normal_idx]

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

def compute_loss(labels, z):

    c_tf = tf.Variable(c, dtype=tf.float32, trainable=False)

    dist = tf.math.reduce_sum((z - c_tf) ** 2, axis=1)

    # print("dist:", dist)

    euclidean_norm = tf.math.sqrt(dist)

    labels = tf.squeeze(labels)

    normal_idx = tf.where(labels == 0)
    abnormal_idx = tf.where(labels == 1)

    normal_idx = tf.squeeze(normal_idx)
    abnormal_idx = tf.squeeze(abnormal_idx)

    normal_labels = 1-tf.gather(labels, normal_idx)
    normal_labels = tf.cast(normal_labels, dtype=tf.float32)
    normal_dist_loss = tf.math.reduce_mean(normal_labels * tf.gather(euclidean_norm, normal_idx))

    # print("abnormal_idx:", abnormal_idx)
    # if(abnormal_idx == tf.Variable([], dtype=tf.int64, trainable=False)):
    if(tf.size(abnormal_idx) == [0,]):
    # if(abnormal_idx.eval() == []):
        # print("abnormal loss is 0")
        abnormal_dist_loss = 0
        abnormal_dist_loss_expansion = 0
    else:
        abnormal_labels = tf.gather(labels, abnormal_idx)
        abnormal_labels = tf.cast(abnormal_labels, dtype=tf.float32)

        abnormal_dist_loss = tf.math.reduce_mean(abnormal_labels * tf.math.maximum(tf.zeros_like(tf.gather(euclidean_norm, abnormal_idx)), margin - tf.gather(euclidean_norm, abnormal_idx)))

        abnormal_dist_loss_expansion = tf.math.reduce_mean(abnormal_labels * (1. / (1. + tf.math.exp(tf.gather(euclidean_norm, abnormal_idx) - margin))))

    loss = normal_dist_loss + theta * (abnormal_dist_loss + abnormal_dist_loss_expansion)

    # print("euclidean_norm:", euclidean_norm)
    print("normal_dist_loss:", normal_dist_loss)
    print("theta * (abnormal_dist_loss + abnormal_dist_loss_expansion):", theta * (abnormal_dist_loss + abnormal_dist_loss_expansion))

    return loss

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

def train_step(inputs, labels, optimizer):
    # print("training......")

    with tf.GradientTape() as tape:
        z = half_model(inputs, training=True)
        loss = compute_loss(labels, z)
        # print(loss)

    gradients = tape.gradient(loss, half_model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, half_model.trainable_variables))

    return np.array(loss), optimizer

def valid_step(inputs, labels):
    # print("validation......")

    z = half_model(inputs, training=False)
    loss = compute_loss(labels, z)
    # print(loss)

    z = np.array(z)

    dist = np.sum((z - c) ** 2, axis=-1)
    dist = np.sqrt(dist)

    predictions = np.zeros_like(dist)

    abnormal_idx = np.where(dist > margin)

    predictions[abnormal_idx] = 1

    metric_results = my_metrics(labels, predictions)

    return np.array(loss), metric_results, predictions

def train(train_images, train_labels, validation_images, validation_labels, epochs, BATCH_SIZE):

    global learning_rate
    global optimizer

    best_auc_roc = 0

    best_val_loss = 0
    best_val_loss_epoch_lr_decay = 0
    best_val_loss_epoch = 0

    first_tag = 1

    iteration = 0
    val_iteration = 0

    for epoch in range(epochs):
        start = time.time()

        idx = np.random.permutation(len(train_labels))
        train_images, train_labels = train_images[[idx]], train_labels[[idx]]

        print("train epoch = ", epoch)
        for index in range(0, len(train_labels)-BATCH_SIZE, BATCH_SIZE):
            label_batch = [] # always load the same batch
            for i in range(BATCH_SIZE):

                img_array = train_images[index+i]

                img_array = img_array / 255.

                # data augmentation
                img_array = tf.keras.preprocessing.image.random_rotation(img_array, 0.2)
                img_array = tf.keras.preprocessing.image.random_shift(img_array, 0.1, 0.1)
                img_array = tf.keras.preprocessing.image.random_shear(img_array, 0.1)
                img_array = tf.keras.preprocessing.image.random_zoom(img_array, (0.7,1))

                img_array = np.array(img_array)
                img_array = np.expand_dims(img_array, axis=0)

                if(i == 0):
                    image_batch = img_array
                else:
                    image_batch = np.concatenate((image_batch, img_array), axis=0)

                label_batch.append(train_labels[index+i])

            label_batch = np.array(label_batch)
            label_batch = np.expand_dims(label_batch, axis=-1)

            loss, optimizer = train_step(image_batch, label_batch, optimizer)

            if(iteration % 30 == 0):

                idx_val = np.random.permutation(len(validation_labels))
                validation_images, validation_labels = validation_images[[idx_val]], validation_labels[[idx_val]]

                val_loss_average = 0
                val_tag = 0

                accuracy_average = 0
                recall_average = 0
                precision_average = 0

                all_val_labels = []
                all_val_predictions = []

                print("validation")
                # validation
                for val_index in range(0, len(validation_labels)-BATCH_SIZE, BATCH_SIZE):
                    label_batch = [] # always load the same batch
                    for i in range(BATCH_SIZE):
                        img_array = validation_images[val_index+i]

                        img_array = img_array / 255.

                        img_array = np.expand_dims(img_array, axis=0)

                        if(i == 0):
                            image_batch = img_array
                        else:
                            image_batch = np.concatenate((image_batch, img_array), axis=0)

                        label_batch.append(validation_labels[val_index+i])

                    label_batch = np.array(label_batch)
                    label_batch = np.expand_dims(label_batch, axis=-1)

                    print("image_batch.shape:", image_batch.shape)
                    print("label_batch.shape:", label_batch.shape)

                    loss, metric_results, val_predictions = valid_step(image_batch, label_batch)

                    val_tag += 1
                    val_loss_average += loss

                    accuracy_average += metric_results[7]
                    recall_average += metric_results[5]
                    precision_average += metric_results[4]

                    all_val_labels.append(label_batch.flatten())
                    all_val_predictions.append(np.array(val_predictions).flatten())
                    # AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'flatten'

                val_loss_average = val_loss_average / val_tag

                accuracy_average = accuracy_average / val_tag
                recall_average = recall_average / val_tag
                precision_average = precision_average / val_tag

                all_val_labels = np.array(all_val_labels)
                all_val_predictions = np.array(all_val_predictions)

                all_val_labels = all_val_labels.flatten()
                all_val_predictions = all_val_predictions.flatten()

                auc_roc_average = sklearn.metrics.roc_auc_score(all_val_labels, all_val_predictions)
                # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.

                print("val_loss_average:", val_loss_average)
                print("accuracy_average:", accuracy_average)
                print("recall_average:", recall_average)
                print("precision_average:", precision_average)
                print("auc_roc_average:", auc_roc_average)
                
                # save the best only
                # best val_auc_roc
                if(first_tag == 1 or auc_roc_average > best_auc_roc):
                    best_auc_roc = auc_roc_average
                    print("saveing model")
                    ckpt_manager.save()

                    f = open(os.path.join(base_dir,"best_auc_roc_binary_classification.txt"), "w")
                    f.write(str(best_auc_roc))
                    f.close()
                
                # best val_loss
                if(first_tag == 1 or val_loss_average < best_val_loss):
                    best_val_loss = val_loss_average
                    best_val_loss_epoch = val_iteration
                    best_val_loss_epoch_lr_decay = best_val_loss_epoch
                    # print("saveing model")
                    # ckpt_manager.save()

                # learning rate decay --> reduce learning rate on plateau
                plateau_patience = 30

                if(val_loss_average > best_val_loss and (val_iteration - best_val_loss_epoch_lr_decay) == plateau_patience):

                    checkpoint.restore(ckpt_manager.latest_checkpoint)

                    best_val_loss_epoch_lr_decay = val_iteration
                    print("learning rate on plateau, reduce learning rate")
                    decay_factor = 0.5
                    learning_rate = learning_rate * decay_factor
                    optimizer = tf.keras.optimizers.Adam(learning_rate)

                # early stopping
                early_stopping_patience = 150
                if(val_loss_average > best_val_loss and (val_iteration - best_val_loss_epoch) == early_stopping_patience):
                    print("early stopping")
                    return

                first_tag = 0

                val_iteration += 1

            iteration += 1

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

        if(dist[0] > margin):
            prediction = 1
        else:
            prediction = 0

        dists.append(dist[0])
        predictions.append(prediction)
        
        labels.append(test_labels[i])

    labels = np.array(labels)
    predictions = np.array(predictions)

    metric_results = my_metrics(labels, predictions)

    auc_roc = sklearn.metrics.roc_auc_score(labels, dists)

    print("TP:", metric_results[0])
    print("TN:", metric_results[1])
    print("FP:", metric_results[2])
    print("FN:", metric_results[3])

    print("precision:", metric_results[4])
    print("recall:", metric_results[5])
    print("f_measure:", metric_results[6])
    print("accuracy:", metric_results[7])
    print("auc_roc:", auc_roc)

epochs = 400
BATCH_SIZE = 128
train(train_images, train_labels, validation_images, validation_labels, epochs, BATCH_SIZE)
evaluation(validation_images, validation_labels)
evaluation(test_images, test_labels)
