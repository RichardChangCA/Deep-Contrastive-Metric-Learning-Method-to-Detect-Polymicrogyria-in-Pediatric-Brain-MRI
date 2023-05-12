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

import tensorflow_addons as tfa

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

model = tf.keras.Sequential(
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

        tf.keras.layers.Dense(latent_dim)
    ]
)

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

checkpoint_dir = os.path.join(base_dir, 'triplet_loss_checkpoints')

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

checkpoint.restore(ckpt_manager.latest_checkpoint) # restore the lastest checkpoints

def my_metrics(y_true, y_pred, threshold=0.5):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    # y_pred = np.where(y_pred >= 0.5, 1, 0)
    y_pred = np.where(y_pred >= threshold, 1, 0)

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
        labels = tf.cast(labels, tf.float32)

        predictions = model(inputs, training=True)
        loss = tfa.losses.TripletSemiHardLoss(margin=1.0)(labels, predictions)
        # print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return np.array(loss), optimizer

def valid_step(inputs, labels):
    # print("validation......")

    labels = tf.cast(labels, tf.float32)

    predictions = model(inputs, training=False)

    loss = tfa.losses.TripletSemiHardLoss(margin=1.0)(labels, predictions)
    # print(loss)

    return np.array(loss), predictions

def train(train_images, train_labels, validation_images, validation_labels, epochs, BATCH_SIZE):

    global learning_rate
    global optimizer

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

                    # print("image_batch.shape:", image_batch.shape)
                    # print("label_batch.shape:", label_batch.shape)

                    loss, val_predictions = valid_step(image_batch, label_batch)

                    val_tag += 1
                    val_loss_average += loss

                val_loss_average = val_loss_average / val_tag

                print("val_loss_average:", val_loss_average)
                
                # best val_loss
                if(first_tag == 1 or val_loss_average < best_val_loss):
                    best_val_loss = val_loss_average
                    best_val_loss_epoch = val_iteration
                    best_val_loss_epoch_lr_decay = best_val_loss_epoch
                    
                    print("saveing model")
                    ckpt_manager.save()

                    f = open(os.path.join(base_dir,"best_triplet_loss.txt"), "w")
                    f.write(str(best_val_loss))
                    f.close()

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

def evaluation(test_images, test_labels, threshold=0.5):
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

        prediction = model(image_batch, training=False)
        
        predictions.append(prediction)
        
        labels.append(test_labels[i])

    labels = np.array(labels)
    predictions = np.array(predictions)

    metric_results = my_metrics(labels, predictions, threshold)

    predictions = np.squeeze(predictions)

    print("labels.shape:", labels.shape)
    print("predictions.shape:", predictions.shape)

    auc_roc = sklearn.metrics.roc_auc_score(labels, predictions)

    print("TP:", metric_results[0])
    print("TN:", metric_results[1])
    print("FP:", metric_results[2])
    print("FN:", metric_results[3])

    print("precision:", metric_results[4])
    print("recall:", metric_results[5])
    print("f_measure:", metric_results[6])
    print("accuracy:", metric_results[7])
    print("auc_roc:", auc_roc)

    f2_measure = (5*metric_results[4]*metric_results[5]) / (4*metric_results[4]+metric_results[5])

    print("f2_measure:", f2_measure)

epochs = 400
BATCH_SIZE = 128
train(train_images, train_labels, validation_images, validation_labels, epochs, BATCH_SIZE)

from sklearn.svm import SVC

def train_svm(train_images, train_labels, validation_images, validation_labels, test_images, test_labels):
    # print("inside train_svm function")
    checkpoint.restore(ckpt_manager.latest_checkpoint)

    labels = []
    predictions = []

    for i in range(len(train_labels)):
        img_array = train_images[i]

        img_array = img_array / 255.

        img_array = np.expand_dims(img_array, axis=0)

        image_batch = img_array

        prediction = model(image_batch, training=False)
        
        predictions.append(prediction)
        
        labels.append(train_labels[i])

    labels = np.array(labels)
    predictions = np.array(predictions)

    print("training SVM")
    svc = SVC()
    # print("predictions:", predictions.shape)
    # print("predictions:", predictions.squeeze().shape)
    # print("labels:", labels.shape)
    svc.fit(predictions.squeeze(), labels)

    test_predictions = []
    labels_test = []

    for i in range(len(test_labels)):
        img_array = test_images[i]

        img_array = img_array / 255.

        img_array = np.expand_dims(img_array, axis=0)

        image_batch = img_array

        prediction = model(image_batch, training=False)
        
        test_predictions.append(prediction)
        
        labels_test.append(test_labels[i])

    labels_test = np.array(labels_test)
    test_predictions = np.array(test_predictions)

    # print("test_predictions:", test_predictions.squeeze().shape)

    test_predictions_class = svc.predict(test_predictions.squeeze())

    metric_results = my_metrics(labels_test, test_predictions_class)

    auc_roc = sklearn.metrics.roc_auc_score(labels_test, test_predictions_class)

    print("TP:", metric_results[0])
    print("TN:", metric_results[1])
    print("FP:", metric_results[2])
    print("FN:", metric_results[3])

    print("precision:", metric_results[4])
    print("recall:", metric_results[5])
    print("f_measure:", metric_results[6])
    print("accuracy:", metric_results[7])
    print("auc_roc:", auc_roc)

    f2_measure = (5*metric_results[4]*metric_results[5]) / (4*metric_results[4]+metric_results[5])

    print("f2_measure:", f2_measure)

train_svm(train_images, train_labels, validation_images, validation_labels, test_images, test_labels)
