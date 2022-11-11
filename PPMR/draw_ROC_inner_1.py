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
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Concatenate, UpSampling2D, Reshape
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import layers

import sklearn
from sklearn.model_selection import train_test_split

from natsort import natsorted

import cv2
import shutil
import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test')
parser.add_argument('--val')
parser.add_argument('--inner_fold')
args = parser.parse_args()

image_shape = (256,256,3)

margin = 5
theta = 5

latent_representation_dim = 128

kernel_size = 3
leaky_relu_alpha = 0.01

latent_dim = 1024

dropout_ratio = 0.2

class model_class():
    def dilated_block(self, x, kernel_size, channel):
        channel = int(channel/2)
        dil_1 = Conv2D(channel, (kernel_size,kernel_size), padding='same', dilation_rate=(1,1))(x)
        dil_1 = BatchNormalization()(dil_1)
        dil_1 = tf.nn.leaky_relu(dil_1, alpha=leaky_relu_alpha)

        dil_2 = Conv2D(channel, (kernel_size,kernel_size), padding='same', dilation_rate=(2,2))(x)
        dil_2 = BatchNormalization()(dil_2)
        dil_2 = tf.nn.leaky_relu(dil_2, alpha=leaky_relu_alpha)

        dil_3 = Conv2D(channel, (kernel_size,kernel_size), padding='same', dilation_rate=(3,3))(x)
        dil_3 = BatchNormalization()(dil_3)
        dil_3 = tf.nn.leaky_relu(dil_3, alpha=leaky_relu_alpha)

        return Concatenate(axis=3)([dil_1, dil_2, dil_3])
    
    def SE_block(self, x, ratio = 16):
        filters = x.shape[-1]
        # print("x:", x)
        se_shape = (1,1,filters)

        squeeze = GlobalAveragePooling2D()(x)
        squeeze = Reshape(se_shape)(squeeze)

        # print("squeeze:", squeeze)
        squeeze = Dense(filters // ratio)(squeeze)
        # print("squeeze:", squeeze)
        squeeze = tf.nn.leaky_relu(squeeze, alpha=leaky_relu_alpha)
        # print("squeeze:", squeeze)
        squeeze = Dense(filters, activation = 'sigmoid')(squeeze)
        # print("squeeze:", squeeze)

        squeeze = x * squeeze

        # print("excite:", squeeze)

        return x + squeeze

# x: Tensor("concatenate/Identity:0", shape=(None, 256, 256, 48), dtype=float32)
# squeeze: Tensor("reshape/Identity:0", shape=(None, 1, 1, 48), dtype=float32)
# squeeze: Tensor("dense/Identity:0", shape=(None, 1, 1, 3), dtype=float32)
# squeeze: Tensor("LeakyRelu_3:0", shape=(None, 1, 1, 3), dtype=float32)
# squeeze: Tensor("dense_1/Identity:0", shape=(None, 1, 1, 48), dtype=float32)
# excite: Tensor("mul:0", shape=(None, 256, 256, 48), dtype=float32)

    def model(self):
        inputs = Input(shape=image_shape)
        x = self.dilated_block(inputs, kernel_size, int(latent_dim/32))
        x = self.SE_block(x)
        x = MaxPooling2D()(x)

        GAP_1 = GlobalAveragePooling2D()(x)

        x = self.dilated_block(x, kernel_size, int(latent_dim/16))
        x = self.SE_block(x)
        x = MaxPooling2D()(x)

        GAP_2 = GlobalAveragePooling2D()(x)

        x = self.dilated_block(x, kernel_size, int(latent_dim/8))
        x = self.SE_block(x)
        x = MaxPooling2D()(x)

        GAP_3 = GlobalAveragePooling2D()(x)

        x = self.dilated_block(x, kernel_size, int(latent_dim/4))
        x = self.SE_block(x)
        x = MaxPooling2D()(x)

        GAP_4 = GlobalAveragePooling2D()(x)

        x = self.dilated_block(x, kernel_size, int(latent_dim/2))
        x = self.SE_block(x)
        x = MaxPooling2D()(x)

        GAP_5 = GlobalAveragePooling2D()(x)

        x = self.dilated_block(x, kernel_size, latent_dim)
        x = self.SE_block(x)
        x = GlobalAveragePooling2D()(x)

        x = Concatenate(axis=-1)([GAP_1, GAP_2, GAP_3, GAP_4, GAP_5, x])

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = tf.nn.leaky_relu(x, alpha=leaky_relu_alpha)
        x = Dropout(dropout_ratio)(x)

        x = Dense(128)(x)

        return Model(inputs,x)

c = np.ones(latent_representation_dim)

half_model = model_class().model()

# print(half_model.summary())

print("model complexity:", half_model.count_params())

learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)

checkpoint_dir = './inner_fold_' + str(args.inner_fold)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=half_model)

ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

# checkpoint.restore(ckpt_manager.latest_checkpoint)

dataset_dir = '/home/lingfeng/Downloads/PMGstudycaseslabelled'
dataset_dir_normal = '/home/lingfeng/Downloads/PMGControlsEditedDec2021'

def test_usable_label(slice_):
    label_num = slice_.split('.')[0].split('_')[-1]
    label_num = int(label_num)
    if(label_num == 3):
        return False
    else:
        return True

def get_label(slices):
    label_num = slices.split('.')[0].split('_')[-1]
    label_num = int(label_num)
    if(label_num == 1):
        label = 'Yes'
    else:
        label = 'No'
    return label

val_value = args.val
val_value = val_value.split(" ")

print("val_value:", val_value)

test_value = args.test
test_value = test_value.split(" ")

print("test_value:", test_value)

def dataset_collection_func(): # or add some patients' normal slices

    train_filepaths=[]
    train_labels=[]
    validation_filepaths=[]
    validation_labels=[]
    test_filepaths=[]
    test_labels=[]

    num = 0

    print("patient id sequence:", natsorted(os.listdir(dataset_dir)))
    for patient_item in tqdm(natsorted(os.listdir(dataset_dir))):
        patient_path_saved = os.path.join(dataset_dir, patient_item)

        normal_abnormal = os.listdir(patient_path_saved)
        if(normal_abnormal[0].__contains__("cor")):
            coronal_path_saved = os.path.join(patient_path_saved, normal_abnormal[0])
            # control_path_saved = os.path.join(patient_path_saved, normal_abnormal[1])

        else:
            coronal_path_saved = os.path.join(patient_path_saved, normal_abnormal[1])
            # control_path_saved = os.path.join(patient_path_saved, normal_abnormal[0])

        coronal_path_list = os.listdir(coronal_path_saved)

        for slice_num in range(len(coronal_path_list)):
            if(test_usable_label(coronal_path_list[slice_num])):
                pass
            else:
                continue
            
            if(patient_item == str(val_value[0]) or patient_item == str(val_value[1]) or patient_item == str(val_value[2]) or patient_item == str(val_value[3])): # validation
                validation_filepaths.append(os.path.join(coronal_path_saved, coronal_path_list[slice_num]))
                validation_labels.append(get_label(coronal_path_list[slice_num]))
            elif(patient_item == str(test_value[0]) or patient_item == str(test_value[1]) or patient_item == str(test_value[2]) or patient_item == str(test_value[3])):
                test_filepaths.append(os.path.join(coronal_path_saved, coronal_path_list[slice_num]))
                test_labels.append(get_label(coronal_path_list[slice_num]))
            elif(patient_item != '33'):
                train_filepaths.append(os.path.join(coronal_path_saved, coronal_path_list[slice_num]))
                train_labels.append(get_label(coronal_path_list[slice_num]))
            
        num += 1

    num = 0

    print("patient id sequence:", natsorted(os.listdir(dataset_dir_normal)))
    for patient_item in tqdm(natsorted(os.listdir(dataset_dir_normal))):
        patient_path_saved = os.path.join(dataset_dir_normal, patient_item)

        patient_sub_dir = os.listdir(patient_path_saved)

        for sub_dir in range(len(patient_sub_dir)):
            control_path_saved = os.path.join(patient_path_saved, patient_sub_dir[sub_dir])
            control_path_list = os.listdir(control_path_saved)
            for slice_num in range(len(control_path_list)):

                if(patient_item == str(val_value[0]) or patient_item == str(val_value[1]) or patient_item == str(val_value[2]) or patient_item == str(val_value[3])): # validation
                    validation_filepaths.append(os.path.join(control_path_saved, control_path_list[slice_num]))
                    validation_labels.append('No')
                elif(patient_item == str(test_value[0]) or patient_item == str(test_value[1]) or patient_item == str(test_value[2]) or patient_item == str(test_value[3])):
                    test_filepaths.append(os.path.join(control_path_saved, control_path_list[slice_num]))
                    test_labels.append('No')
                elif(patient_item != '33'):
                    train_filepaths.append(os.path.join(control_path_saved, control_path_list[slice_num]))
                    train_labels.append('No')

        num += 1

    train_filepaths = np.array(train_filepaths)
    train_labels = np.array(train_labels)
    validation_filepaths = np.array(validation_filepaths)
    validation_labels = np.array(validation_labels)
    test_filepaths = np.array(test_filepaths)
    test_labels = np.array(test_labels)

    # print(train_labels[:10])
    train_labels = np.where(train_labels=='No', 0, 1)
    # print(train_labels[:10])
    validation_labels = np.where(validation_labels=='No', 0, 1)
    test_labels = np.where(test_labels=='No', 0, 1)

    return train_filepaths, train_labels, validation_filepaths, validation_labels, test_filepaths, test_labels

def get_image(img_path):
    img_array = Image.open(img_path)
    img_array.load()
    img_array = img_array.resize((image_shape[0],image_shape[1]))
    img_array = np.asarray(img_array).astype(np.float32)
    # print("img_array.shape:", img_array.shape)

    img_array = img_array / 255.

    return img_array

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

    print("euclidean_norm:", euclidean_norm)
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

    return np.array(loss), metric_results, dist

# add tensorboard
def train(train_filepaths, train_labels, validation_filepaths, validation_labels, epochs, BATCH_SIZE):

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
        train_filepaths, train_labels = train_filepaths[idx], train_labels[idx]

        print("train epoch = ", epoch)
        for index in range(0, len(train_labels)-BATCH_SIZE, BATCH_SIZE):
            label_batch = [] # always load the same batch
            for i in range(BATCH_SIZE):
                img_path = train_filepaths[index+i]

                img_array = get_image(img_path)

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
                validation_filepaths, validation_labels = validation_filepaths[idx_val], validation_labels[idx_val]

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
                        img_path = validation_filepaths[val_index+i]

                        img_array = get_image(img_path)

                        img_array = np.expand_dims(img_array, axis=0)

                        if(i == 0):
                            image_batch = img_array
                        else:
                            image_batch = np.concatenate((image_batch, img_array), axis=0)

                        label_batch.append(validation_labels[val_index+i])

                    label_batch = np.array(label_batch)
                    label_batch = np.expand_dims(label_batch, axis=-1)

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

                    f = open("best_auc_roc_binary_classification.txt", "w")
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

def evaluation(test_filepaths, test_labels, tag_name):

    checkpoint.restore(ckpt_manager.latest_checkpoint)

    labels = []
    predictions = []
    dists = []

    for i in range(len(test_labels)):
        img_path = test_filepaths[i]

        img_array = get_image(img_path)

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

    # print("TP:", metric_results[0])
    # print("TN:", metric_results[1])
    # print("FP:", metric_results[2])
    # print("FN:", metric_results[3])

    # print("precision:", metric_results[4])
    # print("recall:", metric_results[5])
    # print("f_measure:", metric_results[6])
    # print("accuracy:", metric_results[7])
    # print("auc_roc:", auc_roc)

    f = open("results.txt", "a")
    f.write("\ninner_fold_"+str(args.inner_fold)+"\n")
    f.write(tag_name + "\n")

    f.write("TP:"+str(metric_results[0])+"\n")
    f.write("TN:"+str(metric_results[1])+"\n")
    f.write("FP:"+str(metric_results[2])+"\n")
    f.write("FN:"+str(metric_results[3])+"\n")

    f.write("precision:"+str(metric_results[4])+"\n")
    f.write("recall:"+str(metric_results[5])+"\n")
    f.write("f_measure:"+str(metric_results[6])+"\n")
    f.write("accuracy:"+str(metric_results[7])+"\n")
    f.write("auc_roc:"+str(auc_roc)+"\n")
    f.close()

plt.title('Receiver Operating Characteristic')

def evaluation_roc(test_filepaths, test_labels, tag_name):

    checkpoint.restore(ckpt_manager.latest_checkpoint)

    labels = []
    predictions = []
    dists = []

    for i in range(len(test_labels)):
        img_path = test_filepaths[i]

        img_array = get_image(img_path)

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

    # auc_roc = sklearn.metrics.roc_auc_score(labels, dists)

    fpr, tpr, threshold = sklearn.metrics.roc_curve(labels, dists)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, label = tag_name + ' AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    # plt.show()

epochs = 200
BATCH_SIZE = 64
train_filepaths, train_labels, validation_filepaths, validation_labels, test_filepaths, test_labels = dataset_collection_func()
# train(train_filepaths, train_labels, validation_filepaths, validation_labels, epochs, BATCH_SIZE)
# evaluation(validation_filepaths, validation_labels, "validation")
# evaluation(test_filepaths, test_labels, "testing")
evaluation_roc(validation_filepaths, validation_labels, "validation")
evaluation_roc(test_filepaths, test_labels, "testing")

plt.savefig('ROC.png')

# python3 draw_ROC_inner_1.py --test="13 23 34 6" --val="31 29 3 28" --inner_fold=1