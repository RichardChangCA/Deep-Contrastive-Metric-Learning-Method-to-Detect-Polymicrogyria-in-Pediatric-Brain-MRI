# Code reference: https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/blob/main/main.py
# Code reference: https://github.com/youngjae-avikus/PaDiM-EfficientNet

import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

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
parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
args = parser.parse_args()

SEED = 1997

import random
from random import sample

from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# device = torch.device('cpu')

if args.arch == 'resnet18':
    model = resnet18(pretrained=True, progress=True)
    t_d = 448
    d = 100
elif args.arch == 'wide_resnet50_2':
    model = wide_resnet50_2(pretrained=True, progress=True)
    t_d = 1792
    d = 550
model.to(device)
model.eval()
random.seed(1024)
torch.manual_seed(1024)

if use_cuda:
    torch.cuda.manual_seed_all(1024)

idx = torch.tensor(sample(range(0, t_d), d))

# set model's intermediate outputs
outputs = []

def hook(module, input, output):
    outputs.append(output)

model.layer1[-1].register_forward_hook(hook)
model.layer2[-1].register_forward_hook(hook)
model.layer3[-1].register_forward_hook(hook)

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

# image_shape = (256,256,3)
image_shape = (224,224,3)

dataset_dir = '/home/lingfeng/Downloads/PMGstudycaseslabelled'
dataset_dir_normal = '/home/lingfeng/Downloads/PMGControlsEditedDec2021'
# dataset_dir_normal = '/home/lingfeng/Downloads/Controls_5'

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

def my_metrics(y_true, y_pred, threshold=0.5):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

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

    f1_measure = (2. * precision * recall)/(precision + recall + 1.0e-4)

    f2_measure = (5. * precision * recall)/(4. * precision + recall + 1.0e-4)

    accuracy = (TP + TN) / (TP + TN + FP + FN+1.0e-4)

    # print("TP:", TP)
    # print("TN:", TN)
    # print("FP:", FP)
    # print("FN:", FN)

    # print("precision:", precision)
    # print("recall:", recall)
    # print("f1_measure:", f1_measure)
    # print("f2_measure:", f2_measure)
    # print("accuracy:", accuracy)

    return np.array([TP, TN, FP, FN, precision, recall, f1_measure, f2_measure, accuracy])

train_filepaths, train_labels, validation_filepaths, validation_labels, test_filepaths, test_labels = dataset_collection_func()

# print("train_labels:", train_labels)
# print("len(train_labels):", len(train_labels))
train_filepaths, train_labels = train_filepaths[train_labels==0], train_labels[train_labels==0] # only get normal images for training
# print("train_labels:", train_labels)
# print("len(train_labels):", len(train_labels))

idx_normal = np.random.permutation(len(train_labels))
train_filepaths, train_labels = train_filepaths[idx_normal], train_labels[idx_normal]

BATCH_SIZE = 1 # 32

# for index in tqdm(range(0, len(train_labels)-BATCH_SIZE, BATCH_SIZE)): # out of RAM due to embedding_vectors_list
for index in tqdm(range(0, 4000, BATCH_SIZE)):
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    for i in range(BATCH_SIZE):
        img_path = train_filepaths[index+i]

        img_array = get_image(img_path)

        img_array = np.array(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        if(i == 0):
            image_batch = img_array
        else:
            image_batch = np.concatenate((image_batch, img_array), axis=0)

    image_batch = np.transpose(image_batch, (0, 3, 1, 2))

    x = torch.tensor(image_batch)

    # print("outputs before model:", outputs)
    with torch.no_grad():
        _ = model(x.to(device))
    # print("outputs after model:", outputs)

    # get intermediate layer outputs
    for k, v in zip(train_outputs.keys(), outputs):
        train_outputs[k].append(v.cpu().detach())

    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)

    # initialize hook outputs
    outputs = []

    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

    B, C, H, W = embedding_vectors.size() # B is batch size here
    embedding_vectors = embedding_vectors.view(B, C, H * W)

    if(index==0):
        embedding_vectors_list = embedding_vectors
    else:
        embedding_vectors_list = torch.cat((embedding_vectors_list, embedding_vectors), 0)

    # print("embedding_vectors_list.size():", embedding_vectors_list.size())

# # here
mean = torch.mean(embedding_vectors_list, dim=0).numpy()
cov = torch.zeros(C, C, H * W).numpy()
I = np.identity(C)

# print("C:", C)

print("calculating cov")
for i in tqdm(range(H * W)):
    # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
    cov[:, :, i] = np.cov(embedding_vectors_list[:, :, i].numpy(), rowvar=False) + 0.01 * I
# save learned distribution
train_outputs = [mean, cov]

import gc
del embedding_vectors_list
gc.collect()

mean = torch.Tensor(train_outputs[0]).to(device)
cov_inv = torch.Tensor(train_outputs[1]).to(device)

print("validation")

gt_list = []

for index in tqdm(range(0, len(validation_labels)-BATCH_SIZE, BATCH_SIZE)):
    validation_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    label_batch = []
    for i in range(BATCH_SIZE):
        img_path = validation_filepaths[index+i]

        img_array = get_image(img_path)

        img_array = np.array(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        if(i == 0):
            image_batch = img_array
        else:
            image_batch = np.concatenate((image_batch, img_array), axis=0)

        label_batch.append(validation_labels[index+i])

    label_batch = np.array(label_batch)
    label_batch = np.expand_dims(label_batch, axis=-1)

    image_batch = np.transpose(image_batch, (0, 3, 1, 2))

    x = torch.tensor(image_batch)
    y = torch.tensor(label_batch)

    gt_list.extend(y.cpu().detach().numpy())

    with torch.no_grad():
        _ = model(x.to(device))
    # get intermediate layer outputs
    for k, v in zip(validation_outputs.keys(), outputs):
        validation_outputs[k].append(v.cpu().detach())

    for k, v in validation_outputs.items():
        validation_outputs[k] = torch.cat(v, 0)

    # initialize hook outputs
    outputs = []

    embedding_vectors = validation_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, validation_outputs[layer_name])

    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()

    embedding_vectors = torch.Tensor(embedding_vectors).to(device)

    dist_list_one_sample = torch.zeros(size=(H*W, B))
    for i in range(H * W):
        delta = embedding_vectors[:, :, i] - mean[:, i]
        m_dist = torch.sqrt(torch.diag(torch.mm(torch.mm(delta, cov_inv[:, :, i]), delta.t())))
        dist_list_one_sample[i] = m_dist

    dist_list_one_sample = np.array(dist_list_one_sample).transpose(1, 0).reshape(B, H, W)

    if(index == 0):
        dist_list = dist_list_one_sample
    else:
        dist_list = np.concatenate((dist_list, dist_list_one_sample), 0)

# upsample
dist_list = torch.tensor(dist_list)
score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                            align_corners=False).squeeze().numpy()

# apply gaussian smoothing on the score map
for i in range(score_map.shape[0]):
    score_map[i] = gaussian_filter(score_map[i], sigma=4)

# Normalization
max_score = score_map.max()
min_score = score_map.min()
scores = (score_map - min_score) / (max_score - min_score)

img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
gt_list = np.asarray(gt_list)
validation_roc_auc = roc_auc_score(gt_list, img_scores)

precision, recall, thresholds = precision_recall_curve(gt_list, img_scores)
a = 5 * precision * recall
b = 4 * precision + recall
f2 = np.divide(a, b, out=np.zeros_like(a), where=b != 0) # (5 * Precision * Recall) / (4 * Precision + Recall)
threshold = thresholds[np.argmax(f2)]

print("testing")

gt_list = []

for index in tqdm(range(0, len(test_labels)-BATCH_SIZE, BATCH_SIZE)):
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    label_batch = []
    for i in range(BATCH_SIZE):
        img_path = test_filepaths[index+i]

        img_array = get_image(img_path)

        img_array = np.array(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        if(i == 0):
            image_batch = img_array
        else:
            image_batch = np.concatenate((image_batch, img_array), axis=0)

        label_batch.append(test_labels[index+i])

    label_batch = np.array(label_batch)
    label_batch = np.expand_dims(label_batch, axis=-1)

    image_batch = np.transpose(image_batch, (0, 3, 1, 2))

    x = torch.tensor(image_batch)
    y = torch.tensor(label_batch)

    gt_list.extend(y.cpu().detach().numpy())

    with torch.no_grad():
        _ = model(x.to(device))
    # get intermediate layer outputs
    for k, v in zip(test_outputs.keys(), outputs):
        test_outputs[k].append(v.cpu().detach())

    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # initialize hook outputs
    outputs = []

    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()

    embedding_vectors = torch.Tensor(embedding_vectors).to(device)

    dist_list_one_sample = torch.zeros(size=(H*W, B))
    for i in range(H * W):
        delta = embedding_vectors[:, :, i] - mean[:, i]
        m_dist = torch.sqrt(torch.diag(torch.mm(torch.mm(delta, cov_inv[:, :, i]), delta.t())))
        dist_list_one_sample[i] = m_dist

    dist_list_one_sample = np.array(dist_list_one_sample).transpose(1, 0).reshape(B, H, W)

    if(index == 0):
        dist_list = dist_list_one_sample
    else:
        dist_list = np.concatenate((dist_list, dist_list_one_sample), 0)

# upsample
dist_list = torch.tensor(dist_list)
score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                            align_corners=False).squeeze().numpy()

# apply gaussian smoothing on the score map
for i in range(score_map.shape[0]):
    score_map[i] = gaussian_filter(score_map[i], sigma=4)

# Normalization
max_score = score_map.max()
min_score = score_map.min()
scores = (score_map - min_score) / (max_score - min_score)

img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
gt_list = np.asarray(gt_list)
img_roc_auc = roc_auc_score(gt_list, img_scores)
# print('image ROCAUC: %.3f' % (img_roc_auc))

metric_results = my_metrics(gt_list, img_scores, threshold)

f = open("results.txt", "a")
f.write("\ninner_fold_"+str(args.inner_fold)+"\n")

f.write("validation_roc_auc:"+str(validation_roc_auc)+"\n")
f.write("threshold:"+str(threshold)+"\n")

f.write("TP:"+str(metric_results[0])+"\n")
f.write("TN:"+str(metric_results[1])+"\n")
f.write("FP:"+str(metric_results[2])+"\n")
f.write("FN:"+str(metric_results[3])+"\n")

f.write("precision:"+str(metric_results[4])+"\n")
f.write("recall:"+str(metric_results[5])+"\n")
f.write("f1_measure:"+str(metric_results[6])+"\n")
f.write("f2_measure:"+str(metric_results[7])+"\n")
f.write("accuracy:"+str(metric_results[8])+"\n")

f.write("auc_roc:"+str(img_roc_auc)+"\n")
f.close()