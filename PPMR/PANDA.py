# Code reference: https://github.com/talreiss/PANDA/tree/master

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
parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
parser.add_argument('--diag_path', default='../fisher_diagonal.pth', help='fim diagonal path')
# The Fisher file is downloaded from https://drive.google.com/file/d/12PTw4yNqp6bgCHj94vcowwb37m81rvpY/view
# This Fisher is corresponding to resnet_type=152
parser.add_argument('--epochs', default=50, type=int, metavar='epochs', help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
args = parser.parse_args()

import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ResNet
from copy import deepcopy

import faiss # pip install faiss-gpu or pip install faiss-cpu
import torch.optim as optim

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

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

train_filepaths, train_labels = train_filepaths[train_labels==0], train_labels[train_labels==0] # only get normal images for training

BATCH_SIZE = 32

class CompactnessLoss(nn.Module):
    def __init__(self, center):
        super(CompactnessLoss, self).__init__()
        self.center = center

    def forward(self, inputs):
        m = inputs.size(1)
        variances = (inputs - self.center).norm(dim=1).pow(2) / m
        return variances.mean()

class EWCLoss(nn.Module):
    def __init__(self, frozen_model, fisher, lambda_ewc=1e4):
        super(EWCLoss, self).__init__()
        self.frozen_model = frozen_model
        self.fisher = fisher
        self.lambda_ewc = lambda_ewc

    def forward(self, cur_model):
        loss_reg = 0
        for (name, param), (_, param_old) in zip(cur_model.named_parameters(), self.frozen_model.named_parameters()):
            if 'fc' in name:
                continue
            loss_reg += torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        return self.lambda_ewc * loss_reg

def get_resnet_model(resnet_type=152):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=True, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=True, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=True, progress=True)
    else:  #152
        return ResNet.resnet152(pretrained=True, progress=True)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours) # https://github.com/facebookresearch/faiss/issues/493
    return np.sum(D, axis=1)

model = get_resnet_model(resnet_type=args.resnet_type)
model = model.to(device)

ewc_loss = None

# Freezing Pre-trained model for EWC
frozen_model = deepcopy(model).to(device)
frozen_model.eval()
freeze_model(frozen_model)
fisher = torch.load(args.diag_path)
ewc_loss = EWCLoss(frozen_model, fisher)

freeze_parameters(model)

model.eval()

def get_score(model, device, train_filepaths, train_labels, validation_filepaths, validation_labels):

    idx_normal = np.random.permutation(len(train_labels))
    train_filepaths, train_labels = train_filepaths[idx_normal], train_labels[idx_normal]

    train_feature_space = []
    with torch.no_grad():
        for index in tqdm(range(0, len(train_labels)-BATCH_SIZE, BATCH_SIZE)): # out of RAM due to embedding_vectors_list
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
            image_batch = torch.Tensor(image_batch)

            image_batch = image_batch.to(device)
            _, features = model(image_batch)
            # print("features.size():",features.size()) # [32, 2048]
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    test_feature_space = []
    with torch.no_grad():
        
        for index in tqdm(range(0, len(validation_labels))): # out of RAM due to embedding_vectors_list
            img_path = validation_filepaths[index]

            img_array = get_image(img_path)

            img_array = np.array(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            image_batch = img_array

            image_batch = np.transpose(image_batch, (0, 3, 1, 2))
            image_batch = torch.Tensor(image_batch)

            image_batch = image_batch.to(device)
            _, features = model(image_batch)
            features = torch.unsqueeze(features, 0)
            # print("features.size():",features.size()) # [1, 2048]
            test_feature_space.append(features)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()

    # print("train_feature_space.shape:", train_feature_space.shape)
    # print("test_feature_space.shape:", test_feature_space.shape)
    distances = knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(validation_labels, distances)

    return auc, train_feature_space, distances

def run_epoch(model, train_filepaths, optimizer, criterion, device, ewc_loss):
    running_loss = 0.0
    iter_ = 0
    for index in tqdm(range(0, len(train_labels)-BATCH_SIZE, BATCH_SIZE)): # out of RAM due to embedding_vectors_list
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
        image_batch = torch.Tensor(image_batch)

        images = image_batch.to(device)

        optimizer.zero_grad()

        _, features = model(images)

        loss = criterion(features) + ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

        iter_ += 1

    return running_loss / iter_

auc, feature_space, _ = get_score(model, device, train_filepaths, train_labels, validation_filepaths, validation_labels)
print('Epoch: {}, AUROC is: {} \n'.format(0, auc))
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9)

best_auc = auc

center = torch.FloatTensor(feature_space).mean(dim=0)
criterion = CompactnessLoss(center.to(device))
for epoch in tqdm(range(args.epochs)):
    running_loss = run_epoch(model, train_filepaths, optimizer, criterion, device, ewc_loss)
    print('Epoch: {}, Loss: {} \n'.format(epoch + 1, running_loss))
    auc, feature_space, _ = get_score(model, device, train_filepaths, train_labels, validation_filepaths, validation_labels)
    print('Epoch: {}, AUROC is: {} \n'.format(epoch + 1, auc))

    if(epoch == 0 or auc >= best_auc):
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                }, 'inner_fold_' + str(args.inner_fold) + '_best_val_auc.pt')

        best_auc = auc
        best_auc_epoch = epoch

torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
        }, 'inner_fold_' + str(args.inner_fold)+'_final.pt')

checkpoint = torch.load('inner_fold_' + str(args.inner_fold)+'_best_val_auc.pt')
model.load_state_dict(checkpoint['model_state_dict'])

validation_roc_auc, feature_space, distance = get_score(model, device, train_filepaths, train_labels, validation_filepaths, validation_labels)

precision, recall, thresholds = precision_recall_curve(validation_labels, distance)
a = 5 * precision * recall
b = 4 * precision + recall
f2 = np.divide(a, b, out=np.zeros_like(a), where=b != 0) # (5 * Precision * Recall) / (4 * Precision + Recall)
threshold = thresholds[np.argmax(f2)]

test_auc, feature_space, distance = get_score(model, device, train_filepaths, train_labels, test_filepaths, test_labels)

metric_results = my_metrics(test_labels, distance, threshold)

f = open("results.txt", "a")
f.write("\ninner_fold_"+str(args.inner_fold)+"\n")

f.write("validation_roc_auc:"+str(validation_roc_auc)+"\n")
f.write("best_auc_epoch:"+str(best_auc_epoch)+"\n")
f.write("threshold:"+str(threshold)+"\n")

f.write("Test Performance:\n")
f.write("TP:"+str(metric_results[0])+"\n")
f.write("TN:"+str(metric_results[1])+"\n")
f.write("FP:"+str(metric_results[2])+"\n")
f.write("FN:"+str(metric_results[3])+"\n")

f.write("precision:"+str(metric_results[4])+"\n")
f.write("recall:"+str(metric_results[5])+"\n")
f.write("f1_measure:"+str(metric_results[6])+"\n")
f.write("f2_measure:"+str(metric_results[7])+"\n")
f.write("accuracy:"+str(metric_results[8])+"\n")

f.write("auc_roc:"+str(test_auc)+"\n")
f.close()