#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 03:15:59 2024

@author: saiful
"""
learning_rate = 0.00001  # 0001
batchsize=32

import sys
# sys.stdout = open(f"output_lr_{learning_rate}_batchsize_{batchsize}.txt", "w")

epochs = 10 # Number of epochs
print("learning_rate", learning_rate)
print("batchsize", batchsize)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,precision_recall_fscore_support 

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
import pandas as pd
import os
import random
import itertools
from torch.optim.lr_scheduler import StepLR
import time
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import (Array2D, Array3D, ClassLabel, Dataset, Features,
                      Sequence, Value)
from PIL import Image
from sklearn.model_selection import train_test_split 
from tqdm import tqdm
# from transformers import (AdamW, LayoutLMv2FeatureExtractor,
#                           LayoutLMv2ForSequenceClassification,
#                           LayoutLMv2Processor, LayoutLMv2Tokenizer)

# dataset_path = "../input/tobacco3482jpg/Tobacco3482-jpg"
dataset_path ="/data/saiful/pilot projects/Tobacco3482-jpg/Tobacco3482-jpg"

input_size = 224
ch = 3
test_size = 0.2
lr=5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def count_exp(path):
    c = {}
    for expression in os.listdir(path):
        class_path = os.path.join(path, expression)
        c[expression] = len(os.listdir(class_path))
    df = pd.DataFrame(c, index=["count"])
    return df

count = count_exp(dataset_path)
print(count)
count.transpose().plot(kind='bar');

def plot_images(path):
    plt.figure(figsize=(20,50))
    for idx, label in enumerate(os.listdir(path)):
        label_path = os.path.join(path, label)
        images = os.listdir(label_path)
        
        plt.subplot(5,2,idx+1)
        k = np.random.randint(0, len(images))
        im = Image.open(os.path.join(label_path, images[k]))
        plt.imshow(im, cmap='gray')
        plt.axis('off')
        plt.title(label)
        
    plt.show()
        
# plot_images(dataset_path)
#%%
labels = [label for label in os.listdir(dataset_path)]
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}
print('label2id:', label2id)

# loading data
images = []
labels = []

for label_folder, _, file_names in os.walk(dataset_path):
    if label_folder != dataset_path:
        label = label_folder.split("/")[-1]
        for _, _, image_names in os.walk(label_folder):
            relative_image_names = []
            for image_file in image_names:
                relative_image_names.append(
                    dataset_path + "/" + label + "/" + image_file)
            images.extend(relative_image_names)
            labels.extend([label] * len(relative_image_names))

data = pd.DataFrame.from_dict({'image_path': images, 'label': labels})

labels = list(set(labels))
data.head()
#%%
import pandas as pd

# Assuming 'data' is your DataFrame and it's already loaded
# Replace 'label' column values using the label2id mapping
data['label'] = data['label'].map(label2id)

# Display the updated DataFrame
data.head()
#%%
train_df, test_df = train_test_split(data, test_size=test_size)

print(f"Train Len:: {len(train_df)}\tTest Len:: {len(test_df)}")

#%%
train_df.label.value_counts()
test_df.label.value_counts()

#%%
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
#%%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL import Image, UnidentifiedImageError
import pandas as pd
from torch.utils.data import Dataset


class TobaccoDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pandas.DataFrame): Dataframe containing image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Filter the dataframe to include only .jpg files
        self.dataframe = dataframe[dataframe['image_path'].str.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        while True:
            try:
                img_name = self.dataframe.iloc[idx, 0]  # Assuming image paths are in the first column
                image = Image.open(img_name).convert('RGB')
                label = self.dataframe.iloc[idx, 1]  # Assuming labels are in the second column

                if self.transform:
                    image = self.transform(image)

                return image, label
            except UnidentifiedImageError:
                print(f"Skipping file {img_name} as it's not identifiable as an image.")
                idx = (idx + 1) % len(self.dataframe)  # Safely increment index, loop back if necessary
            except IndexError:
                # Handle case where all remaining files might be problematic
                raise RuntimeError("No valid image files found in the dataset.")

#%%DataLoader
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust if needed based on your image channel specifics
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset = TobaccoDataset(dataframe=train_df, transform=train_transform)
testset = TobaccoDataset(dataframe=test_df, transform=test_transform)
train_loader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=4)
val_loader = test_loader

#%%import

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
# %matplotlib inline
# python libraties
import os,itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# to make the results are reproducible
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)
#%% Model building"""

# feature_extract is a boolean that defines if we are finetuning or feature extracting.
# If feature_extract = False, the model is finetuned and all model parameters are updated.
# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    print("initialize_model() :", model_name)
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    elif model_name == "densenet":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "densenet161":
        model_ft = models.densenet161(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

"""You can change your backbone network, here are 4 different networks, each network also has sevaral versions. Considering the limited training data, we used the ImageNet pre-training model for fine-tuning. This can speed up the convergence of the model and improve the accuracy.

There is one thing you need to pay attention to, the input size of Inception is different from the others (299x299), you need to change the setting of compute_img_mean_std() function
"""
#%%model_1 = densenet
# resnet,vgg,densenet,inception
model_name_1 = 'densenet'
num_classes = 10
feature_extract = False
# Initialize the model for this run
model_ft_1, input_size = initialize_model(model_name_1, num_classes, feature_extract, use_pretrained=True)
# Define the device:
device = torch.device('cuda')
# Put the model on the device:
model_1 = model_ft_1.to(device)

# norm_mean = (0.49139968, 0.48215827, 0.44653124)
# norm_std = (0.24703233, 0.24348505, 0.26158768)
# # define the transformation of the train images.
# train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
#                                       transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
#                                       transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
#                                         transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
# # define the transformation of the val images.
# val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
#                                     transforms.Normalize(norm_mean, norm_std)])



#%%
# we use Adam optimizer, use cross entropy loss as our loss function
optimizer_1 = optim.Adam(model_1.parameters(), lr=1e-3)
criterion_1 = nn.CrossEntropyLoss().to(device)

"""## Step 3. Model training"""

# this function is used during training process, to calculation the loss and accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#%%train
total_loss_train, total_acc_train = [],[]
def train(train_loader, model, criterion, optimizer, epoch):

    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        images, labels = data
        N = images.size(0)
        # print('image shape:',images.size(0), 'label shape',labels.size(0))
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % 100 == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
    return train_loss.avg, train_acc.avg
#%%validate
def validate(val_loader, model, criterion, optimizer, epoch):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

            val_loss.update(criterion(outputs, labels).item())

    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
    return val_loss.avg, val_acc.avg

#%%train loop for model_1
print('# training with  :', model_name_1)

epoch_num = 50
best_val_acc = 0
total_loss_val, total_acc_val = [],[]
for epoch in range(1, epoch_num+1):
    loss_train, acc_train = train(train_loader, model_1, criterion_1, optimizer_1, epoch)
    loss_val, acc_val = validate(val_loader, model_1, criterion_1, optimizer_1, epoch)
    total_loss_val.append(loss_val)
    total_acc_val.append(acc_val)
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        # print('*****************************************************')
        # print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
        # print('*****************************************************')



#%%model_name_2 = 'resnet'
"""## Step 2_1. Model building"""
model_name_2 = 'resnet'
feature_extract = False
# Initialize the model for this run
model_ft_2, input_size = initialize_model(model_name_2, num_classes, feature_extract, use_pretrained=True)
# Define the device:
device = torch.device('cuda')
# Put the model on the device:
model_2 = model_ft_2.to(device)

optimizer_2 = optim.Adam(model_2.parameters(), lr=1e-3)
criterion_2 = nn.CrossEntropyLoss().to(device)

"""## Step 3_1. Model training"""

# epoch_num = 5
best_val_acc = 0
total_loss_val, total_acc_val = [],[]
print('training with ..', model_name_2)

for epoch in range(1, epoch_num+1):
    loss_train, acc_train = train(train_loader, model_2, criterion_2, optimizer_2, epoch)
    loss_val, acc_val = validate(val_loader, model_2, criterion_2, optimizer_2, epoch)
    total_loss_val.append(loss_val)
    total_acc_val.append(acc_val)
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        # print('*****************************************************')
        # print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
        # print('*****************************************************')

"""## Step 2_2. Model building"""
#%%model_name_3 = 'densenet161'
model_name_3 = 'densenet161'
num_classes = 10
feature_extract = False
# Initialize the model for this run
model_ft_3, input_size = initialize_model(model_name_3, num_classes, feature_extract, use_pretrained=True)
# Define the device:
device = torch.device('cuda')
# Put the model on the device:
model_3 = model_ft_3.to(device)

optimizer_3 = optim.Adam(model_3.parameters(), lr=1e-3)
criterion_3 = nn.CrossEntropyLoss().to(device)

"""## Step 3_2. Model training"""

# epoch_num = 5
best_val_acc = 0
total_loss_val, total_acc_val = [],[]
print('training with ..', model_name_3)

for epoch in range(1, epoch_num+1):
    loss_train, acc_train = train(train_loader, model_3, criterion_3, optimizer_3, epoch)
    loss_val, acc_val = validate(val_loader, model_3, criterion_3, optimizer_3, epoch)
    total_loss_val.append(loss_val)
    total_acc_val.append(acc_val)
    if acc_val > best_val_acc:
        best_val_acc = acc_val
        # print('*****************************************************')
        # print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
        # print('*****************************************************')

#%%Step 4. Model evaluation"""

import torch.nn.functional as F

def single_model_acc(val_loader, model):
    model.eval()
    with torch.no_grad():
        correctly_identified = 0
        total_images = 0
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            for i in range(N):
                soft_max_output = F.softmax(outputs[i])
                max_index = torch.argmax(soft_max_output)
                max_value = soft_max_output[max_index]
                total_images += 1
                correctly_identified += int(labels[i] == max_index)
        print("Correctly identified = ", correctly_identified, " Total_images = ", total_images, " Accuracy = ",(float(correctly_identified)/total_images)*100)

def combined_two_models_acc(val_loader, model_1, model_2):
    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        correctly_identified = 0
        total_images = 0
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs_1 = model_1(images)
            outputs_2 = model_2(images)
            for i in range(N):
                soft_max_output_1 = F.softmax(outputs_1[i])
                soft_max_output_2 = F.softmax(outputs_2[i])
                max_index_1 = torch.argmax(soft_max_output_1)
                max_index_2 = torch.argmax(soft_max_output_2)
                max_value_1 = soft_max_output_1[max_index_1]
                max_value_2 = soft_max_output_2[max_index_2]
                total_images += 1

                if max_index_1 == max_index_2:
                    correctly_identified += int(labels[i] == max_index_1)
                else:
                    if max_value_1 > max_value_2:
                        correctly_identified += int(labels[i] == max_index_1)
                    else:
                        correctly_identified += int(labels[i] == max_index_2)
        print("Correctly identified = ", correctly_identified, " Total_images = ", total_images, " Accuracy = ",(float(correctly_identified)/total_images)*100)

def combined_two_models_acc_rev(val_loader, model_1, model_2):
    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        correctly_identified_1 = 0
        correctly_identified_2 = 0
        correctly_identified_combined = 0
        total_images = 0
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs_1 = model_1(images)
            outputs_2 = model_2(images)

            for i in range(N):
                soft_max_output_1 = F.softmax(outputs_1[i])
                soft_max_output_2 = F.softmax(outputs_2[i])
                max_index_1 = torch.argmax(soft_max_output_1)
                max_index_2 = torch.argmax(soft_max_output_2)
                max_value_1 = soft_max_output_1[max_index_1]
                max_value_2 = soft_max_output_2[max_index_2]

                correctly_identified_1 += int(labels[i] == max_index_1)
                correctly_identified_2 += int(labels[i] == max_index_2)


                total_images += 1

                if max_index_1 == max_index_2:
                    correctly_identified_combined += int(labels[i] == max_index_1)

                elif max_value_1 > max_value_2:
                    correctly_identified_combined += int(labels[i] == max_index_1)

                else:
                    correctly_identified_combined += int(labels[i] == max_index_2)


        print("Correctly identified by model 1 = ", correctly_identified_1, " Total_images = ", total_images, " Accuracy = ",(float(correctly_identified_1)/total_images)*100)
        print()
        print("Correctly identified by model 2 = ", correctly_identified_2, " Total_images = ", total_images, " Accuracy = ",(float(correctly_identified_2)/total_images)*100)
        print()
        print("Correctly identified by combined model  = ", correctly_identified_combined, " Total_images = ", total_images, " Accuracy = ",(float(correctly_identified_combined)/total_images)*100)

def combined_three_models_acc(val_loader, model_1, model_2, model_3):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    with torch.no_grad():
        correctly_identified_1 = 0
        correctly_identified_2 = 0
        correctly_identified_3 = 0
        correctly_identified_combined = 0
        total_images = 0
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs_1 = model_1(images)
            outputs_2 = model_2(images)
            outputs_3 = model_3(images)
            for i in range(N):
                soft_max_output_1 = F.softmax(outputs_1[i])
                soft_max_output_2 = F.softmax(outputs_2[i])
                soft_max_output_3 = F.softmax(outputs_3[i])
                max_index_1 = torch.argmax(soft_max_output_1)
                max_index_2 = torch.argmax(soft_max_output_2)
                max_index_3 = torch.argmax(soft_max_output_3)
                max_value_1 = soft_max_output_1[max_index_1]
                max_value_2 = soft_max_output_2[max_index_2]
                max_value_3 = soft_max_output_3[max_index_3]

                correctly_identified_1 += int(labels[i] == max_index_1)
                correctly_identified_2 += int(labels[i] == max_index_2)
                correctly_identified_3 += int(labels[i] == max_index_3)


                total_images += 1

                if max_index_1 == max_index_2:
                    correctly_identified_combined += int(labels[i] == max_index_1)

                elif max_index_1 == max_index_3:
                    correctly_identified_combined += int(labels[i] == max_index_1)

                elif max_index_2 == max_index_3:
                    correctly_identified_combined += int(labels[i] == max_index_2)

                elif max_value_1 > max_value_2 and max_value_1 > max_value_3:
                    correctly_identified_combined += int(labels[i] == max_index_1)

                elif max_value_2 > max_value_1 and max_value_2 > max_value_3:
                    correctly_identified_combined += int(labels[i] == max_index_2)

                else:
                    correctly_identified_combined += int(labels[i] == max_index_3)


        print("Correctly identified by model 1 = ", correctly_identified_1, " Total_images = ", total_images, " Accuracy = ",(float(correctly_identified_1)/total_images)*100)
        print()
        print("Correctly identified by model 2 = ", correctly_identified_2, " Total_images = ", total_images, " Accuracy = ",(float(correctly_identified_2)/total_images)*100)
        print()
        print("Correctly identified by model 3 = ", correctly_identified_3, " Total_images = ", total_images, " Accuracy = ",(float(correctly_identified_3)/total_images)*100)
        print()
        print("Correctly identified by combined model  = ", correctly_identified_combined, " Total_images = ", total_images, " Accuracy = ",(float(correctly_identified_combined)/total_images)*100)

single_model_acc(val_loader, model_1)

single_model_acc(val_loader, model_2)

single_model_acc(val_loader, model_3)

#%%

combined_two_models_acc(val_loader, model_1, model_2)

combined_two_models_acc_rev(val_loader, model_1, model_2)

combined_three_models_acc(val_loader, model_1, model_2, model_3)


fig = plt.figure(num = 2)
fig1 = fig.add_subplot(2,1,1)
fig2 = fig.add_subplot(2,1,2)
fig1.plot(total_loss_train, label = 'training loss')
fig1.plot(total_acc_train, label = 'training accuracy')
fig2.plot(total_loss_val, label = 'validation loss')
fig2.plot(total_acc_val, label = 'validation accuracy')
plt.legend()
plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


print('= = = = = = = = execution finished = = = = = = = = = = = = =')
