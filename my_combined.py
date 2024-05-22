#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:37:31 2024

@author: saiful
"""
import os, cv2,itertools
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
import torch.nn.functional as F

# to make the results are reproducible
np.random.seed(10)
torch.manual_seed(10)
torch.cuda.manual_seed(10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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