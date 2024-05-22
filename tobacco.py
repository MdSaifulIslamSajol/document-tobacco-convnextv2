#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 03:15:59 2024

@author: saiful
"""
learning_rate = 0.0001  # 0001
batchsize= 32

import sys
# sys.stdout = open(f"Adam_output_lr_{learning_rate}_batchsize_{batchsize}_b.txt", "w")

epochs = 25 # Number of epochs 25
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
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda import memory_allocated
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

#%%
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset = TobaccoDataset(dataframe=train_df, transform=train_transform)
testset = TobaccoDataset(dataframe=test_df, transform=test_transform)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=4)

#%%

from my_models import ViTForImageClassification, ConvNextV2ForImageClassification, Swinv2ForImageClassification
from my_models import  ImageGPTForImageClassification, CvtForImageClassification
from my_models import  EfficientFormerForImageClassification, PvtV2ForImageClassification, MobileViTV2ForImageClassification
from my_models import  EfficientNetForImageClassification, BeitForImageClassification
from my_models import  BitForImageClassification, FocalNetForImageClassification


images, labels = iter(trainloader).next()
print(images.shape)
print(labels.shape)

#%%


# =============================================================================
# 
# =============================================================================
def train_test_plot(method_name = "_"):
    
    print("\n\nTraining started with ", method_name)

    history=[]
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    epoch_list=[]
    
    if method_name == "Swinv2ForImageClassification":
        model_ft = Swinv2ForImageClassification().to(device)
        
    elif method_name == "ViTForImageClassification":
        model_ft = ViTForImageClassification().to(device)
        
    elif method_name == "ConvNextV2ForImageClassification":
        model_ft = ConvNextV2ForImageClassification().to(device)
        
    elif method_name == "ImageGPTForImageClassification":
        model_ft = ImageGPTForImageClassification().to(device)
 
    elif method_name == "CvtForImageClassification":
        model_ft = CvtForImageClassification().to(device)
 
    elif method_name == "EfficientFormerForImageClassification":
        model_ft = EfficientFormerForImageClassification().to(device)
        
    elif method_name == "PvtV2ForImageClassification":
        model_ft = PvtV2ForImageClassification().to(device)
            
    elif method_name == "MobileViTV2ForImageClassification":
        model_ft = MobileViTV2ForImageClassification().to(device)
        
    elif method_name == "EfficientNetForImageClassification":
        model_ft = EfficientNetForImageClassification().to(device)
        
    elif method_name == "BeitForImageClassification":
        model_ft = BeitForImageClassification().to(device)
        
    elif method_name == "BitForImageClassification":
        model_ft = BitForImageClassification().to(device)
        
    elif method_name == "FocalNetForImageClassification":
        model_ft = FocalNetForImageClassification().to(device)
        

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer  = optim.Adam(model_ft.classifier.parameters(),lr = 0.001)
    optimizer  = optim.Adam(model_ft.parameters(),lr = learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.35)
    best_model_path = "./best_model.pt" 
    best_valid_acc = 0.0
    
    # Initialize timing and memory tracking
    start_time = time.time()
    max_memory_used = 0
    # Tracking number of parameters
    total_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)

    for epoch in range(epochs):
        
        epoch_start = time.time()
        
        # print("Epoch: {}/{}".format(epoch+1, epochs))
        # Set to training mode
        model_ft.train()
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
    
        # training on trainloader
        for i, (inputs, labels) in enumerate(trainloader):
            # inputs = inputs.long()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Track GPU memory
            before_train_memory = memory_allocated(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = model_ft(inputs)
            outputs = outputs.logits
    
            # Compute loss
            loss = criterion(outputs, labels)
            
            #
            # loss.requires_grad = True
            #
            
            # Backpropagate the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
            
            after_train_memory = memory_allocated(device)
            max_memory_used = max(max_memory_used, after_train_memory - before_train_memory)
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            # print(" Training Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
            
            # print("train predictions:",predictions)
            # print("train labels:", labels)
    
    
        # Validation - No gradient tracking needed
        with torch.no_grad():
            # Set to evaluation mode
            model_ft.eval()
            # Validation loop
            for j, (inputs, labels) in enumerate(testloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass - compute outputs on input data using the model
                start_inference_time = time.time()
                outputs = model_ft(inputs)
                inference_time = time.time() - start_inference_time
                outputs = outputs.logits
    
                # Compute loss
                loss = criterion(outputs, labels)
                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)
                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                
                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)
                # print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
                
                
        # Find average training loss and training accuracy per epoch
        avg_train_loss = train_loss/len(trainset) 
        avg_train_acc = train_acc/float(len(trainset))
        
        # Find average validation loss and validation accuracy per epoch
        avg_valid_loss = valid_loss/len(testset) 
        avg_valid_acc = valid_acc/float(len(testset))
        
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        valid_loss_list.append(avg_valid_loss)
        valid_acc_list.append(avg_valid_acc)
        epoch_list.append(epoch+1)
        # history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        epoch_end = time.time()
        # print("\n ##  Training and validation loss and  accuracy per epoch")
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, Validation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        # Check for best accuracy
        if avg_valid_acc > best_valid_acc:
            best_valid_acc = avg_valid_acc
            torch.save(model_ft.state_dict(), best_model_path)  # Save the best model
            print(f"New best model saved with validation accuracy: {best_valid_acc*100:.2f}%")

    
        # Save if the model has best accuracy till now
        # torch.save(model_ft, "/home/saiful/mobilenet_classification/saved models/document_dataset "+'_model_'+str(epoch)+'.pt')
        torch.save(model_ft.state_dict(), "/data/saiful/document_Vit/saved models/document_dataset "+'_model_'+str(epoch)+'.pt')
        
        epoch+=1
        if epoch== epochs-1:
            print("flag1")
            print("Last epoch : ", epoch)
            break
    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    print(f"Maximum GPU memory used during training: {max_memory_used / (1024 ** 2):.2f} MB")
    print(f"Inference time per batch: {inference_time:.4f} seconds")
    print(f"Total number of trainable parameters: {total_params}")
    print("Training Finished for  ", method_name)


    
    #%%
    print("len(train_acc_list) ",len(train_acc_list))
    print("len(valid_acc_list) ",len(valid_acc_list))
    
    import matplotlib.pyplot as plt
    
    #%% 1
    import matplotlib.pyplot as plt
    
    # Assuming your data lists are defined: epoch_list, train_acc_list, valid_acc_list, train_loss_list, valid_loss_list
    
    # Plot for Accuracy
    train_acc_percent = [acc * 100 for acc in train_acc_list]
    valid_acc_percent = [acc * 100 for acc in valid_acc_list]
    
    # Plot for Accuracy
    plt.figure(figsize=(3.5 * 1.5, 2.8 * 1.5))
    plt.plot(epoch_list, train_acc_percent, label="Train Accuracy")
    plt.plot(epoch_list, valid_acc_percent, label="Test Accuracy")
    plt.legend(fontsize=10)
    plt.xlabel('Epoch Number', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Accuracy Curve', fontsize=10)
    plt.ylim(0, 100)  # Set the y-axis range from 0 to 100
    
    plt.tight_layout()
    plt.savefig(f"./results/adam_output_lr_{learning_rate}_batchsize_{batchsize}_{method_name}_document_dataset_accuracy_curve_IEEE.png", dpi=300)  # Save the figure with high resolution
    plt.show()
    
    # Plot for Loss
    plt.figure(figsize=(3.5 * 1.5, 2.8 * 1.5))
    plt.plot(epoch_list, train_loss_list, label="Train Loss")
    plt.plot(epoch_list, valid_loss_list, label="Test Loss")
    plt.legend(fontsize=10)
    plt.xlabel('Epoch Number', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Loss Curve', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"./results/adam_output_lr_{learning_rate}_batchsize_{batchsize}_{method_name}_dataset_loss_curve_IEEE.png", dpi=300)
    plt.show()
    
    
    #%%
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
    import numpy as np
    
    # Load the best model
    model_ft.load_state_dict(torch.load(best_model_path))
    model_ft.eval()  # Ensure the model is in evaluation mode
    
    true_labels = []
    predictions = []
    probabilities = []
    
    best_accuracy = 0.0
    
    # No gradient is needed
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass to get outputs
            outputs = model_ft(inputs)
            outputs = outputs.logits if hasattr(outputs, 'logits') else outputs  # Adjust based on your model's output
            
            # Convert outputs to predictions
            _, preds = torch.max(outputs, 1)
            
            # Move predictions and labels to CPU
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            # Convert logits to probabilities using softmax
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            # Collect for later evaluation
            true_labels.extend(labels)
            predictions.extend(preds)
            probabilities.extend(probs)
    
    # Convert lists to numpy arrays for sklearn metrics
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Calculate metrics
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    accuracy = accuracy_score(true_labels, predictions)
    
    auc_score = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='macro')
    # Print metrics
    print(f'== On test data ==')
    
    print(f'Test Accuracy on best model: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC Score: {auc_score:.4f}')
    
    #%%
    from sklearn.metrics import classification_report
    
    # Assuming true_labels and predictions are already defined as shown previously
    report = classification_report(true_labels, predictions, digits=4)
    
    print('classification report\n')
    print(report)
    
    
    #%%
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import numpy as np
    
    # Assuming 'true_labels' and 'predictions' are available
    
    # You can replace 'unique_classes' with your specific class names if known, e.g., ['class1', 'class2', ...]
    unique_classes = np.unique(np.concatenate([true_labels, predictions]))
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=unique_classes)
    
    # Create a DataFrame for better label handling in seaborn
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)
    
    # Plotting the confusion matrix using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(f"./results/cm_adam_output_lr_{learning_rate}_batchsize_{batchsize}_{method_name}_document_dataset_accuracy_curve_IEEE.png", dpi=300)  # Save the figure with high resolution
    plt.show()


    #%%
    
    import numpy as np
    
    # Assuming true_labels and predictions are numpy arrays obtained from your test dataset
    unique_classes = np.unique(true_labels)
    class_accuracies = {}
    
    for cls in unique_classes:
        # Indices where the current class is the true label
        class_indices = np.where(true_labels == cls)
        
        # Subset of true and predicted labels where the true label is the current class
        true_for_class = true_labels[class_indices]
        preds_for_class = predictions[class_indices]
        
        # Calculate accuracy: the fraction of predictions that match the true labels for this class
        class_accuracy = np.mean(preds_for_class == true_for_class)
        
        # Store the accuracy for this class
        class_accuracies[cls] = class_accuracy
    
    # Print class-wise accuracies
    for cls, acc in class_accuracies.items():
        print(f"Class {cls}: Accuracy = {acc*100:.2f}%")
      
    print('= = = = = = = = flag 1.12 = = = = = = = = = = = = =')
    print('= = = = = = = = flag 1.12 = = = = = = = = = = = = =\n\n')

    return   print('finished for ', method_name)


print('= = = = = = = = flag 1.11 = = = = = = = = = = = = =')

# train_test_plot(method_name= "Swinv2ForImageClassification")
# train_test_plot(method_name= "ViTForImageClassification")
train_test_plot(method_name= "ConvNextV2ForImageClassification")
# train_test_plot(method_name= "CvtForImageClassification")
# train_test_plot(method_name= "EfficientFormerForImageClassification")
# train_test_plot(method_name= "PvtV2ForImageClassification")
# train_test_plot(method_name= "MobileViTV2ForImageClassification")
# train_test_plot(method_name= "EfficientNetForImageClassification")
# train_test_plot(method_name= "BeitForImageClassification")
# train_test_plot(method_name= "BitForImageClassification")
# train_test_plot(method_name= "FocalNetForImageClassification")




print('= = = = = = = = execution finished = = = = = = = = = = = = =')
