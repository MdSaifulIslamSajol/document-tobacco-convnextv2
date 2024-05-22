#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:02:58 2024

@author: saiful
"""
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models

num_classes = 10 

def ViTForImageClassification():
    print("== ViTForImageClassification ==")

    from transformers import ViTForImageClassification, ViTFeatureExtractor
    # feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    model_ft = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    # model_ft.fc = nn.Sequential(
    #                 nn.Linear(768, 128),
    #                 nn.ReLU(inplace=True),
    #                 nn.Linear(128, 2))
    
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)
    
    return model_ft


def EfficientNetForImageClassification():
    print("== EfficientNetForImageClassification ==")

    from transformers import EfficientNetForImageClassification, ViTFeatureExtractor
    model_ft = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b7")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)
    
    return model_ft

def BeitForImageClassification():
    print("== BeitForImageClassification ==")

    from transformers import BeitForImageClassification
    model_ft = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)
    
    return model_ft

def BitForImageClassification():
    print("== BitForImageClassification ==")

    from transformers import BitForImageClassification
    
    model_ft =BitForImageClassification.from_pretrained("google/bit-50")
    # model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)
    num_ftrs = model_ft.classifier[1].in_features  # Update this index based on actual structure

    # Replace the classifier with a new one suitable for your number of classes
    model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)  # Adjust the index as needed
    return model_ft

def FocalNetForImageClassification():
    print("== FocalNetForImageClassification ==")

    from transformers import FocalNetForImageClassification
    model_ft = FocalNetForImageClassification.from_pretrained("microsoft/focalnet-tiny")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)
    
    return model_ft

def ConvNextV2ForImageClassification():
    print("== ConvNextV2ForImageClassification ==")

    from transformers import  ConvNextV2ForImageClassification
    model_ft = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features ,  num_classes)
    
    return model_ft

def Swinv2ForImageClassification():
    print("== Swinv2ForImageClassification ==")

    from transformers import Swinv2ForImageClassification
    model_ft = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features ,  num_classes)
    
    return model_ft

def ImageGPTForImageClassification():
    print("== ImageGPTForImageClassification ==")

    from transformers import  ImageGPTForImageClassification
    model_ft = ImageGPTForImageClassification.from_pretrained("openai/imagegpt-small")
    num_in_features = model_ft.score.in_features  
    model_ft.classifier = nn.Linear(num_in_features ,  num_classes)
    
    return model_ft

def CvtForImageClassification():
    print("== CvtForImageClassification ==")

    from transformers import  CvtForImageClassification
    model_ft = CvtForImageClassification.from_pretrained("microsoft/cvt-13")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features , num_classes)

    return model_ft

def EfficientFormerForImageClassification():
    print("== EfficientFormerForImageClassification ==")

    from transformers import  EfficientFormerForImageClassification
    model_ft = EfficientFormerForImageClassification.from_pretrained("snap-research/efficientformer-l1-300")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features ,  num_classes)

    return model_ft

def PvtV2ForImageClassification():
    print("== PvtV2ForImageClassification ==")
    from transformers import  PvtV2ForImageClassification

    model_ft = PvtV2ForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features ,  num_classes)

    return model_ft

def MobileViTV2ForImageClassification():
    print("== MobileViTV2ForImageClassification ==")
    from transformers import  MobileViTV2ForImageClassification

    model_ft = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
    model_ft.classifier = nn.Linear(model_ft.classifier.in_features ,  num_classes)
    return model_ft




