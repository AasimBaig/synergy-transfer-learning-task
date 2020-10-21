import config
import torch
import torch.nn.functional as F
import torch.nn as nn
import pretrainedmodels
from torchvision import models


def get_model():
    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False

    num_features = resnet.fc.in_features
    print(num_features)

    fc_layers = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(1024, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25),
        nn.Linear(128, 2)
    )

    resnet.fc = fc_layers

    return resnet


'''
model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)
    )

'''
