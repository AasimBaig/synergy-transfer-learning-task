from torchvision import models
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.autograd import Variable

import config
import engine
from dataset import AntsBeesDataset
import nn_model

from sklearn.model_selection import train_test_split


def run_training():

    # get csv file that contains image paths.
    train = pd.read_csv("/home/aasim/synergy-ai-task/src/train_data.csv")
    test = pd.read_csv("/home/aasim/synergy-ai-task/src/val_data.csv")

    # call out custom dataset.
    train_dataset = AntsBeesDataset(train["image_paths"].tolist(),
                                    train["targets"].tolist(),

                                    # various transform to increase the datasize because we are low on data.
                                    transform=[
        transforms.Resize(
            size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH), interpolation=Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.RandomRotation(40),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    test_dataset = AntsBeesDataset(test["image_paths"].tolist(),
                                   test["targets"].tolist(),
                                   transform=[
        transforms.Resize(
            size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    # train_loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True)

    # test_loader or val_loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False)

    # get model
    model = nn_model.get_model()
    torch.cuda.empty_cache()

    # add model to GPU
    if torch.cuda.is_available():
        model.cuda()

    # different loss function.
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # scheduling Learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True)

    model.train()
    for epoch in range(config.EPOCHS):
        train_loss, train_acc = engine.train_fn(model, train_loader, optimizer)
        print(
            f"Epoch: {epoch} --- Training loss : {train_loss} --- Accuracy : {train_acc}\n")

    print("\nTraining Finished \n")
    engine.save_checkpoint(model)

    engine.check_accuracy(test_loader)


if __name__ == "__main__":
    run_training()
