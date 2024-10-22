import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path

from PIL import Image,ImageOps
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import os
import cv2

from tqdm import tqdm

import argparse

import yaml

from utils import (BinarySemanticCityscapes,Trainer)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, help='config.yaml path')
    parser.add_argument('-c', '--cpu', action='store_true', help='run on cpu')
    parser.add_argument('--output_folder', type=str, default="./runs",help='folder where best model at each epoch are saved')
    parser.add_argument('--log_folder', type=str, default="./log",help='folder where log information are saved')
    parser.add_argument('--runs_name', type=str, default="fcn_model",help='name used to save train model')
    args = parser.parse_args()

    config_path=args.config_path
    log_folder=args.log_folder
    output_folder=args.output_folder
    runs_name=args.runs_name

    # Importation of the model and learning parameter
    with open(config_path) as file:
        config=yaml.safe_load(file)
    lr=float(config['learning_param']['lr'])
    batch_size=int(config['learning_param']['batch_size'])
    input_shape=(int(config['input_param']['height']),int(config['input_param']['width']))
    nb_epoch=int(config['learning_param']['nb_epoch'])
    with_data_augmentation=bool(config['learning_param']['with_data_augmentation'])


    if args.cpu:
        device=torch.device("cpu")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Run on "+str(device.type))

    # Importation of the model 
    fcn = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)#pretrained=True)
    fcn.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes
    fcn=fcn.to(device)
    optimizer=torch.optim.Adam(params=fcn.parameters(),lr=lr) # Create adam optimizer

    # Input and target are resized to allow fast training with limited memory
    transform = transforms.Compose([
        transforms.Resize(input_shape,transforms.InterpolationMode.BILINEAR),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Image is normalized
    ])
    target_transform = transforms.Compose([
        transforms.Resize(input_shape,transforms.InterpolationMode.NEAREST),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])

    # Data augmentation
    if with_data_augmentation:

        # Random Transformation only applied to the input (color jitter, auto contrast)
        training_transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.Resize(input_shape,transforms.InterpolationMode.BILINEAR),
            transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=.5, hue=.3)]),p=0.4),
            transforms.RandomAutocontrast(p=0.4),
            
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Random geometrical transformation, applied to both input and target (horizontal flip, cropping)
        shared_training_transform= transforms.Compose([
            # you can add other transformations in this list
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomCrop(size=(input_shape[0]//2, input_shape[1]//2))]),p=0.4),
        ])
    else:
        training_transform=transform
        shared_training_transform=None
        
    # Generation of Dataset and Dataloader
    train_dataset = BinarySemanticCityscapes('./data/cityscapes', split='train',transform=training_transform,target_transform=target_transform,sync_transform=shared_training_transform)
    val_dataset = BinarySemanticCityscapes('./data/cityscapes', split='val',transform=transform,target_transform=target_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Training
    fcn_trainer=Trainer(fcn, optimizer, device,train_dataloader, val_dataloader, output_folder, nb_epoch,log_folder,runs_name=runs_name)
    fcn_trainer.train()



if __name__ == '__main__':
    main()