
import argparse
import time
import os

import torch
import yaml

import torchvision.transforms.v2 as transforms
import torchvision.models as models
from utils import (BinarySemanticCityscapes,accuracy,IoU,precision,recall,f1_score,violinplot,distribution_plot)
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Model path')
    parser.add_argument('--config_path', help='Path to config used for training')
    parser.add_argument('-c', '--cpu', action='store_true', help='run on cpu')
    parser.add_argument('--output_folder', type=str, default="./output/evaluation",help='Ouput folder')
    args = parser.parse_args()

    log_filename="result_on_test_data.txt"

    if args.cpu:
        device=torch.device("cpu")
    else:
       device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print("Run on "+str(device.type))
    model_path = args.model_path

    
    output_folder=args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    config_path=args.config_path
    with open(config_path) as file:
        config=yaml.safe_load(file)
    input_shape=(int(config['input_param']['height']),int(config['input_param']['width']))
    
    # The model is imported
    print("Importing the model...")
    model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)#pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes
    model.load_state_dict(torch.load(model_path,weights_only=True,map_location=device))
    model.to(device)
    model.eval()
    print("The model has been successfully imported.")


    transform = transforms.Compose([
        transforms.Resize(input_shape,transforms.InterpolationMode.BILINEAR),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        
    target_transform = transforms.Compose([
        transforms.Resize(input_shape,transforms.InterpolationMode.NEAREST),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])

    # The test set is imported
    test_dataset = BinarySemanticCityscapes('./data/cityscapes', split='test',transform=transform,target_transform=target_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)#,generator=torch.Generator(device))

    # Metrics are computed on each image of the test set.
    accuracies=[]
    IoU_by_label=[[],[]]
    precision_by_label=[[],[]]
    recall_by_label=[[],[]]
    f1_score_by_label=[[],[]]
    computation_times=[]
    print("The model is running on all the images of the testing set.")
    for i,val in enumerate(tqdm(test_dataloader)):
      img=val[0].to(device)
      label=(val[1].to(device)*255).int()

      start=time.time()
      pred=model(img)['out'].softmax(dim=1).argmax(dim=1)
      computation_times.append(time.time()-start)
      pred=pred.squeeze()
      label=label.squeeze()

      accuracies.append(accuracy(pred,label).item())
      for l in range(2): #IoU is computed for "sky" and "other"
        iou=IoU(pred,label,l)
        prec=precision(pred,label,l)
        rec=recall(pred,label,l)
        f1_s=f1_score(pred,label,l)
        if iou:
            IoU_by_label[l].append(iou.item())
        if prec:
            precision_by_label[l].append(prec.item())
        if rec:
            recall_by_label[l].append(rec.item())
        if f1_s:
            f1_score_by_label[l].append(f1_s.item())

    # The results are saved.
    with open(os.path.join(output_folder,log_filename),"w") as f:
       message=("Metric value for the test split ({} images):\n".format(len(test_dataloader))
                +"\n"
                +"- Accuracy: mean={} std={}\n".format(np.mean(accuracies),np.std(accuracies))
                +"\n"
                +"- Model's computation time (s): mean={} std={}\n".format(np.mean(computation_times),np.std(computation_times))
                +"\n"
                +"- for label 'Sky':\n"
                +"  > IoU : mean={} std={}\n".format(np.mean(IoU_by_label[1]),np.std(IoU_by_label[1]))
                +"  > Precision : mean={} std={}\n".format(np.mean(precision_by_label[1]),np.std(precision_by_label[1]))
                +"  > Recall : mean={} std={}\n".format(np.mean(recall_by_label[1]),np.std(recall_by_label[1]))
                +"  > F1_score : mean={} std={}\n".format(np.mean(f1_score_by_label[1]),np.std(f1_score_by_label[1]))
                +"\n"
                +"- for label 'Other':\n"
                +"  > IoU : mean={} std={}\n".format(np.mean(IoU_by_label[0]),np.std(IoU_by_label[0]))
                +"  > Precision : mean={} std={}\n".format(np.mean(precision_by_label[0]),np.std(precision_by_label[0]))
                +"  > Recall : mean={} std={}\n".format(np.mean(recall_by_label[0]),np.std(recall_by_label[0]))
                +"  > F1_score : mean={} std={}\n".format(np.mean(f1_score_by_label[0]),np.std(f1_score_by_label[0])))
       print(message)
       f.write(message)
       
    distribution_plot(accuracies,"accuracy",filename=os.path.join(output_folder,"accuracy_violinplot.png"))
    violinplot(IoU_by_label,"IoU",["Other","Sky"], filename=os.path.join(output_folder,"IoU_violinplot.png"))
    violinplot(precision_by_label,"Precision",["Other","Sky"], filename=os.path.join(output_folder,"precision_violinplot.png"))
    violinplot(recall_by_label,"Recall",["Other","Sky"], filename=os.path.join(output_folder,"recall_violinplot.png"))
    violinplot(f1_score_by_label,"F1_score",["Other","Sky"], filename=os.path.join(output_folder,"f1_score_violinplot.png"))
    distribution_plot(computation_times,"Computation time (s)",filename=os.path.join(output_folder,"computation_time.png"))

if __name__ == '__main__':
    main()