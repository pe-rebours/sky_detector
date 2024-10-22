import cv2

import argparse
import time
import os

import torch
import yaml

import torchvision.transforms as transforms
import torchvision.models as models
from utils import (accuracy,IoU,violinplot)

import numpy as np
import re

from PIL import Image
import matplotlib.pyplot as plt

ALLOWS_IMG_EXTENSION=["png","jpg","jpeg"] # acceptable file for inference on a single image
ALLOWS_VIDEO_EXTENSION=["mp4"] # acceptable file for inference on a video without real-time display

def pre_process_input(input,required_input_shape):
    transform = transforms.Compose([
        transforms.Resize(required_input_shape,transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(input)

def post_process_output(output,output_shape):
    output=Image.fromarray(output.astype(np.uint8)).resize(output_shape,resample=Image.Resampling.NEAREST)
    return output


def run_inference(model,input):
    return model(input)['out'].softmax(dim=1).argmax(dim=1)




def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Model path')
    parser.add_argument('--config_path', required=True, help='path to config.yaml')
    parser.add_argument('-c', '--cpu', action='store_true', help='run on cpu')
    parser.add_argument('--output_folder', type=str, default="./output/inference",help='Ouput folder')
    parser.add_argument('--input', type=str,default="0", help='Path to input. Accepted file for non-real time inference are .jpg .png or .mp4')
    parser.add_argument('--real_time', action='store_true', help='read the input and show the output in real time')
    args = parser.parse_args()

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
    
    print("Importing the model...")
    model = models.segmentation.fcn_resnet50(weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT)
    model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes
    model.load_state_dict(torch.load(model_path,weights_only=False,map_location=device))
    model.to(device)
    model.eval()
    print("The model is charged.")

    input_path=args.input
    img_ext_pattern=r"(\.{})+$".format("|".join(ALLOWS_IMG_EXTENSION))
    video_ext_pattern=r"(\.{})+$".format("|".join(ALLOWS_VIDEO_EXTENSION))

    real_time=args.real_time

    # Inference for a single image
    if re.search(img_ext_pattern,input_path) and not real_time:
        with torch.no_grad():
            frame=Image.open(input_path)
            initial_img_size=frame.size
            input=pre_process_input(frame,(256,512)).to(device)
            input=input[None,:,:,:]
            pred=run_inference(model,input)
            frame=np.array(frame)
            pred=pred.squeeze().cpu().numpy()
            pred=np.array(post_process_output(pred,initial_img_size))
            
            frame[pred==1]=[255,0,0]
            Image.fromarray(frame).save(os.path.join(output_folder,"out.png").replace("\\","/"))

    # Inference for video without real-time display
    elif re.search(video_ext_pattern,input_path) and not real_time:

        cap=cv2.VideoCapture(input_path)

        # Instentation of the video writer
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join(output_folder,"out.mp4").replace("\\","/"), fourcc, fps, size)

        with torch.no_grad():
            flag, frame = cap.read()
            while flag:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                initial_img_size=size
                input=pre_process_input(frame,(256,512)).to(device)
                input=input[None,:,:,:]
                pred=run_inference(model,input)
                frame=np.array(frame)
                pred=pred.squeeze().cpu().numpy()
                pred=np.array(post_process_output(pred,initial_img_size))
                
                frame[pred==1]=[255,0,0]

                videoWriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                flag, frame = cap.read()
        videoWriter.release()

    # Inference for real-time display comming from a video source (camera, stream url,...)
    else:
        print("Connecting to the video source...")
        if input_path.isdigit():
            input_path=int(input_path)
        cap=cv2.VideoCapture(input_path)
        if cap is None or not cap.isOpened():
            raise FileNotFoundError('Unable to open video source: ' +str(input_path))
        print("Capture is ready.")

        fps=int(cap.get(cv2.CAP_PROP_FPS))
        frame_number=0
        with torch.no_grad():
            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                flag, frame = cap.read()
                if flag==False:
                    break

                start=time.time()
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                initial_img_size=frame.size

                # Pre-processing
                input=pre_process_input(frame,input_shape).to(device)
                input=input[None,:,:,:]

                pred=run_inference(model,input)

                # Post-porcessing
                frame=np.array(frame)
                pred=pred.squeeze().cpu().numpy()
                pred=np.array(post_process_output(pred,initial_img_size))

                # output (the sky is painted in red on the output frame)
                frame[pred==1]=[255,0,0]
                cv2.imshow('Sky detector', np.uint8(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))

                frame_number+=1+int(fps*(time.time()-start)) # If the source is not a stream, the next processed frame is the one captured by the source after the inference process.

                if cv2.waitKey(10) == 27: # If the user press "esc"
                    break
    

if __name__ == '__main__':
    main()