import cv2

import argparse
import time
import os

import torch
import yaml

import torchvision.transforms as transforms
import torchvision.models as models
from utils import (BinaryCityscapes,accuracy,IoU,violinplot)
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import re

from PIL import Image
import matplotlib.pyplot as plt

import tensorrt as trt
import pycuda.driver as cuda
import onnx

ALLOWS_IMG_EXTENSION=["png","jpg"]
ALLOWS_VIDEO_EXTENSION=["mp4"]

# logger to capture errors, warnings, and other information during the build and inference phases
LOGGER = trt.Logger()
 
def build_engine(onnx_file_path):


    # Load the ONNX model
    model_onnx = onnx.load(onnx_file_path)
    
    # Create a TensorRT builder and network
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network()
    
    # Create an ONNX-TensorRT backend
    parser = trt.OnnxParser(network, builder.logger)
    parser.parse(model_onnx.SerializeToString())
    
    # Set up optimization profile and builder parameters
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 10), (1, 10), (1, 10))
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, (1 << 30))
    builder_config.set_flag(trt.BuilderFlag.FP16)
    
    # Build the TensorRT engine from the optimized network
    engine = builder.build_serialized_network(network, builder_config) 
 
    return engine,builder

def pre_process_input(input,required_input_shape):
    transform = transforms.Compose([
        transforms.Resize(required_input_shape,transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(input)
def input_to_rgb_image(input,img_shape):
    invTrans = transforms.Compose([ 
          transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
          transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ]),
          transforms.ToPILImage(),
          transforms.Resize((img_shape[1],img_shape[0]),transforms.InterpolationMode.BILINEAR)
          ])
    return invTrans(input)

def post_process_output(output,output_shape):
    output=Image.fromarray(output.astype(np.uint8)).resize(output_shape,resample=Image.Resampling.NEAREST)
    return output


def run_inference(model,input):
    return model(input)['out'].softmax(dim=1).argmax(dim=1)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_model_path', help='Model path')
    parser.add_argument('-c', '--cpu', action='store_true', help='run on cpu')
    parser.add_argument('--output_folder', type=str, default="./output",help='Ouput folder')
    parser.add_argument('--input_path', type=str, help='input .jpg .png or .mp4')
    args = parser.parse_args()

    log_filename="result_on_test_data.txt"
    onnx_model_path=args.onnx_model_path

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) 

    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    # Set cache
    cache = config.create_timing_cache(b"")
    config.set_timing_cache(cache, ignore_mismatch=False)

    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#version-compat
    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#explicit-implicit-batch
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    path_onnx_model = onnx_model_path
    with open(path_onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            print(f"ERROR: Failed to parse the ONNX file {path_onnx_model}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for input in inputs:
        print(f"Model {input.name} shape: {input.shape} {input.dtype}")
    for output in outputs:
        print(f"Model {output.name} shape: {output.shape} {output.dtype}") 
    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes


    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision
    # https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
    half = True
    int8 = False
    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    elif int8:
        config.set_flag(trt.BuilderFlag.INT8)

    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#weightless-build
    # https://github.com/NVIDIA/TensorRT/tree/main/samples/python/sample_weight_stripping
    strip_weights = False
    if strip_weights:
        config.set_flag(trt.BuilderFlag.STRIP_PLAN)
    # To remove strip plan from config
    # config.flags &= ~(1 << int(trt.BuilderFlag.STRIP_PLAN))

    engine = builder.build_serialized_network(network, config) 
    context = engine.create_execution_context()

    """
    #model_path = args.model_path
    onnx_model_path=args.onnx_model_path

    # initialize TensorRT engine and parse ONNX model
    engine,builder = build_engine(onnx_model_path)

    # Create a TensorRT execution context
    context = engine.create_execution_context()
    
    # Allocate device memory for input and output buffers
    input_name = 'input'
    output_name = 'output'
    input_shape = (1, 10)
    output_shape = (1, 5)
    input_buf = trt.cuda.alloc_buffer(builder.max_batch_size * trt.volume(input_shape) * trt.float32.itemsize)
    output_buf = trt.cuda.alloc_buffer(builder.max_batch_size * trt.volume(output_shape) * trt.float32.itemsize)

    # Run inference on the TensorRT engine
    input_data = torch.randn(1, 10).numpy()
    output_data = np.empty(output_shape, dtype=np.float32)
    input_buf.host = input_data.ravel()
    trt_outputs = [output_buf.device]
    trt_inputs = [input_buf.device]
    context.execute_async_v2(bindings=trt_inputs + trt_outputs, stream_handle=trt.cuda.Stream())
    output_buf.device_to_host()
    output_data[:] = np.reshape(output_buf.host, output_shape)    
    """
    
    output_folder=args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_path=args.input_path
    img_ext_pattern=r"(\.{})+$".format("|".join(ALLOWS_IMG_EXTENSION))
    video_ext_pattern=r"(\.{})+$".format("|".join(ALLOWS_VIDEO_EXTENSION))

    if re.search(img_ext_pattern,input_path):

        # Run inference on the TensorRT engine
        frame=Image.open(input_path)
        initial_img_size=frame.size
        frame=pre_process_input(frame,(256,512)).numpy()

        input_data = frame
        output_data = np.empty(frame.shape, dtype=np.float32)

        input_buf.host = frame.ravel()

        trt_outputs = [output_buf.device]
        trt_inputs = [input_buf.device]

        context.execute_async_v2(bindings=trt_inputs + trt_outputs, stream_handle=trt.cuda.Stream())
        output_buf.device_to_host()
        output_data[:] = np.reshape(output_buf.host, frame.shape)  
        
        pred=np.array(post_process_output(pred,output_data[:]))
        
        frame[pred==1]=[255,0,0]
        Image.fromarray(frame).save("out.png")

    elif re.search(video_ext_pattern,input_path):

        cap=cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        ps = 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter("out.mp4", fourcc, fps, size)
        with torch.no_grad():
            flag, frame = cap.read()
            while flag:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                initial_img_size=frame.size
                frame=pre_process_input(frame,(256,512)).to(device)
                input=frame[None,:,:,:]
                pred=run_inference(model,input)
                frame=np.array(input_to_rgb_image(frame,initial_img_size))
                pred=pred.squeeze().cpu().numpy()
                pred=np.array(post_process_output(pred,initial_img_size))
                
                frame[pred==1]=[255,0,0]
                cv2.imshow('video', np.uint8(frame))

                videoWriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                flag, frame = cap.read()
        videoWriter.release()

    else:
        return
        cap=cv2.VideoCapture(1)


        while True:
            flag, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame=pre_process_input(Image.fromarray(frame),(256,512)).to(device)
            input=frame[None,:,:,:]
            pred=run_inference(model,input)
            frame=np.array(input_to_rgb_image(frame))
            pred=pred.squeeze().cpu().numpy()
            print(np.unique(pred))
            
            frame[pred==1]=[255,0,0]
            #cv2.imshow('video', np.uint8(frame))

            if cv2.waitKey(10) == 27:
                break

if __name__ == '__main__':
    main()