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
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader
import os
import cv2

from tqdm import tqdm

class Trainer(object):

    def __init__(self, model, optimizer, device,
                 train_dataloader, val_dataloader, output_folder, nb_epoch,log_folder,runs_name="fcn_model"):
        self.model=model
        self.optimizer=optimizer
        self.device=device
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader
        self.output_folder=output_folder
        self.nb_epoch=nb_epoch
        self.log_folder=log_folder
        self.runs_name=runs_name
        
    def train(self):
        mean_best_val_loss=np.inf
        #criterion = torch.nn.CrossEntropyLoss() # Set loss function

        torch.cuda.empty_cache()
        criterion = torch.nn.CrossEntropyLoss()
        log_file=os.path.join(self.log_folder,'log.txt').replace("\\","/")
        for epoch in range(1,self.nb_epoch+1):
            val_losses=[]
            train_losses=[]
            print("Epoch "+str(epoch))
            with open(log_file, "a") as f:
                print("Epoch "+str(epoch),file=f)
            
            self.model.train()
            
            for i,val in enumerate(tqdm(self.train_dataloader)):
                img=val[0].to(self.device)
                label=val[1].to(self.device)
                
                pred=self.model(img)['out']
                label=label.reshape((label.shape[0],label.shape[2],label.shape[3]))
                loss=criterion(pred,(label*255).long()) # Calculate cross entropy loss
                self.optimizer.zero_grad()
                loss.backward() # Backpropogate loss
                self.optimizer.step() # Apply gradient descent change to weight

                
                train_losses.append(loss.item())
            with open(log_file, "a") as f:
                print("train Loss {}".format(np.mean(train_losses)),file=f)

            self.model.eval()
            with torch.no_grad():
                for i,val in enumerate(tqdm(self.val_dataloader)):
                    img=val[0].to(self.device)
                    label=val[1].to(self.device)
                    
                    pred=self.model(img)['out']
                    label=label.reshape((label.shape[0],label.shape[2],label.shape[3]))
                    val_loss=criterion(pred,(label*255).long()) 
                    val_losses.append(val_loss.item())

                mean_val_loss=np.mean(val_losses)
                with open(log_file, "a") as f:
                    print("valid Loss {}".format(mean_val_loss),file=f)
                
                if mean_val_loss < mean_best_val_loss:
                    mean_best_val_loss=mean_val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.output_folder,self.runs_name+"_"+str(epoch) + ".pt").replace("\\","/"))