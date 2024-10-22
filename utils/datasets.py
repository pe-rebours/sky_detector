

import numpy as np
from torch.utils import data

from PIL import Image

from torchvision.datasets import Cityscapes
import torchvision.transforms.v2 as transforms

import os



class BinarySemanticCityscapes(Cityscapes):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    This class implement the binarized version (2 labels: one versus the others) of Cityscapes Dataset, only for the "fine" groundtruth of semantic segmentation

    Args:
        root (str): Root directory of dataset where directory "leftImg8bit"
            and "gtFine" are located.
        split (string): dataset split to use ("train","val" or "test")
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. Only applied to the input image.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        sync_transform (callable, optional): A function/transform that transformed both input image and target with the same transformation.
        positive_label (int): label of Cityscapes considered as the positive class (default value is 23 ("Sky")).
    """


    class_names = np.array([
        'other',
        'sky',
    ])

    def __init__(self, root, split,transform,target_transform,sync_transform=None,positive_label=23):
        self.transform_to_binary_label=self.binary_label_transform(positive_label)
        self.sync_transform=sync_transform
        super().__init__(root,split=split,mode="fine",target_type='semantic',transform=transform,target_transform=target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.

        Overwritting of the follow function: https://pytorch.org/vision/stable/_modules/torchvision/datasets/cityscapes.html#Cityscapes.__getitem__)
        """

        image = Image.open(self.images[index]).convert("RGB")

        
        target = Image.open(self.targets[index][0])  # type: ignore[assignment]
        target=self.transform_to_binary_label(target)

        #If  we have transform for data augmentation that should be the same for both input and target
        if self.sync_transform is not None:
            image, target = self.sync_transform(image, target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def target_to_PIL_image(self, target):
        return Image.fromarray((255*target).cpu().numpy().astype(np.uint8).transpose(1, 2, 0))

    class binary_label_transform(object):
        def __init__(self, sky_label):
            self.sky_label = sky_label

        def __call__(self, target):
            target=np.array(target)
            target[target!=self.sky_label]=0
            target[target==self.sky_label]=1
            return Image.fromarray(target)
        
class SkyFinder(data.Dataset):
    """`SkyFinder <https://zenodo.org/records/5884485>`_ Dataset.

    This class implement the Dataset object for SkyFinder Dataset.

    Args:
        root (str): Root directory of the dataset.
        split (string): dataset split to use ("train","val" or "test")
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. Only applied to the input image.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        sync_transform (callable, optional): A function/transform that transformed both input image and target with the same transformation.
    """

    class_names = np.array([
        'other',
        'sky',
    ])
    #mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self,root, split, transform, target_transform, sync_transform=None):
        self.root=root
        self.split=split
        img_gt_path=[]
        for d in os.listdir(os.path.join(root,split,'img').replace("\\","/")):
            for f in os.listdir(os.path.join(root,split,'img',d).replace("\\","/")):
                img_gt_path.append([os.path.join('gt',d+".png"),os.path.join('img',d,f).replace("\\","/")])
        self.img_gt_path=np.array(img_gt_path)
        self.transform=transform
        self.target_transform=target_transform
        self.sync_transform=sync_transform

    def __getitem__(self, index):
        path=self.img_gt_path[index]
        gt=Image.open(os.path.join(self.root,self.split,path[0]).replace("\\","/"))
        img=Image.open(os.path.join(self.root,self.split,path[1]).replace("\\","/"))

        #If  we have transform for data augmentation that should be the same for both input and target
        if self.sync_transform is not None:
            img, gt = self.sync_transform(img, gt)

        return (self.target_transform(img),self.target_transform(gt))
    def __len__(self):
        return len(self.img_gt_path)




    def target_to_PIL_image(self, target):
        return Image.fromarray((255*target).cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
