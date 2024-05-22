import os
import torchvision.datasets as datasets
from IPython.core.debugger import set_trace
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
from pathlib import Path
from PIL import Image
import random
import numpy as np
import re
import csv
import torch

__all__ = ['ImageFolderInstance', 'ImageFolderInstanceSamples']

class ImageFolderInstance(Dataset):
    """: Folder datasets which returns the index of the image as well::
    """
    def __getitem__(self,paths, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = paths[index]
        img = np.load(str(path))['arr_0']
        img = torch.from_numpy(img).float()
        pattern = r"^\d+"
        target = re.search(pattern, str(os.path.dirname(path)))
        target = torch.from_numpy(np.array(int(target)))
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img, target, index

class ImageFolderInstanceSamples(Dataset):
    """: Folder datasets which returns the index of the image as well::
    """
    
    def __init__(self,paths, n_samples=1):        
        self.n_samples = n_samples
        self.paths = paths
    
    def __len__(self):
        return len(self.paths) 
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        
        path = self.paths[index]
        image = np.load(str(path))['arr_0']
        image_torch = torch.from_numpy(image).float()
        image_torch = image_torch
        class_path= os.path.dirname(path)
        pattern = r'(\d+)_class'
        target = re.search(pattern, str(class_path)).group(1)
        target = torch.from_numpy(np.array(int(target)))
        
        imgs= [torch.from_numpy(np.load(str(os.path.join(class_path,random.choice(os.listdir(class_path)))))['arr_0']).float() for i in range(self.n_samples)]
        targets = [target for i in range(self.n_samples)]
        indexs = [index for i in range(self.n_samples)]

        # if self.transform is not None:
        #     img = [self.transform(img) for i in range(self.n_samples)]

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        
        return imgs, targets, indexs  