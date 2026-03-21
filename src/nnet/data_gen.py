import torch
import os
import cv2

from torch.utils.data import Dataset

from src.nnet.wavelet import wavelet_packet_transform
from ..utils.my_utils import augment_image, rbg2ycbcr, resize_img # .. means go up one level (from nnet to src)


class ImageDataset(Dataset):
    def __init__(self, data_root: str, metadata_file: str, augment: bool, ycbcr: bool, resize: bool, wavelet: bool, wavelet_levels: int):
        """
        Custom PyTorch image data set 
        
        Inputs: 
        -  data_root: Path to where all data is stored
        -  metadata_file: Name of the metadata file
        -  augment: Whether or not to perform image augmentation for pre-processing
        -  ycbcr: Whether or not to convert images to YCbCr color space for pre-processing
        -  resize: Whether or not to resize images for pre-processing
        """
        
        self.data_root = data_root
        self.metadata = f"{data_root}\\{metadata_file}"
        self.augment = augment
        self.ycbcr = ycbcr
        self.resize = resize
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels

        self.img_types = os.listdir(self.data_root)

        self.img_paths = []
        self.labels = []

        for label, type in enumerate(self.img_types):

            class_path = os.path.join(self.data_root, type)
            imgs = os.listdir(class_path)

            for img in imgs:
                
                img_path = os.path.join(class_path, img)
                self.img_paths.append(img_path)
                self.labels.append(label)
            
            
    def __len__(self):

        return len(self.img_paths)
    

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path)

        if self.resize:
            img = resize_img(img,448,448)

        if self.ycbcr:
            img = rbg2ycbcr(img)

        if self.augment:
            img = augment_image(img) 

        if self.wavelet:
            img = wavelet_packet_transform(img, maxlevel=self.wavelet_levels)
        
        img =  img / 255.0
        img = img.transpose(2, 0, 1) # Change from H x W x C to C x H x W (Channels First)
        img, label = torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        return img, label