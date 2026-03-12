import numpy as np
import cv2
import torch

from torch.utils.data import random_split
from src.nnet.torchBaselineModel import BaselineModel
from torchinfo import summary


def augment_image(img):
    """ Performs a random data augmentation on image: Flip, Rotate, and Noise
    
    Inputs: 
    - img: H x W x C image
    
    Outputs:
    - img: The image again but augmented"""
    idxFlips = np.random.randint(0, 3)
    idxRotations = np.random.randint(0, 3)
    flips = [0, 1, -1]
    rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
    noise_type = np.random.choice(["gaussian", "laplace"], p=[0.5, 0.5])

    img = cv2.flip(img, flips[idxFlips])
    img = cv2.rotate(img, rotations[idxRotations])

    img = img.astype(np.float32)
    
    if noise_type == "gaussian":
        img += np.random.normal(0, 0.03, img.shape).astype(np.float32)
    else:
        img += np.random.laplace(0, 0.03, img.shape).astype(np.float32)

    return img


def train_test_split(dataset, train_prop, val_prop):
    """Spits full data set into train, val, and test splits
    
    Inputs:
    - dataset: A PyTorch dataset
    - train_prop: Proportion of data to be used for training
    - val_prop: Proportion of data to be used for validation
    
    Outputs:
    - train_dataset, val_dataset, test_dataset: The train, val, and test data sets """

    if np.sum([train_prop, val_prop]) > 1:
        raise ValueError("Provide proper training and testing proportion sizes")
    
    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    return train_dataset, val_dataset, test_dataset



def check_cuda_availability():
    """
    Checks if CUDA-enabled GPU is available and returns the device being used for training.
    """
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"### GPU: {torch.cuda.get_device_name(0)} is available. Using {device} ### \n")
    else:
        device = torch.device("cpu")
        print(f"### No GPU available. Training will run on CPU. Using {device} ### \n")


    return device

    

def rbg2ycbcr(img):
    """ Convert an RGB image to YCbCr color space and reorder channels to YCbCr (instead of OpenCV's default YCrCb)
    Inputs:
    - img: A H x W x 3 RGB image (channels last)
    Outputs:
    - img: A H x W x 3 YCbCr image (channels last) with channels ordered as Y, Cb, Cr"""
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img = img[...,[0, 2, 1]].astype(np.float32) # Reorder channels to YCbCr

    return  img


def resize_img(img, H_new, W_new):
    """ Resize an image to new dimensions using cubic interpolation
    Inputs:
    - img: A H x W x C image
    - H_new: The new height of the image
    - W_new: The new width of the image
    
     Outputs:
     - resized_img: The resized image with dimensions H_new x W_new x C"""

    new_dims = (H_new, W_new)
    resized_img = cv2.resize(img, new_dims, interpolation=cv2.INTER_CUBIC )

    return resized_img


def get_model_summary():
    """ Gets a summary of the model architecture and number of parameters using torchinfo"""

    model = BaselineModel(num_classes=8, num_channels=3)
    input_size = (1, 3, 224, 224)
    summary(model, input_size=input_size)