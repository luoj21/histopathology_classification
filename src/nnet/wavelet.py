import numpy as np
import torch
import torch.nn as nn
import pywt



def wavelet_packet_transform(x, wavelet='haar', maxlevel=2):
        """ Apply 2D wavelet packet transform to an image and return all subbands concatenated along channel dimension 
        Inputs:
        - img: A H x W x C image (channels last)
        - wavelet: The wavelet to use
        - maxlevel: The maximum level of the wavelet packet tree
        Outputs:
        - all_data: A (C * 4^maxlevel) x H' x W' image containing all wavelet subbands concatenated along the channel dimension, where H' and W' are dimensions after downsampling by the wavelet transform"""

        wp = pywt.WaveletPacket2D(data=x, 
                                wavelet=wavelet, 
                                mode='symmetric', 
                                maxlevel=maxlevel, 
                                axes =(0, 1))
        nodes = wp.get_level(maxlevel, order='natural')
        all_data = np.concatenate([n.data for n in nodes], axis=-1)

        return all_data



class MaxEnergySelector(nn.Module):

    """ Expects a WPT transformed image. 
    Selects the channels with the highest squared wavelet coefficients """
    
    def __init__(self, num_selected_channels):
        super(MaxEnergySelector, self).__init__()
        self.num_selected_channels = num_selected_channels


    def forward(self, x):
        energy = torch.sum(x**2, dim=(0, 2, 3)) 
        _, selected_idx = torch.topk(energy, self.num_selected_channels, dim=-1) 
        x = x[:, selected_idx, :, :] 

        return x