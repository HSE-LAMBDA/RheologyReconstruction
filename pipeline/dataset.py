import os
import re
import numpy as np
import torch

from torch.utils.data import Dataset
from transforms import DifferentialTransform
from functools import reduce
from scipy.ndimage import gaussian_filter


class SeismogramBatch():
    """
    Custom batch with memory pinning allows faster data transfer

    """

    def __init__(self, data):

        self.seismograms = torch.stack([i['seismogram'] for i in data], dim=0)
        self.masks       = torch.stack([i['mask']       for i in data], dim=0)
        self.weights     = torch.stack([i['weights']    for i in data], dim=0)

    def pin_memory(self):

        self.seismograms = self.seismograms.pin_memory()
        self.masks       = self.masks.pin_memory()
        self.weights     = self.weights.pin_memory()

        return self

    def to(self, device):

        self.seismograms = self.seismograms.to(device)
        self.masks       = self.masks.to(device)
        self.weights     = self.weights.to(device)

        return self


def compute_weight_map(mask: torch.tensor,
                       border_pixel_weight=2.,
                       gaussian_filter_sigma=1.):
    """
    Computes a weight map for a given mask
    :params:
        mask - numpy array of shape (W, H)
        borders_pixel_weight  - additional weight multiplier for border pixels
        gaussian_filter_sigma - kernel size for gaussian filter
    :returns:   
        weight_map - torch.tensor of shape (1, W, H)
    """

    scale_mult = 1.
    diff       = DifferentialTransform()

    weight_map = mask.view(1, 1, *mask.shape).float()
    
    if torch.sum(mask):
        scale_mult = float(reduce(lambda x, y: x * y, mask.shape)) / torch.sum(weight_map > 0.0).float()

    borders = torch.zeros(*mask.shape)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (i == 0) or (i == mask.shape[0] - 1) or \
               (j == 0) or (j == mask.shape[1] - 1): 
               borders[i, j] = 1. 

    weight_map = scale_mult * (
        weight_map  +\
        border_pixel_weight * (torch.max(diff(weight_map) - borders, torch.zeros(*mask.shape)))
    )
    
    weight_map = torch.from_numpy(gaussian_filter(weight_map.data.numpy(), sigma=gaussian_filter_sigma))
    weight_map += 1.


    return weight_map.view(*mask.shape)



class SeismogramDataset(Dataset):
    """
    Seismogram dataset.
    Implements the interface of torch.utils.dataset.Dataset
    Names of the files must contain numerals, due to the indexing issues
    File with the least numeral is considered to be a 'head' of the dataset
    
    """

    def __init__(self, root_dir):
        """
        :param root_dir (string): 
            A root directory of the dataset. Following layout is assumed:
                /root
                    /seismograms
                    /masks
        :param scaling_factor (float):
            Constant multiplier for seismograms. Default=1e7
            
        """
        self.root_dir    = os.path.join(os.getcwd(), root_dir)
        self.data_dir    = os.path.join(self.root_dir, 'seismograms')
        self.mesh_dir    = os.path.join(self.root_dir, 'masks')

        self.seismograms = [(file, re.findall(r'(\d+)', file)[-1]) for file in os.listdir(self.data_dir)]
        self.seismograms = [x[0] for x in list(sorted(self.seismograms, key= lambda x: x[1]))]
        
        self.masks = [(file, re.findall(r'(\d+)', file)[-1]) for file in os.listdir(self.mesh_dir)]
        self.masks = [x[0] for x in list(sorted(self.masks, key= lambda x: x[1]))]

    def __len__(self):
        return len(self.seismograms)

    def __getitem__(self, idx):

        sname = os.path.join(self.data_dir, self.seismograms[idx])
        mname = os.path.join(self.mesh_dir, self.masks[idx])
        
        s = torch.from_numpy(np.load(sname))
        m = torch.from_numpy(np.load(mname))
        w = compute_weight_map(m)

        result = {
        	'seismogram': s.float(),
        	'mask'      : m.float(),
        	'weights'   : w.float()
        } 

        return result