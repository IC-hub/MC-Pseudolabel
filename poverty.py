# Adapted from the code base: https://github.com/p-lambda/wilds

# MIT License

# Copyright (c) 2020 WILDS team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.poverty_dataset import PovertyMapDataset

from resnet_multispectral import ResNet18, ResNet18Label

import torch
from torch.utils.data import Dataset
from sklearn.neighbors import KernelDensity

from model import Output


class Poverty_Batched_Dataset(Dataset):
    """
    Batched dataset for Poverty. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, dataset, split, batch_size, transform=None, args = None, domain2idx = None):
        self.split_array = dataset.split_array
        self.split_dict = dataset.split_dict
        self.args = args
        split_mask = self.split_array == self.split_dict[split]
        # split_idx = [ 2390  2391  2392 ... 19665 19666 19667]
        self.split_idx = np.where(split_mask)[0] 

        self.root = dataset.root
        self.no_nl = dataset.no_nl

        self.metadata_array = torch.stack([dataset.metadata_array[self.split_idx, i] for i in [0, 2]], -1)
        
        self.y_array = dataset.y_array[self.split_idx]

        self.eval = dataset.eval
        self.collate = dataset.collate
        # metadata_fields:['urban', 'y', 'country']
        self.metadata_fields = dataset.metadata_fields
        self.data_dir = dataset.data_dir

        self.transform = transform if transform is not None else lambda x: x

        domains = self.metadata_array[:, 0] # domain column for csv # 1

        self.domain_indices = [torch.nonzero(domains == loc).squeeze(-1)
                               for loc in domains.unique()] # split into different domain idx, domain_indices is 2D
        # visualization #
        print('='*20 + f' Data information ' + '='*20)
        print(f'len(self.domain_indices):{len(self.domain_indices)}')

        self.domains = domains
        if domain2idx is None:
            self.domain2idx = {loc.item(): i for i, loc in enumerate(self.domains.unique())} 
        else:
            self.domain2idx = domain2idx
        
        self.num_envs = len(domains.unique())
        self.targets = self.y_array
        self.batch_size = batch_size


    def get_input(self, idx):
        """Returns x for a given idx."""
        img = np.load(self.root / 'images' / f'landsat_poverty_img_{self.split_idx[idx]}.npz')['x']
        if self.no_nl:
            img[-1] = 0
        img = torch.from_numpy(img).float()
        return img


    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domain2idx[self.domains[idx].item()], idx

    def __len__(self):
        return len(self.targets)



IMG_HEIGHT = 224
NUM_CLASSES = 1
target_resolution = (224, 224)


def initialize_poverty_train_transform():
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""

    def ms_cutout(ms_img):
        def _sample_uniform(a, b):
            return torch.empty(1).uniform_(a, b).item()

        assert ms_img.shape[1] == ms_img.shape[2]
        img_width = ms_img.shape[1]
        cutout_width = _sample_uniform(0, img_width/2)
        cutout_center_x = _sample_uniform(0, img_width)
        cutout_center_y = _sample_uniform(0, img_width)
        x0 = int(max(0, cutout_center_x - cutout_width/2))
        y0 = int(max(0, cutout_center_y - cutout_width/2))
        x1 = int(min(img_width, cutout_center_x + cutout_width/2))
        y1 = int(min(img_width, cutout_center_y + cutout_width/2))

        # Fill with 0 because the data is already normalized to mean zero
        ms_img[:, x0:x1, y0:y1] = 0
        return ms_img

    #transform_step = get_image_base_transform_steps()
    transforms_ls = [

        transforms.ToPILImage(),
        transforms.Resize(target_resolution),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(size=target_resolution,),
        transforms.RandomVerticalFlip(),
        #wyp add affine,ms_color and ms_img
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1, scale=(0.9, 1.1)),
        #transforms.Lambda(lambda ms_img: poverty_color_jitter(ms_img)),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),
        transforms.Lambda(lambda ms_img:ms_cutout(ms_img)),

        transforms.ToTensor()]
    rgb_transform = transforms.Compose(transforms_ls)
    
    return transforms.Compose([]) # empty transform


class ResnetMS(nn.Module):
    def __init__(self, weights=None, num_classes=1, input_label=False):
        super(ResnetMS, self).__init__()

        #resnet18_ms
        if input_label:
            self.enc = ResNet18Label(num_classes=num_classes, num_channels=8)
        else:
            self.enc = ResNet18(num_classes=num_classes, num_channels=8)
        
        self.output_layer = Output(num_classes, num_classes, init=1)

    def forward(self, x, label=None):
        if label is None:
            phi =  self.enc(x)
        else:
            phi = self.enc(x, label)
        y = self.output_layer(phi)
        return y, phi, None
    
    def model(self):
        return self.enc

    def head(self):
        return self.output_layer

class PovertyDataLoader:
    def __init__(self, batch_size=64, path=None, workers=4, fold='A'):
        dataset = PovertyMapDataset(root_dir=path,
                                    download=True, no_nl=False, fold=fold, use_ood_val=False)
        # get all train data
        transform = initialize_poverty_train_transform()

        train_sets = Poverty_Batched_Dataset(dataset, 'train', batch_size, transform)
        val_sets = Poverty_Batched_Dataset(dataset, 'val', batch_size, domain2idx=train_sets.domain2idx)
        test_sets = Poverty_Batched_Dataset(dataset, 'test', batch_size)
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
               datasets[split] = dataset.get_subset(split, transform=transform)
        self._training_loader = DataLoader(
            train_sets, shuffle=True, # Shuffle training dataset                    
            sampler=None, collate_fn=train_sets.collate, batch_size=batch_size,
            num_workers=workers, pin_memory=True
        )
        self._training_loader_sequential = DataLoader(
            train_sets, 
            shuffle=False,              
            sampler=None, 
            collate_fn=train_sets.collate, 
            batch_size=batch_size,
            num_workers=workers, 
            pin_memory=True
        )
        self._validation_loader = DataLoader(
            val_sets, shuffle=False,              
            sampler=None, collate_fn=datasets['val'].collate, batch_size=batch_size,
            num_workers=workers, pin_memory=True
        )
        self._test_loader = DataLoader(
            test_sets, shuffle=False,              
            sampler=None, collate_fn=datasets['test'].collate, batch_size=batch_size,
            num_workers=workers, pin_memory=True
        )
        self.training_dataset = train_sets
        self.training_yarray = train_sets.targets.squeeze().cpu().numpy()

    @property
    def training_loader(self):
        return self._training_loader
    
    @property
    def training_loader_sequential(self):
        return self._training_loader_sequential

    @property
    def validation_loader(self):
        return self._validation_loader
    
    @property
    def test_loader(self):
        return self._test_loader
    

