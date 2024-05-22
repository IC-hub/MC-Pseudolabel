import os
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.poverty_dataset import PovertyMapDataset

from resnet_multispectral import ResNet18

import torch
from torch.utils.data import Dataset
from sklearn.neighbors import KernelDensity

from model import Output

# code base: https://github.com/p-lambda/wilds

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


    # def reset_batch(self):
    #     """Reset batch indices for each domain."""
    #     self.batch_indices, self.batches_left = {}, {}
    #     for loc, d_idx in enumerate(self.domain_indices):
    #         self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
    #         self.batches_left[loc] = len(self.batch_indices[loc])

    # def get_batch(self, domain):
    #     """Return the next batch of the specified domain."""
    #     batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
    #     return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
    #            self.targets[batch_index], self.domains[batch_index]

    def get_input(self, idx):
        """Returns x for a given idx."""
        img = np.load(self.root / 'images' / f'landsat_poverty_img_{self.split_idx[idx]}.npz')['x']
        if self.no_nl:
            img[-1] = 0
        img = torch.from_numpy(img).float()
        return img


    def __getitem__(self, idx):
        # print (self.domains[idx])
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
    def __init__(self, weights=None, num_classes=1):
        super(ResnetMS, self).__init__()

        #resnet18_ms
        self.enc = ResNet18(num_classes=num_classes, num_channels=8)
        # if weights is not None:
        #     self.load_state_dict(deepcopy(weights))
        
        self.output_layer = Output(num_classes, num_classes, init=1)

    # def reset_weights(self, weights):
    #     self.load_state_dict(deepcopy(weights))

    def forward(self, x):
        phi =  self.enc(x)
        y = self.output_layer(phi)
        return y, phi, None
    
    def model(self):
        return self.enc

    def head(self):
        return self.output_layer

class PovertyDataLoader:
    def __init__(self, batch_size=64, path='/home/jiayunwu/multicalibration/C-Mixup/PovertyMap/wilds', workers=4, fold='A'):
        dataset = PovertyMapDataset(root_dir=path,
                                    download=True, no_nl=False, fold=fold, use_ood_val=False)
        # get all train data
        transform = initialize_poverty_train_transform()

        train_sets = Poverty_Batched_Dataset(dataset, 'train', batch_size, transform)
        # print (train_sets.domain2idx)
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
    

