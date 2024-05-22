import os
import torch
from torch.utils.data import DataLoader, TensorDataset


class TensorLoader(object):

    def __init__(self, batch_size, path, split, workers, data_tensors=None):
        ''' @param data_tensors (train, val, test1, test2, ...)
        '''
        if data_tensors is None:
            data, label, env = torch.load(os.path.join(path, 'train.pt'))
        else:
            data, label, env = data_tensors[0]
        ids = torch.arange(len(label))
        self.training_dataset = TensorDataset(data, label, env, ids)
        if data_tensors is None:
            self.validation_dataset = TensorDataset(*torch.load(os.path.join(path, 'val.pt')))
        else:
            self.validation_dataset = TensorDataset(*data_tensors[1])
        self.test_dataset = {}
        for i, group in enumerate(split):
            if data_tensors is None:
                self.test_dataset[group] = TensorDataset(*torch.load(os.path.join(path, f'{group}.pt')))
            else:
                self.test_dataset[group] = TensorDataset(*data_tensors[i+2])
        self._training_loader = DataLoader(
            dataset=self.training_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=False
        )
        self._training_loader_sequential = DataLoader(
            dataset=self.training_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False
        )
        
        self._validation_loader = DataLoader(
            dataset=self.validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False
        )

        self._test_loader = {}
        for key in self.test_dataset.keys():
            self._test_loader[key] =  DataLoader(
                dataset=self.test_dataset[key],
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
                drop_last=False
            )        

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
    
    @property
    def feature_dim(self):
        return len(self.training_dataset[0][0])
    
