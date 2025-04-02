# This code defines an `AnimalDataModule` class for managing data in a deep learning project 
# using PyTorch Lightning. This class allows for loading, randomly splitting, and preparing
# data for training, validation, and testing of models, using a specific dataset 
# (in this case, an animal dataset). It simplifies the management of different phases 
# of the data pipeline, ensuring that the datasets are correctly divided into training, 
# validation, and test sets, while providing automation of the data loading process via DataLoader.

import os, sys
sys.path = [r'..\ML_Projects\Image_Category_Classification\Pytorch_Lightning_Version'] + sys.path
from dataset.process import AnimalDataset
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
from hydra.utils import instantiate
import random

class AnimalDataModule(pl.LightningDataModule):

    def __init__(self, dataset: DictConfig, batch_size: DictConfig, train_split: DictConfig, val_split: DictConfig, test_split: DictConfig, train_workers, val_workers,):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.train_workers = train_workers
        self.val_workers = val_workers

    def setup(self, stage: str = None):

        full_dataset: AnimalDataset = instantiate(self.dataset.train, 
                                                  _recursive_=False,
                                                  )

        dataset_size = len(full_dataset)
        train_size = int(self.train_split * dataset_size)
        val_size = int(self.val_split * dataset_size)
        test_size = dataset_size - train_size - val_size

        indices = list(range(dataset_size))
        random.shuffle(indices)  

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        if stage == 'fit' or stage is None:
            self.train_dataset = Subset(full_dataset, train_indices)
            self.val_dataset = Subset(full_dataset, val_indices)
        
        if stage == 'test' or stage is None:
            self.test_dataset = Subset(full_dataset, test_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.train_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.val_workers, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

