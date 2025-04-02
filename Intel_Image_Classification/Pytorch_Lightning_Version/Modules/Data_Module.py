import os, sys
sys.path = [r'..\ML_Projects\Intel_Image_Classification\Pytorch_Lightning_Version'] + sys.path
from dataset.intel import Intel_Dataset
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Subset
from typing import Optional
from hydra.utils import instantiate

class IntelDataModule(pl.LightningDataModule):

    def __init__(self, dataset: DictConfig, batch_size: DictConfig, train_workers: int):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.train_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit' or stage is None:
     
            self.train_dataset: Intel_Dataset = instantiate(self.dataset.train, 
                                                            _recursive_=False,
                                                            )
        
            self.test_dataset: Intel_Dataset = instantiate(self.dataset.val, 
                                                    _recursive_=False,
                                                    )
        if stage == 'predict' or stage is None:

            self.predict_dataset: Intel_Dataset = instantiate(self.dataset.predict, 
                                                        _recursive_=False,
                                                        )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.train_workers)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)


