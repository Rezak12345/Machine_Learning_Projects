
import os, sys
sys.path = [r'..\ML_Projects\Semantique_Segmentation_With_UNet\Pytorch_Lightning_Version'] + sys.path
from dataset.city import CityscapeDataset
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing import Optional
from hydra.utils import instantiate
os.environ["MKL_THREADING_LAYER"] = "GNU"

class CityDataModule(pl.LightningDataModule):

    def __init__(self, dataset: DictConfig, batch_size: DictConfig, train_workers: int):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_workers = train_workers
        self.train_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str):

        if stage == 'fit':
     
            self.train_dataset: CityscapeDataset = instantiate(self.dataset.train, 
                                                            _recursive_=False,
                                                            )
            
            print(f"Train dataset: {len(self.train_dataset)} samples")

            assert self.train_dataset is not None, "Erreur : train_dataset n'a pas été correctement instancié"
            print(f"Train dataset: {len(self.train_dataset)} samples")
            
        if stage == 'predict':
        
            self.predict_dataset: CityscapeDataset = instantiate(self.dataset.val, 
                                                    _recursive_=False,
                                                    )
            
            print(f"Instantiating val dataset with config: {self.dataset.val}")
            try:
                self.predict_dataset: CityscapeDataset = instantiate(self.dataset.val, 
                                                                 _recursive_=False,
                                                                 )
            except Exception as e:
                print(f"Erreur lors de l'instanciation du predict_dataset: {e}")
                raise e  # Relancer l'exception après l'avoir loggée
            assert self.predict_dataset is not None, "Erreur : predict_dataset n'a pas été correctement instancié"
            print(f"Validation dataset: {len(self.predict_dataset)} samples")

    def train_dataloader(self) -> DataLoader:

        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.train_workers)
        print(f"Train DataLoader:")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Number of workers: {self.train_workers}")
        print(f"- Shuffle: True")
        print(f"- Persistent workers: True")
        print(f"- Number of batches: {len(dataloader)}")
        return dataloader
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.train_workers)



