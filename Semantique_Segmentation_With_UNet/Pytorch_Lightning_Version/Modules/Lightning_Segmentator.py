
import sys, os
sys.path = [r'..\ML_Projects\Semantique_Segmentation_With_UNet\Pytorch_Lightning_Version'] + sys.path
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import nn, optim
from Models.model import UNet
import torchmetrics
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassCohenKappa
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassJaccardIndex

class City_Segmentator(pl.LightningModule):
    def __init__(self, 
                 train_config: DictConfig,
                 num_classes,        
                 ):

        super().__init__()
        
        self.train_config = train_config
        self.classifier = UNet(num_classes=num_classes)
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = train_config.lr
        self.batch_size = train_config.batch_size
        self.scheduler = train_config.scheduler
        self.optimizer = train_config.optimizer
        self.accuracy  = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="micro", multidim_average="global")
        self.F1_score = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.F1_score_classe = MulticlassF1Score(num_classes=num_classes, average="none")
        self.Kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.jaccard = MulticlassJaccardIndex(num_classes=num_classes)
        self.classes_metrics = {}

    def forward(self, x: torch.Tensor):

        x = self.classifier(x)
    
        return x

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer,
                                params=self.parameters(),
                                lr=self.lr)
        scheduler = instantiate(self.scheduler, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.train_config.optimizer_monitor,
                "strict": False
            }
        }
    def _common_step(self, batch: dict, batch_idx):

        x = batch["data"]
        y = batch["labels"]
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        
        return loss, scores, y


    def training_step(self, batch: dict, batch_idx):

        loss, scores, y = self._common_step(batch, batch_idx)
        preds = torch.argmax(scores, dim=1)
        self.log_dict(
                {
                    "train_loss": loss,
                    
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.batch_size
            )
        return {"loss": loss, "scores": scores, "y": y}

    def predict_step(self, batch, batch_idx):
        x = batch['data']
        y = batch['labels']
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
                
    



            


