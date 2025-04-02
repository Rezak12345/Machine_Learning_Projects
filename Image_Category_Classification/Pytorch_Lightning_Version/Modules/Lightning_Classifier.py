# This code defines a PyTorch Lightning class for a custom DINOv2. 
# The model uses a backbone and a classifier to make predictions and includes several performance metrics, 
# such as Accuracy, F1 Score, Kappa, and Jaccard Index, to evaluate the quality of the predictions. 
# The code handles the training, validation, and testing of the model, with automated logging to track performance 
# in real-time and hyperparameter adjustments, making it a complete pipeline for evaluating classification models.

import sys
sys.path = [r'..\ML_Projects\Image_Category_Classification\Pytorch_Lightning_Version'] + sys.path
import torch
from cmath import nan
import torch.nn as nn
import pytorch_lightning as pl
from torch import nn, optim
from Models.model import ConvNet
from labels import ANIMAL_LABELS
import torchmetrics
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassCohenKappa
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassJaccardIndex

class CNN_Classifier(pl.LightningModule):
    def __init__(self, 
                 train_config: DictConfig,
                 num_classes,        
                 ):

        super().__init__()
        
        self.train_config = train_config
        self.classifier = ConvNet(num_classes=num_classes)
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
        all_zeros = torch.all(y == 0)
        if not all_zeros: 
            loss = self.loss_fn(scores, y)               
            return loss, scores, y
        else:
            return None, scores, y    


    def training_step(self, batch: dict, batch_idx):

        loss, scores, y = self._common_step(batch, batch_idx)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, y)

        if loss is not None and not torch.isnan(loss):

            self.log_dict(
                    {
                        "train_loss": loss,
                        "train_accuracy": accuracy,
                        
                    },
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=self.batch_size
                )
            return {"loss": loss, "scores": scores, "y": y}


    def validation_step(self, batch, batch_idx):

        loss, scores, y = self._common_step(batch, batch_idx)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, y)
    
        if loss is not None and not torch.isnan(loss):

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
            self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
            return loss, accuracy
    
    def test_step(self, batch, batch_idx):

        loss, scores, y = self._common_step(batch, batch_idx)
        
        all_zeros = torch.all(y == 0)

        if not all_zeros: 
            self.accuracy.update(scores, y)
            self.F1_score.update(scores, y)
            self.F1_score_classe.update(scores, y)
            self.Kappa.update(scores, y)
            self.jaccard.update(scores, y)    
          
        else:
            return None

    def on_test_epoch_end(self):
        
        accuracy = self.accuracy.compute()
        print('acc', accuracy)
        f1_score = self.F1_score.compute()
        f1_score_classe = self.F1_score_classe.compute()
        print("f1_score_classe", f1_score_classe.shape)
        print("f1_score_classe", f1_score_classe)
        kappa = self.Kappa.compute()
        jaccard = self.jaccard.compute()

        for class_idx, class_name in ANIMAL_LABELS.items():
            f1_score_class = f1_score_classe[class_idx]  
            self.log(f"f1_score_{class_name}", f1_score_class, prog_bar=True)

        self.log('accuracy', accuracy, prog_bar=True)
        self.log('kappa', kappa, prog_bar=True)
        self.log('f1', f1_score, prog_bar=True)
        self.log('jaccard', jaccard, prog_bar=True)
        print(self.classes_metrics)

        return accuracy, f1_score, f1_score_classe, kappa, jaccard
            


