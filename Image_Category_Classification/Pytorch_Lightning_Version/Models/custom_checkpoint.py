import sys
sys.path = [r'..\ML_Projects\Image_Category_Classification\Pytorch_Lightning_Version'] + sys.path
from labels import ANIMAL_LABELS
import pytorch_lightning as pl
import torch

class ModelCheckpoint(pl.Callback):
    def __init__(self, save_dir, best_model_name):
        super().__init__()
        self.save_dir = save_dir
        self.best_model_name = best_model_name
        self.best_epoch = -1
        self.best_val_loss = float('inf')
    
    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = trainer.current_epoch
            
            model_state = pl_module.state_dict()
            optimizer_state = trainer.optimizers[0].state_dict()
            loss_dict = {
                "train_loss": trainer.callback_metrics.get("train_loss", None),
                "val_loss": val_loss.item(),
            }
            accuracy_dict = {
                "train_accuracy": trainer.callback_metrics.get("train_accuracy", None),
                "val_accuracy": trainer.callback_metrics.get("val_accuracy", None),
            }

            file_path = f"{self.save_dir}/{self.best_model_name}_Ep_{self.best_epoch}.pt"
            
            torch.save(
                {
                    "epoch": self.best_epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer_state,
                    "loss_dict": loss_dict,
                    "accuracy_dict": accuracy_dict,
                    "CLASS_NAMES": ANIMAL_LABELS,
                },
                file_path
            )
            
            print(f"Model saved at epoch {self.best_epoch} with val_loss {self.best_val_loss:.4f}")


