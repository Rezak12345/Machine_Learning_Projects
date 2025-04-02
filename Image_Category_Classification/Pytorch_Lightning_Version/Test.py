import sys
sys.path = [r'..\ML_Projects\Image_Category_Classification\Pytorch_Lightning_Version'] + sys.path
from Modules.Lightning_Classifier import Classifier
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
pl.seed_everything(2)
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

@hydra.main(config_path=r"..\Pytorch_Lightning_Version\config", config_name="Train_Model.yaml", version_base="1.3.2")

def main(config: DictConfig):
    
    train_config = config.train

    model_test = Classifier.load_from_checkpoint(
                        config.saves.load_from_checkpoint.best_model_path, 
                        train_config=train_config,
                        num_classes=config.module.lightningmodel.num_classes, 
                        backbone=config.module.lightningmodel.backbone,
                       )
    
    dataset = config.dataset
    batch_size = config.train.batch_size
    train_split = config.train.train_split
    val_split = config.train.val_split
    test_split = config.train.test_split
    datamodule_config = config.data.animal_datamodule

    data_module = instantiate(datamodule_config,
                              dataset=dataset,
                              batch_size=batch_size,
                              train_split=train_split,
                              val_split=val_split,
                              test_split=test_split,
                              _recursive_=False,
                             )

    csv_logger = CSVLogger(name="csv_logs", save_dir=config.saves.logging.logs)
    tb_logger = TensorBoardLogger(name="tensorboard",save_dir=config.saves.logging.logs)

    trainer = instantiate(config.train.trainer,
                          logger=[csv_logger, tb_logger],
                         )
        
    trainer.test(model_test, data_module)
if __name__ == "__main__":
    main()

