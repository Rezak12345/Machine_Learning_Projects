import os, sys
sys.path = [r'..\ML_Projects\Semantique_Segmentation_With_UNet\Pytorch_Lightning_Version'] + sys.path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def split_image(image):
    image = np.array(image)
    cityscape, label = image[:, :256, :], image[:, 256:, :]
    return cityscape, label

class CityscapeDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, image_size: tuple):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_fns = os.listdir(image_dir)
        self.label_fns = os.listdir(label_dir)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),  # Convertir en tensor
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Normalisation
        ])

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
      
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        cityscape, _ = split_image(image)  
        cityscape = Image.fromarray(cityscape)
        cityscape = self.transform(cityscape)
        label_fn = os.path.splitext(image_fn)[0] 
        label_fp = os.path.join(self.label_dir, f"{label_fn}_label_kmeans.pt")
        label_class = torch.load(label_fp)
        assert label_class.shape == (256, 256), "Les dimensions des labels doivent être [256, 256]"

        return {"data": cityscape, "labels": label_class}