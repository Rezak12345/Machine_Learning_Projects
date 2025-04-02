# This code defines a custom dataset for the image classification model using PyTorch.
# The main purpose of this code is to create an `AnimalDataset` class that efficiently handles
# the preparation of training data for DINOv2. This class allows for loading images,
# transforming them into a format usable by the model (with resizing, normalization),
# and associating each image with a label corresponding to its class (cats, dogs, snakes). 
# Additionally, it includes a feature to limit the number of images per class, 
# which is useful for balancing the data or testing specific configurations.

import sys,os
from PIL import Image
sys.path = [r'..\ML_Projects\Image_Category_Classification\Pytorch_Lightning_Version'] + sys.path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Dict, Optional

class AnimalDataset(Dataset):
    def __init__(self, data_dir, image_size: int, limit_classes: Optional[List[Tuple[str, int]]] = None, *args, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = (image_size, image_size)  
        self.limit_classes = limit_classes or []
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_paths, self.labels = self._get_image_paths_and_labels()
        self.num_classes = len(set(self.labels))
        self.class_names = ['cats', 'dogs', 'snakes']

    def _get_image_paths_and_labels(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []
        class_to_idx = {'cats': 0, 'dogs': 1, 'snakes': 2}

        class_limits = {cls: limit for cls, limit in self.limit_classes}

        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                class_idx = class_to_idx.get(class_name, -1)
                paths = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

                if class_name in class_limits:
                    limited_paths = paths[:class_limits[class_name]]
                    image_paths.extend(limited_paths)
                    labels.extend([class_idx] * len(limited_paths))
                    print(f"Après limitation, nombre d'images dans '{class_name}': {len(limited_paths)}")
                else:
                    image_paths.extend(paths)
                    labels.extend([class_idx] * len(paths))
                    print(f"Nombre d'images dans '{class_name}': {len(paths)}")

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform(image)
        return {"data": image, "labels": label}
 