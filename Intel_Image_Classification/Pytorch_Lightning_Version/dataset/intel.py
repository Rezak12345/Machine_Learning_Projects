
import sys, os
from PIL import Image
sys.path = [r'..\ML_Projects\Intel_Image_Classification\Pytorch_Lightning_Version'] + sys.path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Tuple, Dict

class Intel_Dataset(Dataset):
    def __init__(self, data_dir: str, image_size: int, class_names: List[str], data_type: str, *args, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.image_size = (image_size, image_size)
        self.class_names = class_names
        self.data_type = data_type  
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.image_paths, self.labels = self._get_image_paths_and_labels()
        self.num_classes = len(set(self.labels))

    def _get_image_paths_and_labels(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []

        if self.data_type == 'pred':
            pred_path = os.path.join(self.data_dir, f'seg_{self.data_type}', f'seg_{self.data_type}')
            image_paths = [os.path.join(pred_path, fname) for fname in os.listdir(pred_path)
                           if fname.endswith(('.png', '.jpg', '.jpeg'))]
            labels = [-1] * len(image_paths)  
        else:
            class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
            data_path = os.path.join(self.data_dir, f'seg_{self.data_type}', f'seg_{self.data_type}')

            for class_name in os.listdir(data_path):
                class_dir = os.path.join(data_path, class_name)
                if os.path.isdir(class_dir):
                    class_idx = class_to_idx.get(class_name, -1)
                    paths = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)
                             if fname.endswith(('.png', '.jpg', '.jpeg'))]

                    image_paths.extend(paths)
                    labels.extend([class_idx] * len(paths))
                    print(f"Nombre d'images dans '{class_name}': {len(paths)}")

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        image = self.transform(image)
        return {"data": image, "labels": label}