# 🧠 Machine Learning Projects with PyTorch Lightning

This repository is a collection of machine learning projects implemented using **[PyTorch Lightning](https://www.pytorchlightning.ai/)**. It includes both **classification** and **semantic segmentation** tasks, leveraging architectures like **Convolutional Neural Networks (CNNs)** and **UNet**.

The goal is to provide modular, reproducible, and scalable ML pipelines using Lightning's clean and flexible framework.

---

## ⚙️ Framework Overview: PyTorch Lightning

[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) is a lightweight wrapper on top of PyTorch that abstracts away much of the engineering boilerplate while preserving full control over research logic. It simplifies:

- training loops,
- model checkpoints,
- distributed training,
- logging & visualization.

All projects here follow LightningModule-based architecture to ensure clarity and maintainability.

---

## 📂 Project Structure


├── Image_Category_Classification

    ├── models
            ├── custom_checkpoint.py         
            ├── model.py

    ├── Modules
            ├── Data_Module.py
            ├── Lightning_Classifier.py
            
    ├── config
            ├── Data
            ├── Dataset
            ├── module
            ├── saves
            ├── train

    ├── dataset
            ├── process.py

    ├── Test.py
    ├── Train.py
    ├── labels.py

├── Intel_Image_Classification

    ├── models
            ├── custom_checkpoint.py
            ├── model.py

    ├── Modules
            ├── Data_Module.py
            ├── Lightning_Classifier.py

    ├── config
            ├── Data
            ├── Dataset
            ├── module
            ├── saves
            ├── train

    ├── dataset
            ├── intel.py
    
    ├── Train.py
    ├── labels.py

├── Semantique_Segmentation_With_UNet

    ├── models
            ├── custom_checkpoint.py
            ├── model.py

    ├── Modules
            ├── Data_Module.py
            ├── Lightning_Segmentator.py

    ├── config
            ├── Data
            ├── Dataset
            ├── module
            ├── saves
            ├── train
            
    ├── dataset
            ├── city.py
    
    ├── Train.py

## 📊 Datasets Used

🖼️ Image Classification
Dataset:https://www.kaggle.com/datasets/puneet6060/intel-image-classification

Alternative datasets can be integrated easily (e.g., CIFAR-10, ImageNet subsets)

🏙️ Semantic Segmentation
Cityscapes Dataset (urban scene understanding)

Source: https://www.cityscapes-dataset.com

Includes .zip downloads, not versioned here due to size

Note: Datasets are referenced or loaded using PyTorch Datasets or custom loaders. Large files are ignored via .gitignore.

## 🚀 How to Run

1. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
2. Train a model
bash
Copy
Edit
python train.py --config config.yaml
Each project includes a config file and script tailored to the task.

## ✅ Features

📦 PyTorch Lightning training structure

🔁 Training/validation/test split handling

📉 Early stopping and checkpointing

📈 TensorBoard/CSV logging

⚙️ Configurable hyperparameters

🧪 Reproducible experiments

## 🙏 Credits / Inspiration

These implementations are inspired by the excellent open-source contributions from:

- 📘 [Gokul Karthik – Image Segmentation with UNet (Kaggle)](https://www.kaggle.com/code/gokulkarthik/image-segmentation-with-unet-pytorch)  

- 🧠 [@milesial — PyTorch-UNet](https://github.com/milesial/Pytorch-UNet)  

- ⚡ [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)  

Thanks to all contributors in the open-source ML community 🙌

## 📄 License

This repository is released under the MIT License. Please review the license file before reuse.
