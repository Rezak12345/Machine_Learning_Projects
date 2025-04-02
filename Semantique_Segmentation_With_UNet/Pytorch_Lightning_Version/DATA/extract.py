import tarfile
import zipfile
import os

# Chemin vers le fichier .tgz
zip_path = r'..\ML_Projects\Semantique_Segmentation_With_UNet\Pytorch_Lightning_Version\DATA\ZipFile\City_Data.zip'

# Répertoire dans lequel les fichiers seront extraits
extract_path = r'..\ML_Projects\Semantique_Segmentation_With_UNet\Pytorch_Lightning_Version\DATA'

# Créer le répertoire s'il n'existe pas déjà
os.makedirs(extract_path, exist_ok=True)

# Ouvrir et extraire le fichier .tgz
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction terminée.")
