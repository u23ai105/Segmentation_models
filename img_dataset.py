import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

def is_image_file(filename):
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return filename.lower().endswith(IMG_EXTENSIONS)

class ImgDataset(Dataset):
    def __init__(self,root_path, transforms=None,test=False,val=False):
        if val==False and test ==False:
            image_folder = os.path.join(root_path, "train")
            mask_folder = os.path.join(root_path, "train", "train_masks")
        elif val:
            image_folder = os.path.join(root_path, "valid")
            mask_folder = os.path.join(root_path, "valid", "valid_masks")
        elif test:
            image_folder = os.path.join(root_path, "test")
            mask_folder = os.path.join(root_path, "test", "test_masks")
        else:
            raise ValueError("Specify one of train, val, or test as True")

        self.images = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if os.path.isfile(os.path.join(image_folder, f)) and is_image_file(f)
        ])
        self.masks = sorted([
            os.path.join(mask_folder, f)
            for f in os.listdir(mask_folder)
            if os.path.isfile(os.path.join(mask_folder, f)) and is_image_file(f)
        ])

        # Slice the lists to only include the first 20 files.
        self.images = self.images[:20]
        self.masks = self.masks[:20]

        self.transform=transforms

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])

        if self.transform:
            augmented = self.transform(image=np.array(img), mask=np.array(mask))
            img = augmented["image"]
            mask = augmented["mask"]

        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = mask.unsqueeze(0)

        return img, mask

    def __len__(self):
        return len(self.images)