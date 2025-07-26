import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Augmentation for training set
def get_training_augmentation():
    train_transform = A.Compose([
        A.Resize(572, 572),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # standard ImageNet stats
        ToTensorV2()
    ])
    return train_transform

# Augmentation for validation/test set
def get_validation_augmentation():
    val_transform = A.Compose([
        A.Resize(572, 572),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return val_transform
