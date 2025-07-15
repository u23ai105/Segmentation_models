import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

def is_image_file(filename):
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return filename.lower().endswith(IMG_EXTENSIONS)

class ImgDataset(Dataset):
    def __init__(self,root_path,test=False,val=False):
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

        self.transform=transforms.Compose([
                transforms.Resize((572,572)),
                transforms.ToTensor()
        ])

    def __getitem__(self,index):
        img=Image.open(self.images[index]).convert("RGB")
        mask=Image.open(self.masks[index]).convert("L")

        return self.transform(img),self.transform(mask)

    def __len__(self):
        return len(self.images)