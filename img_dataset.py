import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class ImgDataset(Dataset):
    def _init__(self,root_path,test=False,val=False):
        self.root_path=root_path
        if test:
            self.images=sorted([os.path.join(root_path,i) for i in os.listdir(root_path)])
            self.masks=sorted([os.path.join(root_path,"test_masks",i) for i in os.listdir(os.path.join(root_path,"test_masks"))])
        elif val:
            self.images = sorted([os.path.join(root_path, i) for i in os.listdir(root_path)])
            self.masks = sorted([os.path.join(root_path, "valid_masks", i) for i in os.listdir(os.path.join(root_path, "valid_masks"))])
        else:
            self.images=sorted([os.path.join(root_path,i) for i in os.listdir(root_path)])
            self.masks=sorted([os.path.join(root_path,"train_masks",i) for i in os.listdir(os.path.join(root_path,"train_masks"))])

        self.transform=transforms.Compose([
                transforms.Resize((512,512)),
                transforms.ToTensor()
        ])

    def _getitem_(self,index):
        img=Image.open(self.images[index]).convert("RGB")
        mask=Image.open(self.masks[index]).convert("L")

        return self.transform(img),self.transform(mask)

    def _len__(self):
        return len(self.images)