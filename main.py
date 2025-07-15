import torch
from torch import optim,nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from img_dataset import ImgDataset

if __name__ == "__main__":
    LEARNING_RATE=3E-4
    BATCH_SIZE=32
    EPOCHS=10
    DATA_PATH="/content/brain-tumor-image-dataset-semantic-segmentation/train"
    MODEL_SAVE_PATH = "/content/gdrive/My Drive/brain_tumor_unet.pth"
    VAL_DATA_PATH="/content/brain-tumor-image-dataset-semantic-segmentation/valid"

    device="cuda" if torch.cuda.is_available() else "cpu"
    train_dataset=ImgDataset(DATA_PATH)
    val_dataset=ImgDataset(VAL_DATA_PATH,val=True)

    train_dataloader=DataLoader(datset=train_dataset,batch_size=BATCH_SIZE,shffle=True)
    val_dataloader=DataLoader(datset=val_dataset,batch_size=BATCH_SIZE,shuffle=True)

    model=UNet(in_channles=3,num_classes=1).to(device)
    optimizer=optim.AdamW(model.parameters(),lr=LEARNING_RATE)
    criterion=nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss=0
        for idx,img_mask in enumerate(tqdm(train_dataloader)):
            img=img_mask[0].float().to(device)
            mask=img_mask[1].float().to(device)

            y_pred=model(img)
            optimizer.zero_grad()

            loss=criterion(y_pred,mask)
            train_running_loss+=loss.item()

            loss.backward()
            optimizer.step()

        train_loss=train_running_loss/(idx+1)

        model.eval()
        val_running_loss=0
        with torch.no_grad():
            for idx,img_mask in enumerate(tqdm(val_dataloader)):
                img=img_mask[0].float().to(device)
                mask=img_mask[1].float().to(device)

                y_pred=model(img)
                loss=criterion(y_pred,mask)

                val_running_loss+=loss.item()

            val_loss=val_running_loss/(idx+1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    torch.save(model.state_dict(),MODEL_SAVE_PATH)
