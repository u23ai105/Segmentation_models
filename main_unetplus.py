import torch
import csv
import os
from torch import optim,nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import DiceBCELoss, deep_supervision_loss
from tqdm import tqdm
from unet_plus import UNetPlus
from img_dataset import ImgDataset
from augmentations import get_training_augmentation, get_validation_augmentation

CHECKPOINT_PATH = "/content/gdrive/My Drive/segmentation_models/unet_plus_checkpoint.pth"
CSV_LOG_PATH = "/content/gdrive/My Drive/segmentation_models/training_log.csv"

def save_checkpoint(epoch, model, optimizer, scheduler, loss):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'loss': loss
    }, CHECKPOINT_PATH)

def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean()

def load_checkpoint(model, optimizer, scheduler):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0

def log_to_csv(epoch, train_loss, val_loss, dice_score):
    file_exists = os.path.isfile(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "Train Loss", "Val Loss", "Dice Score"])
        writer.writerow([epoch, train_loss, val_loss, dice_score])

if __name__ == "__main__":
    LEARNING_RATE=1E-3
    BATCH_SIZE=2
    EPOCHS=15
    DATA_PATH="/content/brain-tumor-image-dataset-semantic-segmentation"

    device="cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = ImgDataset(root_path=DATA_PATH, transforms=get_training_augmentation())
    val_dataset = ImgDataset(root_path=DATA_PATH, transforms=get_validation_augmentation(), val=True)

    train_dataloader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    val_dataloader=DataLoader(dataset=val_dataset,batch_size=BATCH_SIZE,shuffle=False)

    model=UNetPlus(in_channels=3,out_channels=1).to(device)
    optimizer=optim.AdamW(model.parameters(),lr=LEARNING_RATE)
    criterion=DiceBCELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    start_epoch = load_checkpoint(model, optimizer, scheduler)

    try:
     for epoch in range(start_epoch,EPOCHS):
        print(f"Starting Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training")

        for images, masks in pbar:
            images = images.float().to(device)
            masks = masks.float().to(device)

            outputs = model(images)
            loss = deep_supervision_loss(outputs, masks, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} finished. Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        dice_total = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation")
            for images, masks in val_pbar:
                images = images.float().to(device)
                masks = masks.float().to(device)

                outputs = model(images)
                loss = deep_supervision_loss(outputs, masks, criterion)
                final_pred = torch.sigmoid(outputs[-1])
                dice = dice_score(final_pred, masks)

                val_loss += loss.item()
                dice_total += dice.item()

                val_pbar.set_postfix({"Val Loss": loss.item(), "Dice": dice.item()})
                print("Mask sum:", masks.sum().item(), "Pred sum:", (final_pred > 0.5).float().sum().item())

        avg_val_loss = val_loss / len(val_dataloader)
        avg_dice = dice_total / len(val_dataloader)

        scheduler.step(avg_val_loss)

        print(f"\n[Epoch {epoch + 1}] Train Loss: {total_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Dice: {avg_dice:.4f}\n")
        save_checkpoint(epoch, model, optimizer, scheduler, avg_val_loss)
        log_to_csv(epoch + 1, avg_train_loss, avg_val_loss, avg_dice)

        torch.save(model.state_dict(), f"/content/gdrive/My Drive/segmentation_models/unet_plus_epoch_{epoch + 1}.pth")
        print(f"Model saved to"+f"/content/gdrive/My Drive/segmentation_models/unet_plus_epoch_{epoch + 1}.pth")

    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        save_checkpoint(epoch, model, optimizer, scheduler, avg_val_loss)
        print("Checkpoint saved. You can resume training later.")