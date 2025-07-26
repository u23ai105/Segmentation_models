import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
import sys

# --- Imports from your project files ---
# This assumes your script is run from a directory where it can see /content/
# Or that you've added /content to your PYTHONPATH
if '/content' not in sys.path:
    sys.path.append('/content')

from custom_unet import Unet
from dataset import SemanticSegmentationDataset
from losses import DiceLoss  # We can reuse the loss function to calculate the score


def test_model():
    """
    Loads a trained model and evaluates it on a test dataset.
    Calculates the average Dice score and saves prediction masks.
    """
    # --- Configuration ---
    IMG_SIZE = 224  # Must match the training image size
    BATCH_SIZE = 4  # Can be larger for testing
    # ERROR FIX: Updated the model path to match the save location from your training log.
    MODEL_PATH = "./content/gdrive/My Drive/segmentation_model/best_model.pth"

    # --- Paths for Test Data ---
    # IMPORTANT: Replace these with the paths to your actual test data
    test_img_path = '/content/brain-tumor-image-dataset-semantic-segmentation/test'
    test_mask_path = '/content/brain-tumor-image-dataset-semantic-segmentation/test_masks'

    # Directory to save the visual predictions
    prediction_save_dir = "./test_predictions"
    os.makedirs(prediction_save_dir, exist_ok=True)

    # --- Device Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        print("Please ensure your Google Drive is mounted and the path is correct.")
        return

    # Instantiate the model with the same architecture as during training
    model = Unet(
        backbone='swin_base_patch4_window7_224',
        in_channels=3,
        num_classes=1,
        pretrained=False,  # No need to download pretrained weights again
        use_transformer_bottleneck=False
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- Data Loading ---
    # Create dummy test data if it doesn't exist
    if not os.path.exists(test_img_path):
        print("Creating dummy test dataset for demonstration...")
        for p in [test_img_path, test_mask_path]: os.makedirs(p, exist_ok=True)
        for i in range(10):
            Image.new('RGB', (IMG_SIZE, IMG_SIZE)).save(os.path.join(test_img_path, f'test_img_{i}.jpg'))
            Image.new('L', (IMG_SIZE, IMG_SIZE), color=255 if i % 2 != 0 else 0).save(
                os.path.join(test_mask_path, f'test_mask_{i}.png'))

    image_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor()
    ])

    test_dataset = SemanticSegmentationDataset(
        test_img_path,
        test_mask_path,
        image_transforms=image_transforms,
        mask_transforms=mask_transforms
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # --- Evaluation Loop ---
    criterion = DiceLoss(from_logits=True)
    total_dice_score = 0.0

    with torch.no_grad():  # Disable gradient calculations for inference
        progress_bar = tqdm(test_loader, desc="Testing")
        for i, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Calculate Dice Loss, then convert to Dice Score
            loss = criterion(outputs, masks)
            dice_score = 1.0 - loss.item()
            total_dice_score += dice_score

            # --- Save Prediction Masks ---
            # Apply sigmoid to get probabilities, then threshold to get binary mask
            preds = torch.sigmoid(outputs) > 0.5

            for j in range(preds.shape[0]):
                # Get the original filename to name the prediction
                original_img_path = test_dataset.img_paths[i * BATCH_SIZE + j]
                original_filename = os.path.basename(original_img_path)

                # Convert tensor to numpy array and then to a PIL image
                pred_mask_np = preds[j].squeeze().cpu().numpy().astype(np.uint8) * 255
                pred_mask_img = Image.fromarray(pred_mask_np, mode='L')

                # Save the image
                pred_mask_img.save(os.path.join(prediction_save_dir, original_filename))

    # --- Final Score ---
    avg_dice_score = total_dice_score / len(test_loader)
    print("\n--- Testing Complete ---")
    print(f"Average Dice Score on Test Set: {avg_dice_score:.4f}")
    print(f"Predicted masks have been saved to: {prediction_save_dir}")


if __name__ == '__main__':
    test_model()
