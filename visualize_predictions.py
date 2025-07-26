import torch
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T
import sys
import matplotlib.pyplot as plt
import random

# --- Imports from your project files ---
# This assumes your script is run from a directory where it can see /content/
# Or that you've added /content to your PYTHONPATH
if '/content' not in sys.path:
    sys.path.append('/content')

from custom_unet import Unet
from dataset import SemanticSegmentationDataset


def unnormalize(tensor, mean, std):
    """Reverses the normalization on a tensor image for visualization."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_predictions():
    """
    Loads a trained model and saves visual comparisons of its predictions
    against the ground truth for a few random samples from the test set.
    """
    # --- Configuration ---
    IMG_SIZE = 224  # Must match the training image size
    NUM_SAMPLES_TO_SHOW = 10
    MODEL_PATH = "./content/gdrive/My Drive/segmentation_model/best_model.pth"

    # --- Paths for Test Data ---
    test_img_path = '/content/brain-tumor-image-dataset-semantic-segmentation/test'
    test_mask_path = '/content/brain-tumor-image-dataset-semantic-segmentation/test_masks'

    # **NEW**: Directory to save the output plot images
    visualization_save_dir = "./sample_visualizations"
    os.makedirs(visualization_save_dir, exist_ok=True)

    # --- Device Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        print("Please ensure your Google Drive is mounted and the path is correct.")
        return

    model = Unet(
        backbone='swin_base_patch4_window7_224',
        in_channels=3,
        num_classes=1,
        pretrained=False,
        use_transformer_bottleneck=False
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- Data Loading ---
    if not os.path.exists(test_img_path):
        print(f"Error: Test image directory not found at {test_img_path}")
        return

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    image_transforms = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
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

    # --- Visualization Loop ---
    print(f"Generating and saving {NUM_SAMPLES_TO_SHOW} random samples...")

    # Get random indices for samples
    sample_indices = random.sample(range(len(test_dataset)), min(NUM_SAMPLES_TO_SHOW, len(test_dataset)))

    with torch.no_grad():
        for idx in sample_indices:
            image, true_mask = test_dataset[idx]

            # Add batch dimension and send to device
            image_tensor = image.unsqueeze(0).to(device)

            # Get model prediction
            output = model(image_tensor)
            pred_mask = torch.sigmoid(output) > 0.5

            # --- Plotting ---
            img_to_plot = unnormalize(image, mean=norm_mean, std=norm_std)
            img_to_plot = img_to_plot.cpu().numpy().transpose(1, 2, 0)

            true_mask_to_plot = true_mask.squeeze().cpu().numpy()
            pred_mask_to_plot = pred_mask.squeeze().cpu().numpy()

            original_filename = os.path.basename(test_dataset.img_paths[idx])

            # Create subplot
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Sample: {original_filename}', fontsize=16)

            ax[0].imshow(img_to_plot)
            ax[0].set_title("Input Image")
            ax[0].axis('off')

            ax[1].imshow(true_mask_to_plot, cmap='gray')
            ax[1].set_title("Ground Truth Mask")
            ax[1].axis('off')

            ax[2].imshow(pred_mask_to_plot, cmap='gray')
            ax[2].set_title("Predicted Mask")
            ax[2].axis('off')

            # **FIX**: Instead of plt.show(), save the figure to a file
            save_path = os.path.join(visualization_save_dir, f"comparison_{original_filename}")
            plt.savefig(save_path)
            plt.close(fig)  # Close the figure to free up memory

    print("\n--- Visualization Complete ---")
    print(f"Saved {NUM_SAMPLES_TO_SHOW} sample plots to: {visualization_save_dir}")


if __name__ == '__main__':
    visualize_predictions()
