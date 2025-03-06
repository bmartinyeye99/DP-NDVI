import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns

def plot_regression(model, dataloader, image_size=256, patch_size=64, stride=32, epoch=None):
    """
    Args:
        model: The trained model.
        dataloader: DataLoader for validation or test data.
        image_size: Size of the full image (default: 256).
        patch_size: Size of each patch (default: 64).
        stride: Stride for patch extraction (default: 32).
        epoch: Current epoch number (for title).
    """
    model.eval()
    device = next(model.parameters()).device  # Get the device of the model

    # Process one batch (one image) from the dataloader
    batch_patches, batch_targets = next(iter(dataloader))
    inputs, targets = batch_patches[0], batch_targets[0]  # Get the first image's patches

    # Move inputs to the correct device
    inputs = inputs.to(device)

    # Get predictions
    with torch.no_grad():
        preds = model(inputs).cpu().numpy()  # Shape: [num_patches, 1, patch_size, patch_size]

    # Reshape predictions and targets
    preds = preds[:, 0, :, :]  # Remove channel dimension
    targets = targets.numpy()[:, 0, :, :]  # Remove channel dimension

    # Stitch patches into full-size images
    pred_ndvi = stitch_patches(preds, image_size, patch_size, stride)
    gt_ndvi = stitch_patches(targets, image_size, patch_size, stride)

        # Scatter plot with regression line
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.regplot(
        x=gt_ndvi.flatten(), 
        y=pred_ndvi.flatten(), 
        scatter_kws={'alpha': 0.5}, 
        line_kws={'color': 'red'}  # Set regression line color to red
    )
    plt.xlabel("Ground Truth NDVI")
    plt.ylabel("Predicted NDVI")

    # The plot_regression function processes one batch from the validation DataLoader.
    # It selects the first image in the batch (batch_patches[0] and batch_targets[0]).
    # The patches for this image are stitched back into a full-size image, and the colormap
    #     is generated for both the ground truth and predicted NDVI.
    # Colormap comparison
    plt.subplot(1, 2, 2)
    plt.imshow(gt_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar()
    plt.axis('off')
    plt.title("Ground Truth NDVI")
    plt.show()

    plt.imshow(pred_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title("Predicted NDVI")
    plt.colorbar()
    plt.axis('off')
    plt.show()

# Helper functions
def compute_rgb_indices(rgb, eps=1e-6):
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ngrdi = (G - R) / (G + R + eps)
    vari  = (G - R) / (G + R - B + eps)
    gli   = (2 * G - R - B) / (2 * G + R + B + eps)
    return {'NGRDI': ngrdi, 'VARI': vari, 'GLI': gli}

def compute_ndvi(nir, red, eps=1e-6):
    return (nir - red) / (nir + red + eps)

def stitch_patches(patch_predictions, image_size=256, patch_size=64, stride=32):
    """
    Stitch patches into a full-size image.
    
    Args:
        patch_predictions: Array of shape [num_patches, patch_size, patch_size].
        image_size: Size of the full image.
        patch_size: Size of each patch.
        stride: Stride used for patch extraction.
    
    Returns:
        Full-size image of shape [image_size, image_size].
    """
    ndvi_full = np.zeros((image_size, image_size), dtype=np.float32)
    weight_matrix = np.zeros((image_size, image_size), dtype=np.float32)

    patch_idx = 0
    for i in range(0, image_size - patch_size + 1, stride):
        for j in range(0, image_size - patch_size + 1, stride):
            ndvi_full[i:i+patch_size, j:j+patch_size] += patch_predictions[patch_idx]
            weight_matrix[i:i+patch_size, j:j+patch_size] += 1
            patch_idx += 1
    
    ndvi_full /= np.maximum(weight_matrix, 1)  # Avoid division by zero
    return ndvi_full