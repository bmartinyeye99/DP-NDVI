import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Helper functions
def compute_rgb_indices(rgb, eps=1e-6):
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    ngrdi = (G - R) / (G + R + eps)
    vari  = (G - R) / (G + R - B + eps)
    gli   = (2 * G - R - B) / (2 * G + R + B + eps)
    return {'NGRDI': ngrdi, 'VARI': vari, 'GLI': gli}

def compute_ndvi(nir, red, eps=1e-6):
    return (nir - red) / (nir + red + eps)

# Dataset with Overlapping Patches
class NDVIDataset(Dataset):
    def __init__(self, dataset_dir, image_size=256, patch_size=64, stride=32):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.filenames = sorted(os.listdir(dataset_dir))
        self.rgb_files = [f for f in self.filenames if f.lower().endswith('.jpg')]
        self.nir_files = [f.replace('.JPG', '.TIF') for f in self.rgb_files]

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.dataset_dir, self.rgb_files[idx])
        nir_path = os.path.join(self.dataset_dir, self.nir_files[idx])

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        if rgb_bgr is None or nir_img is None:
            raise ValueError(f"Failed to load images for index {idx}")
        
        rgb_bgr = cv2.resize(rgb_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        nir_img = cv2.resize(nir_img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        nir = nir_img.astype(np.float32) / 255.0
        indices = compute_rgb_indices(rgb)
        red = rgb[..., 0]
        gt_ndvi = compute_ndvi(nir, red)

        input_img = np.concatenate([rgb, nir[..., None], indices['NGRDI'][..., None],
                                    indices['VARI'][..., None], indices['GLI'][..., None]], axis=-1)

        patches = []
        gt_patches = []
        for i in range(0, self.image_size - self.patch_size + 1, self.stride):
            for j in range(0, self.image_size - self.patch_size + 1, self.stride):
                patch = input_img[i:i+self.patch_size, j:j+self.patch_size]
                gt_patch = gt_ndvi[i:i+self.patch_size, j:j+self.patch_size]
                
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float()
                gt_tensor = torch.from_numpy(gt_patch).unsqueeze(0).float()

                patches.append(patch_tensor)
                gt_patches.append(gt_tensor)

        return torch.stack(patches), torch.stack(gt_patches)

# Patch stitching function
def stitch_patches(patch_predictions, image_size=256, patch_size=64, stride=32):
    ndvi_full = np.zeros((image_size, image_size), dtype=np.float32)
    weight_matrix = np.zeros((image_size, image_size), dtype=np.float32)

    patch_idx = 0
    for i in range(0, image_size - patch_size + 1, stride):
        for j in range(0, image_size - patch_size + 1, stride):
            ndvi_full[i:i+patch_size, j:j+patch_size] += patch_predictions[patch_idx]
            weight_matrix[i:i+patch_size, j:j+patch_size] += 1
            patch_idx += 1
    
    ndvi_full /= np.maximum(weight_matrix, 1)
    return ndvi_full

# Function to evaluate model
def test_model(model, test_dataset):
    model.eval()
    device = next(model.parameters()).device
    reconstructed_ndvi_images = []
    
    with torch.no_grad():
        for batch_patches, _ in test_dataset:
            predictions = []
            for patch in batch_patches:
                patch = patch.unsqueeze(0).to(device)
                pred_patch = model(patch).cpu().numpy().squeeze()
                predictions.append(pred_patch)
            
            reconstructed_ndvi = stitch_patches(predictions, image_size=256, patch_size=64, stride=32)
            reconstructed_ndvi_images.append(reconstructed_ndvi)

    return reconstructed_ndvi_images

# Training function
def train_model(dataset_dir, num_epochs=4, batch_size=8, lr=1e-3):
    dataset = NDVIDataset(dataset_dir, patch_size=64, stride=32)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NDVICNN(in_channels=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_patches, batch_targets in train_dataloader:
            batch_loss = 0.0
            for inputs, targets in zip(batch_patches, batch_targets):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            epoch_loss += batch_loss / len(batch_patches)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataset):.6f}")
    
    return model, test_dataset

if __name__ == "__main__":
    dataset_dir = "your_dataset_path"
    trained_model, test_dataset = train_model(dataset_dir, num_epochs=4, batch_size=2, lr=1e-3)
    test_ndvi_images = test_model(trained_model, test_dataset)
