import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from utils import *

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
        
        # Resize to fixed size (256x256)
        rgb_bgr = cv2.resize(rgb_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        nir_img = cv2.resize(nir_img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        nir = nir_img.astype(np.float32) / 255.0
        indices = compute_rgb_indices(rgb)
        red = rgb[..., 0]
        gt_ndvi = compute_ndvi(nir, red)

        input_img = np.concatenate([rgb, nir[..., None], indices['NGRDI'][..., None],
                                    indices['VARI'][..., None], indices['GLI'][..., None]], axis=-1)

    #  Patching:  sliding a 64x64 window over the 256x256 image with a stride of 32. 
    #  Each window extracts a patch for training.
    #  Stitching: After predicting NDVI for each patch, imagine placing the patches back
    #  into their original positions and averaging overlapping regions to create a smooth full-size image.

    #  The dataset extracts overlapping patches from the full image using a sliding window approach. For example:
    #  Patch Size: 64x64
    # Stride: 32 (overlap of 32 pixels between patches)
    # Eachpatch is a 7-channel input (RGB + NIR + NGRDI + VARI + GLI), and the corresponding ground truth is the NDVI patch.

    # The overlapping regions (e.g., pixels) are averaged during stitching.

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

class DataModule:
    def __init__(self, dataset_dir, batch_size=8):
        self.dataset = NDVIDataset(dataset_dir)
        train_size = int(0.7 * len(self.dataset))  # 70% for training
        val_size = int(0.15 * len(self.dataset))   # 15% for validation
        test_size = len(self.dataset) - train_size - val_size  # 15% for testing
        
        # Perform the split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        
        # Create DataLoaders for each set
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)