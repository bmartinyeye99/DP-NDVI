import os
import cv2
import numpy as np
import torch
import torch.nn 
import torch.nn.functional 
from torch.utils.data import Dataset, DataLoader, Subset
from utils import *
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import albumentations

class NDVIDataset(Dataset):
    def __init__(self, dataset_dir, image_size=256, patch_size=64, stride=32, augment=True, noise_std=0.0):
        assert (image_size - patch_size) % stride == 0, \
            f"image_size={image_size}, patch_size={patch_size}, stride={stride} are incompatible."
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.noise_std = noise_std
        self.filenames = os.listdir(dataset_dir)
        self.rgb_files = sorted([f for f in self.filenames if f.lower().endswith('.jpg')])
        self.nir_files = sorted([f for f in self.filenames if f.lower().endswith('.tif')])
        if len(self.rgb_files) != len(self.nir_files):
            raise ValueError(f"Mismatch between the number of RGB and NIR files in {dataset_dir}")
        

        if self.augment:
            # Define an augmentation pipeline using albumentations.
            # The additional_targets parameter ensures that 'nir' gets the same transforms as 'image'.
            self.aug_pipeline = albumentations.Compose([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.Rotate(limit=30, p=0.5),
                albumentations.GaussianBlur(blur_limit=(3, 7), p=0.3),
                #albumentations.RandomScale(scale_limit=0.3, p=0.5),
                # # RandomCrop ensures the output size remains image_size x image_size.
                albumentations.RandomCrop(height=self.image_size/3, width=self.image_size/3, p=0.5),
            ], additional_targets={'nir': 'image'})

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.dataset_dir, self.rgb_files[idx])
        nir_path = os.path.join(self.dataset_dir, self.nir_files[idx])
        if rgb_path.split('_')[0] != nir_path.split('_')[0]:
            raise ValueError("Mismatch between paired RGB and NIR files.")

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        if rgb_bgr is None or nir_img is None:
            raise ValueError(f"Failed to load images for index {idx}")

        rgb_bgr = cv2.resize(rgb_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        nir_img = cv2.resize(nir_img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        
        # Convert images to [0, 1]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        nir = nir_img.astype(np.float32) / 255.0

        # --- Augmentation: Apply the same augmentation to both RGB and NIR ---
        if self.augment:
            # Albumentations expects uint8 images in [0,255].
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            nir_uint8 = (nir * 255).astype(np.uint8)
            
            # Apply the augmentation pipeline to both images.
            augmented = self.aug_pipeline(image=rgb_uint8, nir=nir_uint8)
            rgb = augmented["image"].astype(np.float32) / 255.0
            nir = augmented["nir"].astype(np.float32) / 255.0

        # --- Now, compute the indices from the (augmented) reflectance values (in [0,1]) ---
        RGBindices = compute_rgb_indices(rgb)
        red = rgb[..., 0]
        gt_ndvi = compute_ndvi(nir, red)
        
        # Validate NDVI values (optional)
        for row in gt_ndvi:
            for element in row:
                if element > 1 or element < -1:
                    raise ValueError(f"NDVI matrix contains an invalid value: {element}")

        # --- Now normalize the raw images to [-1, 1] for input to the network ---
        rgb_norm = rgb * 2 - 1
        nir_norm = nir * 2 - 1

        # --- Reconstruct input image ---
        # Combine normalized RGB with the computed indices.
        input_img = np.concatenate([
            rgb_norm,
            RGBindices['NGRDI'][..., None],
            RGBindices['MGRVI'][..., None],
            RGBindices['VARI'][..., None],
            RGBindices['RGBVI'][..., None],
            RGBindices['TGI'][..., None]
        ], axis=-1)

        # Convert to tensors (channels-first)
        input_img_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float()
        gt_ndvi_tensor = torch.from_numpy(gt_ndvi).unsqueeze(0).float()

        # --- Patching (using sliding window) ---
        patches = []
        gt_patches = []
        for i in range(0, self.image_size - self.patch_size + 1, self.stride):
            for j in range(0, self.image_size - self.patch_size + 1, self.stride):
                patch = input_img_tensor[:, i:i+self.patch_size, j:j+self.patch_size]
                gt_patch = gt_ndvi_tensor[:, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
                gt_patches.append(gt_patch)

        return torch.stack(patches), torch.stack(gt_patches)

class DataModule:
    def __init__(self, dataset_dir, image_size=256, patch_size=64, stride=32, batch_size=8, noise_std=0.0):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.noise_std = noise_std

        base_dataset = NDVIDataset(dataset_dir, image_size, patch_size, stride, augment=False, noise_std=noise_std)
        all_indices = np.arange(len(base_dataset))
        np.random.shuffle(all_indices)
        train_size = int(0.8 * len(base_dataset))
        val_size = int(0.2 * len(base_dataset))
        #test_size = len(base_dataset) - train_size - val_size
        
        train_indices = all_indices[:train_size].tolist()
        val_indices = all_indices[train_size:train_size+val_size].tolist()
        #test_indices = all_indices[train_size+val_size:].tolist()

        train_dataset_full = NDVIDataset(dataset_dir, image_size, patch_size, stride, augment=True, noise_std=noise_std)
        val_dataset_full = NDVIDataset(dataset_dir, image_size, patch_size, stride, augment=False, noise_std=noise_std)
        #test_dataset_full = NDVIDataset(dataset_dir, image_size, patch_size, stride, augment=False, noise_std=noise_std)

        self.train_dataset = Subset(train_dataset_full, train_indices)
        self.val_dataset = Subset(val_dataset_full, val_indices)
        #self.test_dataset = Subset(test_dataset_full, test_indices)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        #self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)


