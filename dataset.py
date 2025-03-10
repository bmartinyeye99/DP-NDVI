import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from utils import *
from torchvision import transforms
import torchvision.transforms.functional as TF


def add_gaussian_noise(image, mean=0, std=0.1):
    """
    Add Gaussian noise to an image.
    
    Args:
        image: Input image (numpy array).
        mean: Mean of the Gaussian distribution.
        std: Standard deviation of the Gaussian distribution.
    
    Returns:
        Noisy image.
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1) 


# Dataset with Overlapping Patches
class NDVIDataset(Dataset):
    def __init__(self, dataset_dir, image_size=256, patch_size=64, stride=32,augment=True, noise_std=0.0):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.noise_std = noise_std  # Standard deviation for Gaussian noise
        self.filenames = os.listdir(dataset_dir)
        self.rgb_files = sorted([f for f in self.filenames if f.lower().endswith('.jpg')])
        self.nir_files = sorted([f for f in self.filenames if f.lower().endswith('.tif')])
        # Ensure that RGB and NIR files are paired correctly
        if len(self.rgb_files) != len(self.nir_files):
            raise ValueError (f"Mismatch between the number of RGB and NIR files. RGB {len(self.rgb_files)}  NIR  {len(self.nir_files)} {dataset_dir}")

        #self.nir_files = [f.replace('.JPG', '.TIF') for f in self.rgb_files]
            # Define augmentation transforms
        self.augment_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                   # transforms.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.1),
                ])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Load and preprocess images
        rgb_path = os.path.join(self.dataset_dir, self.rgb_files[idx])
        nir_path = os.path.join(self.dataset_dir, self.nir_files[idx])

        print(f"RGB file : {rgb_path} NIR file :{nir_path}")

        if rgb_path.split('_')[0] != nir_path.split('_')[0]:
            raise ValueError (f"Mismatch between scene images RGB {len(rgb_path)}  NIR  {len(nir_path)}")

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        nir_img = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        if rgb_bgr is None or nir_img is None:
            raise ValueError(f"Failed to load images for index {idx}")
        
        # Resize to fixed size (256x256)
        rgb_bgr = cv2.resize(rgb_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        nir_img = cv2.resize(nir_img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # Loaded normalised RGB image
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        nir = nir_img.astype(np.float32) / 255.0

        # Add Gaussian noise
        if self.noise_std > 0:
            rgb = add_gaussian_noise(rgb, std=self.noise_std)
            nir = add_gaussian_noise(nir, std=self.noise_std)

        RGBindices = compute_rgb_indices(rgb)
        red = rgb[..., 0]
        gt_ndvi = compute_ndvi(nir, red) # ground truth ndvi matrix
        for row in gt_ndvi:
                for element in row:
                    if element > 1 or element < -1:
                        raise ValueError(f"Matrix contains an invalid value: {element}")


        # Create input image of 8 chanels. RGB, NIR + NDVI + RGB indices
        input_img = np.concatenate([
        rgb,
        # nir[..., None],
        RGBindices['NGRDI'][..., None],
        RGBindices['MGRVI'][..., None],
        RGBindices['VARI'][..., None],
        RGBindices['RGBVI'][..., None],
        RGBindices['TGI'][..., None]


        ], axis=-1)


    #  Patching:  sliding a 64x64 window over the 256x256 image with a stride of 32. 
    #  Each window extracts a patch for training.
    #  Stitching: After predicting NDVI for each patch, imagine placing the patches back
    #  into their original positions and averaging overlapping regions to create a smooth full-size image.

    #  The dataset extracts overlapping patches from the full image using a sliding window approach. For example:
    #  Patch Size: 64x64
    # Stride: 32 (overlap of 32 pixels between patches)
    # Each patch is a 7-channel input (RGB + NIR + NGRDI + VARI + GLI), and the corresponding ground truth is the NDVI patch.

    # The overlapping regions (e.g., pixels) are averaged during stitching.

        input_img_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float()  
        gt_ndvi_tensor = torch.from_numpy(gt_ndvi).unsqueeze(0).float()              

        # Apply augmentation to both input and ground truth
        if self.augment:
            if np.random.rand() > 0.5:
                input_img_tensor = TF.hflip(input_img_tensor)
                gt_ndvi_tensor = TF.hflip(gt_ndvi_tensor)
            if np.random.rand() > 0.5:
                input_img_tensor = TF.vflip(input_img_tensor)
                gt_ndvi_tensor = TF.vflip(gt_ndvi_tensor)
            angle = np.random.uniform(-30, 30)
            input_img_tensor = TF.rotate(input_img_tensor, angle)
            gt_ndvi_tensor = TF.rotate(gt_ndvi_tensor, angle)


        input_img_aug = input_img_tensor.numpy().transpose(1, 2, 0)  # Shape: [H, W, 7]
        gt_ndvi_aug = gt_ndvi_tensor.numpy()[0, :, :]                   # Shape: [H, W]



        # Extract patches from the augmented images
        patches = []
        gt_patches = []
        for i in range(0, self.image_size - self.patch_size + 1, self.stride):
            for j in range(0, self.image_size - self.patch_size + 1, self.stride):
                patch = input_img_aug[i:i+self.patch_size, j:j+self.patch_size]
                gt_patch = gt_ndvi_aug[i:i+self.patch_size, j:j+self.patch_size]
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float()
                gt_tensor = torch.from_numpy(gt_patch).unsqueeze(0).float()
                patches.append(patch_tensor)
                gt_patches.append(gt_tensor)

        return torch.stack(patches), torch.stack(gt_patches)


    
class DataModule:
    def __init__(self, dataset_dir, image_size=256, patch_size=64, stride=32, batch_size=8, noise_std=0.0):
        # self.dataset_dir = dataset_dir
        # self.image_size = image_size
        # self.patch_size = patch_size
        # self.stride = stride
        self.batch_size = batch_size
        # self.noise_std = noise_std

        # Load the full dataset
        self.dataset = NDVIDataset(
            dataset_dir, 
            image_size, 
            patch_size, 
            stride, 
            augment=True, 
            noise_std=noise_std
        )
        
        # Split the dataset into training, validation, and test sets
        train_size = int(0.7 * len(self.dataset))  # 70% for training
        val_size = int(0.15 * len(self.dataset))   # 15% for validation
        test_size = len(self.dataset) - train_size - val_size  # Remaining 15% for testing
        
        # Perform the split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        
        # Create DataLoaders for each set
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)