import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from dataset import NDVIDataset
from model import NDVICNN
from train import Trainer


class Experiment:
    def __init__(self, dataset_dir, image_size=256, patch_size=64, stride=32, 
                 batch_size=8, num_epochs=4, lr=1e-3):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        # Initialize dataset and dataloaders
        self._prepare_data()
        
        # Initialize model and trainer
        self.model = self._create_model()
        self.trainer = Trainer(self.model, self, lr=self.lr)
        
    def _prepare_data(self):
        # Load the full dataset
        dataset = NDVIDataset(self.dataset_dir, self.image_size, self.patch_size, self.stride)
        
        # Split the dataset into training, validation, and test sets
        train_size = int(0.7 * len(dataset))  # 70% for training
        val_size = int(0.15 * len(dataset))   # 15% for validation
        test_size = len(dataset) - train_size - val_size  # Remaining 15% for testing
        
        # Perform the split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create DataLoaders for each set
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

    def _create_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return NDVICNN(in_channels=7).to(device)
    
    def run(self):
        self.trainer.train(self.num_epochs)
        self.evaluate()

    def evaluate(self):
        gt_ndvi, pred_ndvi = self.trainer.validate()
        
        # Plot regression
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=gt_ndvi, y=pred_ndvi, alpha=0.5)
        plt.xlabel("Ground Truth NDVI")
        plt.ylabel("Predicted NDVI")
        plt.title("Regression Plot: NDVI Prediction")
        plt.show()