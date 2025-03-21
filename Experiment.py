import torch
import matplotlib.pyplot as plt
import seaborn as sns
from model import NDVICNN
from train import Trainer
import wandb
from dataset import DataModule

class Experiment:
    def __init__(self, dataset_dir, image_size=256, patch_size=64, stride=32, 
                 batch_size=16, num_epochs=4, lr=1e-3,noise_std=0.0):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.noise_std = noise_std  # Standard deviation for Gaussian noise

        # Initialize DataModule - handle dataset loading, splitting, and dataloader creation.
        self.datamodule = DataModule(
            dataset_dir=dataset_dir,
            image_size=image_size,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            noise_std=noise_std
        )
        
        # Initialize model and trainer
        self.model = self._create_model()
        self.trainer = Trainer(self.model, self.datamodule, lr=self.lr)
        
    def _create_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return NDVICNN(in_channels=8).to(device)
    
    def run(self):
        self.trainer.train(self.num_epochs)
        self.evaluate()

    # evaluate method uses the test data to compute final metrics
    # The test data is used to evaluate the final model's performance on unseen data after training is complete.
    # It provides an unbiased estimate of the model's generalization ability.

    def evaluate(self):
        # uses the results from the validate method to visualize the model's performance
        gt_ndvi, pred_ndvi = self.trainer.validation()  # ground truth (gt_ndvi) and predicted NDVI values (pred_ndvi) from validation data.
        
        # Plot regression
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=gt_ndvi, y=pred_ndvi, alpha=0.5)
        plt.xlabel("Ground Truth NDVI")
        plt.ylabel("Predicted NDVI")
        plt.title("Regression Plot: NDVI Prediction")

            # Log plot to wandb
        wandb.log({"Regression Plot": wandb.Image(plt)})
        plt.show()