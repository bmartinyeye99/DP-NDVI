import torch
import torch.nn as nn
import torch.optim as optim
from utils import plot_regression
import matplotlib.pyplot as plt
import wandb
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

class Trainer:
    def __init__(self, model, datamodule, lr=1e-3):
        self.model = model
        self.datamodule = datamodule  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.HuberLoss()

        #self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()

        self.train_loss_history = []
        self.val_loss_history = []
        wandb.init(project="ndvi-prediction ", config={"lr": lr})
    
    def train(self, num_epochs=5):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            total_patches = 0
            for batch_patches, batch_targets in self.datamodule.train_dataloader:
                # --- Batching Patches Improvement ---
                # Instead of looping over each image’s patches individually,
                # concatenate all patches from the batch into one large mini-batch.
                # For instance, if batch_patches is a list of tensors each with shape [n_patches, C, H, W],
                # we combine them along dimension 0.
                inputs = batch_patches.flatten(0, 1).to(self.device)
                targets = batch_targets.flatten(0, 1).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if torch.any(outputs > 1) or torch.any(outputs < -1):
                    invalid_values = outputs[(outputs > 1) | (outputs < -1)]
                    raise ValueError(f"Outputs contains invalid values: {invalid_values}")
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                total_patches += inputs.size(0)

            avg_train_loss = total_loss / total_patches
            avg_val_loss = self._compute_validation_loss()
            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_val_loss)
            wandb.log({"epoch": epoch+1, "avg train_loss": avg_train_loss, "avg val_loss": avg_val_loss})
            
            # Log regression plot and loss plot (existing plotting code)
            plot_regression(
                self.model,
                self.datamodule.val_dataloader,
                image_size=self.datamodule.image_size,
                patch_size=self.datamodule.patch_size,
                stride=self.datamodule.stride,
                epoch=epoch
            )
            epochs_list = list(range(1, len(self.train_loss_history)+1))
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(epochs_list, self.train_loss_history, label="Train Loss", marker="o")
            ax.plot(epochs_list, self.val_loss_history, label="Validation Loss", marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss vs. Epoch")
            ax.legend()
            plt.show()
            
            wandb.log({"loss_plot": wandb.Image(fig)})
            plt.close(fig)
            
            gt_ndvi, pred_ndvi = self.validation()
            mse = ((gt_ndvi - pred_ndvi)**2).mean()
            mae = (np.abs(gt_ndvi - pred_ndvi)).mean()
            r2 = r2_score(gt_ndvi, pred_ndvi)
            # For simplicity, using MSE and MAE here.
            wandb.log({"val_mse": mse, "val_mae": mae, "val_r2": r2})
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, R2: {r2}")
            
    def _compute_validation_loss(self):
        self.model.eval()
        total_val_loss = 0.0
        total_val_patches = 0
        with torch.no_grad():
            for batch_patches, batch_targets in self.datamodule.val_dataloader:
                # Flatten batch and patch dimensions: [B, n_patches, ...] -> [B*n_patches, ...]
                inputs = batch_patches.flatten(0, 1).to(self.device)
                targets = batch_targets.flatten(0, 1).to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_val_loss += loss.item() * inputs.size(0)
                total_val_patches += inputs.size(0)
        self.model.train()
        return total_val_loss / total_val_patches

    def validation(self):
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch_patches, batch_targets in self.datamodule.val_dataloader:
                inputs = batch_patches.flatten(0, 1).to(self.device)
                outputs = self.model(inputs).cpu().numpy()
                preds.extend(outputs.flatten())
                targets.extend(batch_targets.flatten(0, 1).cpu().numpy().flatten())
        self.model.train()
        return np.array(targets), np.array(preds)
