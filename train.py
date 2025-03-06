import torch
import torch.nn as nn
import torch.optim as optim
from utils import plot_regression
import matplotlib.pyplot as plt
import wandb
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
class Trainer:
    def __init__(self, model, datamodule, lr=1e-3):
        self.model = model
        self.datamodule = datamodule  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_loss_history = []
        self.val_loss_history = []
        wandb.init(project="ndvi-prediction", config={"lr": lr})
    
    def train(self, num_epochs=5):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            total_patches = 0  # Count patches
            for batch_patches, batch_targets in self.datamodule.train_dataloader:
                    # batch_patches and batch_targets are lists of patches for each image in the batch
                for inputs, targets in zip(batch_patches, batch_targets):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    total_patches += 1

            avg_train_loss = total_loss / total_patches

            avg_val_loss = self._compute_validation_loss()
            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_val_loss)

            current_epoch_count = len(self.train_loss_history)
            epochs = list(range(1, current_epoch_count + 1))

            #     for inputs, targets in zip(batch_patches, batch_targets):
            #         inputs, targets = inputs.to(self.device), targets.to(self.device)
            #         self.optimizer.zero_grad()
            #         outputs = self.model(inputs)
            #         loss = self.criterion(outputs, targets)
            #         loss.backward()
            #         self.optimizer.step()
            #         total_loss += loss.item()
            
            # avg_train_loss = total_loss / len(self.datamodule.train_dataset)
            # avg_val_loss = self._compute_validation_loss()
            
            # # Append history
            # self.train_loss_history.append(avg_train_loss)
            # self.val_loss_history.append(avg_val_loss)
            
            # Log scalar metrics to wandb
            wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
            
            # Log regression plot
            plot_regression(
                self.model,
                self.datamodule.val_dataloader,
                image_size=256,
                patch_size=64,
                stride=32,
                epoch=epoch
            )
            
            # Create and log a loss plot for the current epoch
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(epochs, self.train_loss_history, label="Train Loss", marker="o")
            ax.plot(epochs, self.val_loss_history, label="Validation Loss", marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss vs. Epoch")
            ax.legend()
            plt.show()
            wandb.log({"loss_plot": wandb.Image(fig)})
            plt.close(fig)
            
            gt_ndvi, pred_ndvi = self.validation()  # Get ground truth and predictions from validation data
        
            mse = mean_squared_error(gt_ndvi,pred_ndvi)
            mae = mean_absolute_error(gt_ndvi,pred_ndvi)
            # Calculate R-squared: handle case where variance is zero
            variance = np.mean((gt_ndvi - np.mean(gt_ndvi)) ** 2)
            r2 = 1 - (np.sum((gt_ndvi - pred_ndvi) ** 2) / variance) if variance != 0 else float('nan')
            
            # Log additional metrics to wandb
            wandb.log({"val_mse": mse, "val_mae": mae, "val_r2": r2})
            
            # Print metrics to console
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Validation Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    def _compute_validation_loss(self):
        self.model.eval()
        total_val_loss = 0.0
        total_val_patches = 0

        with torch.no_grad():
            for batch_patches, batch_targets in self.datamodule.val_dataloader:
                for inputs, targets in zip(batch_patches, batch_targets):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    total_val_loss += loss.item()
                    total_val_patches += 1

        avg_val_loss = total_val_loss / total_val_patches
        return avg_val_loss
        # self.model.eval()
        # total_val_loss = 0
        # with torch.no_grad():
        #     for batch_patches, batch_targets in self.datamodule.val_dataloader:
        #         for inputs, targets in zip(batch_patches, batch_targets):
        #             inputs, targets = inputs.to(self.device), targets.to(self.device)
        #             outputs = self.model(inputs)
        #             loss = self.criterion(outputs, targets)
        #             total_val_loss += loss.item()
        # avg_val_loss = total_val_loss / len(self.datamodule.val_dataset)
        # return avg_val_loss
    
    def validation(self):
        """
        Evaluate the model on the validation set and return ground truth NDVI and predicted NDVI values.
        """
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch_patches, batch_targets in self.datamodule.val_dataloader:
                for inputs, targets_batch in zip(batch_patches, batch_targets):
                    inputs = inputs.to(self.device)
                    pred = self.model(inputs).cpu().numpy().flatten()
                    preds.extend(pred)
                    targets.extend(targets_batch.numpy().flatten())
        return np.array(targets), np.array(preds)