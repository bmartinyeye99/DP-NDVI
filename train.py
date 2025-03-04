import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import plot_regression

class Trainer:
    def __init__(self, model, datamodule, lr=1e-3):
        self.model = model
        self.datamodule = datamodule
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, num_epochs=5):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_patches, batch_targets in self.datamodule.train_dataloader:
                for inputs, targets in zip(batch_patches, batch_targets):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(self.datamodule.train_dataset)}")
            plot_regression(
                self.model,
                self.datamodule.val_dataloader,
                image_size=256,
                patch_size=64,
                stride=32,
                epoch=epoch
            )

    def validate(self):
        """
        Vyhodnotí model na validačnej množine a vráti ground truth NDVI a predikované NDVI hodnoty.
        """
        self.model.eval()
        preds, targets = [], []
        device = self.device

        with torch.no_grad():
            for batch_patches, batch_targets in self.datamodule.val_dataloader:
                for inputs, targets_batch in zip(batch_patches, batch_targets):
                    inputs = inputs.to(device)
                    pred = self.model(inputs).cpu().numpy().flatten()
                    preds.extend(pred)
                    targets.extend(targets_batch.numpy().flatten())

        return np.array(targets), np.array(preds)
