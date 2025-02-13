import datasets
import numpy
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.nn import Sequential, MSELoss
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from typing import Any

# Load dataset from Hugging Face
dataset: datasets.arrow_dataset.Dataset = load_dataset("hangyeol522/anomaly-detection-model", split="train")

# Convert dataset to DataFrame
df: DataFrame = pd.DataFrame(dataset)

# Select numerical features
numerical_columns: pandas.core.indexes.base.Index = df.select_dtypes(include=[np.number]).columns

df: DataFrame = df[numerical_columns]
print(f"before normalizing {df.head(3)}")
# Normalize data
scaler: StandardScaler = StandardScaler()
df[numerical_columns]: pandas.core.frame.DataFrame = scaler.fit_transform(df[numerical_columns])
print(f"after normalizing {df.head(3)}")
# Convert to PyTorch tensors
data_tensor: torch.Tensor = torch.tensor(df.values, dtype=torch.float32)


# Custom PyTorch Dataset
class NetworkDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Create DataLoader
batch_size: int = 256
data_loader: DataLoader = DataLoader(NetworkDataset(data_tensor), batch_size=batch_size, shuffle=True)


# Define the Masked Autoencoder
class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, mask_ratio=0.3):
        # Randomly mask input features
        from torch import Tensor
        mask: Tensor = torch.rand_like(x) > mask_ratio
        masked_x = x * mask.float()

        # Encode & Decode
        encoded: Sequential = self.encoder(masked_x)
        reconstructed: Sequential = self.decoder(encoded)

        return reconstructed, mask


# Initialize model
input_dim: int = data_tensor.shape[1]
model: MaskedAutoencoder = MaskedAutoencoder(input_dim)
criterion: MSELoss = nn.MSELoss()
optimizer: torch.optim.Adam = optim.Adam(model.parameters(), lr=0.001)

# Train Model
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        reconstructed, mask = model(batch)
        loss = criterion(reconstructed * mask, batch * mask)  # Compute loss only on unmasked data
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Detect Anomalies (Reconstruction Error)
with torch.no_grad():
    reconstructed_data, _ = model(data_tensor)
    anomaly_scores: numpy.ndarray = torch.mean((data_tensor - reconstructed_data) ** 2, dim=1).numpy()

# Plot anomaly scores
plt.hist(anomaly_scores, bins=50)
plt.title("Anomaly Score Distribution")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.show()

# Define anomaly threshold
threshold: numpy.float64 = np.percentile(anomaly_scores, 95)  # Top 5% are anomalies
print(f"the anomaly threshold: {threshold}")
anomalies: np.ndarray[Any, np.dtype[np.bool_]] = anomaly_scores > threshold
print(f"Detected {sum(anomalies)} anomalies out of {len(anomalies)} samples.")
