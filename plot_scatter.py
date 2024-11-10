import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
from pathlib import Path
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import seaborn as sns
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

def transform(img, epsilon=1e-8):
    # Convert the input image (Pandas DataFrame) to a torch tensor
    img_tensor = torch.tensor(img.values, dtype=torch.float32)
    
    # Add batch and channel dimensions for the CNN
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, H, W]
    
    # Resize the image to the required size (5000, 20)
    img_resized = F.interpolate(img_tensor, size=(5000, 20), mode='bilinear', align_corners=False)
    
    # Normalize each column (feature) individually
    for i in range(img_resized.shape[2]):  # Iterate through the columns (features)
        column = img_resized[:, :, i, :]  # Extract each column
        mean = column.mean()
        std = column.std()
        
        # Prevent division by zero
        std = std + epsilon
        
        # Normalize the column (feature)
        img_resized[:, :, i, :] = (column - mean) / std
    
    return img_resized

from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
artists = ['Asha_Bhosle', 'Michael_Jackson', 'Kishore_Kumar', 'Indian_National_Anthem', 'Marathi_Bhavgeet', 'Marathi_Lavani']
num_classes = len(artists)

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int = 1, output_shape: int = num_classes):
        super(SimpleCNN, self).__init__()
        
        # Define more convolutional layers
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.3)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.3)
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.3)
        )
        
        # Calculate the output size after convolutional layers
        self._dummy_input = torch.zeros(1, input_channels, 20, 5000)
        self._conv_out_size = self._get_conv_output(self._dummy_input)
        print("Calculated output size after conv layers:", self._conv_out_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._conv_out_size, 512)
        self.fc2 = nn.Linear(512, output_shape)

    def _get_conv_output(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        return int(torch.prod(torch.tensor(x.size()[1:])))  # Flatten the dimensions

    def forward(self, x, extract_features=False):
        # Pass through convolutional blocks
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        
        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)
        
        # If we need to extract features, return after the first fully connected layer
        if extract_features:
            features = self.fc1(x)
            return features
        
        # Otherwise, continue to the final fully connected layer and return the output
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
X = []
for i in tqdm(range(1, 117)):
    song = pd.read_csv(f"MFCC_T/{i:02d}-MFCC.csv")
    X.append(transform(song))

X1 = torch.stack(X)
X1 = X1.squeeze(1)
X1.shape

model = SimpleCNN(input_channels=1, output_shape=num_classes)
model.load_state_dict(torch.load("models/model_final.pth"))
model.eval()

with torch.no_grad():
    pred_logits = model(X1)
    pred_probs = F.softmax(pred_logits, dim=1)
    pred_labels = torch.argmax(pred_probs, dim=1)

artist_short = ['AB', 'MJ', 'KK', 'INA', 'MB', 'ML']

# Print the header with artist names and aligned columns
header = f"{'Song Number':<12} | " + " | ".join([f"Prob {artist:<4}" for artist in artist_short]) + " | Predicted Artist"
print(header)
print("-" * len(header))  # Separator line

# Print each song's probabilities and predicted artist
for i in range(1, 117):
    # Format probabilities with 2 decimal places and percent sign
    probabilities = " | ".join([f"{pred_probs[i-1][j] * 100:8.2f}%" for j in range(6)])  
    predicted_artist = "    "+artist_short[pred_labels[i-1]]
    
    # Print the row with aligned columns
    print(f"{i:<12} | {probabilities} | {predicted_artist}")

songs_by_artist = {artist: [] for artist in artist_short}

for i in range(1, 117):
    predicted_artist = artist_short[pred_labels[i - 1]]
    songs_by_artist[predicted_artist].append(i)

for artist, songs in songs_by_artist.items():
    print(f"{artist}: {songs}")

# Reshape X1 to be 2D if necessary
# Assuming X1 has a shape like (n_samples, 1, n_features) or similar

X2 = X1.view(X1.size(0), -1)  # Flatten to (n_samples, n_features)

# Check shape to ensure it's 2D now
print("X1 shape after reshaping:", X2.shape)

from sklearn.manifold import TSNE
import matplotlib.patches as mpatches

# Proceed with t-SNE on the reshaped 2D array
tsne = TSNE(n_components=3, random_state=42)
X1_3d = tsne.fit_transform(X2.cpu().numpy())  # Convert to numpy for t-SNE

# Plotting (same as before)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

artist_short = ['AB', 'MJ', 'KK', 'INA', 'MB', 'ML']
unique_labels = torch.unique(pred_labels)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

# Plot each predicted artist with a unique color
for idx, label in enumerate(unique_labels):
    points = X1_3d[pred_labels.cpu() == label]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors[idx], label=artist_short[label], s=20, alpha=0.7)

# Add legend for artists
legend_handles = [mpatches.Patch(color=colors[i], label=artist_short[unique_labels[i]]) for i in range(len(unique_labels))]
plt.legend(handles=legend_handles, title="Predicted Artist", loc="upper right")

ax.set_title("3D t-SNE of Song Predictions by Predicted Artist")
plt.show()

