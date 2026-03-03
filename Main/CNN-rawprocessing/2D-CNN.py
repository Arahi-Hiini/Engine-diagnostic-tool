import os
import scipy.io
import scipy.signal
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# ==========================================
# STEP 1: LOAD & CONVERT TO SPECTROGRAMS
# ==========================================
def process_folder_to_images(folder_path, label):
    all_images = []
    print(f"\n--- Scanning folder: {folder_path} ---")
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            file_path = os.path.join(folder_path, file_name)
            
            # Load raw data
            mat_data = scipy.io.loadmat(file_path)
            key = [k for k in mat_data.keys() if 'DE_time' in k][0]
            signal = mat_data[key].flatten()
            
            # Normalize
            signal = (signal - np.mean(signal)) / np.std(signal)
            
            # Chop into windows of 1024
            window_size = 1024
            num_windows = len(signal) // window_size
            windows = signal[:num_windows * window_size].reshape(num_windows, window_size)
            
            # Convert each 1D window into a 2D Spectrogram Image
            file_images = []
            for w in windows:
                # nperseg=64 creates a nice, compact 33x33 pixel image for our AI
                freqs, times, Sxx = scipy.signal.spectrogram(w, fs=12000, nperseg=64, noverlap=32)
                Sxx_dB = 10 * np.log10(Sxx + 1e-10) # Convert to Decibels
                file_images.append(Sxx_dB)
                
            all_images.extend(file_images)
            print(f" -> {file_name}: Created {num_windows} spectrogram images")
            
    return np.array(all_images)

# Load the images
h_images = process_folder_to_images('healthy', label=0)
f_images = process_folder_to_images('faulty', label=1)

# ==========================================
# STEP 2: BALANCE THE DECK & BUILD PIPELINE
# ==========================================
# Duplicate healthy images if we have fewer of them
if len(h_images) < len(f_images):
    multiplier = len(f_images) // len(h_images)
    h_images = np.tile(h_images, (multiplier, 1, 1))
    print(f"\n⚠️ Duplicated healthy images {multiplier}x to balance the dataset!")

h_labels = np.zeros(len(h_images))
f_labels = np.ones(len(f_images))

X_train = np.concatenate((h_images, f_images), axis=0)
y_train = np.concatenate((h_labels, f_labels), axis=0)

# Reshape for 2D-CNN: (Batch, Channels, Height, Width)
# Our images are 33x33 pixels, Grayscale (1 channel)
X_train = X_train.reshape(-1, 1, 33, 31) 
print(f"\n✅ Final Master Image Matrix shape: {X_train.shape}")

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.long)

train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==========================================
# STEP 3: THE 2D-CNN ARCHITECTURE
# ==========================================
class VisionEngineCNN(nn.Module):
    def __init__(self):
        super(VisionEngineCNN, self).__init__()
        
        # Notice we are using Conv2d and MaxPool2d now!
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate Flatten Size automatically based on a 33x33 image
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 33, 31)
            x = self.pool1(self.relu1(self.conv1(dummy)))
            x = self.pool2(self.relu2(self.conv2(x)))
            self.flattened_size = x.view(1, -1).size(1)
            
        self.fc = nn.Linear(self.flattened_size, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

model = VisionEngineCNN()
print(f"Vision AI Built Successfully! Flatten size: {model.flattened_size}")

# ==========================================
# STEP 4: THE TRAINING LOOP
# ==========================================
print("\n--- Starting the Training Process ---")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10 
for epoch in range(epochs):
    model.train()
    running_loss, correct_guesses, total_guesses = 0.0, 0, 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted_classes = torch.max(outputs, 1)
        correct_guesses += (predicted_classes == y_batch).sum().item()
        total_guesses += y_batch.size(0)
        
    epoch_accuracy = (correct_guesses / total_guesses) * 100
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {epoch_accuracy:.2f}%")

# ==========================================
# STEP 5: SAVE THE NEW "VISION" BRAIN
# ==========================================
print("\n--- Saving the Trained Vision Model ---")
torch.save(model.state_dict(), "spectrogram_cnn_weights.pth")
print("✅ New visual weights saved successfully!")