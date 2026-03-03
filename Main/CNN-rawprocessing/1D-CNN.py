import os
import scipy.io
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==========================================
# PART 1 & 2: MULTI-FILE LOADER & BALANCER
# ==========================================
def process_folder(folder_path, label):
    all_windows = []
    
    print(f"\n--- Scanning folder: {folder_path} ---")
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            file_path = os.path.join(folder_path, file_name)
            
            # Load
            mat_data = scipy.io.loadmat(file_path)
            key = [k for k in mat_data.keys() if 'DE_time' in k][0]
            signal = mat_data[key].flatten()
            
            # Normalize
            signal = (signal - np.mean(signal)) / np.std(signal)
            
            # Window
            window_size = 1024
            num_windows = len(signal) // window_size
            windows = signal[:num_windows * window_size].reshape(num_windows, window_size)
            
            all_windows.append(windows)
            print(f" -> {file_name}: Created {num_windows} windows")
            
    # Smash all the files in this folder into one giant matrix
    return np.vstack(all_windows)

# 1. Load the data
h_windows = process_folder('healthy', label=0)
f_windows = process_folder('faulty', label=1)

# 2. Fix the Class Imbalance
# We have roughly 470 healthy windows and 940 faulty windows.
# We will duplicate the healthy windows so the AI doesn't cheat.
if len(h_windows) < len(f_windows):
    multiplier = len(f_windows) // len(h_windows)
    h_windows = np.tile(h_windows, (multiplier, 1))
    print(f"\n⚠️ Duplicated healthy data {multiplier}x to balance the dataset!")

print(f"\nTotal Healthy Windows: {len(h_windows)}")
print(f"Total Faulty Windows:  {len(f_windows)}")

# 3. Create the Labels and Combine
h_labels = np.zeros(len(h_windows))
f_labels = np.ones(len(f_windows))

X_train = np.concatenate((h_windows, f_windows), axis=0)
y_train = np.concatenate((h_labels, f_labels), axis=0)

# Reshape for 1D-CNN
X_train = X_train.reshape(-1, 1, 1024)
print(f"✅ Final Master X_train shape: {X_train.shape}")

# ==========================================
# PART 3: PYTORCH DATALOADER (THE DEALER)
# ==========================================
print("\n--- Building PyTorch Data Pipeline ---")
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.long)

train_dataset = TensorDataset(X_tensor, y_tensor)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"DataLoader ready with {len(train_loader)} batches of {batch_size}.")

# ==========================================
# PART 4: THE 1D-CNN ARCHITECTURE
# ==========================================
class VibrationCNN(nn.Module):
    def __init__(self):
        super(VibrationCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=16, stride=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate Flatten Size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 1024)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
            self.flattened_size = x.view(1, -1).size(1)
            
        self.fc = nn.Linear(self.flattened_size, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

print("\n--- Building AI Architecture ---")
model = VibrationCNN()
print(f"AI Built Successfully! Automatically calculated flatten size: {model.flattened_size}")

# ==========================================
# PART 5: THE SMOKE TEST
# ==========================================
print("\n--- Running the Smoke Test ---")
# Grab exactly one batch from the DataLoader we built in Part 3
X_batch, y_batch = next(iter(train_loader))

# Pass it through the model
predictions = model(X_batch)

print(f"Input shape:  {X_batch.shape}  -> (Batch_Size, Channels, Length)")
print(f"Output shape: {predictions.shape}    -> (Batch_Size, Num_Classes)")
print(f"Raw Output for first window:\n {predictions[0].detach().numpy()}")

import torch.optim as optim

# ==========================================
# PART 6: THE TRAINING LOOP
# ==========================================
print("\n--- Starting the Training Process ---")

# 1. Define the Grader and the Mechanic
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of times to review the entire deck of flashcards
epochs = 10 

for epoch in range(epochs):
    model.train() # Put the model in training mode
    
    running_loss = 0.0
    correct_guesses = 0
    total_guesses = 0
    
    # Deal the flashcards one batch at a time
    for X_batch, y_batch in train_loader:
        
        # Step A: Clear the mechanic's previous adjustments
        optimizer.zero_grad()
        
        # Step B: Make a prediction (Forward Pass)
        outputs = model(X_batch)
        
        # Step C: Compare prediction to the answer key (Calculate Loss)
        loss = criterion(outputs, y_batch)
        
        # Step D: Figure out exactly what went wrong (Backpropagation)
        loss.backward()
        
        # Step E: Adjust the math filters to be smarter (Optimizer Step)
        optimizer.step()
        
        # --- Track our progress ---
        running_loss += loss.item()
        
        # Figure out if the model guessed 0 or 1 by finding the highest probability
        _, predicted_classes = torch.max(outputs, 1)
        
        # Tally up the correct answers
        correct_guesses += (predicted_classes == y_batch).sum().item()
        total_guesses += y_batch.size(0)
        
    # Calculate the average score for this Epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = (correct_guesses / total_guesses) * 100
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

print("\n✅ Training Complete! The AI has learned the fault signatures.")

# ==========================================
# PART 7: SAVING THE MODEL
# ==========================================
print("\n--- Saving the Trained Model ---")

# We save the 'state_dict', which is a dictionary containing all the 
# optimized weights and biases the AI just learned.
save_path = "bearing_cnn_weights.pth"
torch.save(model.state_dict(), save_path)

print(f"✅ Model successfully saved to: {save_path}")