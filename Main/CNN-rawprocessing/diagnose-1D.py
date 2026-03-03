import torch
import torch.nn as nn
import scipy.io
import numpy as np
import os

# ==========================================
# 1. REBUILD THE EMPTY ARCHITECTURE
# PyTorch needs the blueprint to know where to put the weights
# ==========================================
class VibrationCNN(nn.Module):
    def __init__(self):
        super(VibrationCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=16, stride=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 1024)
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

# ==========================================
# 2. LOAD THE TRAINED "BRAIN"
# ==========================================
print("--- Initializing Diagnostic System ---")
model = VibrationCNN()
model.load_state_dict(torch.load("bearing_cnn_weights.pth"))

# CRITICAL: Put the model in 'Evaluation Mode'
# This turns off training-specific behaviors so it runs faster and more predictably
model.eval() 
print("✅ Trained weights loaded successfully.")

# ==========================================
# 3. THE DIAGNOSTIC FUNCTION
# ==========================================
def diagnose_engine(file_path):
    print(f"\n--- Analyzing File: {os.path.basename(file_path)} ---")
    
    try:
        # A. Load the raw data
        mat_data = scipy.io.loadmat(file_path)
        key = [k for k in mat_data.keys() if 'DE_time' in k][0]
        signal = mat_data[key].flatten()
        
        # --- THE FIX: NORMALIZATION ---
        # Center the data around 0 and scale the amplitude
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        # B. Chop into windows
        window_size = 1024
        num_windows = len(signal) // window_size
        clean_signal = signal[:num_windows * window_size]
        windows = clean_signal.reshape(num_windows, window_size)
        
        # C. Convert to PyTorch Tensor: Shape (Batch, 1 Channel, 1024 Length)
        # We use .unsqueeze(1) to add that middle channel dimension instantly
        X_tensor = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)
        
        # D. Make the Predictions!
        # torch.no_grad() tells the GPU "Do not calculate gradients, we aren't training!"
        # This saves massive amounts of memory.
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predictions = torch.max(outputs, 1)
            
        # E. Tally the votes
        healthy_votes = (predictions == 0).sum().item()
        faulty_votes = (predictions == 1).sum().item()
        
        print(f"Total Windows Scanned: {num_windows}")
        print(f" -> Healthy Signatures Found: {healthy_votes}")
        print(f" -> Faulty Signatures Found:  {faulty_votes}")
        
        # F. Final Verdict
        print("\n==================================")
        if faulty_votes > healthy_votes:
            confidence = (faulty_votes / num_windows) * 100
            print(f"🚨 DIAGNOSIS: FAULTY BEARING DETECTED 🚨")
            print(f"Confidence: {confidence:.1f}%")
        else:
            confidence = (healthy_votes / num_windows) * 100
            print(f"✅ DIAGNOSIS: ENGINE IS HEALTHY ✅")
            print(f"Confidence: {confidence:.1f}%")
        print("==================================\n")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file at {file_path}")

# ==========================================
# 4. RUN THE TEST!
# ==========================================
# Grab a brand new file from your dataset that the AI has NEVER seen before!
# For example, try a healthy file from a different horsepower load, 
# or a different Inner Race fault file.

test_file = os.path.join('testing_data', 'IR014_1_175.mat') # Swap this out!
diagnose_engine(test_file)