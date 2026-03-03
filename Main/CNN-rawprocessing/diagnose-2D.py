import torch
import torch.nn as nn
import scipy.io
import scipy.signal
import numpy as np
import os

# ==========================================
# 1. REBUILD THE 2D VISION ARCHITECTURE
# ==========================================
class VisionEngineCNN(nn.Module):
    def __init__(self):
        super(VisionEngineCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
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

# ==========================================
# 2. LOAD THE TRAINED VISION WEIGHTS
# ==========================================
print("--- Initializing Vision Diagnostic System ---")
model = VisionEngineCNN()

# Load the new weights we just created
model.load_state_dict(torch.load("spectrogram_cnn_weights.pth"))
model.eval() 
print("✅ Visual weights loaded successfully.")

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
        
        # B. Normalize
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        # C. Chop into windows
        window_size = 1024
        num_windows = len(signal) // window_size
        clean_signal = signal[:num_windows * window_size]
        windows = clean_signal.reshape(num_windows, window_size)
        
        # D. Convert to Spectrogram Images on the fly!
        live_images = []
        for w in windows:
            freqs, times, Sxx = scipy.signal.spectrogram(w, fs=12000, nperseg=64, noverlap=32)
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)
            live_images.append(Sxx_dB)
            
        # E. Reshape to match the AI's format: (Batch, Channels, Height, Width)
        X_matrix = np.array(live_images).reshape(-1, 1, 33, 31)
        X_tensor = torch.tensor(X_matrix, dtype=torch.float32)
        
        # F. Make the Predictions
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predictions = torch.max(outputs, 1)
            
        # G. Tally the votes
        healthy_votes = (predictions == 0).sum().item()
        faulty_votes = (predictions == 1).sum().item()
        
        print(f"Total Windows Scanned: {num_windows}")
        print(f" -> Healthy Signatures Found: {healthy_votes}")
        print(f" -> Faulty Signatures Found:  {faulty_votes}")
        
        # H. Final Verdict
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
# 4. RUN THE ULTIMATE TEST!
# ==========================================
# Point this directly at your unseen 0.014" fault file in your testing folder
test_file = os.path.join('testing_data', 'IR014_1_175.mat') 
diagnose_engine(test_file)