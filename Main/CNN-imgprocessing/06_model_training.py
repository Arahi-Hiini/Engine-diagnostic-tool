import os
import sys

# Flush output immediately so you see progress (avoids "nothing returned" when script errors)
print("06_model_training: starting...", flush=True)
sys.stdout.flush()
sys.stderr.flush()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

print("Starting model training...", flush=True)

# Run from script directory so paths like 'processed_data' resolve correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# --- 1. Setup & Data Loading ---
# PyTorch needs the images to be converted into math tensors and normalized
transform = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = 'processed_data'  # The folder you created in the last step
print("Loading images from directory...", flush=True)

if not os.path.isdir(data_dir):
    print(f"ERROR: Directory not found: {os.path.abspath(data_dir)}", flush=True)
    print("Create 'processed_data' with subfolders (e.g. healthy/, faulty/) and run 05_*.py first.", flush=True)
    sys.exit(1)

# ImageFolder automatically labels images based on the folder name (healthy=0, faulty=1)
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split into 80% Training and 20% Validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Classes found: {full_dataset.classes}", flush=True)
print(f"Training on {train_size} images, Validating on {val_size} images.", flush=True)

# --- 2. Transfer Learning (Card 7 & 8) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Load pre-trained ResNet-18
model = models.resnet18(weights='DEFAULT')

# Freeze the early layers so we don't destroy what it already knows
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer (Card 8)
# ResNet-18's final layer normally outputs 1000 classes. We change it to 2 (Healthy/Faulty)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# Define Loss (CrossEntropy) and Optimizer (Adam - only updating the new final layer)
# Force the AI to care 11x more about the rare faulty class
class_weights = torch.tensor([11.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# --- 3. The Training Loop (Card 9) ---
epochs = 20 # Start small for the Proof of Concept

print("\nStarting Training...", flush=True)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    # Train the model
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()      # Clear old calculations
        outputs = model(inputs)    # Make predictions
        loss = criterion(outputs, labels) # Calculate how wrong it was
        loss.backward()            # Learn from mistakes
        optimizer.step()           # Update weights
        
        running_loss += loss.item()
        
    # Test the model on the Validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results for this epoch
    epoch_loss = running_loss / len(train_loader)
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Validation Accuracy: {val_accuracy:.2f}%", flush=True)

# Save the trained model to your hard drive
torch.save(model.state_dict(), 'engine_fault_model.pth')
print("\nTraining complete! Model saved as 'engine_fault_model.pth'.", flush=True)