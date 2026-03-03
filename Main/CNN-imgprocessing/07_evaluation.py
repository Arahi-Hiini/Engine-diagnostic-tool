import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Setup ---
# Must match the exact transformations used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
data_dir = 'processed_data'
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_loader = DataLoader(full_dataset, batch_size=16, shuffle=False)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None) # We don't need the Microsoft weights anymore
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)     # Match our 2-class setup
model.load_state_dict(torch.load('engine_fault_model.pth')) # Load YOUR learned weights
model = model.to(device)
model.eval() # Set model to evaluation mode (turns off learning)

# --- 2. Run Predictions (Card 10) ---
all_preds = []
all_labels = []

print("Running AI on all audio files...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Apply Softmax to get actual percentages (0.0 to 1.0)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Class 0 is 'faulty'. Let's say: "If you are even 30% sure it's faulty, flag it!"
        # (Assuming 'faulty' is at index 0 and 'healthy' is at index 1)
        faulty_probs = probabilities[:, 0]
        
        # Create custom predictions based on our new paranoid threshold
        custom_threshold = 0.30
        preds = (faulty_probs < custom_threshold).long() 
        # If faulty prob is >= 0.30, it gets marked as 0 (faulty). 
        # If faulty prob is < 0.30, it gets marked as 1 (healthy).
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# --- 3. Generate Metrics & Error Analysis (Card 11) ---
class_names = full_dataset.classes # ['faulty', 'healthy']

print("\n--- AI PERFORMANCE REPORT ---")
# The classification report automatically calculates Precision, Recall, and F1-Score
print(classification_report(all_labels, all_preds, target_names=class_names))

# Calculate a single F1-Score for the "faulty" class
# (Assuming 'faulty' is class 0 based on alphabetical folder sorting)
f1 = f1_score(all_labels, all_preds, pos_label=0) 
print(f"Critical Fault Detection F1-Score: {f1:.2f}")

# --- 4. Plot Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Engine Fault Confusion Matrix')
plt.ylabel('Actual True Status')
plt.xlabel('What the AI Predicted')
plt.show()