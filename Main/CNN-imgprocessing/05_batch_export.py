import os
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- 1. Core Functions ---
def standardize_audio(file_path, target_sr=16000, duration=10):
    audio, sr = librosa.load(file_path, sr=target_sr)
    target_samples = target_sr * duration
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), 'constant')
    return audio, sr

def create_melspectrogram(audio, sr):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=128, hop_length=512, n_fft=2048
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

def save_spectrogram_image(mel_db, out_path):
    """Saves the 2D array as a pure image file with no axes or borders."""
    # origin='lower' ensures the low frequencies stay at the bottom of the image
    plt.imsave(out_path, mel_db, cmap='magma', origin='lower')

# --- 2. Setup Directories ---
# This creates the folders where the images will be saved
output_base_dir = 'processed_data'
healthy_dir = os.path.join(output_base_dir, 'healthy')
faulty_dir = os.path.join(output_base_dir, 'faulty')

os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(faulty_dir, exist_ok=True)

# --- 3. Batch Process ---
def process_dataset(csv_file):
    print("Loading CSV and mapping files...")
    df = pd.read_csv(csv_file)
    
    # tqdm creates a nice progress bar so you know it hasn't frozen
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Audio"):
        file_path = row['file_name']
        
        # Check if the file actually exists before processing
        if not os.path.exists(file_path):
            continue
            
        # Determine if it's healthy or faulty based on the file name
        if 'anomaly' in file_path:
            target_folder = faulty_dir
        else:
            target_folder = healthy_dir
            
        # Create the new image file name (change .wav to .png)
        # e.g., section_00_source_test_anomaly_0000.wav -> section_00_source_test_anomaly_0000.png
        base_name = os.path.basename(file_path)
        image_name = base_name.replace('.wav', '.png')
        out_path = os.path.join(target_folder, image_name)
        
        # Process and Save
        audio, sr = standardize_audio(file_path)
        mel_db = create_melspectrogram(audio, sr)
        save_spectrogram_image(mel_db, out_path)

if __name__ == '__main__':
    # Ensure this points to your downloaded CSV
    process_dataset('attributes_00.csv')
    print("\nBatch processing complete! Images are saved in the 'processed_data' folder.")