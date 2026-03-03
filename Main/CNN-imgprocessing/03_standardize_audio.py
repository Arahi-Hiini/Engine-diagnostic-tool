import librosa
import numpy as np
import os

def standardize_audio(file_path, target_sr=16000, duration=10):
    """
    Loads, resamples, and adjusts the length of an audio file.
    """
    # 1. Load and Resample
    # sr=target_sr forces librosa to resample the file upon loading
    audio, sr = librosa.load(file_path, sr=target_sr)
    
    # Calculate target number of samples
    target_samples = target_sr * duration
    
    # 2. Trim if too long
    if len(audio) > target_samples:
        audio = audio[:target_samples]
        
    # 3. Pad with zeros if too short
    elif len(audio) < target_samples:
        padding = target_samples - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
        
    return audio, sr

# --- TEST SECTION (Checklist Item 3) ---

# Replace these with actual paths from your 'bearing' folder
test_normal = 'bearing/train/section_00_source_train_normal_0000_noAttribute.wav'
test_anomaly = 'bearing/test/section_00_source_test_anomaly_0000_noAttribute.wav'

for file in [test_normal, test_anomaly]:
    if os.path.exists(file):
        processed_audio, sample_rate = standardize_audio(file)
        print(f"File: {file}")
        print(f"Sample Rate: {sample_rate}Hz")
        print(f"Total Samples: {len(processed_audio)} (Expected: 160000)")
        print("-" * 30)
    else:
        print(f"Could not find {file}. Check your folder paths!")