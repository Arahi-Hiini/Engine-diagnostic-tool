import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# 1 Standardize the audio
def standardize_audio(file_path, target_sr=16000, duration=10):
    audio, sr = librosa.load(file_path, sr=target_sr)
    target_samples = target_sr * duration
    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), 'constant')
    return audio, sr

# 2 Create a Mel Spectrogram
def create_mel_spectrogram(audio, sr=16000):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr,
        n_mels=128, 
        hop_length=512, 
        n_fft=2048
    )
    
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return log_mel_spectrogram

# 3 visualize the Mel Spectrogram
def plot_comparison(normal_path, anomaly_path, save_path=None):
    audio_norm, sr_norm = standardize_audio(normal_path)
    mel_spectrogram_norm = create_mel_spectrogram(audio_norm, sr_norm)

    audio_anom, sr_anom = standardize_audio(anomaly_path)
    mel_spectrogram_anom = create_mel_spectrogram(audio_anom, sr_anom)

    # draw normal
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    img1 = librosa.display.specshow(mel_spectrogram_norm, sr=sr_norm, x_axis='time', y_axis='mel', ax=ax[0], cmap='magma')
    ax[0].set_title('Healthy Bearing (Normal)')
    fig.colorbar(img1, ax=[ax[0]], format="%+2.0f dB")

    # draw anomaly
    img2 = librosa.display.specshow(mel_spectrogram_anom, sr=sr_anom, x_axis='time', y_axis='mel', ax=ax[1], cmap='magma')
    ax[1].set_title('Faulty Bearing (Anomaly)')
    fig.colorbar(img2, ax=[ax[1]], format="%+2.0f dB")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.abspath(save_path)}", flush=True)
    plt.show()

# TEST
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    test_normal = 'bearing/train/section_00_source_train_normal_0000_noAttribute.wav'
    test_anomaly = 'bearing/test/section_00_source_test_anomaly_0000_noAttribute.wav'
    out_image = 'spectrogram_comparison.png'

    print("04_generate_spectrograms: checking for audio files...", flush=True)

    if os.path.exists(test_normal) and os.path.exists(test_anomaly):
        print("Generating visual comparison...", flush=True)
        plot_comparison(test_normal, test_anomaly, save_path=out_image)
        print("Done.", flush=True)
    else:
        missing = []
        if not os.path.exists(test_normal):
            missing.append(test_normal)
        if not os.path.exists(test_anomaly):
            missing.append(test_anomaly)
        print("Files not found:", flush=True)
        for p in missing:
            print(f"  - {os.path.abspath(p)}", flush=True)
        print("Place WAV files in bearing/train/ and bearing/test/ or update paths.", flush=True)