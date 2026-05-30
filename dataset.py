# import os
# import glob
# import torch
# import librosa
# import numpy as np
# from torch.utils.data import Dataset

# class NoisySpeakerPCENDataset(Dataset):
#     def __init__(self, data_dir, n_mels=80, max_frames=300, file_ext="flac", target_snr=-5):
#         """
#         Loads clean audio, injects noise on-the-fly to hit a strict target SNR, 
#         and extracts PCEN features.
#         """
#         self.files = glob.glob(f"{data_dir}/**/*.{file_ext}", recursive=True)
#         if len(self.files) == 0:
#             raise ValueError(f"No .{file_ext} files found in {data_dir}")
            
#         self.speakers = sorted(list(set([path.split(os.sep)[-3] for path in self.files])))
#         self.spk_to_id = {spk: i for i, spk in enumerate(self.speakers)}
        
#         self.n_mels = n_mels
#         self.max_frames = max_frames
#         self.target_snr = target_snr
        
#         print(f"Initialized dataset with {len(self.files)} files.")
#         print(f"Targeting a noisy floor of {self.target_snr}dB SNR.")

#     def inject_noise(self, y):
#         """Calculates audio power and injects noise to hit the exact target SNR."""
#         signal_power = np.mean(y ** 2)
#         if signal_power == 0:
#             return y  
            
#         noise_power = signal_power / (10 ** (self.target_snr / 10))
#         noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        
#         return y + noise

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         file_path = self.files[idx]
#         spk_name = file_path.split(os.sep)[-3]
#         label = self.spk_to_id[spk_name]
        
#         # Load and corrupt audio
#         y, sr = librosa.load(file_path, sr=16000)
#         y_noisy = self.inject_noise(y)
        
#         # Compute Mel and apply PCEN
#         mel = librosa.feature.melspectrogram(y=y_noisy, sr=sr, n_mels=self.n_mels, hop_length=256)
#         pcen = librosa.pcen(S=mel * (2**31), sr=sr, hop_length=256).T 
        
#         # Pad or truncate for uniform batch sizing
#         if pcen.shape[0] < self.max_frames:
#             pad_width = self.max_frames - pcen.shape[0]
#             pcen = np.pad(pcen, ((0, pad_width), (0, 0)), mode='constant')
#         else:
#             pcen = pcen[:self.max_frames, :]
            
#         return torch.FloatTensor(pcen), torch.tensor(label, dtype=torch.long)

import os
import glob
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class NoisySpeakerRawDataset(Dataset):
    def __init__(self, data_dir, max_audio_length=48000, file_ext="flac"):
        """
        Loads clean audio, strips dead silence, injects DYNAMIC noise, 
        and loops (tiles) short audio to return robust RAW waveforms.
        max_audio_length=48000 equals exactly 3 seconds of audio at 16kHz.
        """
        self.files = glob.glob(f"{data_dir}/**/*.{file_ext}", recursive=True)
        if len(self.files) == 0:
            raise ValueError(f"No .{file_ext} files found in {data_dir}")
            
        self.speakers = sorted(list(set([path.split(os.sep)[-3] for path in self.files])))
        self.spk_to_id = {spk: i for i, spk in enumerate(self.speakers)}
        
        self.max_audio_length = max_audio_length
        
        print(f"Initialized dataset with {len(self.files)} files.")
        print("Using dynamic SNR augmentation, silence trimming, and audio tiling.")

    def inject_noise(self, y):
        """Injects a random level of noise to prevent shortcut learning."""
        # Randomly choose an SNR for this specific clip. 
        # None = Clean. Lower numbers = more noise.
        # 40% Clean, 30% Light Noise, 30% Heavy Noise
        target_snr = np.random.choice(
            [None, 20, 15, 10, 5, 0, -5], 
            p=[0.40, 0.15, 0.15, 0.10, 0.10, 0.05, 0.05]
        )
        
        if target_snr is None:
            return y
            
        signal_power = np.mean(y ** 2)
        if signal_power == 0:
            return y  
            
        noise_power = signal_power / (10 ** (target_snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        y_noisy = y + noise
        
        # FIX: Prevent clipping by restricting values to standard float audio range
        y_noisy = np.clip(y_noisy, -1.0, 1.0) 
        
        return y_noisy

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        spk_name = file_path.split(os.sep)[-3]
        label = self.spk_to_id[spk_name]
        
        # 1. Load audio
        y, sr = librosa.load(file_path, sr=16000)
        
        # 1.5. Strip dead silence (Prevent the model from learning dead air)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Safety check: If the file was 100% silence and got completely trimmed, fallback to original
        if len(y_trimmed) > 0:
            y = y_trimmed
            
        # 2. Inject random noise
        y_noisy = self.inject_noise(y)
        
        # 3. Tile (loop) or truncate RAW AUDIO for uniform batching
        if len(y_noisy) < self.max_audio_length:
            # FIX: Loop the audio to fill 3 seconds (e.g., "Hello" -> "Hello Hello Hello")
            repeats = int(np.ceil(self.max_audio_length / len(y_noisy)))
            y_noisy = np.tile(y_noisy, repeats)[:self.max_audio_length]
        else:
            y_noisy = y_noisy[:self.max_audio_length]
            
        # Return the 1D raw audio tensor
        return torch.FloatTensor(y_noisy), torch.tensor(label, dtype=torch.long)