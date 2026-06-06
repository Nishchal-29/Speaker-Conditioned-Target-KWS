import os
import random
import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from TCResNet import KWSTrainer
import soundfile

DATASET_PATH = "./tts_data"
NOISE_PATH = "./noise_dataset"

def process_file(file_path: str, add_noise: bool = False) -> torch.Tensor:
    waveform, sr = soundfile.read(file_path)
    waveform = torch.from_numpy(waveform).float()
    
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=1, keepdim=True).T
    else:
        waveform = waveform.unsqueeze(0)    
        
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    peak = torch.max(torch.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    try:
        y_trimmed = torchaudio.functional.vad(waveform, sample_rate=16000)
        if y_trimmed.numel() > 1600:
            waveform = y_trimmed
    except Exception as e:
        pass
        
    max_len = 16000
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    elif waveform.shape[1] < max_len:
        waveform = F.pad(waveform, (0, max_len - waveform.shape[1]))
    if add_noise and os.path.exists(NOISE_PATH):
        noise_files = [os.path.join(NOISE_PATH, f) for f in os.listdir(NOISE_PATH) if f.endswith('.wav')]
        if noise_files:
            noise_file = random.choice(noise_files)
            noise_wav, n_sr = soundfile.read(noise_file)
            noise_wav=torch.from_numpy(noise_wav).float()
            if noise_wav.ndim > 1:
                noise_wav = torch.mean(noise_wav, dim=1, keepdim=True).T
            else:
                noise_wav = noise_wav.unsqueeze(0)
            if noise_wav.shape[0] > 1:
                noise_wav = torch.mean(noise_wav, dim=0, keepdim=True)
            if n_sr != 16000:
                noise_wav = torchaudio.transforms.Resample(n_sr, 16000)(noise_wav)
                
            if noise_wav.shape[1] > max_len:
                start = random.randint(0, noise_wav.shape[1] - max_len)
                noise_wav = noise_wav[:, start : start + max_len]
            else:
                noise_wav = F.pad(noise_wav, (0, max_len - noise_wav.shape[1]))
                
            speech_power = torch.mean(waveform ** 2) + 1e-8
            noise_power = torch.mean(noise_wav ** 2) + 1e-8
            target_snr = random.uniform(-5.0, 15.0)
            
            target_noise_power = speech_power / (10 ** (target_snr / 10.0))
            scale_factor = torch.sqrt(target_noise_power / noise_power)
            
            waveform = torch.clamp(waveform + (noise_wav * scale_factor), -1.0, 1.0)

    return waveform

def verify_tc_resNet():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running verification on: {device.upper()}")
    
    word1 = "accept"
    word2 = "activate"
    
    files = [os.path.join(DATASET_PATH, word1, f) for f in os.listdir(os.path.join(DATASET_PATH, word1)) if f.endswith('.wav')]
    anchor_file = files[0]
    pos_file = files[1]
    
    files = [os.path.join(DATASET_PATH, word2, f) for f in os.listdir(os.path.join(DATASET_PATH, word2)) if f.endswith('.wav')]
    neg_file = files[0]
    anchor = process_file("./accept 1.wav", add_noise=False).to(device)
    pos = process_file("./accept 2.wav", add_noise=False).to(device)
    neg = process_file(neg_file, add_noise=True).to(device)
    
    model = KWSTrainer().to(device)
    model.load_state_dict(torch.load("./checkpoints/tc_resnet_weights_ep50.pth", map_location=device))
    model.eval()    
    with torch.no_grad():
        emb_anch = model(anchor)
        emb_pos = model(pos)
        emb_neg = model(neg)
        sim_pos = F.cosine_similarity(emb_anch, emb_pos)
        sim_neg = F.cosine_similarity(emb_anch, emb_neg)

    print("\n--- Verification Results ---")
    print(f"Anchor Word:   {word1} (Clean)")
    print(f"Positive Word: {word1} (Noisy)")
    print(f"Negative Word: {word2} (Noisy)\n")
    print(f"Similarity (Anchor vs Positive): {sim_pos.item():.4f} (Expected: High, near 1.0)")
    print(f"Similarity (Anchor vs Negative): {sim_neg.item():.4f} (Expected: Low, near 0.0)")

if __name__=="__main__":
    verify_tc_resNet()