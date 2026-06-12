import torch
import torch.nn as nn
import soundfile as sf
import torchaudio
import torch.nn.functional as F
import os
from TCResNet import KWSTrainer

# --- CONFIGURATION ---
LIVE_AUDIO = "./accept 1.wav"

WC_PROFILE = "./accept.pt"
ES_PROFILE = "./accept_speaker.pt"

# Trained Weight References
# BACKBONE_WEIGHTS = "./checkpoints/tc_resnet_weights_ep50.pth"
BACKBONE_WEIGHTS = "./Film_checkpoints/tuned_backbone_ep30.pth"
FILM_WEIGHTS = "./Film_checkpoints/film_ep30.pth"
COMP_WEIGHTS = "./Film_checkpoints/comparator_ep30.pth"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- VALIDATOR NETWORK STRUCTURE ---
class FiLM(nn.Module):
    def __init__(self, speaker_dim=192, target_channels=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(speaker_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, target_channels * 2)
        )
        self.target_channels = target_channels

    def forward(self, e_s):
        out = self.fc(e_s)
        gamma = out[:, :self.target_channels]
        beta = out[:, self.target_channels:]
        return gamma, beta
    
class NeuralComparator(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(256, 128),
            nn.LayerNorm(128), 
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, w_c, w_live):
        diff = torch.abs(w_c - w_live)
        mul = w_c * w_live
        vec = torch.cat([w_c, w_live, diff, mul], dim=1)
        return self.net(vec)
    
class Validator(nn.Module):
    def __init__(self, tc_resnet_backbone):
        super().__init__()
        self.film = FiLM(speaker_dim=192, target_channels=128)
        self.n_comparator = NeuralComparator(input_dim=512)
        self.backbone = tc_resnet_backbone

    def forward(self, audio, w_c, e_s):
        wc_raw = self.backbone(audio)
        wc_raw = F.normalize(wc_raw, p=2, dim=-1) 
        
        gamma, beta = self.film(e_s)
        w_live = (gamma * wc_raw) + beta
        w_live = F.normalize(w_live, p=2, dim=-1) 
        
        pred = self.n_comparator(w_c, w_live)
        return pred

def evaluate_live_entry():
    print(f"Loading Security Profiles from Disk...")
    # Add batch axis [1, Dim] to match network requirements
    if not os.path.exists(WC_PROFILE) or not os.path.exists(ES_PROFILE):
        raise FileNotFoundError("Keys not found. Please run save_profiles.py first!")
        
    w_c = torch.load(WC_PROFILE, map_location=DEVICE).unsqueeze(0)
    e_s = torch.load(ES_PROFILE, map_location=DEVICE).unsqueeze(0)

    print("Building Model Pipeline Architecture...")
    tc_resnet = KWSTrainer().to(DEVICE)
    tc_resnet.load_state_dict(torch.load(BACKBONE_WEIGHTS, map_location=DEVICE))
    
    model = Validator(tc_resnet_backbone=tc_resnet).to(DEVICE)
    model.film.load_state_dict(torch.load(FILM_WEIGHTS, map_location=DEVICE))
    model.n_comparator.load_state_dict(torch.load(COMP_WEIGHTS, map_location=DEVICE))
    model.eval()

    print(f"Loading and Preprocessing Live Audio Stream: {LIVE_AUDIO}")
    data, sr = sf.read(LIVE_AUDIO)
    waveform = torch.from_numpy(data).float()

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1).unsqueeze(0)
    else:
        waveform = waveform.unsqueeze(0)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    try:
        y_trimmed = torchaudio.functional.vad(waveform, sample_rate=16000)
        if y_trimmed.numel() > 1600:
            waveform = y_trimmed
    except Exception:
        pass

    target_len = 16000
    current_len = waveform.shape[1]
    if current_len > target_len:
        waveform = waveform[:, :target_len]
    elif current_len < target_len:
        waveform = F.pad(waveform, (0, target_len - current_len))

    waveform = waveform.to(DEVICE)

    print("Executing Multi-Verification Scoring...")
    print("Executing Multi-Verification Scoring...")
    with torch.no_grad():
        logits = model(waveform, w_c, e_s)
        confidence_score = torch.sigmoid(logits).item()

    print("\n" + "="*50)
    print(f"SYSTEM VERIFICATION CONFIDENCE: {confidence_score * 100:.2f}%")
    if confidence_score >= 0.50:
        print("MATCH VERIFIED: Access Granted.")
    else:
        print("ANOMALY DETECTED: Access Denied.")
    print("="*50)

if __name__ == "__main__":
    evaluate_live_entry()