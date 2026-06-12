import os
import json
import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import onnxruntime as ort
from TCResNet import KWSTrainer

WAV_FILE = "./accept 1.wav"
WORD_NAME = "accept"
SPEAKER_NAME = "accept_speaker"

WC_PATH = "."
ES_PATH = "."
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Executing profile extraction on: {DEVICE.upper()}")

# --- TRACK 1: SPEAKER EMBEDDING (e_s) COMPONENTS ---
class learnablePCEN(nn.Module):
    def __init__(self, n_mels=80, sample_rate=16000, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.n_mels = n_mels
        win_length = 480
        hop_length = 160
        n_fft = 512
        self.mel_spectogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, n_mels=n_mels, power=2.0
        )
        self.alpha_logit = nn.Parameter(torch.empty(1, n_mels, 1).fill_(math.log(0.98/(1-0.98))))
        self.log_delta = nn.Parameter(torch.empty(1, n_mels, 1).fill_(math.log(2.0)))
        self.r_logit = nn.Parameter(torch.empty(1, n_mels, 1).fill_(math.log(0.5/(1-0.5))))
        self.s_logit = nn.Parameter(torch.empty(1, n_mels, 1).fill_(math.log(0.25/(1-0.25))))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_spectogram(waveforms)
        alpha = torch.sigmoid(self.alpha_logit)
        delta = torch.exp(self.log_delta)
        r = torch.sigmoid(self.r_logit)
        s = torch.sigmoid(self.s_logit)
        batch_size, n_mels, time_steps = mel_spec.shape
        s_2d = s.squeeze(-1) 
        M = [mel_spec[:, :, 0]]
        for i in range(1, time_steps):
            m_t = (1 - s_2d) * M[-1] + s_2d * mel_spec[:, :, i]
            M.append(m_t)
        M = torch.stack(M, dim=-1)
        smoth_denom = (self.eps + M) ** alpha
        pcen = ((mel_spec + self.eps) / (smoth_denom + delta)) ** r - delta ** r
        return pcen

# Initialize PCEN
pcen_layer = learnablePCEN().to(DEVICE)
if os.path.exists("pcen_params.json"):
    with open("pcen_params.json", "r") as f:
        params = json.load(f)
        with torch.no_grad():
            pcen_layer.alpha_logit.copy_(torch.tensor(params['alpha']).view(1, -1, 1))
            pcen_layer.log_delta.copy_(torch.tensor(params['delta']).view(1, -1, 1))
            pcen_layer.r_logit.copy_(torch.tensor(params['r']).view(1, -1, 1))
            pcen_layer.s_logit.copy_(torch.tensor(params['s']).view(1, -1, 1))
pcen_layer.eval()

providers = ['CUDAExecutionProvider'] if DEVICE == 'cuda' else ['CPUExecutionProvider']
ort_session = ort.InferenceSession("ecapa_backbone.onnx", providers=providers)
input_name = ort_session.get_inputs()[0].name

# --- COMMON AUDIO LOADING ---
data, sr = sf.read(WAV_FILE)
raw_waveform = torch.from_numpy(data).float()
if raw_waveform.ndim > 1:
    raw_waveform = raw_waveform.mean(dim=1).unsqueeze(0)
else:
    raw_waveform = raw_waveform.unsqueeze(0)

if sr != 16000:
    raw_waveform = torchaudio.transforms.Resample(sr, 16000)(raw_waveform)

peak = raw_waveform.abs().max()
if peak > 0:
    raw_waveform = raw_waveform / peak


# --- EXTRACT SPEAKER EMBEDDING (e_s) ---
print("Extracting Biometric Speaker Profile (e_s)...")
try:
    front_trimmed = torchaudio.functional.vad(raw_waveform, sample_rate=16000)
    reversed_wav = torch.flip(front_trimmed, dims=[1])
    back_trimmed = torchaudio.functional.vad(reversed_wav, sample_rate=16000)
    pure_speech = torch.flip(back_trimmed, dims=[1])
    es_wave = pure_speech if pure_speech.numel() > 1600 else raw_waveform
except Exception:
    es_wave = raw_waveform

target_len_es = 48000
current_len_es = es_wave.shape[1]
if current_len_es > target_len_es:
    es_wave = es_wave[:, :target_len_es]
elif current_len_es < target_len_es:
    repeats = math.ceil(target_len_es / current_len_es)
    es_wave = es_wave.repeat(1, repeats)[:, :target_len_es]

es_wave = es_wave.to(DEVICE)
with torch.no_grad():
    pcen_features = pcen_layer(es_wave)
    pcen_np = pcen_features.cpu().numpy()
    ort_outs = ort_session.run(None, {input_name: pcen_np})
    emb_es = torch.from_numpy(ort_outs[0]).to(DEVICE)
    e_s = F.normalize(emb_es, p=2, dim=-1).squeeze(0).cpu()

save_path_es = os.path.join(ES_PATH, f"{SPEAKER_NAME}.pt")
torch.save(e_s, save_path_es)
print(f"-> Saved Speaker Embedding: {save_path_es} | Shape: {e_s.shape}")


print("Extracting Keyword Target Profile (w_c)...")
model = KWSTrainer().to(DEVICE)
# model.load_state_dict(torch.load("./checkpoints/tc_resnet_weights_ep50.pth", map_location=DEVICE))
model.load_state_dict(torch.load("./Film_checkpoints/tuned_backbone_ep30.pth", map_location=DEVICE))
model.eval()

try:
    y_trimmed = torchaudio.functional.vad(raw_waveform, sample_rate=16000)
    wc_wave = y_trimmed if y_trimmed.numel() > 1600 else raw_waveform
except Exception:
    wc_wave = raw_waveform

target_len_wc = 16000
current_len_wc = wc_wave.shape[1]
if current_len_wc > target_len_wc:
    wc_wave = wc_wave[:, :target_len_wc]
elif current_len_wc < target_len_wc:
    wc_wave = F.pad(wc_wave, (0, target_len_wc - current_len_wc))

wc_wave = wc_wave.to(DEVICE)
with torch.no_grad():
    emb_wc = model(wc_wave)
    w_c = F.normalize(emb_wc, p=2, dim=-1).squeeze(0).cpu()

save_path_wc = os.path.join(WC_PATH, f"{WORD_NAME}.pt")
torch.save(w_c, save_path_wc)
print(f"-> Saved Keyword Template: {save_path_wc} | Shape: {w_c.shape}")
print("Initialization Complete.")