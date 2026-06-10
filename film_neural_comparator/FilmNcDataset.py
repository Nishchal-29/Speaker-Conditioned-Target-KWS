import os
import random
import math
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf
from collections import defaultdict

NOISE_DATASET = "./noise_dataset"
SPEAKER_DATASET = "./tts_data"
ES_PATH = "./speaker_embeddings"
WC_PATH = "./kws_wc"

class QuadStateDataset(Dataset):
    def __init__(self, virtual_length=5000):
        self.virtual_length = virtual_length
        self.target_sr = 16000
        self.max_audio_length = 16000
        
        self.noise_files = [os.path.join(NOISE_DATASET, f) for f in os.listdir(NOISE_DATASET) if f.endswith('.wav')] if os.path.exists(NOISE_DATASET) else []
        
        self.speakers = [f.replace('.pt', '') for f in os.listdir(ES_PATH) if f.endswith('.pt')]
        self.words = [d for d in os.listdir(SPEAKER_DATASET) if os.path.isdir(os.path.join(SPEAKER_DATASET, d))]
        
        self.index = defaultdict(lambda: defaultdict(list))
        self.all_valid_samples = [] 

        print("Indexing dataset files...")
        for word in self.words:
            word_dir = os.path.join(SPEAKER_DATASET, word)
            for f in os.listdir(word_dir):
                if f.endswith(".wav"):
                    full_path = os.path.join(word_dir, f)
                    
                    speaker_id = f.split('_')[0] 
                    
                    if speaker_id in self.speakers:
                        self.index[speaker_id][word].append(full_path)
                        
        for spk in list(self.index.keys()):
            if len(self.index[spk]) >= 2:
                for word in self.index[spk].keys():
                    for path in self.index[spk][word]:
                        self.all_valid_samples.append({
                            "speaker": spk,
                            "word": word,
                            "path": path
                        })
                        
        print(f"Dataset successfully indexed! Found {len(self.all_valid_samples)} valid anchor combinations.")
        if len(self.all_valid_samples) == 0:
            raise ValueError("CRITICAL: Zero valid samples found. Check if filename prefixes match your .pt files in speaker_embeddings!")

    def _load_and_resample(self, file_path):
        data, sr = sf.read(file_path)
        y = torch.from_numpy(data).float()
        if y.ndim > 1:
            y = torch.mean(y, dim=1, keepdim=True).T
        else:
            y = y.unsqueeze(0)
        if sr != self.target_sr:
            y = torchaudio.transforms.Resample(sr, self.target_sr)(y)
        return y

    def _prepare_audio(self, file_path):
        clean_speech = self._load_and_resample(file_path)
        peak = clean_speech.abs().max()
        if peak > 0:
            clean_speech = clean_speech / peak
        try:
            y_trimmed = torchaudio.functional.vad(clean_speech, sample_rate=self.target_sr)
            if y_trimmed.numel() > 1600:
                clean_speech = y_trimmed
        except Exception:
            pass
            
        if clean_speech.shape[1] > self.max_audio_length:
            clean_speech = clean_speech[:, :self.max_audio_length]
        elif clean_speech.shape[1] < self.max_audio_length:
            clean_speech = F.pad(clean_speech, (0, self.max_audio_length - clean_speech.shape[1]))
            
        if self.noise_files and random.random() < 0.8: 
            noise_path = random.choice(self.noise_files)
            noise_wav = self._load_and_resample(noise_path)
            if noise_wav.shape[1] > self.max_audio_length:
                start = random.randint(0, noise_wav.shape[1] - self.max_audio_length)
                noise_wav = noise_wav[:, start : start + self.max_audio_length]
            else:
                repeats = math.ceil(self.max_audio_length / noise_wav.shape[1])
                noise_wav = noise_wav.repeat(1, repeats)[:, :self.max_audio_length]
                
            speech_power = torch.mean(clean_speech ** 2) + 1e-8
            noise_power = torch.mean(noise_wav ** 2) + 1e-8
            target_snr = random.uniform(-5.0, 15.0) 
            target_noise_power = speech_power / (10 ** (target_snr / 10.0))
            scale_factor = torch.sqrt(target_noise_power / noise_power)
            clean_speech = torch.clamp(clean_speech + (noise_wav * scale_factor), -1.0, 1.0)
            
        return clean_speech.squeeze(0)

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, idx):
        anchor = random.choice(self.all_valid_samples)
        pos_speaker = anchor["speaker"]
        pos_word = anchor["word"]
        tp_path = anchor["path"]
        alt_speakers = [s for s in self.index.keys() if s != pos_speaker and pos_word in self.index[s]]
        neg_speaker = random.choice(alt_speakers) if alt_speakers else pos_speaker
        in_path = random.choice(self.index[neg_speaker][pos_word])
        alt_words = [w for w in self.index[pos_speaker].keys() if w != pos_word]
        neg_word = random.choice(alt_words)
        pd_path = random.choice(self.index[pos_speaker][neg_word])
        alt_en_speakers = [s for s in self.index.keys() if s != pos_speaker and neg_word in self.index[s]]
        en_speaker = random.choice(alt_en_speakers) if alt_en_speakers else neg_speaker
        en_path = random.choice(self.index[en_speaker][neg_word])        
        audio_tp = self._prepare_audio(tp_path)
        audio_in = self._prepare_audio(in_path)
        audio_pd = self._prepare_audio(pd_path)
        audio_en = self._prepare_audio(en_path)
        
        audio_block = torch.stack([audio_tp, audio_in, audio_pd, audio_en], dim=0)
        e_s = torch.load(os.path.join(ES_PATH, f"{pos_speaker}.pt"), map_location='cpu')
        w_c = torch.load(os.path.join(WC_PATH, f"{pos_word}.pt"), map_location='cpu')
        
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        
        return {"audio": audio_block, "e_s": e_s, "w_c": w_c, "labels": labels}
