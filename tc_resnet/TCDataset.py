import os
import random
import math
import torch
import torchaudio
import soundfile
from torch.utils.data import Dataset

class TTS_Triplet_Dataset(Dataset):
    def __init__(self, data_dir, noise_dir, target_sr=16000, max_seconds=1.0, virtual_length=5000):
        """
        Dynamically constructs Anchor, Positive, and Negative triplets.
        Trims words to a single isolated utterance using VAD, and embeds them 
        randomly into a continuous 1-second noise canvas at a target SNR.
        """
        self.target_sr = target_sr
        self.max_audio_length = int(target_sr * max_seconds) 
        self.virtual_length = virtual_length 
        self.word_to_files = {}
        self.words = []
        for word in os.listdir(data_dir):
            word_path = os.path.join(data_dir, word)
            if os.path.isdir(word_path):
                files = [os.path.join(word_path, f) for f in os.listdir(word_path) if f.endswith('.wav')]
                if len(files) >= 2: 
                    self.word_to_files[word] = files
                    self.words.append(word)
        if not self.words:
            raise ValueError(f"No valid word folders with >= 2 audio files found in {data_dir}")
        self.noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]
        if not self.noise_files:
            raise FileNotFoundError(f"No .wav files found in noise directory: {noise_dir}")

    def __load_and_resample(self, file_path: str) -> torch.Tensor:
        """Loads audio and returns a 2D tensor of shape (1, samples)."""
        data, sr = soundfile.read(str(file_path))
        y = torch.from_numpy(data).float()
        if y.ndim > 1:
            y = torch.mean(y, dim=1, keepdim=True).T
        else:
            y = y.unsqueeze(0)
        if sr != self.target_sr:
            y = torchaudio.transforms.Resample(sr, self.target_sr)(y)
        return y 

    def __get_one_second_noise_canvas(self) -> torch.Tensor:
        """Extracts or tiles a background noise segment to be exactly 1 second long."""
        noise_path = random.choice(self.noise_files)
        noise = self.__load_and_resample(noise_path)
        noise_len = noise.shape[-1]
        
        if noise_len < self.max_audio_length:
            repeats = math.ceil(self.max_audio_length / noise_len)
            noise = torch.tile(noise, (1, repeats))
            return noise[:, :self.max_audio_length]
        else:
            start_idx = random.randint(0, noise_len - self.max_audio_length)
            return noise[:, start_idx : start_idx + self.max_audio_length]

    def __prepare_audio(self, file_path: str) -> torch.Tensor:
        """
        Extracts a clean single utterance of a word and embeds it 
        randomly onto a continuous background noise frame.
        """
        clean_speech = self.__load_and_resample(file_path)
        y_trimmed = torchaudio.functional.vad(clean_speech, sample_rate=self.target_sr) 
        if y_trimmed.numel() > 0:
            clean_speech = y_trimmed
        speech_len = clean_speech.shape[-1]
        if speech_len > self.max_audio_length:
            clean_speech = clean_speech[:, :self.max_audio_length]
            speech_len = self.max_audio_length
        if random.random() < 0.2:
            final_audio = torch.zeros((1, self.max_audio_length))
            start_idx = random.randint(0, self.max_audio_length - speech_len)
            final_audio[:, start_idx : start_idx + speech_len] = clean_speech
            return final_audio.squeeze(0)
        noise_canvas = self.__get_one_second_noise_canvas()
        start_idx = random.randint(0, self.max_audio_length - speech_len)
        noise_segment = noise_canvas[:, start_idx : start_idx + speech_len]
        speech_power = torch.mean(clean_speech ** 2) + 1e-8
        noise_power = torch.mean(noise_segment ** 2) + 1e-8
        target_snr = random.uniform(-5.0, 15.0)
        target_noise_power = speech_power / (10 ** (target_snr / 10.0))
        scale_factor = torch.sqrt(target_noise_power / noise_power)
        scaled_noise_canvas = noise_canvas * scale_factor
        scaled_noise_canvas[:, start_idx : start_idx + speech_len] += clean_speech
        return torch.clamp(scaled_noise_canvas, -1.0, 1.0).squeeze(0)

    def __len__(self):
        return self.virtual_length 

    def __getitem__(self, idx):
        anchor_word = random.choice(self.words)
        anchor_file, pos_file = random.sample(self.word_to_files[anchor_word], 2)
        neg_word = random.choice(self.words)
        while neg_word == anchor_word:
            neg_word = random.choice(self.words)
        neg_file = random.choice(self.word_to_files[neg_word])
        anchor_audio = self.__prepare_audio(anchor_file)
        pos_audio = self.__prepare_audio(pos_file)
        neg_audio = self.__prepare_audio(neg_file)
        return anchor_audio, pos_audio, neg_audio