import os
import math
import random
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf

NOISE_DATASET = "./noise_dataset"
SPEAKER_DATASET = "./tts_data"
ES_PATH = "./speaker_embeddings"
WC_PATH = "./kws_wc"


def load_tensor_or_state(path: str, map_location: str = "cpu") -> torch.Tensor:
    """
    Loads an embedding saved as:
      - raw tensor
      - dict with tensor under common keys
    """
    obj = torch.load(path, map_location=map_location)

    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, dict):
        for key in ("embedding", "feat", "features", "tensor", "w_c", "e_s", "vector"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]

        tensor_vals = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if len(tensor_vals) == 1:
            return tensor_vals[0]

    raise ValueError(f"Could not extract tensor from file: {path}")


class QuadStateDataset(Dataset):
    """
    Quad-state training sample:
      1) True Positive   -> target speaker + target word
      2) Imposter Neg    -> wrong speaker + target word
      3) Phonetic Decoy  -> target speaker + wrong word
      4) Easy Negative   -> wrong speaker + wrong word
    """

    def __init__(self, virtual_length: int = 5000, target_sr: int = 16000, max_audio_length: int = 16000):
        super().__init__()
        self.virtual_length = virtual_length
        self.target_sr = target_sr
        self.max_audio_length = max_audio_length

        self.noise_files = (
            [os.path.join(NOISE_DATASET, f) for f in os.listdir(NOISE_DATASET) if f.endswith(".wav")]
            if os.path.exists(NOISE_DATASET)
            else []
        )

        self.speakers = sorted([f[:-3] for f in os.listdir(ES_PATH) if f.endswith(".pt")]) if os.path.exists(ES_PATH) else []
        self.words = sorted([d for d in os.listdir(SPEAKER_DATASET) if os.path.isdir(os.path.join(SPEAKER_DATASET, d))]) if os.path.exists(SPEAKER_DATASET) else []

        self.index: Dict[str, Dict[str, List[str]]] = {}
        self.all_valid_samples: List[Dict[str, str]] = []

        print("Indexing dataset files...")
        for word in self.words:
            word_dir = os.path.join(SPEAKER_DATASET, word)
            for fname in os.listdir(word_dir):
                if not fname.endswith(".wav"):
                    continue

                speaker_id = fname.split("_")[0]
                full_path = os.path.join(word_dir, fname)

                if speaker_id not in self.speakers:
                    continue

                self.index.setdefault(speaker_id, {}).setdefault(word, []).append(full_path)

        for spk, words_dict in self.index.items():
            if len(words_dict) >= 2:
                for word, files in words_dict.items():
                    for path in files:
                        self.all_valid_samples.append(
                            {
                                "speaker": spk,
                                "word": word,
                                "path": path,
                            }
                        )

        print(f"Dataset successfully indexed! Found {len(self.all_valid_samples)} valid anchor combinations.")
        if len(self.all_valid_samples) == 0:
            raise ValueError(
                "CRITICAL: Zero valid samples found. Check filename prefixes and the speaker_embeddings / tts_data folders."
            )

    def __len__(self) -> int:
        return self.virtual_length

    def _load_and_resample(self, file_path: str) -> torch.Tensor:
        data, sr = sf.read(file_path, always_2d=False)

        if data.ndim == 1:
            wav = torch.from_numpy(data).float().unsqueeze(0)
        else:
            wav = torch.from_numpy(data).float().mean(dim=1, keepdim=True).transpose(0, 1)

        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)

        return wav

    def _apply_vad_and_pad(self, wav: torch.Tensor) -> torch.Tensor:
        try:
            y_trimmed = torchaudio.functional.vad(wav, sample_rate=self.target_sr)
            if y_trimmed.numel() > 1600:
                wav = y_trimmed
        except Exception:
            pass

        if wav.shape[1] > self.max_audio_length:
            start = random.randint(0, wav.shape[1] - self.max_audio_length)
            wav = wav[:, start:start + self.max_audio_length]
        elif wav.shape[1] < self.max_audio_length:
            wav = F.pad(wav, (0, self.max_audio_length - wav.shape[1]))

        return wav

    def _prepare_audio(self, file_path: str) -> torch.Tensor:
        wav = self._load_and_resample(file_path)

        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak

        wav = self._apply_vad_and_pad(wav)

        if self.noise_files and random.random() < 0.8:
            noise_path = random.choice(self.noise_files)
            noise_wav = self._load_and_resample(noise_path)

            if noise_wav.shape[1] > self.max_audio_length:
                start = random.randint(0, noise_wav.shape[1] - self.max_audio_length)
                noise_wav = noise_wav[:, start:start + self.max_audio_length]
            else:
                repeats = math.ceil(self.max_audio_length / noise_wav.shape[1])
                noise_wav = noise_wav.repeat(1, repeats)[:, :self.max_audio_length]

            speech_power = torch.mean(wav ** 2) + 1e-8
            noise_power = torch.mean(noise_wav ** 2) + 1e-8
            target_snr = random.uniform(-5.0, 15.0)
            target_noise_power = speech_power / (10 ** (target_snr / 10.0))
            scale_factor = torch.sqrt(target_noise_power / noise_power)

            wav = torch.clamp(wav + noise_wav * scale_factor, -1.0, 1.0)

        return wav.squeeze(0)

    def _load_embedding(self, path: str, expected_dim: Optional[int] = None) -> torch.Tensor:
        emb = load_tensor_or_state(path, map_location="cpu").float().squeeze()

        if emb.ndim != 1:
            emb = emb.view(-1)

        if expected_dim is not None and emb.numel() != expected_dim:
            raise ValueError(f"Embedding at {path} has dim {emb.numel()}, expected {expected_dim}")

        return F.normalize(emb, p=2, dim=0)

    def _sample_other_speaker_same_word(self, speaker: str, word: str) -> Optional[Tuple[str, str]]:
        candidates = [s for s in self.index.keys() if s != speaker and word in self.index[s]]
        if not candidates:
            return None
        spk = random.choice(candidates)
        return spk, random.choice(self.index[spk][word])

    def _sample_other_word_same_speaker(self, speaker: str, word: str) -> Optional[Tuple[str, str]]:
        candidates = [w for w in self.index.get(speaker, {}).keys() if w != word]
        if not candidates:
            return None
        w = random.choice(candidates)
        return w, random.choice(self.index[speaker][w])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor = self.all_valid_samples[idx % len(self.all_valid_samples)]
        pos_speaker = anchor["speaker"]
        pos_word = anchor["word"]
        tp_path = anchor["path"]

        for _ in range(30):
            imposter = self._sample_other_speaker_same_word(pos_speaker, pos_word)
            decoy = self._sample_other_word_same_speaker(pos_speaker, pos_word)

            if imposter is None or decoy is None:
                anchor = random.choice(self.all_valid_samples)
                pos_speaker = anchor["speaker"]
                pos_word = anchor["word"]
                tp_path = anchor["path"]
                continue

            neg_speaker_1, in_path = imposter
            neg_word_2, pd_path = decoy

            easy_candidates = [s for s in self.index.keys() if s != pos_speaker and neg_word_2 in self.index[s]]
            if not easy_candidates:
                anchor = random.choice(self.all_valid_samples)
                pos_speaker = anchor["speaker"]
                pos_word = anchor["word"]
                tp_path = anchor["path"]
                continue

            en_speaker = random.choice(easy_candidates)
            en_path = random.choice(self.index[en_speaker][neg_word_2])

            audio_tp = self._prepare_audio(tp_path)
            audio_in = self._prepare_audio(in_path)
            audio_pd = self._prepare_audio(pd_path)
            audio_en = self._prepare_audio(en_path)

            audio_block = torch.stack([audio_tp, audio_in, audio_pd, audio_en], dim=0)

            e_s = self._load_embedding(os.path.join(ES_PATH, f"{pos_speaker}.pt"), expected_dim=192)
            w_c = self._load_embedding(os.path.join(WC_PATH, f"{pos_word}.pt"), expected_dim=128)

            labels = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

            return {
                "audio": audio_block,
                "e_s": e_s,
                "w_c": w_c,
                "labels": labels,
            }

        raise RuntimeError("Failed to construct a valid quad-state sample after many retries.")
